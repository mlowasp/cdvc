import os
import math
from typing import Dict, List, Tuple

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_safe import SafeVideoDatasetTwoStream, compute_pos_weight
from model_twostream_i3d import TwoStreamI3D
from losses import BinaryFocalLoss
from utils import seed_everything, ensure_dir, AverageMeter, compute_metrics_from_preds, save_checkpoint, load_checkpoint


def build_scheduler(optimizer, cfg: Dict, steps_per_epoch: int):
    name = cfg.get("name", "none")
    if name == "none":
        return None

    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", 8)),
            gamma=float(cfg.get("gamma", 0.1)),
        )

    if name == "cosine":
        warmup_epochs = int(cfg.get("warmup_epochs", 1))
        total_epochs = int(cfg["_total_epochs"])
        total_steps = total_epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            prog = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(f"Unknown scheduler: {name}")


@torch.no_grad()
def eval_per_video_aggregation(
    model: nn.Module,
    ds: SafeVideoDatasetTwoStream,
    device: str,
    clips_per_video: int,
    aggregation: str,
    threshold: float,
    batch_size: int,
    num_workers: int,
):
    model.eval()

    pairs: List[Tuple[int, int]] = []
    for vidx in range(len(ds)):
        for c in range(clips_per_video):
            pairs.append((vidx, c))

    class PairDataset(torch.utils.data.Dataset):
        def __init__(self, base_ds, pairs_):
            self.base = base_ds
            self.pairs = pairs_

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, i):
            vidx, cid = self.pairs[i]
            rgb, flow, label, path = self.base.get_clip(vidx, clip_key=f"eval_{cid}")
            return rgb, flow, label, path

    pair_ds = PairDataset(ds, pairs)
    loader = DataLoader(
        pair_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=1,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False,
    )

    by_path = {}
    for rgb, flow, labels, paths in loader:
        rgb = rgb.to(device, non_blocking=True)
        flow = flow.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        logits = model(rgb, flow).clamp(-10, 10)
        probs = torch.sigmoid(logits)

        for pth, prob, logit, lab in zip(paths, probs.detach().cpu(), logits.detach().cpu(), labels.detach().cpu()):
            pth = str(pth)
            if pth not in by_path:
                by_path[pth] = {"probs": [], "logits": [], "label": int(lab.item())}
            by_path[pth]["probs"].append(float(prob.item()))
            by_path[pth]["logits"].append(float(logit.item()))

    agg_probs = []
    agg_labels = []
    for pth, rec in by_path.items():
        if aggregation == "mean_logit":
            mlog = sum(rec["logits"]) / max(1, len(rec["logits"]))
            mprob = float(torch.sigmoid(torch.tensor(mlog)).item())
        else:
            mprob = sum(rec["probs"]) / max(1, len(rec["probs"]))
        agg_probs.append(mprob)
        agg_labels.append(rec["label"])

    agg_probs_t = torch.tensor(agg_probs)
    agg_labels_t = torch.tensor(agg_labels).long()
    agg_preds_t = (agg_probs_t >= float(threshold)).long()

    metrics = compute_metrics_from_preds(agg_preds_t, agg_labels_t)

    # reporting loss only
    bce = torch.nn.BCELoss()
    rep_loss = float(bce(agg_probs_t.clamp(1e-6, 1-1e-6), agg_labels_t.float()).item())

    return rep_loss, metrics


def main(config_path: str = "config.yaml"):
    cfg = yaml.safe_load(open(config_path, "r"))

    seed_everything(int(cfg["seed"]))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    save_dir = cfg["train"]["save_dir"]
    ensure_dir(save_dir)

    root = cfg["data"]["root"]
    train_dir = os.path.join(root, cfg["data"]["train_split"])
    val_dir = os.path.join(root, cfg["data"]["val_split"])
    class_names = cfg["data"]["class_names"]

    flow_cfg = cfg.get("flow", {})

    ds_train = SafeVideoDatasetTwoStream(
        root_split_dir=train_dir,
        class_names=class_names,
        num_frames=int(cfg["video"]["num_frames"]),
        sampling=str(cfg["video"]["sampling_train"]),
        resize_short=int(cfg["video"]["resize_short"]),
        crop_size=int(cfg["video"]["crop_size"]),
        fixed_seed=None,
        flow_cfg=flow_cfg,
    )

    ds_val = SafeVideoDatasetTwoStream(
        root_split_dir=val_dir,
        class_names=class_names,
        num_frames=int(cfg["video"]["num_frames"]),
        sampling="random",
        resize_short=int(cfg["video"]["resize_short"]),
        crop_size=int(cfg["video"]["crop_size"]),
        fixed_seed=12345,
        flow_cfg=flow_cfg,
    )

    pw = compute_pos_weight(ds_train.samples)
    print(f"[imbalance] Using pos_weight={pw:.4f} (neg/pos)")

    model = TwoStreamI3D(
        rgb_weights_path=str(cfg["model"]["rgb_weights"]),
        fusion=str(cfg["model"]["fusion"]),
        fusion_alpha=float(cfg["model"].get("fusion_alpha", 0.5)),
    ).to(device)

    # freeze/unfreeze scheduling
    freeze_epochs = int(cfg["model"].get("freeze_backbones_epochs", 0))
    unfreeze_rgb = bool(cfg["model"].get("unfreeze_rgb", True))
    unfreeze_flow = bool(cfg["model"].get("unfreeze_flow", True))

    # start frozen if requested
    if freeze_epochs > 0:
        model.set_backbone_trainable(False, False)

    criterion = BinaryFocalLoss(
        alpha=float(cfg["focal_loss"]["alpha"]),
        gamma=float(cfg["focal_loss"]["gamma"]),
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        prefetch_factor=1,
        pin_memory=True,
        persistent_workers=True if int(cfg["num_workers"]) > 0 else False,
        drop_last=True,
    )

    cfg["scheduler"]["_total_epochs"] = int(cfg["train"]["epochs"])
    scheduler = build_scheduler(optimizer, cfg["scheduler"], steps_per_epoch=len(train_loader))

    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"].get("amp", True)))

    start_epoch = 0
    best_f1 = -1.0

    resume_path = cfg["train"].get("resume", None)
    if resume_path:
        ckpt = load_checkpoint(resume_path, device=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_f1 = float(ckpt.get("best_f1", best_f1))
        print(f"[resume] Loaded checkpoint from {resume_path} at epoch {start_epoch}")

    log_every = int(cfg["train"]["log_every"])
    grad_clip = float(cfg["train"]["grad_clip"])

    for epoch in range(start_epoch, int(cfg["train"]["epochs"])):

        # Unfreeze after N epochs
        if freeze_epochs > 0 and epoch == freeze_epochs:
            model.set_backbone_trainable(unfreeze_rgb, unfreeze_flow)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=float(cfg["train"]["lr"]) * 0.5,      # safer after unfreeze
                weight_decay=float(cfg["train"]["weight_decay"]),
            )
            scheduler = build_scheduler(optimizer, cfg["scheduler"], steps_per_epoch=len(train_loader))
            print(f"[train] Unfroze backbones at epoch {epoch+1}")

        model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}", dynamic_ncols=True)

        for step, (rgb, flow, labels, _paths) in enumerate(pbar):
            rgb = rgb.to(device, non_blocking=True)
            flow = flow.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=bool(cfg["train"].get("amp", True))):
                logits = model(rgb, flow).clamp(-10, 10)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            loss_meter.update(loss.item(), n=int(labels.size(0)))

            if (step + 1) % log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "lr": f"{lr:.2e}"})

        val_loss, val_metrics = eval_per_video_aggregation(
            model=model,
            ds=ds_val,
            device=device,
            clips_per_video=int(cfg["eval"]["clips_per_video"]),
            aggregation=str(cfg["eval"]["aggregation"]),
            threshold=float(cfg["eval"]["threshold"]),
            batch_size=max(2, int(cfg["train"]["batch_size"]) * 2),
            num_workers=min(8, int(cfg["num_workers"])),
        )

        print(f"[val/video-agg] loss={val_loss:.4f} acc={val_metrics['acc']:.3f} "
              f"prec={val_metrics['precision']:.3f} rec={val_metrics['recall']:.3f} f1={val_metrics['f1']:.3f}")

        last_path = os.path.join(save_dir, "last.pt")
        save_checkpoint(last_path, {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(),
            "best_f1": best_f1,
            "config": cfg,
        })

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_path = os.path.join(save_dir, "best.pt")
            save_checkpoint(best_path, {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict(),
                "best_f1": best_f1,
                "config": cfg,
                "val_metrics": val_metrics,
            })
            print(f"[save] New best f1={best_f1:.4f} -> {best_path}")

    print(f"Done. Best f1={best_f1:.4f}. Checkpoints in: {save_dir}")


if __name__ == "__main__":
    main("config.yaml")

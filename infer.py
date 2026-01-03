import sys
import yaml
import torch

from dataset_safe import SafeVideoDataset
from model import CNNLSTMVideoClassifier
from utils import load_checkpoint


@torch.no_grad()
def infer_video(video_path: str, ckpt_path: str, config_path: str = "config.yaml"):
    cfg = yaml.safe_load(open(config_path, "r"))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(ckpt_path, device=device)
    model = CNNLSTMVideoClassifier(
        encoder_name="resnet18",
        pretrained=False,
        lstm_hidden=128,
        lstm_layers=1,
        dropout=0.4,
        unfreeze_layer4=True,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Build a tiny dataset object just for consistent preprocessing
    tmp = SafeVideoDataset(
        root_split_dir=".",  # unused
        class_names=["not_controlled", "controlled"],
        num_frames=int(cfg["video"]["num_frames"]),
        sampling="random",
        resize_short=int(cfg["video"]["resize_short"]),
        crop_size=int(cfg["video"]["crop_size"]),
        fixed_seed=12345,
    )
    # hack: replace samples with this one path
    tmp.samples = [type(tmp.samples[0]) (video_path, 0)] if tmp.samples else [type("S", (), {"path": video_path, "label": 0})]

    K = int(cfg["eval"]["clips_per_video"])
    aggregation = str(cfg["eval"]["aggregation"])

    probs = []
    logits_list = []
    for i in range(K):
        frames, _label, _pth = tmp.get_clip(0, clip_key=f"infer_{i}")
        frames = frames.unsqueeze(0).to(device)
        logit = model(frames).clamp(-10, 10)[0].item()
        prob = float(torch.sigmoid(torch.tensor(logit)).item())
        probs.append(prob)
        logits_list.append(logit)

    if aggregation == "mean_logit":
        mlog = sum(logits_list) / max(1, len(logits_list))
        mprob = float(torch.sigmoid(torch.tensor(mlog)).item())
    else:
        mprob = sum(probs) / max(1, len(probs))

    return mprob


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python infer.py /path/to/video.mp4 runs/demolition_cls/best.pt")
        sys.exit(1)

    prob = infer_video(sys.argv[1], sys.argv[2], "config.yaml")
    print(f"controlled_demolition_probability={prob:.4f}")

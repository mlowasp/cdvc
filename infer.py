import sys
import yaml
import torch

from dataset_safe import SingleVideoInferTwoStream
from model_twostream_i3d import TwoStreamI3D
from utils import load_checkpoint


@torch.no_grad()
def infer_video(video_path: str, ckpt_path: str, config_path: str = "config.yaml"):
    cfg = yaml.safe_load(open(config_path, "r"))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(ckpt_path, device=device)

    model = TwoStreamI3D(
        rgb_weights_path=str(cfg["model"]["rgb_weights"]),
        fusion=str(cfg["model"]["fusion"]),
        fusion_alpha=float(cfg["model"].get("fusion_alpha", 0.5)),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    infer_ds = SingleVideoInferTwoStream(
        video_path=video_path,
        num_frames=int(cfg["video"]["num_frames"]),
        resize_short=int(cfg["video"]["resize_short"]),
        crop_size=int(cfg["video"]["crop_size"]),
        fixed_seed=12345,
        flow_cfg=cfg.get("flow", {}),
    )

    K = int(cfg["eval"]["clips_per_video"])
    aggregation = cfg["eval"].get("aggregation", "mean_prob")
    threshold = float(cfg["eval"].get("threshold", 0.5))

    probs = []
    logits_list = []

    for i in range(K):
        rgb, flow, _label, _pth = infer_ds.get_clip(clip_key=f"infer_{i}")
        rgb = rgb.unsqueeze(0).to(device)
        flow = flow.unsqueeze(0).to(device)

        logit = model(rgb, flow).clamp(-10, 10)[0].item()
        prob = float(torch.sigmoid(torch.tensor(logit)).item())

        logits_list.append(logit)
        probs.append(prob)

    if aggregation == "mean_logit":
        mean_logit = sum(logits_list) / len(logits_list)
        final_prob = float(torch.sigmoid(torch.tensor(mean_logit)).item())
    else:
        final_prob = sum(probs) / len(probs)

    pred = "controlled" if final_prob >= threshold else "not_controlled"
    return final_prob, pred


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 infer.py /path/to/video.mp4 runs/demolition_cls_twostream/best.pt")
        sys.exit(1)

    prob, pred = infer_video(sys.argv[1], sys.argv[2], "config.yaml")
    print(f"prediction: {pred}")
    print(f"controlled_demolition_probability: {prob:.4f}")

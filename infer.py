import sys
import yaml
import torch

from dataset_safe import SingleVideoInfer
from model import CNNLSTMVideoClassifier
from utils import load_checkpoint


@torch.no_grad()
def infer_video(video_path: str, ckpt_path: str, config_path: str = "config.yaml"):
    cfg = yaml.safe_load(open(config_path, "r"))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(ckpt_path, device=device)
    state = ckpt["model"]

    # ------------------------------------------------------------
    # 🔒 DERIVE ARCHITECTURE FROM CHECKPOINT (IMPOSSIBLE TO MISMATCH)
    # ------------------------------------------------------------
    # lstm.weight_ih_l0 has shape [4*H, input_dim]
    lstm_w = state["lstm.weight_ih_l0"]
    lstm_hidden = lstm_w.shape[0] // 4

    # head.weight has shape [1, H]
    head_hidden = state["head.weight"].shape[1]

    assert lstm_hidden == head_hidden, "Checkpoint is inconsistent!"

    print(f"[infer] Detected LSTM hidden size = {lstm_hidden}")

    model = CNNLSTMVideoClassifier(
        encoder_name="resnet18",
        pretrained=False,
        lstm_hidden=lstm_hidden,
        lstm_layers=1,
        dropout=0.5,
        unfreeze_layer4=False,
    ).to(device)

    model.load_state_dict(state)
    model.eval()

    # ------------------------------------------------------------
    # Single-video inference helper
    # ------------------------------------------------------------
    infer_ds = SingleVideoInfer(
        video_path=video_path,
        num_frames=int(cfg["video"]["num_frames"]),
        resize_short=int(cfg["video"]["resize_short"]),
        crop_size=int(cfg["video"]["crop_size"]),
        fixed_seed=12345,
    )

    K = int(cfg["eval"]["clips_per_video"])
    aggregation = cfg["eval"].get("aggregation", "mean_prob")
    threshold = float(cfg["eval"].get("threshold", 0.5))

    probs = []
    logits = []

    for i in range(K):
        frames, _, _ = infer_ds.get_clip(clip_key=f"infer_{i}")
        frames = frames.unsqueeze(0).to(device)

        logit = model(frames).clamp(-10, 10)[0].item()
        prob = torch.sigmoid(torch.tensor(logit)).item()

        logits.append(logit)
        probs.append(prob)

    if aggregation == "mean_logit":
        mean_logit = sum(logits) / len(logits)
        final_prob = torch.sigmoid(torch.tensor(mean_logit)).item()
    else:
        final_prob = sum(probs) / len(probs)

    pred = "controlled" if final_prob >= threshold else "not_controlled"
    return final_prob, pred


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 infer.py /path/video.mp4 runs/demolition_cls/best.pt")
        sys.exit(1)

    prob, pred = infer_video(sys.argv[1], sys.argv[2], "config.yaml")
    print(f"prediction: {pred} demolition")
    print(f"controlled_demolition_probability: {prob:.4f}")

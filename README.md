# Controlled Demolition Video Classifier

A **PyTorch-based video classification pipeline** that detects whether a video depicts a **controlled building demolition** (implosion) or **not** (fire collapse, earthquake, accident, partial failure, CGI, etc.).

The model uses a **CNN + LSTM temporal architecture**:
- A **2D CNN** (ResNet) encodes individual frames
- An **LSTM** models temporal collapse dynamics
- The output is a **binary probability** indicating controlled demolition

This repository is designed to be:
- 🔁 **Reproducible** (config-driven)
- 🚀 **GPU-ready** (AMP, checkpointing)
- 🧱 **Extendable** (optical flow, audio, transformers)
- ⚖️ **Robust to class imbalance**

---

## Problem Definition

**Binary classification**

| Label | Meaning |
|------|--------|
| `1` | Controlled demolition (engineered implosion) |
| `0` | Not controlled (fire, earthquake, accidental failure, CGI, etc.) |

The classifier learns **temporal collapse patterns**, not just explosions:
- Vertical symmetry
- Near-simultaneous structural failure
- Collapse into footprint
- Consistent downward velocity

---

## Project Structure

```
.
├── dataset.py        # Video loading & preprocessing
├── model.py          # CNN + LSTM architecture
├── train.py          # Training & validation loop
├── infer.py          # Single-video inference
├── utils.py          # Metrics, seeding, checkpoints
├── config.yaml       # All hyperparameters
└── README.md
```

---

## Dataset Layout

Videos must be organized as:

```
dataset/
  train/
    controlled/
      video_001.mp4
      ...
    not_controlled/
      video_101.mp4
  val/
    controlled/
    not_controlled/
  test/               # optional
    controlled/
    not_controlled/
```

**Important**
- Split at the **event level** (no clips from the same collapse in both train and val)
- Negative samples should outnumber positives (2–3× recommended)

You can find the current dataset files here:

https://www.kaggle.com/datasets/mlabelle/controlled-demolition-classification-dataset/data

https://huggingface.co/datasets/mlowasp/controlled-demolition-classification

https://mega.nz/file/7IlxGYrB#qZ6tLZye605grlNBt1F0qLl3I1UIHoM0QtLQkopERLk

---

## Installation

### Requirements
- Python **3.9+**
- CUDA-capable GPU strongly recommended

### Install dependencies

```bash
pip install torch torchvision torchaudio tqdm pyyaml scikit-learn opencv-python
```

You also need **ffmpeg** installed system-wide:

```bash
# Ubuntu / Debian
sudo apt install ffmpeg
```

---

## Configuration

All parameters live in **`config.yaml`**.

Key sections:
- `video`: frame sampling, resolution
- `model`: encoder & LSTM size
- `train`: optimizer, AMP, batch size
- `imbalance`: class weighting
- `scheduler`: LR schedule

---

## Training

Start training with:

```bash
python train.py
```

Checkpoints are written to:

```
runs/demolition_cls/
  ├── last.pt
  └── best.pt
```

**F1 score** is used for model selection.

---

## Inference

Run inference on a single video:

```bash
python infer.py /path/to/video.mp4 runs/demolition_cls/best.pt
```

Output:
```
controlled_demolition_probability=0.93
```

---

## Model Architecture

```
Video
  ↓
Frame Sampling
  ↓
ResNet (per-frame CNN)
  ↓
LSTM (temporal modeling)
  ↓
Binary Classifier
```

---

## Ethical & Legal Considerations

⚠️ Misclassification can fuel misinformation.
Always output **confidence scores** and use this model for **analysis and research only**.

---

## License

Provided as-is for research and educational use.

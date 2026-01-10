# Two-Stream I3D for Controlled Building Demolition Detection

**Author:** Maxime Labelle  
**Contact:** maxime.labelle@owasp.org  
**License:** MIT  

---

## Overview

This repository contains a **two-stream I3D (Inflated 3D ConvNet)** implementation for **binary classification of building demolition videos**:

- **Controlled demolition**
- **Uncontrolled / accidental collapse**

The model uses:

- **RGB stream** (appearance + motion cues)
- **Optical Flow stream** (explicit motion modeling)
- **Late fusion** at the logit level
- **Per-video aggregation** from multiple temporal clips

This setup is designed for **small, high-quality datasets** and prioritizes **robust generalization over raw accuracy**.

---

## Model Architecture

### Backbone
- **Inception I3D (Inflated 3D ConvNet)**
- Based on the DeepMind I3D architecture
- RGB stream initialized from **ImageNet-inflated I3D weights**
- Optical Flow stream initialized from RGB weights (first-layer channel adaptation)

### Streams
| Stream | Input | Purpose |
|------|------|---------|
| RGB | 3 × T × H × W | Appearance + implicit motion |
| Flow | 2 × T × H × W | Explicit motion dynamics |

### Head
- Binary classification head (1 logit)
- Fusion strategies:
  - `mean_logit` (default)
  - `weighted_logit` (configurable)

---

## Dataset Structure

```
dataset/
├── train/
│   ├── controlled/
│   └── not_controlled/
└── val/
    ├── controlled/
    └── not_controlled/
```

Each folder contains video files (`.mp4`, `.mov`, `.mkv`, etc.).

**Important**

- Split at the **event level** (no clips from the same collapse in both train and val)
- Negative samples should outnumber positives (2–3× recommended)

You can find the current dataset files here:

https://www.kaggle.com/datasets/mlabelle/controlled-demolition-classification-dataset/data

https://huggingface.co/datasets/mlowasp/controlled-demolition-classification

https://mega.nz/file/7IlxGYrB#qZ6tLZye605grlNBt1F0qLl3I1UIHoM0QtLQkopERLk

https://drive.google.com/file/d/1ZxGlJQUZJFscW9T0liSOGVWqtDCTVFZg/view?usp=sharing

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

## Training Configuration

### Input Parameters
- Frames per clip: **64**
- Resolution: **224 × 224**
- Sampling: random contiguous clips
- Normalization:
  - RGB: `[0,1] → [-1,1]`
  - Flow: clipped and normalized to `[-1,1]`

### Loss
- **Binary Focal Loss**
  - `alpha = 0.75`
  - `gamma = 2.0`

### Optimization
- Optimizer: **AdamW**
- Learning rate: **3e-4**
- Weight decay: **0.02**
- Automatic Mixed Precision (AMP): **enabled**
- Gradient clipping: **1.0**

### Training Strategy
- Backbone frozen for first few epochs
- Gradual unfreezing for stability
- Best model selected by **per-video F1 score**

---

## Hardware Used for Training

The baseline model was trained on the following hardware:

| Component | Specification |
|--------|---------------|
| GPU | NVIDIA A40 (48 GB VRAM) |
| GPUs | 1 |
| CPU | 12 vCPUs |
| System RAM | 32 GB |
| Precision | FP16 (AMP) |

---

## Evaluation

### Per-Video Aggregation

Each video is evaluated using **K temporal clips**:

- Model predicts a logit per clip
- Clip predictions are aggregated:
  - Mean probability (default)
  - or Mean logit

Final classification is done using a configurable threshold (default `0.5`).

Metrics reported:
- Accuracy
- Precision
- Recall
- F1-score

---

## Inference

Single-video inference is supported without directory scanning:

```bash
python3 infer.py /path/to/video.mp4 runs/demolition_cls_twostream/best.pt
```

Output:
```
prediction: controlled
controlled_demolition_probability: 0.92
```

---

## Notes & Limitations

- Dataset size is relatively small; results should be interpreted carefully.
- Optical flow is computed on-the-fly (CPU-bound).
- For large-scale training, precomputing flow is recommended.

---

## License

This project is released under the **MIT License**.

---

## Citation

If you use this work in research or production, please cite:

```
Two-Stream I3D for Controlled Building Demolition Detection
Maxime Labelle, 2026
```

---

## Acknowledgements

- DeepMind I3D architecture
- PyTorch implementation adapted from `piergiaj/pytorch-i3d`
- OpenCV for optical flow computation

import os
import random
from typing import Dict

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(path: str, state: Dict):
    torch.save(state, path)


def load_checkpoint(path: str, device: str = "cpu") -> Dict:
    return torch.load(path, map_location=device)


@torch.no_grad()
def compute_metrics_from_preds(preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    preds = preds.long()
    labels = labels.long()

    tp = int(((preds == 1) & (labels == 1)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())

    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

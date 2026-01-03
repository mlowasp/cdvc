import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Binary focal loss for logits.

    alpha: balance toward positives (0.5=no bias). Try 0.75–0.9 if positives are rare.
    gamma: focusing parameter. 2.0 is standard.
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)

        pt = torch.where(targets == 1.0, probs, 1.0 - probs)
        focal = self.alpha * (1.0 - pt).pow(self.gamma) * bce

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal

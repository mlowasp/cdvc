import torch
import torch.nn as nn
from i3d.pytorch_i3d import InceptionI3d


class I3DBinaryClassifier(nn.Module):
    """
    I3D RGB stream pretrained on ImageNet inflation.
    Binary classifier head (1 logit).
    Input: [B, 3, T, H, W]
    Output: [B]
    """

    def __init__(self, pretrained_path: str):
        super().__init__()

        # Original I3D has 400 classes (Kinetics-400)
        self.backbone = InceptionI3d(
            num_classes=400,
            in_channels=3,
        )

        # Load pretrained RGB weights
        state = torch.load(pretrained_path, map_location="cpu")

        # Some checkpoints wrap weights
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # Load backbone weights, ignore classifier mismatch
        self.backbone.load_state_dict(state, strict=False)

        # Replace classification head → binary
        self.backbone.replace_logits(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, T, H, W]
        returns: logits [B]
        """
        out = self.backbone(x)

        # Output shape varies slightly across I3D impls:
        # Usually [B, 1, t, 1, 1]
        if out.dim() == 5:
            out = out.mean(dim=2)  # average over time
            out = out.squeeze(-1).squeeze(-1).squeeze(1)
        else:
            out = out.squeeze(1)

        return out

import torch
import torch.nn as nn

from i3d.pytorch_i3d import InceptionI3d


def _load_state(path: str):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


class TwoStreamI3D(nn.Module):
    """
    Two-stream I3D:
      - RGB stream: InceptionI3d(in_channels=3) with pretrained weights
      - Flow stream: InceptionI3d(in_channels=2) initialized from RGB conv1 inflation
    Input:
      rgb  [B,3,T,H,W] in [-1,1]
      flow [B,2,T,H,W] in [-1,1]
    Output:
      logits [B]
    """
    def __init__(self, rgb_weights_path: str, fusion: str = "mean_logit", fusion_alpha: float = 0.5):
        super().__init__()
        self.fusion = fusion
        self.fusion_alpha = float(fusion_alpha)

        # Build streams with original num_classes=400 to load weights
        self.rgb = InceptionI3d(num_classes=400, in_channels=3)
        self.flow = InceptionI3d(num_classes=400, in_channels=2)

        # Load pretrained RGB
        rgb_state = _load_state(rgb_weights_path)
        self.rgb.load_state_dict(rgb_state, strict=False)

        # Initialize flow stream from rgb_state where possible
        # - for conv3d_1a_7x7: average first 2 channels from RGB weights
        flow_state = self.flow.state_dict()
        for k, v in rgb_state.items():
            if k in flow_state and flow_state[k].shape == v.shape:
                flow_state[k] = v.clone()

        # Special case: first conv weight differs in in_channels
        # Many I3D impls name it: "Conv3d_1a_7x7.conv3d.weight"
        key = "Conv3d_1a_7x7.conv3d.weight"
        if key in rgb_state and key in flow_state:
            w = rgb_state[key]  # [out, 3, kt, kh, kw]
            if w.shape[1] == 3 and flow_state[key].shape[1] == 2:
                # Take first 2 channels and renormalize
                flow_w = w[:, :2, :, :, :].clone()
                flow_state[key] = flow_w

        self.flow.load_state_dict(flow_state, strict=False)

        # Replace logits with 1-class head for both streams
        self.rgb.replace_logits(1)
        self.flow.replace_logits(1)

    def _reduce_logits(self, out: torch.Tensor) -> torch.Tensor:
        """
        Reduce I3D output to a single scalar logit per sample.
        Accepts shapes:
        [B, C]
        [B, C, T]
        [B, C, T, 1, 1]
        Returns:
        [B]
        """
        # Remove spatial dims if present
        if out.dim() == 5:
            out = out.squeeze(-1).squeeze(-1)  # [B, C, T]

        # Average over temporal dimension if present
        if out.dim() == 3:
            out = out.mean(dim=2)  # [B, C]

        # If still multi-channel, average channels
        if out.dim() == 2 and out.size(1) > 1:
            out = out.mean(dim=1)  # [B]

        # Final safety squeeze
        return out.view(-1)

    def forward(self, rgb: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        rgb_out = self._reduce_logits(self.rgb(rgb))
        flow_out = self._reduce_logits(self.flow(flow))

        if self.fusion == "weighted_logit":
            a = self.fusion_alpha
            return a * rgb_out + (1.0 - a) * flow_out

        return 0.5 * (rgb_out + flow_out)

    def set_backbone_trainable(self, rgb_trainable: bool, flow_trainable: bool):
        """
        Freeze/unfreeze backbone feature extractors while
        keeping classification heads trainable.
        """

        def _is_logit_param(name: str) -> bool:
            # I3D replace_logits creates layers with 'logits' in name
            return "logits" in name or "logit" in name

        for name, p in self.rgb.named_parameters():
            if _is_logit_param(name):
                p.requires_grad = True
            else:
                p.requires_grad = bool(rgb_trainable)

        for name, p in self.flow.named_parameters():
            if _is_logit_param(name):
                p.requires_grad = True
            else:
                p.requires_grad = bool(flow_trainable)

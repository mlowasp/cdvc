import torch
import torch.nn as nn
import torchvision.models as tvm


def build_resnet_encoder(name: str = "resnet18", pretrained: bool = True):
    if name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        dim = 512
    elif name == "resnet34":
        m = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT if pretrained else None)
        dim = 512
    elif name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT if pretrained else None)
        dim = 2048
    else:
        raise ValueError(f"Unsupported encoder: {name}")

    m.fc = nn.Identity()
    return m, dim


class CNNLSTMVideoClassifier(nn.Module):
    """
    Input:  frames [B, T, 3, H, W]
    Output: logits [B]
    """
    def __init__(
        self,
        encoder_name: str = "resnet18",
        pretrained: bool = True,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.4,
        unfreeze_layer4: bool = True,
    ):
        super().__init__()
        self.encoder, feat_dim = build_resnet_encoder(encoder_name, pretrained)

        # Freeze all, optionally unfreeze layer4 for light adaptation
        for name, p in self.encoder.named_parameters():
            if unfreeze_layer4 and "layer4" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden, 1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)
        feat = self.encoder(x)          # [B*T, feat_dim]
        feat = feat.view(B, T, -1)      # [B, T, feat_dim]
        out, _ = self.lstm(feat)        # [B, T, hidden]
        last = out[:, -1, :]            # [B, hidden]
        last = self.drop(last)
        logits = self.head(last).squeeze(1)  # [B]
        return logits

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class NIMA(nn.Module):
    """Simple PyTorch NIMA (MobileNetV2 backbone)."""

    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v2(pretrained=True)


        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.75),
            nn.Linear(1280, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        return self.backbone(x)


def compute_nima_score(model: NIMA, img: torch.Tensor) -> torch.Tensor:
    """
    Args:
        model: NIMA model
        img: (B, 3, H, W) in [0,1]

    Returns:
        scalar mean aesthetic score
    """
    model.eval()
    with torch.no_grad():
        out = model(img)  # (B, 10)

        weights = torch.arange(1, 11, device=img.device, dtype=out.dtype)
        score = (out * weights).sum(dim=1)  # (B,)

    return score.mean()
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ColorConstancyLoss(nn.Module):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        pred: (B, 3, H, W)
        target: 不用，但必须接收
        """

        mean_rgb = pred.mean(dim=(2, 3))  # (B, 3)

        mr = mean_rgb[:, 0]
        mg = mean_rgb[:, 1]
        mb = mean_rgb[:, 2]

        loss = torch.sqrt(
            (mr - mg) ** 2 +
            (mr - mb) ** 2 +
            (mg - mb) ** 2 +
            1e-8
        )

        return self.weight * loss.mean()
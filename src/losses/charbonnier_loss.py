"""Charbonnier (smooth L1) pixel-wise reconstruction loss."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CharbonnierLoss(nn.Module):
    """Charbonnier loss: weight * mean(sqrt((pred - target)^2 + eps^2)).

    A smooth approximation of L1 that penalises small errors less harshly
    than L1 while remaining differentiable at zero. Empirically yields
    higher PSNR than plain L1 on image restoration tasks.

    Args:
        weight: Scalar multiplier applied to the raw loss value.
        eps: Smoothing constant. Smaller values approach L1; larger values
             approach L2. Typical range: 1e-6 to 1e-2.
    """

    def __init__(self, weight: float = 1.0, eps: float = 1e-3) -> None:
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.weight * torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))

"""MSE pixel-wise reconstruction loss (directly optimises PSNR)."""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MSELoss(nn.Module):
    """Weighted mean squared error loss.

    Minimising MSE is mathematically equivalent to maximising PSNR
    (PSNR = 10 · log10(MAX² / MSE)).

    Args:
        weight: Scalar multiplier applied to the raw MSE value.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.weight * F.mse_loss(pred, target)

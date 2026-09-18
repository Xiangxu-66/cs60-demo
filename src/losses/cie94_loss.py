"""CIE94 perceptual color-difference loss.

Computes the CIE 1994 color-difference formula (graphics-arts parametrization)
per pixel, then takes the mean as a loss term.

    ΔE94 = sqrt( (ΔL / SL)²  +  (ΔC / SC)²  +  (ΔH / SH)² )

    SL = 1
    SC = 1 + 0.045 · C1
    SH = 1 + 0.015 · C1

where C1 is the chroma of the *reference* (target) pixel.

Reference:
    CIE Publication 116-1995, "Industrial Colour-Difference Evaluation"
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.losses.ms_swc_loss import rgb_to_lab


class CIE94Loss(nn.Module):
    """Mean per-pixel CIE94 color difference loss (graphics-arts parametrization).

    Args:
        weight: Scalar multiplier applied to the mean ΔE94.
        eps:    Numerical stability term added inside the square root.
    """

    def __init__(self, weight: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute weighted mean CIE94 color difference.

        Args:
            pred:   Predicted image (B, 3, H, W) in [0, 1].
            target: Ground-truth image (B, 3, H, W) in [0, 1].

        Returns:
            Scalar loss = weight × mean(ΔE94).
        """
        pred_lab = rgb_to_lab(pred)      # (B, 3, H, W): L, a, b
        tgt_lab  = rgb_to_lab(target)

        L1, a1, b1 = tgt_lab[:, 0], tgt_lab[:, 1], tgt_lab[:, 2]    # reference
        L2, a2, b2 = pred_lab[:, 0], pred_lab[:, 1], pred_lab[:, 2]  # prediction

        C1 = torch.sqrt(a1 ** 2 + b1 ** 2 + self.eps)  # chroma of reference
        C2 = torch.sqrt(a2 ** 2 + b2 ** 2 + self.eps)  # chroma of prediction

        dL = L2 - L1
        dC = C2 - C1
        # ΔH² = Δa² + Δb² − ΔC²  (avoids atan2, always non-negative in theory)
        dH2 = (a2 - a1) ** 2 + (b2 - b1) ** 2 - dC ** 2

        SL = 1.0
        SC = 1.0 + 0.045 * C1
        SH = 1.0 + 0.015 * C1

        delta_e = torch.sqrt(
            (dL / SL) ** 2
            + (dC / SC) ** 2
            + dH2.clamp(min=0.0) / (SH ** 2)
            + self.eps
        )
        return self.weight * delta_e.mean()

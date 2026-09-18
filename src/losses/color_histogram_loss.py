"""Soft color histogram matching loss — placeholder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def rgb_to_lab(img):
    # img: [B,3,H,W] in [0,1]

    mask = (img > 0.04045).float()
    img = (((img + 0.055) / 1.055) ** 2.4) * mask + img / 12.92 * (1 - mask)

    # RGB → XYZ
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]

    x = 0.4124*r + 0.3576*g + 0.1805*b
    y = 0.2126*r + 0.7152*g + 0.0722*b
    z = 0.0193*r + 0.1192*g + 0.9505*b

    # normalize (D65)
    x /= 0.95047
    z /= 1.08883

    def f(t):
        return torch.where(t > 0.008856, t ** (1/3), 7.787*t + 16/116)

    fx, fy, fz = f(x), f(y), f(z)

    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)

    return torch.cat([L, a, b], dim=1)


class ColorHistogramLoss(nn.Module):
    """Differentiable soft color histogram matching loss.

    Encourages the output color distribution to match the target's
    per-channel histogram via an RBF soft-binning approach.

    TODO: Tune sigma and num_bins; consider per-image normalisation.

    Args:
        weight: Scalar multiplier.
        num_bins: Histogram bins per channel.
    """
    def __init__(self, weight=0.1, num_bins=64, sigma=0.02):
        super().__init__()
        self.weight = weight
        self.num_bins = num_bins
        self.sigma = sigma

    def _soft_histogram(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1)

        bins = torch.linspace(-1, 1, self.num_bins, device=x.device)

        diff = x.unsqueeze(-1) - bins
        weights = torch.exp(-0.5 * (diff / self.sigma) ** 2)

        hist = weights.sum(dim=2)
        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-6)

        return hist

    def forward(self, pred, target):
        pred = pred.clamp(0, 1)
        target = target.clamp(0, 1)

        pred_lab = rgb_to_lab(pred)
        target_lab = rgb_to_lab(target)

        pred_ab = pred_lab[:, 1:3]
        target_ab = target_lab[:, 1:3]

        pred_hist = self._soft_histogram(pred_ab)
        target_hist = self._soft_histogram(target_ab)

        loss = F.l1_loss(pred_hist, target_hist)

        return self.weight * loss
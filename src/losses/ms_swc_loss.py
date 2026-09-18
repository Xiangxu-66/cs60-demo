"""Multiscale Sliced-Wasserstein Contrastive loss (positive-only variant).

Compares per-pixel CIELAB color distributions of pred vs target across a
Gaussian pyramid using sliced 1-Wasserstein distance with random unit
projections. Distributional in nature — optimises overall color statistics
rather than per-pixel alignment, so typically combined with L1/SSIM.

Reference:
    He et al., "Multiscale Sliced Wasserstein Distances as Perceptual
    Color Difference Measures", ECCV 2024. arXiv:2407.10181.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# sRGB D65 → XYZ (IEC 61966-2-1)
_XYZ_FROM_LINEAR_RGB: tuple[tuple[float, ...], ...] = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
_D65_WHITE: tuple[float, float, float] = (0.95047, 1.0, 1.08883)


def _srgb_to_linear(c: Tensor) -> Tensor:
    c = c.clamp(min=0.0)
    low = c / 12.92
    high = ((c + 0.055) / 1.055).pow(2.4)
    return torch.where(c <= 0.04045, low, high)


def _f_lab(t: Tensor) -> Tensor:
    delta = 6.0 / 29.0
    t_safe = t.clamp(min=1e-12)
    cube_root = t_safe.pow(1.0 / 3.0)
    linear = t / (3.0 * delta * delta) + 4.0 / 29.0
    return torch.where(t > delta ** 3, cube_root, linear)


def rgb_to_lab(rgb: Tensor) -> Tensor:
    """sRGB image (B,3,H,W) in [0,1] → CIELAB. L∈[0,100], a/b≈[-128,128]."""
    linear = _srgb_to_linear(rgb)
    mat = torch.tensor(_XYZ_FROM_LINEAR_RGB, dtype=rgb.dtype, device=rgb.device)
    xyz = torch.einsum("ij,bjhw->bihw", mat, linear)
    wn = torch.tensor(_D65_WHITE, dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
    f = _f_lab(xyz / wn)
    fx, fy, fz = f[:, 0:1], f[:, 1:2], f[:, 2:3]
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.cat([L, a, b], dim=1)


def _make_gauss_kernel(sigma: float, ksize: int) -> Tensor:
    half = ksize // 2
    xs = torch.arange(ksize, dtype=torch.float32) - half
    k1d = torch.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    k1d = k1d / k1d.sum()
    return k1d[:, None] * k1d[None, :]


class _GaussianBlur(nn.Module):
    def __init__(self, channels: int, sigma: float, ksize: int) -> None:
        super().__init__()
        k = _make_gauss_kernel(sigma, ksize)
        k = k.view(1, 1, ksize, ksize).expand(channels, 1, ksize, ksize).contiguous()
        self.register_buffer("kernel", k)
        self.pad = ksize // 2
        self.channels = channels

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.kernel, padding=self.pad, groups=self.channels)


def _gaussian_pyramid(x: Tensor, num_scales: int, blur: _GaussianBlur) -> list[Tensor]:
    pyr = [x]
    for _ in range(num_scales - 1):
        x = F.avg_pool2d(blur(x), kernel_size=2, stride=2)
        pyr.append(x)
    return pyr


def _sliced_wasserstein(pred: Tensor, target: Tensor, num_proj: int) -> Tensor:
    """1-Wasserstein distance averaged over random directions on S^(C-1)."""
    B, C, H, W = pred.shape
    N = H * W
    dirs = torch.randn(num_proj, C, device=pred.device, dtype=pred.dtype)
    dirs = F.normalize(dirs, dim=1)
    p = torch.einsum("bcn,kc->bkn", pred.reshape(B, C, N), dirs)
    t = torch.einsum("bcn,kc->bkn", target.reshape(B, C, N), dirs)
    p_sorted, _ = p.sort(dim=-1)
    t_sorted, _ = t.sort(dim=-1)
    return (p_sorted - t_sorted).abs().mean()


class MSSWCLoss(nn.Module):
    """Positive-only MS-SWD on CIELAB Gaussian pyramid.

    Args:
        weight:          scalar multiplier.
        num_scales:      pyramid depth.
        num_projections: random directions per SWD call; resampled each call.
        pyr_sigma:       Gaussian blur sigma before ↓2.
        pyr_ksize:       Gaussian kernel size.
    """

    def __init__(
        self,
        weight: float = 1.0,
        num_scales: int = 5,
        num_projections: int = 128,
        pyr_sigma: float = 1.0,
        pyr_ksize: int = 5,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.num_scales = num_scales
        self.num_projections = num_projections
        self.blur = _GaussianBlur(channels=3, sigma=pyr_sigma, ksize=pyr_ksize)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Weighted mean of sliced-W1 distances across CIELAB pyramid.

        Args:
            pred, target: (B, 3, H, W) in sRGB [0, 1].
        Returns:
            Scalar loss = weight × mean_{s} SWD(lab_pyr_s(pred), lab_pyr_s(target)).
        """
        pred_lab = rgb_to_lab(pred)
        target_lab = rgb_to_lab(target)
        pyr_p = _gaussian_pyramid(pred_lab, self.num_scales, self.blur)
        pyr_t = _gaussian_pyramid(target_lab, self.num_scales, self.blur)
        total = pred.new_zeros(1).squeeze()
        for p, t in zip(pyr_p, pyr_t):
            total = total + _sliced_wasserstein(p, t, self.num_projections)
        return self.weight * total / self.num_scales

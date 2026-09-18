"""Image quality metrics: PSNR, SSIM, LPIPS, VGG perceptual, ΔE (CIEDE2000)."""
from __future__ import annotations

import importlib
import math
from typing import Any

from src.evaluation.nima import NIMA, compute_nima_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def compute_vgg_perceptual(
    pred: Tensor,
    target: Tensor,
    layers: list[str] | None = None,
) -> Tensor:
    """Compute VGG16-based perceptual distance (matches PerceptualLoss).

    Uses the same VGG16 model and layers as PerceptualLoss for consistent
    training-validation alignment. Returns mean L1 distance in VGG feature space.

    Args:
        pred: Predicted images (B, 3, H, W) in [0, 1].
        target: Ground truth images (B, 3, H, W) in [0, 1].
        layers: VGG16 layers to use. Defaults to ['relu1_2', 'relu2_2', 'relu3_3'].

    Returns:
        Mean L1 distance in VGG feature space (lower is better).
    """
    # Lazy-load VGG16 to avoid overhead if not used
    cache = compute_vgg_perceptual.__dict__.setdefault("_vgg_cache", {})

    layer_key = tuple(layers or ["relu1_2", "relu2_2", "relu3_3"])
    device_key = str(pred.device)
    cache_key = (layer_key, device_key)

    if cache_key not in cache:
        import torchvision.models as models

        layer_names = layers or ["relu1_2", "relu2_2", "relu3_3"]
        _LAYER_IDX: dict[str, int] = {
            "relu1_2": 4,
            "relu2_2": 9,
            "relu3_3": 16,
            "relu4_3": 23,
        }

        vgg_feats = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        indices = sorted(_LAYER_IDX[n] for n in layer_names)

        # Build sequential slices
        slices = []
        prev = 0
        for idx in indices:
            slices.append(nn.Sequential(*list(vgg_feats.children())[prev:idx]).eval())
            prev = idx

        # Freeze and move to device
        for slice_net in slices:
            for p in slice_net.parameters():
                p.requires_grad = False
            slice_net.to(pred.device)

        # ImageNet normalization constants
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)

        cache[cache_key] = {"slices": slices, "mean": mean, "std": std}

    cached = cache[cache_key]
    slices, mean, std = cached["slices"], cached["mean"], cached["std"]

    # Normalize to ImageNet space
    p = (pred - mean) / std
    t = (target - mean) / std

    # Compute L1 distance at each layer
    distance = pred.new_zeros(1).squeeze()
    for slice_net in slices:
        with torch.no_grad():
            p = slice_net(p)
            t = slice_net(t)
            distance = distance + F.l1_loss(p, t)

    return distance


def compute_psnr(pred: Tensor, target: Tensor, data_range: float = 1.0) -> Tensor:
    """Compute mean Peak Signal-to-Noise Ratio across a batch.

    Args:
        pred:       Predicted images (B, 3, H, W) in [0, 1].
        target:     Ground truth images (B, 3, H, W) in [0, 1].
        data_range: Pixel value range (1.0 for normalised images).

    Returns:
        Mean PSNR scalar (dB).
    """
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    psnr = 10.0 * torch.log10(data_range ** 2 / (mse + 1e-10))
    return psnr.mean()


def compute_ssim(pred: Tensor, target: Tensor, window_size: int = 11) -> Tensor:
    """Compute mean SSIM across a batch.

    Reuses the Gaussian kernel from SSIMLoss to avoid code duplication.

    Args:
        pred:        Predicted images (B, 3, H, W) in [0, 1].
        target:      Ground truth images (B, 3, H, W) in [0, 1].
        window_size: Gaussian window size.

    Returns:
        Mean SSIM scalar.
    """
    from src.losses.ssim_loss import _gaussian_kernel, _ssim
    window = _gaussian_kernel(window_size, 1.5, 3).to(pred.device).to(pred.dtype)
    return _ssim(pred, target, window, window_size, 3)


def compute_delta_e(pred: Tensor, target: Tensor) -> Tensor:
    """Mean CIEDE2000 color difference, averaged over all pixels in the batch.

    Args:
        pred:   Predicted images (B, 3, H, W) in [0, 1].
        target: Ground truth images (B, 3, H, W) in [0, 1].

    Returns:
        Scalar mean ΔE00 (lower is better; 0 for identical images).

    Reference:
        Luo et al., "The Development of the CIE 2000 Colour-Difference Formula",
        Color Research & Application, 26(5), 2001.
    """
    from src.losses.ms_swc_loss import rgb_to_lab

    _R = math.pi / 180.0  # degrees → radians
    _25_7 = 25.0 ** 7

    lab_p = rgb_to_lab(pred)
    lab_t = rgb_to_lab(target)

    Lp, ap, bp = lab_p[:, 0], lab_p[:, 1], lab_p[:, 2]
    Lt, at, bt = lab_t[:, 0], lab_t[:, 1], lab_t[:, 2]

    # Step 1: a' correction factor G
    C_p = torch.sqrt(ap ** 2 + bp ** 2)
    C_t = torch.sqrt(at ** 2 + bt ** 2)
    C_mean7 = ((C_p + C_t) * 0.5).pow(7)
    G = 0.5 * (1.0 - torch.sqrt(C_mean7 / (C_mean7 + _25_7)))
    ap2 = ap * (1.0 + G)
    at2 = at * (1.0 + G)

    # Step 2: adjusted C' and h' (degrees, [0, 360))
    Cp = torch.sqrt(ap2 ** 2 + bp ** 2)
    Ct = torch.sqrt(at2 ** 2 + bt ** 2)
    hp = torch.atan2(bp, ap2).remainder(2.0 * math.pi) / _R
    ht = torch.atan2(bt, at2).remainder(2.0 * math.pi) / _R

    # Step 3: ΔL', ΔC', Δh', ΔH'
    dLp = Lt - Lp
    dCp = Ct - Cp

    CC = Cp * Ct
    dh = ht - hp
    # When one C' is zero, Δh' = 0
    dh = torch.where(CC == 0.0, torch.zeros_like(dh), dh)
    # Wrap to [-180, 180]
    dh = torch.where((CC > 0.0) & (dh > 180.0), dh - 360.0, dh)
    dh = torch.where((CC > 0.0) & (dh < -180.0), dh + 360.0, dh)
    dHp = 2.0 * torch.sqrt(CC.clamp(min=0.0)) * torch.sin(dh * _R * 0.5)

    # Step 4: mean L', C', h'
    Lbar = (Lp + Lt) * 0.5
    Cbar = (Cp + Ct) * 0.5

    hsum = hp + ht
    hbar = torch.where(
        CC == 0.0,
        hsum,
        torch.where(
            (hp - ht).abs() <= 180.0,
            hsum * 0.5,
            torch.where(hsum < 360.0, (hsum + 360.0) * 0.5, (hsum - 360.0) * 0.5),
        ),
    )

    # Step 5: weighting functions SL, SC, SH
    T = (
        1.0
        - 0.17 * torch.cos((hbar - 30.0) * _R)
        + 0.24 * torch.cos(2.0 * hbar * _R)
        + 0.32 * torch.cos((3.0 * hbar + 6.0) * _R)
        - 0.20 * torch.cos((4.0 * hbar - 63.0) * _R)
    )
    L50sq = (Lbar - 50.0) ** 2
    SL = 1.0 + 0.015 * L50sq / torch.sqrt(20.0 + L50sq)
    SC = 1.0 + 0.045 * Cbar
    SH = 1.0 + 0.015 * Cbar * T

    # Step 6: rotation term RT
    Cbar7 = Cbar.pow(7)
    RC = 2.0 * torch.sqrt(Cbar7 / (Cbar7 + _25_7))
    dtheta = 30.0 * torch.exp(-((hbar - 275.0) / 25.0) ** 2)
    RT = -torch.sin(2.0 * dtheta * _R) * RC

    # Step 7: ΔE00
    dE = torch.sqrt(
        (dLp / SL) ** 2
        + (dCp / SC) ** 2
        + (dHp / SH) ** 2
        + RT * (dCp / SC) * (dHp / SH)
        + 1e-12,
    )
    return dE.mean()


def compute_lpips(pred: Tensor, target: Tensor) -> Tensor:
    """Compute mean LPIPS perceptual distance across a batch."""
    try:
        import lpips as _lpips_lib
        device_key = str(pred.device)
        cache = compute_lpips.__dict__.setdefault("_cache", {})
        if device_key not in cache:
            cache[device_key] = _lpips_lib.LPIPS(net="alex").to(pred.device)
        lpips_fn = cache[device_key]

        with torch.no_grad():
            return lpips_fn(pred * 2 - 1, target * 2 - 1).mean()
    except ImportError:
        return pred.new_zeros(1).squeeze()


class MetricCollection:
    """Convenience wrapper that computes PSNR, SSIM, LPIPS, ΔE, VGG perceptual, NIMA, CLIP."""

    def __init__(self, cfg: Any = None) -> None:
        self.cfg = cfg
        evaluation_cfg = self.cfg.get("evaluation") if self.cfg else {}

        self.enable_nima = evaluation_cfg.get("enable_nima", True)
        self.enable_clip = evaluation_cfg.get("enable_clip", True)
        self.clip = None

        if self.enable_nima:
            self.nima_model = NIMA()

    def _gate(self, key: str, default: bool) -> bool:
        """Read a boolean gate from evaluation config, falling back to default."""
        if self.cfg is None:
            return default
        evaluation_cfg = self.cfg.get("evaluation")
        if evaluation_cfg is None:
            return default
        return bool(evaluation_cfg.get(key, default))

    def _lpips_enabled(self) -> bool:
        return self._gate("enable_lpips", True)

    def _get_clip_metric(self):
        if self.clip is None:
            try:
                clip_module = importlib.import_module("src.evaluation.clip_score")
            except ImportError as exc:
                raise ImportError(
                    "CLIP metric requested but its optional dependencies are not installed. "
                    "Install the 'clip' package or disable CLIP metrics via "
                    "'evaluation.enable_clip=false' or a loss config without 'clip'."
                ) from exc
            self.clip = clip_module.CLIPScore(device="cpu")
        return self.clip

    def __call__(
        self,
        pred: Tensor,
        target: Tensor,
        enable_psnr: bool = True,
        enable_ssim: bool | None = None,
        enable_vgg_perceptual: bool | None = None,
        enable_lpips: bool | None = None,
        enable_delta_e: bool | None = None,
        enable_nima: bool | None = None,
        enable_clip: bool | None = None,
    ) -> dict[str, Tensor]:
        """Compute metrics.

        Args:
            pred: Predicted images (B, 3, H, W) in [0, 1].
            target: Ground truth images (B, 3, H, W) in [0, 1].
            enable_psnr: Always compute (fast, baseline metric).
            enable_ssim: Override; None = always compute.
            enable_vgg_perceptual: Override; None = skip.
            enable_lpips: Override; None = follow config.
            enable_delta_e: Override ΔE CIEDE2000; None = follow config.
            enable_nima: Override; None = follow config.
            enable_clip: Override; None = follow config.
        """
        should_compute_ssim = enable_ssim if enable_ssim is not None else True
        should_compute_vgg_perceptual = (
            enable_vgg_perceptual if enable_vgg_perceptual is not None else False
        )
        should_compute_lpips = (
            enable_lpips if enable_lpips is not None else self._lpips_enabled()
        )
        should_compute_delta_e = (
            enable_delta_e if enable_delta_e is not None else self._gate("enable_delta_e", False)
        )
        should_compute_nima = (
            enable_nima if enable_nima is not None else self.enable_nima
        )
        should_compute_clip = (
            enable_clip if enable_clip is not None else self.enable_clip
        )

        results = {
            "psnr": compute_psnr(pred, target) if enable_psnr else pred.new_zeros(()),
            "ssim": compute_ssim(pred, target) if should_compute_ssim else pred.new_zeros(()),
            "vgg_perceptual": compute_vgg_perceptual(pred, target) if should_compute_vgg_perceptual else pred.new_zeros(()),
            "lpips": compute_lpips(pred, target) if should_compute_lpips else pred.new_zeros(()),
            "delta_e": compute_delta_e(pred, target) if should_compute_delta_e else pred.new_zeros(()),
        }

        if should_compute_nima:
            results["nima"] = compute_nima_score(self.nima_model.to(pred.device), pred)
        else:
            results["nima"] = pred.new_zeros(())

        if should_compute_clip:
            results["clip"] = self._get_clip_metric()(pred)
        else:
            results["clip"] = pred.new_zeros(())

        return results

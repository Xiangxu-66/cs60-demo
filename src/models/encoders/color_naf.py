"""ColorNAF: NAFNet-style encoder for color enhancement.

Combines multi-colorspace input (RGB+Lab) with NAFNet's efficient architecture
and specialized color-aware components (GRN, SimpleGate, SoftHistogram).

Architecture:
    RGB/RGB+Lab -> ColorBlock stages -> H/8 spatial tokens + histogram token
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Literal

from src.models.encoders.base import BaseEncoder


class ChanLayerNorm(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).to(x.dtype).permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    """SimpleGate from NAFNet: split channels and multiply.

    Splits input into two halves along channel dimension and multiplies them.
    This is more efficient than using a separate gate network.
    """

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class GRN(nn.Module):
    """Global Response Normalization from ConvNeXt V2.

    Models competition and cooperation among channels by normalizing
    the response across channels. Particularly effective for color enhancement
    where channels represent different color/feature components.

    Formula: output = x * (1 + gamma * (nx - 1))
    where nx = x / L2 norm across channels
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        # Compute L2 norm across channels
        gx = torch.norm(x, p=2, dim=1, keepdim=True)
        # Normalize
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        # Apply affine transformation
        return x * (1 + self.gamma * (nx - 1)) + self.beta


class SoftHistogram(nn.Module):
    """Differentiable soft histogram for color statistics.

    Uses soft bin assignment via softmax instead of hard binning.
    This allows gradient flow and end-to-end training.

    Args:
        num_bins: Number of histogram bins per channel.
        temperature: Softmax temperature for soft assignment.
                     Lower = sharper assignment.
    """

    def __init__(self, num_bins: int = 8, temperature: float = 0.1) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.temperature = temperature
        # Learnable bin centers (initialized to evenly spaced)
        self.register_buffer(
            "bin_centers",
            torch.linspace(0, 1, num_bins).view(1, 1, 1, -1),
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute soft histogram.

        Args:
            x: (B, C, H, W), values assumed to be in [0, 1] range.

        Returns:
            Histogram tensor: (B, C * num_bins)
        """
        B, C, H, W = x.shape
        device = x.device

        # Reshape for computation
        x_flat = x.reshape(B, C, -1, 1)  # (B, C, H*W, 1)

        # Get bin centers on correct device
        bin_centers = self.bin_centers.to(device)  # (1, 1, 1, num_bins)

        # Compute squared distance to each bin center
        dist = (x_flat - bin_centers) ** 2  # (B, C, H*W, num_bins)

        # Soft bin assignment via softmax
        # Negative distance + temperature sharpening
        weights = torch.softmax(-dist / self.temperature, dim=-1)  # (B, C, H*W, num_bins)

        # Sum weights to get histogram (soft count per bin)
        hist = weights.sum(dim=2)  # (B, C, num_bins)

        # Normalize to sum to 1 per channel
        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-6)

        return hist.flatten(1)  # (B, C * num_bins)


class SoftHistogram2D(nn.Module):
    """Differentiable 2-D soft histogram for joint color statistics."""

    def __init__(self, num_bins: int = 12, temperature: float = 0.1) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.temperature = temperature
        self.register_buffer(
            "bin_centers",
            torch.linspace(0, 1, num_bins).view(1, 1, num_bins),
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute a joint histogram over the two input channels.

        Args:
            x: (B, 2, H, W), values assumed to be in [0, 1].

        Returns:
            Flattened joint histogram tensor: (B, num_bins * num_bins)
        """
        if x.shape[1] != 2:
            raise ValueError(
                f"SoftHistogram2D expects 2 channels, got {x.shape[1]}"
            )

        B, _, H, W = x.shape
        x_flat = x.reshape(B, 2, H * W)
        centers = self.bin_centers.to(device=x.device, dtype=x.dtype)

        a = x_flat[:, 0].unsqueeze(-1)  # (B, P, 1)
        b = x_flat[:, 1].unsqueeze(-1)  # (B, P, 1)
        weights_a = torch.softmax(-((a - centers) ** 2) / self.temperature, dim=-1)
        weights_b = torch.softmax(-((b - centers) ** 2) / self.temperature, dim=-1)

        joint = torch.einsum("bpi,bpj->bij", weights_a, weights_b)
        joint = joint / (joint.sum(dim=(1, 2), keepdim=True) + 1e-6)
        return joint.flatten(1)


class ColorBlock(nn.Module):
    """NAFNet-style block for color enhancement.

    Combines:
    - SimpleGate for efficient feature gating
    - GRN for channel competition modeling
    - Dual residual structure for stability

    Args:
        dim: Number of input/output channels.
        expansion: Channel expansion factor (default 2).
    """

    def __init__(self, dim: int, expansion: int = 2) -> None:
        super().__init__()

        # First stage: spatial + channel modulation
        self.norm1 = ChanLayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim * expansion, 1)
        self.dwconv = nn.Conv2d(
            dim * expansion, dim * expansion, 3, padding=1, groups=dim * expansion
        )
        self.gate1 = SimpleGate()
        self.grn = GRN(dim)
        self.proj1 = nn.Conv2d(dim, dim, 1)

        # Second stage: channel recalibration
        self.norm2 = ChanLayerNorm(dim)
        self.conv2 = nn.Conv2d(dim, dim * expansion, 1)
        self.gate2 = SimpleGate()
        self.proj2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        # First residual branch
        residual = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.gate1(x)  # dim*expansion -> dim
        x = self.grn(x)
        x = self.proj1(x)
        x = x + residual

        # Second residual branch
        residual = x
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.gate2(x)
        x = self.proj2(x)
        x = x + residual

        return x


class ColorNAFEncoder(BaseEncoder):
    """NAFNet-style encoder for color enhancement.

    Features:
    - Multi-colorspace input (RGB + Lab)
    - Multi-scale skip connections (stage 0, 1, 2)
    - Global histogram token for color statistics
    - Semi-frozen training strategy (early stages frozen)

    Args:
        in_channels: Input channels (default 6 for RGB+Lab).
        base_channels: Base channel count.
        num_stages: Number of stages (default 4).
        channels: Channel config per stage. If None, auto-computed.
        skip_stages: Which stages to output for skip connections.
        frozen_stages: Which stages to freeze (semi-frozen strategy).
        histogram_bins: Number of histogram bins.
        token_dim: Output token dimension.
        input_mode: Input color mode ("rgb", "rgb_lab", "rgb_lab_hsv").
        use_global_token: When False, no histogram/global token is built or
            prepended; the encoder returns only the (B, N, token_dim) spatial
            tokens. Used as an ablation to test whether the global token helps.
    """

    def __init__(
        self,
        in_channels: int = 6,
        base_channels: int = 64,
        num_stages: int = 4,
        channels: list[int] | None = None,
        skip_stages: list[int] = (0, 1, 2),
        frozen_stages: list[int] = (0, 1),
        histogram_bins: int = 8,
        token_dim: int = 128,
        input_mode: Literal["rgb", "rgb_lab", "rgb_lab_hsv"] = "rgb_lab",
        marginal_hist_bins: int | None = None,
        joint_hist_bins: int = 0,
        quantile_levels: tuple[float, ...] = (),
        include_mean_std: bool = False,
        global_stats_size: int = 0,
        normalize_rgb_input: bool = False,
        normalize_lab_input: bool = False,
        use_input_calibration: bool = False,
        align_skip_keys: bool = False,
        use_global_token: bool = True,
    ) -> None:
        super().__init__()

        self._patch_size = 8  # Final spatial token stride after stem + two downsample stages.
        self._embed_dim = token_dim
        self.skip_stages = skip_stages
        self.frozen_stages = frozen_stages
        self.align_skip_keys = align_skip_keys
        self.input_mode = input_mode
        self.num_stages = num_stages
        self.marginal_hist_bins = int(marginal_hist_bins or histogram_bins)
        self.joint_hist_bins = int(joint_hist_bins)
        self.quantile_levels = tuple(float(q) for q in quantile_levels)
        self.include_mean_std = bool(include_mean_std)
        self.global_stats_size = int(global_stats_size)
        self.normalize_rgb_input = bool(normalize_rgb_input)
        self.normalize_lab_input = bool(normalize_lab_input)
        self.use_input_calibration = bool(use_input_calibration)
        self.use_global_token = bool(use_global_token)

        self.register_buffer(
            "rgb_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        # Compute channel config
        if channels is None:
            channels = [base_channels] * 2 + [base_channels * 2] * 2
            channels = channels[:num_stages]

        # Input projection based on mode
        if input_mode == "rgb":
            actual_in = 3
        elif input_mode == "rgb_lab":
            actual_in = 6
        elif input_mode == "rgb_lab_hsv":
            actual_in = 5  # RGB + L + V
        else:
            actual_in = in_channels

        self.rgb_input_calibration: nn.Module | None = None
        self.lab_input_calibration: nn.Module | None = None
        if self.use_input_calibration and input_mode == "rgb_lab":
            self.rgb_input_calibration = nn.Sequential(
                nn.Conv2d(3, 3, 1, bias=True),
                ChanLayerNorm(3),
                nn.GELU(),
            )
            self.lab_input_calibration = nn.Sequential(
                nn.Conv2d(3, 3, 1, bias=True),
                ChanLayerNorm(3),
                nn.GELU(),
            )

        # Stem: initial feature extraction with downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(actual_in, channels[0], 3, stride=2, padding=1, bias=True),
            ChanLayerNorm(channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], channels[0], 3, padding=1, bias=True),
            ChanLayerNorm(channels[0]),
            nn.GELU(),
        )

        # Stages
        self.stages = nn.ModuleList()
        current_ch = channels[0]  # Start with stem output channels

        for i in range(num_stages):
            in_ch = current_ch
            out_ch = channels[i]
            stride = 2 if i < 2 else 1  # Downsample first two stages

            if stride > 1:
                # Downsample stage: conv + norm + gelu + colorblock
                self.stages.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=True),
                        ChanLayerNorm(out_ch),
                        nn.GELU(),
                        ColorBlock(out_ch),
                    )
                )
            elif in_ch != out_ch:
                # No downsample but need channel change
                self.stages.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 1, bias=True),  # 1x1 projection
                        ChanLayerNorm(out_ch),
                        nn.GELU(),
                        ColorBlock(out_ch),
                    )
                )
            else:
                # No downsample, no channel change: just colorblock
                self.stages.append(ColorBlock(in_ch))

            current_ch = out_ch

        # Global token computation (skipped entirely when use_global_token=False)
        if self.use_global_token:
            self.histogram = SoftHistogram(num_bins=self.marginal_hist_bins, temperature=0.1)
            self.joint_histogram = (
                SoftHistogram2D(num_bins=self.joint_hist_bins, temperature=0.1)
                if self.joint_hist_bins > 0
                else None
            )
            quantile_dim = len(self.quantile_levels) * 3
            summary_dim = quantile_dim + (6 if self.include_mean_std else 0)
            global_dim = (
                3 * self.marginal_hist_bins
                + (self.joint_hist_bins * self.joint_hist_bins if self.joint_histogram is not None else 0)
                + summary_dim
            )
            self.global_token_proj = nn.Sequential(
                nn.LayerNorm(global_dim),
                nn.Linear(global_dim, token_dim),
                nn.GELU(),
            )
        else:
            self.histogram = None
            self.joint_histogram = None
            self.global_token_proj = None

        # Token projection for stage output
        self.token_proj = nn.Conv2d(channels[-1], token_dim, 1)

        # Apply the configured semi-frozen strategy once construction finishes.
        self.freeze()

    def _apply_frozen_stages(self) -> None:
        """Freeze specified stages."""
        for idx in self.frozen_stages:
            if idx == 0:
                for param in self.stem.parameters():
                    param.requires_grad_(False)
            elif idx - 1 < len(self.stages):
                for param in self.stages[idx - 1].parameters():
                    param.requires_grad_(False)

    def freeze(self) -> None:
        """Apply the configured semi-frozen policy.

        Unlike fully frozen transformer encoders, ColorNAF keeps later stages
        trainable by default and only freezes the stages listed in
        ``frozen_stages``.
        """
        for param in self.parameters():
            param.requires_grad_(True)
        self._apply_frozen_stages()

    def _prepare_input(self, x_rgb: Tensor) -> Tensor:
        """Prepare multi-colorspace input.

        Args:
            x_rgb: (B, 3, H, W), [0, 1]

        Returns:
            Multi-channel input: (B, C, H, W)
        """
        if self.input_mode == "rgb":
            return self._normalize_rgb_branch(x_rgb)

        # Convert to Lab color space
        x_lab = rgb_to_lab(x_rgb)  # (B, 3, H, W)

        if self.input_mode == "rgb_lab":
            rgb_feat = self._normalize_rgb_branch(x_rgb)
            lab_feat = self._normalize_lab_branch(x_lab)
            if self.rgb_input_calibration is not None:
                rgb_feat = self.rgb_input_calibration(rgb_feat)
            if self.lab_input_calibration is not None:
                lab_feat = self.lab_input_calibration(lab_feat)
            return torch.cat([rgb_feat, lab_feat], dim=1)  # (B, 6, H, W)

        # rgb_lab_hsv mode
        rgb_feat = self._normalize_rgb_branch(x_rgb)
        x_lab_norm = self._normalize_lab_branch(x_lab)
        x_v = rgb_to_value(x_rgb)  # (B, 1, H, W)
        return torch.cat([rgb_feat, x_lab_norm[:, 0:1], x_v], dim=1)  # (B, 5, H, W)

    def _normalize_rgb_branch(self, x_rgb: Tensor) -> Tensor:
        if not self.normalize_rgb_input:
            return x_rgb

        mean = self.rgb_mean.to(device=x_rgb.device, dtype=x_rgb.dtype)
        std = self.rgb_std.to(device=x_rgb.device, dtype=x_rgb.dtype)
        return (x_rgb - mean) / std

    def _normalize_lab_branch(self, x_lab: Tensor) -> Tensor:
        if not self.normalize_lab_input:
            return x_lab

        L = (x_lab[:, 0:1] / 100.0).clamp(0.0, 1.0)
        a = (x_lab[:, 1:2] / 128.0).clamp(-1.0, 1.0)
        b = (x_lab[:, 2:3] / 128.0).clamp(-1.0, 1.0)
        return torch.cat([L, a, b], dim=1)

    def _compute_global_token(self, x_rgb: Tensor) -> Tensor:
        x_lab = normalize_lab_for_histogram(rgb_to_lab(x_rgb))
        if self.global_stats_size > 0:
            x_lab = F.adaptive_avg_pool2d(
                x_lab, (self.global_stats_size, self.global_stats_size)
            )

        features = [self.histogram(x_lab)]

        if self.joint_histogram is not None:
            features.append(self.joint_histogram(x_lab[:, 1:3]))

        if self.quantile_levels or self.include_mean_std:
            flat = x_lab.reshape(x_lab.shape[0], 3, -1)
            summary_parts = []

            if self.quantile_levels:
                q = torch.tensor(
                    self.quantile_levels,
                    device=flat.device,
                    dtype=torch.float32,
                )
                quantiles = torch.quantile(flat.float(), q, dim=-1)
                quantiles = quantiles.permute(1, 2, 0).reshape(flat.shape[0], -1)
                summary_parts.append(quantiles.to(flat.dtype))

            if self.include_mean_std:
                mean = flat.mean(dim=-1)
                std = flat.std(dim=-1, unbiased=False)
                summary_parts.append(torch.cat([mean, std], dim=1))

            features.append(torch.cat(summary_parts, dim=1))

        global_feat = torch.cat(features, dim=1)
        return self.global_token_proj(global_feat).unsqueeze(1)

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        """Extract multi-scale features and tokens.

        Args:
            x: Input image (B, 3, H, W), [0, 1]

        Returns:
            (tokens, skip_dict) where:
            - tokens: (B, N+1, token_dim) when use_global_token=True (histogram
              token prepended), otherwise (B, N, token_dim).
            - skip_dict: Multi-scale skip features
        """
        x_rgb = x

        # Prepare multi-colorspace input
        x_input = self._prepare_input(x_rgb)

        # Stem
        x = self.stem(x_input)
        x_stem = x  # H/2, channels[0]
        skips = {}

        stage_outputs: list[Tensor] = []
        for stage in self.stages:
            x = stage(x)
            stage_outputs.append(x)

        if self.align_skip_keys:
            # Keys named by decoder upsample-block index so spatial resolutions
            # match without any bilinear resize:
            #   "stage0" = H/8 (deepest)  → consumed at decoder block 0 (H/8)
            #   "stage1" = H/4            → consumed at decoder block 1 (H/4)
            #   "stage2" = H/2 (stem)     → consumed at decoder block 2 (H/2)
            if 2 in self.skip_stages:
                skips["stage0"] = stage_outputs[-1] if stage_outputs else x
            if 1 in self.skip_stages and stage_outputs:
                skips["stage1"] = stage_outputs[0]
            if 0 in self.skip_stages:
                skips["stage2"] = x_stem
        else:
            # Legacy naming: stage index = encoder depth order (H/2=stage0, H/8=stage2)
            if 0 in self.skip_stages:
                skips["stage0"] = x_stem
            if 1 in self.skip_stages and stage_outputs:
                skips["stage1"] = stage_outputs[0]
            if 2 in self.skip_stages:
                skips["stage2"] = stage_outputs[-1] if stage_outputs else x

        # Project stage output to token dimension
        x_tokens = self.token_proj(x)  # (B, token_dim, h, w)
        x_tokens = x_tokens.flatten(2).transpose(1, 2)  # (B, N, token_dim)

        if self.use_global_token:
            global_token = self._compute_global_token(x_rgb)
            tokens = torch.cat([global_token, x_tokens], dim=1)  # (B, N+1, token_dim)
        else:
            tokens = x_tokens  # (B, N, token_dim)

        return tokens, {"color_naf": skips}

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size


def rgb_to_lab(x: Tensor) -> Tensor:
    """Convert RGB to Lab color space.

    Args:
        x: (B, 3, H, W), [0, 1]

    Returns:
        Lab: (B, 3, H, W), L in [0, 100], a,b in [-128, 127]
    """
    # First convert to linear RGB if sRGB (simplified, assume linear)
    x_linear = x

    # RGB to XYZ (sRGB D65 illuminant)
    r, g, b = x_linear[:, 0:1], x_linear[:, 1:2], x_linear[:, 2:3]

    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    # XYZ to Lab (D65 reference white: Xn=0.95047, Yn=1.0, Zn=1.08883)
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883

    eps = 0.008856
    delta = 6.0 / 29.0

    # Vectorized f function
    def f(t: Tensor) -> Tensor:
        return torch.where(
            t > eps,
            t ** (1/3),
            delta * t + 4.0 / 29.0
        )

    fx = f(X / (Xn + 1e-6))
    fy = f(Y / (Yn + 1e-6))
    fz = f(Z / (Zn + 1e-6))

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_lab = 200 * (fy - fz)  # renamed to avoid conflict

    return torch.cat([L, a, b_lab], dim=1)


def rgb_to_value(x: Tensor) -> Tensor:
    """Extract Value from RGB (HSV color space).

    Args:
        x: (B, 3, H, W), [0, 1]

    Returns:
        V: (B, 1, H, W), [0, 1]
    """
    return x.max(dim=1, keepdim=True).values


def normalize_lab_for_histogram(x_lab: Tensor) -> Tensor:
    """Map Lab channels to [0, 1] for the histogram branch."""
    L = (x_lab[:, 0:1] / 100.0).clamp(0.0, 1.0)
    a = ((x_lab[:, 1:2] + 128.0) / 255.0).clamp(0.0, 1.0)
    b = ((x_lab[:, 2:3] + 128.0) / 255.0).clamp(0.0, 1.0)
    return torch.cat([L, a, b], dim=1)


__all__ = [
    "ColorNAFEncoder",
    "ColorBlock",
    "GRN",
    "SimpleGate",
    "SoftHistogram",
    "SoftHistogram2D",
    "ChanLayerNorm",
    "rgb_to_lab",
    "rgb_to_value",
    "normalize_lab_for_histogram",
]

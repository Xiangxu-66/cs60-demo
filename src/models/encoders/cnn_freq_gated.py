"""Frequency-gated shallow CNN encoder with dense internal U-Net skip fusion."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.encoders.base import BaseEncoder


def _valid_num_groups(channels: int, max_groups: int = 8) -> int:
    """Return a valid GroupNorm group count for the given channel size."""
    for groups in reversed(range(1, max_groups + 1)):
        if channels % groups == 0:
            return groups
    return 1


class ChanLayerNorm(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps (B, C, H, W)."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).to(x.dtype).permute(0, 3, 1, 2)


class GhostDWBlock(nn.Module):
    """Ghost-style depthwise separable block for efficient local extraction."""

    def __init__(self, in_channels: int, out_channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden_channels = max(out_channels // expansion, 8)

        self.primary = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_valid_num_groups(hidden_channels), hidden_channels),
            nn.GELU(),
        )

        self.cheap = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(_valid_num_groups(hidden_channels), hidden_channels),
            nn.GELU(),
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_valid_num_groups(out_channels), out_channels),
            nn.GELU(),
        )

        self.use_residual = in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        primary_feat = self.primary(x)
        cheap_feat = self.cheap(primary_feat)
        out = self.out_proj(torch.cat([primary_feat, cheap_feat], dim=1))
        return out + x if self.use_residual else out


class ResidualChannelSpatialGate(nn.Module):
    """Residual channel-spatial attention for task-aware feature reweighting."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 16)

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        channel_weight = self.channel_gate(x)

        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        spatial_weight = self.spatial_gate(torch.cat([avg_map, max_map], dim=1))

        return x + x * channel_weight * spatial_weight


class FrequencyInjectionBlock(nn.Module):
    """Multi-scale frequency extraction with learnable scale weights and gated injection."""

    def __init__(
        self,
        in_channels: int,
        feature_channels: int,
        num_scales: int = 3,
        scale_init: float = 0.05,
    ) -> None:
        super().__init__()
        if num_scales < 1 or num_scales > 3:
            raise ValueError(f"num_scales must be in [1, 3], got {num_scales}")

        self.num_scales = num_scales
        self.kernels = [3, 5, 7][:num_scales]
        self.scale_logits = nn.Parameter(torch.zeros(num_scales))

        freq_in_channels = in_channels * num_scales

        self.freq_encoder = nn.Sequential(
            GhostDWBlock(freq_in_channels, feature_channels),
            GhostDWBlock(feature_channels, feature_channels),
            ResidualChannelSpatialGate(feature_channels),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.freq_scale = nn.Parameter(torch.tensor(scale_init))

    def _high_frequency_maps(self, x: Tensor) -> Tensor:
        """Extract weighted multi-scale high-frequency maps."""
        weights = torch.softmax(self.scale_logits, dim=0).to(device=x.device, dtype=x.dtype)
        high_maps: list[Tensor] = []

        for i, kernel in enumerate(self.kernels):
            padding = kernel // 2
            blur = F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=padding)
            high_maps.append(weights[i] * (x - blur))

        return torch.cat(high_maps, dim=1)

    def forward(self, image: Tensor, local_feat: Tensor) -> Tensor:
        # Resize first, then extract high-frequency details at the target feature scale.
        image_resized = F.interpolate(
            image,
            size=local_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        high = self._high_frequency_maps(image_resized)
        freq_feat = self.freq_encoder(high)

        gate = self.gate(torch.cat([local_feat, freq_feat], dim=1))
        return local_feat + self.freq_scale * gate * freq_feat


class DenseUNetSkipFusion(nn.Module):
    """Dense internal U-Net skip fusion using all previous encoder features."""

    def __init__(
        self,
        channels: int,
        num_skips: int,
        scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_skips = num_skips

        self.skip_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.GroupNorm(_valid_num_groups(channels), channels),
                nn.GELU(),
            )
            for _ in range(num_skips)
        ])

        self.skip_logits = nn.Parameter(torch.zeros(num_skips))

        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(_valid_num_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )

        self.skip_scale = nn.Parameter(torch.tensor(scale_init))

    def forward(self, x: Tensor, skips: list[Tensor]) -> Tensor:
        if len(skips) != self.num_skips:
            raise ValueError(f"Expected {self.num_skips} skip features, got {len(skips)}")

        weights = torch.softmax(self.skip_logits, dim=0).to(device=x.device, dtype=x.dtype)
        fused_skip = torch.zeros_like(x)

        for i, skip in enumerate(skips):
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(
                    skip,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            skip = self.skip_projs[i](skip)
            fused_skip = fused_skip + weights[i] * skip

        gate = self.gate(torch.cat([x, fused_skip], dim=1))
        fused = x + self.skip_scale * gate * fused_skip

        return fused + self.refine(fused)


class EnhancedShallowCNNStage(nn.Module):
    """Single enhanced stage with local extraction, frequency injection, and attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_channels: int = 3,
        stride: int = 1,
        num_frequency_scales: int = 3,
        freq_scale_init: float = 0.05,
    ) -> None:
        super().__init__()

        self.downsample = None
        if stride > 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                ChanLayerNorm(out_channels),
                nn.GELU(),
            )

        self.local_blocks = nn.Sequential(
            GhostDWBlock(out_channels, out_channels),
            GhostDWBlock(out_channels, out_channels),
        )

        self.freq_injection = FrequencyInjectionBlock(
            in_channels=image_channels,
            feature_channels=out_channels,
            num_scales=num_frequency_scales,
            scale_init=freq_scale_init,
        )

        self.attention = ResidualChannelSpatialGate(out_channels)

    def forward(self, x: Tensor, image: Tensor) -> Tensor:
        if self.downsample is not None:
            x = self.downsample(x)

        x = self.local_blocks(x)
        x = self.freq_injection(image, x)
        x = self.attention(x)

        return x


class CNNFreqGatedEncoder(BaseEncoder):
    """Frequency-gated shallow CNN encoder with dense internal U-Net skip fusion."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_stages: int = 3,
        output_channels: int = 64,
        num_frequency_scales: int = 3,
        freq_scale_init: float = 0.05,
        skip_scale_init: float = 0.1,
    ) -> None:
        super().__init__()

        # stem (stride 2) + stage0 (stride 2) + stage1 (stride 2) = 8× total
        self._patch_size = 8
        self._embed_dim = output_channels
        self._num_frequency_scales = num_frequency_scales

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            ChanLayerNorm(base_channels),
            nn.GELU(),
            GhostDWBlock(base_channels, output_channels),
        )

        self.stem_freq_injection = FrequencyInjectionBlock(
            in_channels=in_channels,
            feature_channels=output_channels,
            num_scales=num_frequency_scales,
            scale_init=freq_scale_init,
        )

        self.stem_attention = ResidualChannelSpatialGate(output_channels)

        self.stages = nn.ModuleList()
        self.dense_skip_fusions = nn.ModuleList()

        current_channels = output_channels
        for i in range(num_stages):
            if i == 0:
                stride = 2
            elif i == 1:
                stride = 2
            else:
                stride = 1

            self.stages.append(
                EnhancedShallowCNNStage(
                    in_channels=current_channels,
                    out_channels=output_channels,
                    image_channels=in_channels,
                    stride=stride,
                    num_frequency_scales=num_frequency_scales,
                    freq_scale_init=freq_scale_init,
                )
            )

            # At stage i, all previously produced features are used as dense U-Net skips.
            self.dense_skip_fusions.append(
                DenseUNetSkipFusion(
                    channels=output_channels,
                    num_skips=i + 1,
                    scale_init=skip_scale_init,
                )
            )

            current_channels = output_channels

        self.final_reweight = ResidualChannelSpatialGate(output_channels)

        # Project final features to token format for DINO fusion in pipeline
        self.token_proj = nn.Conv2d(output_channels, output_channels, 1, bias=True)

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        """Extract enhanced multi-scale local features.

        Returns:
            (tokens, skip_dict) where:
            - tokens: (B, H/8 * W/8, output_channels) spatial tokens for DINO fusion
            - skip_dict: {"color_naf": {"stage0": H/2, "stage1": H/4, "stage2": H/8}}
        """
        x = x.clamp(0.0, 1.0)
        image = x

        features: list[Tensor] = []

        x = self.stem(image)
        x = self.stem_freq_injection(image, x)
        x = self.stem_attention(x)
        features.append(x)   # features[0]: H/2

        for i, stage in enumerate(self.stages):
            x_new = stage(x, image)

            # Dense internal U-Net skip fusion.
            x_new = self.dense_skip_fusions[i](x_new, features)

            if i == len(self.stages) - 1:
                x_new = self.final_reweight(x_new)

            x = x_new
            features.append(x)
        # features[1]: H/4, features[2]: H/8, features[3]: H/8 (final w/ attention)

        skip_dict = {
            "color_naf": {
                "stage0": features[0],  # H/2
                "stage1": features[1],  # H/4
                "stage2": features[2],  # H/8
            }
        }

        tokens = self.token_proj(features[-1]).flatten(2).transpose(1, 2)  # (B, N, C)
        return tokens, skip_dict

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def num_stages(self) -> int:
        return len(self.stages) + 1


__all__ = [
    "ShallowCNNEncoder",
    "ChanLayerNorm",
    "GhostDWBlock",
    "ResidualChannelSpatialGate",
    "FrequencyInjectionBlock",
    "DenseUNetSkipFusion",
    "EnhancedShallowCNNStage",
]
"""Legacy LUT + Shallow CNN decoder preserved for ablation studies.

This module keeps the pre-`693826b` spatial-branch structure side-by-side with
the current coarse-to-fine decoder so experiments can switch between them via
Hydra config without rewriting files.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.decoders.base import BaseDecoder
from src.models.decoders.cnn_decoder_lut_cnn import (
    ChanLayerNorm,
    GlobalLUTBranch,
    ResidualRefineBlock,
    UpsampleBlock,
)


class SpatialBranchWithCNNLegacy(nn.Module):
    """Original Shallow-CNN spatial branch.

    The legacy design starts from the finest ShallowCNN feature (`f0`, H/2),
    then applies progressively coarser skip features while upsampling. This is
    the structure that produced the original `num_upsample_blocks=1` baseline.
    """

    def __init__(
        self,
        cnn_feature_channels: list[int],
        base_channels: int = 64,
        num_upsample_blocks: int = 4,
        num_refine_blocks: int = 1,
    ) -> None:
        super().__init__()

        self.cnn_feature_channels = cnn_feature_channels
        self.num_upsample_blocks = num_upsample_blocks

        channels = self._build_channel_schedule(base_channels, num_upsample_blocks)

        # Legacy behavior: start directly from the finest H/2 feature.
        self.init_proj = nn.Sequential(
            nn.Conv2d(cnn_feature_channels[0], channels[0], 3, padding=1, bias=True),
            ChanLayerNorm(channels[0]),
            nn.GELU(),
        )

        num_pixelshuffle = num_upsample_blocks - 1
        self.upsample_blocks = nn.ModuleList()
        for i in range(num_pixelshuffle):
            cnn_idx = min(i + 1, len(cnn_feature_channels) - 1)
            block = UpsampleBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                skip_channels=cnn_feature_channels[cnn_idx],
                num_refine_blocks=num_refine_blocks,
            )
            self.upsample_blocks.append(block)

        self.final_proj = nn.Sequential(
            ChanLayerNorm(channels[num_pixelshuffle]),
            nn.Conv2d(channels[num_pixelshuffle], channels[-1], 3, padding=1, bias=True),
            nn.GELU(),
        )

        self.final_refine = nn.ModuleList([
            ResidualRefineBlock(channels[-1])
            for _ in range(2)
        ])

        self.weight_head = nn.Sequential(
            ChanLayerNorm(channels[-1]),
            nn.Conv2d(channels[-1], channels[-1] // 2, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels[-1] // 2, 1, 1, bias=True),
            nn.Sigmoid(),
        )

        self.residual_head = nn.Sequential(
            ChanLayerNorm(channels[-1]),
            nn.Conv2d(channels[-1], channels[-1], 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels[-1], 3, 1, bias=True),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    @staticmethod
    def _build_channel_schedule(base: int, num_blocks: int) -> list[int]:
        channels = [base]
        for i in range(1, num_blocks + 1):
            ratio = 1.0 - (0.25 * i / num_blocks)
            channels.append(max(round(base * ratio), 32))
        return channels

    def forward(
        self,
        cnn_features: list[Tensor],
        target_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        x = self.init_proj(cnn_features[0])

        for i, block in enumerate(self.upsample_blocks):
            skip_idx = min(i + 1, len(cnn_features) - 1)
            x = block(x, cnn_features[skip_idx])

        x = self.final_proj(x)

        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

        for block in self.final_refine:
            x = block(x)

        return self.weight_head(x), self.residual_head(x)


class CNNDecoderLUTCNNLegacy(BaseDecoder):
    """Original LUT + Shallow CNN decoder kept as a legacy baseline."""

    def __init__(
        self,
        input_dim: int,
        cnn_feature_channels: list[int] | None = None,
        patch_size: int = 16,
        num_upsample_blocks: int = 4,
        base_channels: int = 64,
        residual_scale: float = 0.5,
        num_luts: int = 5,
        lut_size: int = 33,
        use_1d_lut: bool = True,
    ) -> None:
        super().__init__()

        if cnn_feature_channels is None:
            cnn_feature_channels = [64, 64, 64, 64]

        self.patch_size = patch_size
        self.residual_scale = residual_scale
        self.predict_residual = True

        self.global_branch = GlobalLUTBranch(
            input_dim=input_dim,
            num_luts=num_luts,
            lut_size=lut_size,
            use_1d_lut=use_1d_lut,
        )

        self.spatial_branch = SpatialBranchWithCNNLegacy(
            cnn_feature_channels=cnn_feature_channels,
            base_channels=base_channels,
            num_upsample_blocks=num_upsample_blocks,
        )

    def forward(
        self,
        x: Tensor,
        h: int,
        w: int,
        img: Tensor | None = None,
        cnn_features: list[Tensor] | None = None,
        **kwargs,
    ) -> Tensor:
        B = x.shape[0]

        if img is None:
            img = x.new_zeros(B, 3, h * self.patch_size, w * self.patch_size)
        _, _, H, W = img.shape

        if cnn_features is None:
            cnn_features = [
                torch.zeros(
                    B,
                    64,
                    max(H // (2 ** (i + 1)), 1),
                    max(W // (2 ** (i + 1)), 1),
                    device=x.device,
                    dtype=x.dtype,
                )
                for i in range(4)
            ]

        lut_output = self.global_branch(x, img)
        M, R = self.spatial_branch(cnn_features, target_size=(H, W))

        R_scaled = torch.tanh(R) * self.residual_scale
        enhanced = (1 - M) * lut_output + M * (img + R_scaled)
        enhanced = enhanced.clamp(0, 1)
        return enhanced - img


__all__ = [
    "SpatialBranchWithCNNLegacy",
    "CNNDecoderLUTCNNLegacy",
]

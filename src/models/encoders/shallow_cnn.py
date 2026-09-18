"""Shallow CNN Encoder for multi-scale local texture extraction.

This encoder extracts low-level texture, edge, and color details from the input image.
It serves as a complement to the DINOv3 semantic encoder, providing features that
preserve fine-grained spatial information.

Architecture:
    Input (B, 3, H, W)
        │
        ▼
    Stem: Conv + LayerNorm + GELU
        │
        ├─────────────────────────────────────┐
        ▼                                     ▼
    Stage 1: Conv(stride=2) → Conv           Stage 2: Conv(stride=2) → Conv
        │                                     │
        └─────────────────┬───────────────────┘
                          │
                    Stage 3: Conv(stride=1) → Conv
                          │
                    Output: [f0, f1, f2, f3]
                    - f0: stem output (H/2, W/2)
                    - f1: stage1 output (H/4, W/4)
                    - f2: stage2 output (H/8, W/8)
                    - f3: stage3 output (H/8, W/8)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.encoders.base import BaseEncoder


class ChanLayerNorm(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps (B, C, H, W)."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        # Preserve input dtype to avoid memory spike
        return self.norm(x.permute(0, 2, 3, 1)).to(x.dtype).permute(0, 3, 1, 2)


class ShallowCNNStage(nn.Module):
    """Single stage in the shallow CNN encoder.

    Each stage consists of:
    - Optional downsampling conv
    - Two 3x3 convolutions with LayerNorm and GELU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        layers = []

        # Downsampling conv (if needed)
        if stride > 1:
            layers.append(
                nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=True)
            )
            layers.append(ChanLayerNorm(out_channels))
            layers.append(nn.GELU())
            current_channels = out_channels
        else:
            current_channels = in_channels

        # Two 3x3 convs
        layers.extend([
            nn.Conv2d(current_channels, out_channels, 3, padding=1, bias=True),
            ChanLayerNorm(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
            ChanLayerNorm(out_channels),
            nn.GELU(),
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ShallowCNNEncoder(BaseEncoder):
    """Shallow CNN encoder for multi-scale local texture extraction.

    This encoder extracts low-level features that complement the semantic features
    from DINOv3. It is designed to be lightweight and preserve spatial details.

    Args:
        in_channels: Input image channels (default: 3 for RGB).
        base_channels: Base channel count for the encoder.
        num_stages: Number of stages (default: 3).
        output_channels: Output channels for all stages (unified for easy fusion).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_stages: int = 3,
        output_channels: int = 64,
    ) -> None:
        super().__init__()

        self._patch_size = 4  # Effective downsampling factor
        self._embed_dim = output_channels

        # Stem: initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1, bias=True),
            ChanLayerNorm(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, output_channels, 3, padding=1, bias=True),
            ChanLayerNorm(output_channels),
            nn.GELU(),
        )

        # Stages
        self.stages = nn.ModuleList()
        current_channels = output_channels

        for i in range(num_stages):
            if i == 0:
                # Stage 1: downsample
                stride = 2
                out_ch = output_channels
            elif i == 1:
                # Stage 2: downsample
                stride = 2
                out_ch = output_channels
            else:
                # Stage 3+: no downsampling
                stride = 1
                out_ch = output_channels

            stage = ShallowCNNStage(
                in_channels=current_channels,
                out_channels=out_ch,
                stride=stride,
            )
            self.stages.append(stage)
            current_channels = out_ch

    def forward(self, x: Tensor) -> list[Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input image (B, 3, H, W), values in [0, 1].

        Returns:
            List of feature maps at different scales:
            - f0: (B, output_channels, H/2, W/2) - stem output
            - f1: (B, output_channels, H/4, W/4) - stage 1 output
            - f2: (B, output_channels, H/8, W/8) - stage 2 output
            - f3: (B, output_channels, H/8, W/8) - stage 3 output
        """
        features = []

        # Stem
        x = self.stem(x)
        features.append(x)  # f0: H/2, W/2

        # Stages
        for stage in self.stages:
            x = stage(x)
            features.append(x)  # f1, f2, f3...

        return features

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def num_stages(self) -> int:
        return len(self.stages) + 1  # +1 for stem


__all__ = ["ShallowCNNEncoder", "ChanLayerNorm", "ShallowCNNStage"]

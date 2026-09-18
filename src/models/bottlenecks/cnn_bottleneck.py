"""B-1: CNN bottleneck — local context modeling baseline (O(K²N))."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.models.bottlenecks.base import BaseBottleneck


class ChanLayerNorm(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps (B, C, H, W).

    Per-sample statistics — no batch dependency, consistent with the
    CNN decoder and other restoration-style modules in this codebase.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1).float()).to(x.dtype).permute(0, 3, 1, 2)


class ResConvBlock(nn.Module):
    """Two-layer residual conv block with GELU activation."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=True)
        self.norm1 = ChanLayerNorm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=True)
        self.norm2 = ChanLayerNorm(channels)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class CNNBottleneck(BaseBottleneck):
    """B-1: CNN bottleneck with O(K²N) complexity.

    Reshapes the (B, N, C) token sequence to a 2D feature map
    (B, C, h, w), applies stacked residual conv blocks, then
    flattens back to (B, N, C). Captures local spatial context only.

    Args:
        input_dim: Encoder output dimension (injected by pipeline).
        hidden_dim: Internal channel dimension.
        num_layers: Number of residual conv blocks.
        kernel_size: Conv kernel size.
        drop_rate: Dropout rate after each block.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        num_layers: int = 4,
        kernel_size: int = 3,
        drop_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self._output_dim = hidden_dim

        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResConvBlock(hidden_dim, kernel_size) for _ in range(num_layers)
        ])
        self.drop = nn.Dropout2d(drop_rate)
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply CNN blocks on 2D feature map reshaped from token sequence.

        Args:
            x: Token features (B, N, C_in). N must be a perfect square.

        Returns:
            Processed features (B, N, hidden_dim).
        """
        B, N, _ = x.shape
        h = w = int(math.isqrt(N))
        if h * w != N:
            # Non-square grid: find closest integer factors
            for h in range(int(N ** 0.5), 0, -1):
                if N % h == 0:
                    w = N // h
                    break

        # (B, N, C_in) → (B, hidden_dim, h, w)
        x = self.proj_in(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        for block in self.blocks:
            x = self.drop(block(x))

        # (B, hidden_dim, h, w) → (B, N, hidden_dim)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.norm(x)
        return self.proj_out(x)

    @property
    def output_dim(self) -> int:
        return self._output_dim

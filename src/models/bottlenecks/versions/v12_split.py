"""V12: parameter-efficient RWKV+CNN split bottleneck."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from src.models.bottlenecks.base import BaseBottleneck
from src.models.bottlenecks.cnn_bottleneck import ChanLayerNorm
from src.models.bottlenecks.rwkv_bottleneck import RWKVChannelMix, RWKVTimeMix


def _factor_hw(num_tokens: int) -> tuple[int, int]:
    """Recover a plausible 2D grid for token reshaping."""
    h = int(math.isqrt(num_tokens))
    if h * h == num_tokens:
        return h, h

    for h_try in range(h, 0, -1):
        if num_tokens % h_try == 0:
            return h_try, num_tokens // h_try
    return 1, num_tokens


def _split_dim(hidden_dim: int, rwkv_ratio: float, min_branch_dim: int = 16) -> int:
    """Choose a stable RWKV branch width while leaving room for the CNN branch."""
    if hidden_dim < 2 * min_branch_dim:
        raise ValueError(
            f"hidden_dim must be at least {2 * min_branch_dim}, got {hidden_dim}"
        )

    proposed = int(round(hidden_dim * rwkv_ratio / 16.0) * 16)
    proposed = max(min_branch_dim, proposed)
    proposed = min(proposed, hidden_dim - min_branch_dim)
    return proposed


class LightweightCNNMix(nn.Module):
    """Cheap local spatial mixer for the CNN branch."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
            bias=True,
        )
        self.norm = ChanLayerNorm(channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.GELU()

        nn.init.normal_(self.depthwise.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.depthwise.bias)
        nn.init.normal_(self.pointwise.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x: Tensor, patch_resolution: tuple[int, int]) -> Tensor:
        b, n, c = x.shape
        h, w = patch_resolution
        x_2d = x.transpose(1, 2).reshape(b, c, h, w)
        y = self.depthwise(x_2d)
        y = self.act(self.norm(y))
        y = self.pointwise(y)
        return y.flatten(2).transpose(1, 2).contiguous()


class SplitRWKVBranch(nn.Module):
    """RWKV branch that operates on a reduced channel subset."""

    def __init__(self, dim: int, drop_rate: float, channel_expansion: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.time_mix = RWKVTimeMix(dim)
        self.drop1 = nn.Dropout(drop_rate)

        self.ln2 = nn.LayerNorm(dim)
        self.channel_mix = RWKVChannelMix(dim, expansion_factor=channel_expansion)
        self.drop2 = nn.Dropout(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop1(self.time_mix(self.ln1(x)))
        x = x + self.drop2(self.channel_mix(self.ln2(x)))
        return x


class SplitCNNBranch(nn.Module):
    """CNN branch that keeps local texture modeling cheap."""

    def __init__(self, dim: int, drop_rate: float, kernel_size: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.local_mix = LightweightCNNMix(dim, kernel_size=kernel_size)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: Tensor, patch_resolution: tuple[int, int]) -> Tensor:
        return x + self.drop(self.local_mix(self.ln(x), patch_resolution))


class SplitFusionBlock(nn.Module):
    """Hybrid block: small RWKV branch + larger CNN branch + gated fusion."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        rwkv_dim: int,
        drop_rate: float,
        kernel_size: int,
        rwkv_channel_expansion: int,
        layer_scale_init: float | None,
    ) -> None:
        super().__init__()
        self.rwkv_dim = rwkv_dim
        self.cnn_dim = hidden_dim - rwkv_dim

        self.rwkv_branch = SplitRWKVBranch(
            self.rwkv_dim,
            drop_rate=drop_rate,
            channel_expansion=rwkv_channel_expansion,
        )
        self.cnn_branch = SplitCNNBranch(
            self.cnn_dim,
            drop_rate=drop_rate,
            kernel_size=kernel_size,
        )

        self.fuse_norm = nn.LayerNorm(hidden_dim)
        self.fuse_gate = nn.Linear(hidden_dim, hidden_dim)
        self.fuse_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.drop = nn.Dropout(drop_rate)

        nn.init.zeros_(self.fuse_gate.weight)
        nn.init.zeros_(self.fuse_gate.bias)
        nn.init.normal_(self.fuse_proj.weight, mean=0.0, std=1e-4)

        if layer_scale_init is None:
            self.gamma = None
        else:
            self.gamma = nn.Parameter(torch.full((hidden_dim,), layer_scale_init))

    def forward(self, x: Tensor, patch_resolution: tuple[int, int]) -> Tensor:
        x_rwkv, x_cnn = torch.split(x, [self.rwkv_dim, self.cnn_dim], dim=-1)

        rwkv_out = self.rwkv_branch(x_rwkv)
        cnn_out = self.cnn_branch(x_cnn, patch_resolution)

        branch_state = torch.cat([rwkv_out, cnn_out], dim=-1)
        branch_delta = torch.cat([rwkv_out - x_rwkv, cnn_out - x_cnn], dim=-1)

        gate = torch.sigmoid(self.fuse_gate(self.fuse_norm(branch_state)))
        fused = gate * self.fuse_proj(branch_delta)
        if self.gamma is not None:
            fused = fused * self.gamma

        return x + self.drop(fused)


class RWKVCNNSplitBottleneckV12(BaseBottleneck):
    """Stable first-pass split bottleneck for reduced parameter count."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        num_layers: int = 6,
        rwkv_ratio: float = 1.0 / 3.0,
        rwkv_channel_expansion: int = 2,
        kernel_size: int = 3,
        drop_rate: float = 0.1,
        layer_scale_init: float | None = 1.0e-4,
    ) -> None:
        super().__init__()
        self._output_dim = hidden_dim
        self._hidden_dim = hidden_dim
        self._rwkv_dim = _split_dim(hidden_dim, rwkv_ratio)

        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                SplitFusionBlock(
                    hidden_dim,
                    rwkv_dim=self._rwkv_dim,
                    drop_rate=drop_rate,
                    kernel_size=kernel_size,
                    rwkv_channel_expansion=rwkv_channel_expansion,
                    layer_scale_init=layer_scale_init,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)

        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.normal_(self.proj_out.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: Tensor) -> Tensor:
        patch_resolution = _factor_hw(x.shape[1])
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x, patch_resolution)
        x = self.norm(x)
        return self.proj_out(x)

    @property
    def output_dim(self) -> int:
        return self._output_dim


__all__ = ["RWKVCNNSplitBottleneckV12"]

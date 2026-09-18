"""V12-lite + Axial attention on the CNN branch."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.bottlenecks.base import BaseBottleneck
from src.models.bottlenecks.versions.v12_split import (
    SplitRWKVBranch,
    _factor_hw,
    _split_dim,
)


class AxialAttention2D(nn.Module):
    """Row-column attention over a 2D token grid."""

    def __init__(self, dim: int, num_heads: int, drop_rate: float) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.row_norm = nn.LayerNorm(dim)
        self.row_attn = nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            dropout=drop_rate,
            batch_first=True,
        )
        self.row_drop = nn.Dropout(drop_rate)

        self.col_norm = nn.LayerNorm(dim)
        self.col_attn = nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            dropout=drop_rate,
            batch_first=True,
        )
        self.col_drop = nn.Dropout(drop_rate)

    def forward(self, x: Tensor, patch_resolution: tuple[int, int]) -> Tensor:
        b, n, c = x.shape
        h, w = patch_resolution
        x_2d = x.reshape(b, h, w, c)

        row_tokens = self.row_norm(x_2d).reshape(b * h, w, c)
        row_out = self.row_attn(row_tokens, row_tokens, row_tokens, need_weights=False)[0]
        x_2d = x_2d + self.row_drop(row_out.reshape(b, h, w, c))

        col_tokens = self.col_norm(x_2d.permute(0, 2, 1, 3).contiguous()).reshape(b * w, h, c)
        col_out = self.col_attn(col_tokens, col_tokens, col_tokens, need_weights=False)[0]
        col_out = col_out.reshape(b, w, h, c).permute(0, 2, 1, 3).contiguous()
        x_2d = x_2d + self.col_drop(col_out)

        return x_2d.reshape(b, n, c)


class SplitCNNBranchWithAxial(nn.Module):
    """CNN branch augmented with axial attention."""

    def __init__(
        self,
        dim: int,
        *,
        drop_rate: float,
        kernel_size: int,
        axial_num_heads: int,
    ) -> None:
        super().__init__()
        from src.models.bottlenecks.versions.v12_split import SplitCNNBranch

        self.local_branch = SplitCNNBranch(dim, drop_rate=drop_rate, kernel_size=kernel_size)
        self.axial = AxialAttention2D(dim, num_heads=axial_num_heads, drop_rate=drop_rate)
        self.axial_norm = nn.LayerNorm(dim)
        self.axial_drop = nn.Dropout(drop_rate)

    def forward(self, x: Tensor, patch_resolution: tuple[int, int]) -> Tensor:
        x = self.local_branch(x, patch_resolution)
        return x + self.axial_drop(self.axial(self.axial_norm(x), patch_resolution))


class SplitFusionAxialBlock(nn.Module):
    """Hybrid split block with axial attention injected into the CNN branch."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        rwkv_dim: int,
        drop_rate: float,
        kernel_size: int,
        rwkv_channel_expansion: int,
        layer_scale_init: float | None,
        axial_num_heads: int,
    ) -> None:
        super().__init__()
        self.rwkv_dim = rwkv_dim
        self.cnn_dim = hidden_dim - rwkv_dim

        self.rwkv_branch = SplitRWKVBranch(
            self.rwkv_dim,
            drop_rate=drop_rate,
            channel_expansion=rwkv_channel_expansion,
        )
        self.cnn_branch = SplitCNNBranchWithAxial(
            self.cnn_dim,
            drop_rate=drop_rate,
            kernel_size=kernel_size,
            axial_num_heads=axial_num_heads,
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


class RWKVCNNSplitAxialBottleneckV12(BaseBottleneck):
    """V12-lite variant with axial attention inside the CNN branch."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        num_layers: int = 6,
        rwkv_ratio: float = 1.0 / 3.0,
        rwkv_channel_expansion: int = 2,
        kernel_size: int = 3,
        axial_num_heads: int = 4,
        drop_rate: float = 0.1,
        layer_scale_init: float | None = 1.0e-4,
    ) -> None:
        super().__init__()
        self._output_dim = hidden_dim
        self._hidden_dim = hidden_dim
        self._rwkv_dim = _split_dim(hidden_dim, rwkv_ratio)
        self._cnn_dim = hidden_dim - self._rwkv_dim
        if self._cnn_dim % axial_num_heads != 0:
            raise ValueError(
                f"CNN branch dim {self._cnn_dim} must be divisible by axial_num_heads={axial_num_heads}"
            )

        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                SplitFusionAxialBlock(
                    hidden_dim,
                    rwkv_dim=self._rwkv_dim,
                    drop_rate=drop_rate,
                    kernel_size=kernel_size,
                    rwkv_channel_expansion=rwkv_channel_expansion,
                    layer_scale_init=layer_scale_init,
                    axial_num_heads=axial_num_heads,
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


__all__ = ["RWKVCNNSplitAxialBottleneckV12"]

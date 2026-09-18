"""V12-lite + late MHSA on the last bottleneck blocks."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.bottlenecks.base import BaseBottleneck
from src.models.bottlenecks.versions.v12_split import (
    SplitFusionBlock,
    _factor_hw,
    _split_dim,
)


class LateMHSABlock(nn.Module):
    """Light Transformer-style residual block used only in late layers."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        mlp_ratio: float,
        drop_rate: float,
        layer_scale_init: float | None,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        hidden_dim = int(dim * mlp_ratio)

        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            dropout=drop_rate,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(drop_rate)

        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, dim),
        )
        self.drop2 = nn.Dropout(drop_rate)

        if layer_scale_init is None:
            self.gamma1 = None
            self.gamma2 = None
        else:
            self.gamma1 = nn.Parameter(torch.full((dim,), layer_scale_init))
            self.gamma2 = nn.Parameter(torch.full((dim,), layer_scale_init))

    def forward(self, x: Tensor) -> Tensor:
        attn_out = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
        if self.gamma1 is not None:
            attn_out = attn_out * self.gamma1
        x = x + self.drop1(attn_out)

        mlp_out = self.mlp(self.ln2(x))
        if self.gamma2 is not None:
            mlp_out = mlp_out * self.gamma2
        x = x + self.drop2(mlp_out)
        return x


class RWKVCNNSplitLateMHSABottleneckV12(BaseBottleneck):
    """V12-lite variant with MHSA injected only in the late blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        num_layers: int = 6,
        rwkv_ratio: float = 1.0 / 3.0,
        rwkv_channel_expansion: int = 2,
        kernel_size: int = 3,
        num_late_attention_blocks: int = 2,
        mhsa_num_heads: int = 6,
        mhsa_mlp_ratio: float = 2.0,
        drop_rate: float = 0.1,
        layer_scale_init: float | None = 1.0e-4,
    ) -> None:
        super().__init__()
        if num_late_attention_blocks < 1 or num_late_attention_blocks > num_layers:
            raise ValueError(
                f"num_late_attention_blocks must be in [1, {num_layers}], got {num_late_attention_blocks}"
            )

        self._output_dim = hidden_dim
        self._hidden_dim = hidden_dim
        self._rwkv_dim = _split_dim(hidden_dim, rwkv_ratio)
        self._late_start = num_layers - num_late_attention_blocks

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
        self.late_attention_blocks = nn.ModuleList(
            [
                LateMHSABlock(
                    hidden_dim,
                    num_heads=mhsa_num_heads,
                    mlp_ratio=mhsa_mlp_ratio,
                    drop_rate=drop_rate,
                    layer_scale_init=layer_scale_init,
                )
                for _ in range(num_late_attention_blocks)
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
        for i, block in enumerate(self.blocks):
            x = block(x, patch_resolution)
            if i >= self._late_start:
                x = self.late_attention_blocks[i - self._late_start](x)
        x = self.norm(x)
        return self.proj_out(x)

    @property
    def output_dim(self) -> int:
        return self._output_dim


__all__ = ["RWKVCNNSplitLateMHSABottleneckV12"]

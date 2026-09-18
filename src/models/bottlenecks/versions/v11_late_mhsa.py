"""V11 + Late MHSA: keep the shixin V11 core and add late global attention."""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from src.models.bottlenecks.versions.v11_quad_rwkv import RWKVBottleneckV11, _factor_hw
from src.models.bottlenecks.versions.v12_split_late_mhsa import LateMHSABlock


class RWKVBottleneckV11LateMHSA(RWKVBottleneckV11):
    """Performance-oriented V11 variant with MHSA injected only in late blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        num_layers: int = 6,
        drop_path: float = 0.0,
        drop_rate: float = 0.1,
        num_late_attention_blocks: int = 2,
        mhsa_num_heads: int = 6,
        mhsa_mlp_ratio: float = 2.0,
        layer_scale_init: float | None = 1.0e-4,
    ) -> None:
        if num_late_attention_blocks < 1 or num_late_attention_blocks > num_layers:
            raise ValueError(
                f"num_late_attention_blocks must be in [1, {num_layers}], got {num_late_attention_blocks}"
            )
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            drop_path=drop_path,
            drop_rate=drop_rate,
        )
        self._late_start = num_layers - num_late_attention_blocks
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

    def forward(self, x: Tensor) -> Tensor:
        resolution = _factor_hw(x.shape[1])
        x = self.proj_in(x)
        for i, block in enumerate(self.blocks):
            x = block(x, resolution)
            if i >= self._late_start:
                x = self.late_attention_blocks[i - self._late_start](x)
        x = self.norm(x)
        return self.proj_out(x)


__all__ = ["RWKVBottleneckV11LateMHSA"]

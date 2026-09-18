"""B-2: Transformer bottleneck — O(N²) quality upper bound."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.bottlenecks.base import BaseBottleneck


class TransformerBottleneck(BaseBottleneck):
    """B-2: Full self-attention Transformer bottleneck.

    Serves as the O(N²) quality anchor in the bottleneck comparison.
    Structure mirrors RWKVBottleneckV1: proj_in → blocks → norm → proj_out,
    with standard self-attention + FFN replacing TimeMix + ChannelMix.

    Args:
        input_dim: Encoder output dimension (injected by pipeline).
        hidden_dim: Internal feature dimension.
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.
        mlp_ratio: FFN hidden dimension multiplier.
        drop_rate: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self._output_dim = hidden_dim

        self.proj_in = nn.Linear(input_dim, hidden_dim)

        # Learnable positional embedding (max 4096 tokens, covers up to 64×64 patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, 4096, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=drop_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-Norm, consistent with RWKV blocks
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Process token features through self-attention layers.

        Args:
            x: Encoder output token features of shape (B, N, C_in).

        Returns:
            Processed token features of shape (B, N, hidden_dim).
        """
        x = self.proj_in(x)
        x = x + self.pos_embed[:, : x.shape[1], :]
        x = self.blocks(x)
        x = self.norm(x)
        return self.proj_out(x)

    @property
    def output_dim(self) -> int:
        return self._output_dim

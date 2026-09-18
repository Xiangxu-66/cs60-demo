"""B-2: Transformer bottleneck — O(N²) quality upper bound."""
from __future__ import annotations

# TODO: Implement TransformerBottleneck
#
# Architecture:
#   proj_in  : Linear(input_dim, hidden_dim)
#   blocks   : N × TransformerBlock(hidden_dim, num_heads, mlp_ratio, drop_rate)
#              Each block: pre-norm, nn.MultiheadAttention (batch_first=True), FFN
#   norm     : LayerNorm(hidden_dim)
#   proj_out : Linear(hidden_dim, hidden_dim)
#
# Input/output: (B, N, C_in) → (B, N, hidden_dim)
# Complexity: O(N²) — use as quality anchor in bottleneck comparison

from src.models.bottlenecks.base import BaseBottleneck
from torch import Tensor


class TransformerBottleneck(BaseBottleneck):
    """B-2: Full self-attention Transformer bottleneck.

    TODO: Full implementation pending.
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
        # TODO: build proj_in, blocks, norm, proj_out
        raise NotImplementedError("TransformerBottleneck not yet implemented.")

    def forward(self, x: Tensor) -> Tensor:
        # TODO: implement
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        return self._output_dim

"""B-4: VMamba bottleneck — O(N) SSM peer comparison."""
from __future__ import annotations

# TODO: Implement VMambaBottleneck
#
# Architecture (VMamba / Mamba-inspired):
#   proj_in  : Linear(input_dim, hidden_dim)
#   norms    : N × LayerNorm(hidden_dim)
#   blocks   : N × SSMBlock(hidden_dim, d_state, d_conv, expand, drop_rate)
#              Each SSMBlock:
#                - in_proj: Linear(dim, d_inner * 2) → split into x, z (gating)
#                - conv1d:  depthwise Conv1d for local context
#                - SSM:     selective state-space scan (A matrix, dt projection)
#                - out_proj: Linear(d_inner, dim)
#   norm_out : LayerNorm(hidden_dim)
#   proj_out : Linear(hidden_dim, hidden_dim)
#
# Complexity: O(N) — same order as RWKV, different recurrence mechanism
# Use as same-complexity comparison point (B-4 vs B-3)

from src.models.bottlenecks.base import BaseBottleneck
from torch import Tensor


class VMambaBottleneck(BaseBottleneck):
    """B-4: VMamba selective SSM bottleneck.

    TODO: Full implementation pending.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        num_layers: int = 6,
        drop_rate: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self._output_dim = hidden_dim
        # TODO: build proj_in, norms, blocks, norm_out, proj_out
        raise NotImplementedError("VMambaBottleneck not yet implemented.")

    def forward(self, x: Tensor) -> Tensor:
        # TODO: implement
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        return self._output_dim

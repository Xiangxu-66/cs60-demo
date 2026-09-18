"""A-lstm: Bidirectional LSTM bottleneck (ablation comparison)."""
from __future__ import annotations

# TODO: Implement BiLSTMBottleneck
#
# Architecture:
#   proj_in  : Linear(input_dim, hidden_dim)
#   lstm     : nn.LSTM(hidden_dim, hidden_dim//2, num_layers,
#                      batch_first=True, bidirectional=True,
#                      dropout=drop_rate if num_layers>1 else 0)
#              bidirectional doubles hidden to hidden_dim
#   drop     : Dropout(drop_rate)
#   norm     : LayerNorm(hidden_dim)
#   proj_out : Linear(hidden_dim, hidden_dim)
#
# Purpose: ablation to compare RWKV vs Bi-LSTM as sequence models
# Expected result: RWKV >= Bi-LSTM in quality, better in efficiency

from src.models.bottlenecks.base import BaseBottleneck
from torch import Tensor


class BiLSTMBottleneck(BaseBottleneck):
    """Bi-LSTM bottleneck for A-lstm ablation.

    TODO: Full implementation pending.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        num_layers: int = 3,
        drop_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self._output_dim = hidden_dim
        # TODO: build proj_in, lstm, drop, norm, proj_out
        raise NotImplementedError("BiLSTMBottleneck not yet implemented.")

    def forward(self, x: Tensor) -> Tensor:
        # TODO: implement
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        return self._output_dim

"""B-0: No bottleneck — direct pass-through with linear projection."""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from src.models.bottlenecks.base import BaseBottleneck


class NoneBottleneck(BaseBottleneck):
    """B-0: Pass-through bottleneck.

    Projects encoder features to hidden_dim via a single linear layer
    followed by LayerNorm. No sequence modeling is performed.

    Serves as the lower bound: proves that a bottleneck with sequence
    context improves upon a simple projection.

    Args:
        input_dim: Encoder output dimension (injected by pipeline).
        hidden_dim: Output dimension.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 384) -> None:
        super().__init__()
        self._output_dim = hidden_dim
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Linear projection + LayerNorm.

        Args:
            x: Token features (B, N, C_in).

        Returns:
            Projected features (B, N, hidden_dim).
        """
        return self.norm(self.proj(x))

    @property
    def output_dim(self) -> int:
        return self._output_dim

"""V9: Full DyRSRNet-style RWKV Bottleneck (V7 + V8 Combined).

This version combines all enhancements from V7 and V8 to fully align with
DyRSRNet's RCSSblock architecture:

From V7 (Squared ReLU FFN):
  - EnhancedChannelMix with OSRM + Squared ReLU activation

From V8 (Bidirectional RWKV):
  - BidirectionalRWKVTimeMix with horizontal ↔ vertical recurrence

Together, this creates a complete DyRSRNet-style block with:
  1. Bidirectional spatial recurrence in TimeMix
  2. OSRM for multi-scale spatial modeling (both TimeMix and FFN)
  3. Squared ReLU activation for better feature representation

Reference: DyRSRNet (PRCV'25 Oral)
Source: DyRSRNet/basicsr/archs/DyRSRNet_arch.py:679-736 (RCSSblock)

Problem solved: V7 and V8 only provide partial enhancements.
  - V7: Enhanced FFN only, single-direction TimeMix
  - V8: Enhanced TimeMix only, standard FFN
  - V9: Both TimeMix and FFN fully enhanced

Solution: Combine BidirectionalRWKVTimeMix (V8) + EnhancedChannelMix (V7)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.models.bottlenecks.rwkv_bottleneck import (
    GPTConfig,
    RWKVBottleneckV1,
    RUN_CUDA_OR_CPU,
)
from src.models.bottlenecks.versions.v7_squared_relu import EnhancedChannelMix
from src.models.bottlenecks.versions.v8_bidirectional import BidirectionalRWKVTimeMix
from src.models.common.osrm import OSRM


class FullDyRSRNetBlock(nn.Module):
    """Complete DyRSRNet-style block combining V7 + V8 enhancements.

    Architecture:
      1. TimeMix: BidirectionalRWKVTimeMix
         - OSRM for multi-scale spatial modeling
         - Bidirectional recurrence (H ↔ W)
         - Independent parameters per direction

      2. FFN: EnhancedChannelMix
         - OSRM for multi-scale spatial modeling
         - Squared ReLU activation (k² = (ReLU(k))²)
         - Receptance gating mechanism

    This aligns with DyRSRNet's RCSSblock which combines:
    - Dual DSSM for bidirectional state space modeling (≈ our BidirectionalRWKVTimeMix)
    - VRSE module with Spatial_Receptive_Fusion + Channel_Receptive_Interaction
    """

    def __init__(self, config: GPTConfig, layer_id: int, drop_rate: float = 0.1) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)

        # TimeMix: Bidirectional RWKV + OSRM (from V8)
        self.att = BidirectionalRWKVTimeMix(config, layer_id)

        # FFN: EnhancedChannelMix with OSRM + Squared ReLU (from V7)
        self.ffn = EnhancedChannelMix(config, layer_id)

        self.drop1 = nn.Dropout(drop_rate)
        self.drop2 = nn.Dropout(drop_rate)

    def forward(self, x: Tensor, resolution: tuple[int, int]) -> Tensor:
        """Forward pass with full DyRSRNet enhancements.

        Args:
            x: Input tokens (B, H*W, C)
            resolution: Spatial resolution (h, w)

        Returns:
            Output tokens (B, H*W, C)
        """
        if self.layer_id == 0:
            x = self.ln0(x)
        x = x + self.drop1(self.att(self.ln1(x), resolution))
        x = x + self.drop2(self.ffn(self.ln2(x), resolution))
        return x


class RWKVBottleneckV9(RWKVBottleneckV1):
    """V9: Full DyRSRNet-style RWKV bottleneck (V7 + V8 combined).

    This version combines all enhancements to fully align with DyRSRNet:

    From V7 (Squared ReLU FFN):
      ✓ EnhancedChannelMix with OSRM + Squared ReLU activation

    From V8 (Bidirectional RWKV):
      ✓ BidirectionalRWKVTimeMix with H ↔ W spatial recurrence

    Key features:
    1. Bidirectional spatial recurrence in TimeMix (horizontal + vertical)
    2. OSRM for multi-scale spatial modeling in both TimeMix and FFN
    3. Squared ReLU activation in FFN for better feature representation
    4. Independent spatial parameters per direction

    This is the most complete DyRSRNet integration among all versions.

    Args:
        input_dim: Input feature dimension from encoder
        hidden_dim: Hidden dimension for RWKV blocks
        num_layers: Number of RWKV blocks
        drop_rate: Dropout rate

    Example:
        >>> bottleneck = RWKVBottleneckV9(input_dim=768, hidden_dim=384, num_layers=6)
        >>> x = torch.randn(2, 64, 768)  # (B, H*W, C)
        >>> h, w = 8, 8  # Spatial resolution
        >>> y = bottleneck(x, resolution=(h, w))
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        num_layers: int = 6,
        drop_rate: float = 0.1,
    ) -> None:
        # Initialize parent class (but we'll override blocks)
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            drop_rate=drop_rate,
        )

        # Replace standard blocks with full DyRSRNet blocks
        config = GPTConfig(
            vocab_size=1,
            ctx_len=4096,
            n_embd=hidden_dim,
            n_layer=num_layers,
            model_type="RWKV",
        )

        self.blocks = nn.ModuleList(
            [FullDyRSRNetBlock(config, i, drop_rate=drop_rate) for i in range(num_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Process token features through full DyRSRNet bottleneck.

        Args:
            x: Encoder output token features of shape (B, N, C_in).
               N should be h * w for some spatial resolution.

        Returns:
            Processed token features of shape (B, N, hidden_dim).
        """
        B, N, C = x.shape

        # Infer spatial resolution from sequence length
        # Assume N = h * w, try to find reasonable h, w
        h = w = int(N ** 0.5)
        if h * w != N:
            # Fall back to treating as 1D sequence
            resolution = (1, N)
        else:
            resolution = (h, w)

        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x, resolution)
        x = self.norm(x)
        return self.proj_out(x)


__all__ = ["RWKVBottleneckV9", "FullDyRSRNetBlock"]

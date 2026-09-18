"""V10: Complete DyRSRNet RCSSblock (DSSM + VRSE Combined).

This version implements the FULL DyRSRNet RCSSblock architecture combining:
1. Dual DSSM (4-directional selective scan) - from DSSM bottleneck
2. VRSE module (Visual Recurrent Shift Encoding) - from V9
   - BidirectionalRWKVTimeMix (H ↔ W spatial recurrence + OSRM)
   - EnhancedChannelMix (OSRM + Squared ReLU activation)

This is the MOST COMPLETE DyRSRNet implementation possible, fully aligning
with the DyRSRNet paper's RCSSblock architecture.

Reference: DyRSRNet (PRCV'25 Oral)
Source: DyRSRNet/basicsr/archs/DyRSRNet_arch.py:679-795 (RCSSblock)

Architecture:
  ln_1
  ├─ DSSM_1 (dim//2, 4-directional selective scan)
  ├─ DSSM_2 (dim//2, 4-directional selective scan)
  └─ concat → residual connection
  ln_2
  └─ VRSE (Block with BidirectionalRWKVTimeMix + EnhancedChannelMix)
      └─ residual connection

Note: Requires mamba_ssm for DSSM selective scan.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import repeat
from torch import Tensor

from src.models.bottlenecks.dssm_bottleneck import DSSM, SELECTIVE_SCAN_AVAILABLE
from src.models.bottlenecks.rwkv_bottleneck import (
    GPTConfig,
    RWKVBottleneckV1,
    RUN_CUDA_OR_CPU,
)
from src.models.bottlenecks.versions.v7_squared_relu import EnhancedChannelMix
from src.models.bottlenecks.versions.v8_bidirectional import BidirectionalRWKVTimeMix
from src.models.common.osrm import OSRM


class CompleteRCSSBlock(nn.Module):
    """Complete DyRSRNet RCSSblock combining DSSM + VRSE.

    This is the full implementation of DyRSRNet's RCSSblock:

    Part 1: Dual DSSM (4-directional selective scan)
      - DSSM_1: Processes first half of channels with 4-way scan
      - DSSM_2: Processes second half of channels with 4-way scan
      - Concatenated with residual connection

    Part 2: VRSE (Visual Recurrent Shift Encoding)
      - BidirectionalRWKVTimeMix: H ↔ W spatial recurrence + OSRM
      - EnhancedChannelMix: OSRM + Squared ReLU activation
      - Residual connection

    Args:
        hidden_dim: Hidden dimension
        d_state: SSM state dimension for DSSM
        expand: Expansion factor for DSSM
        drop_path: Drop path rate
        config: RWKV GPTConfig for VRSE components
        layer_id: Layer index for VRSE components
    """

    def __init__(
        self,
        hidden_dim: int,
        d_state: int = 16,
        expand: float = 2.0,
        drop_path: float = 0.0,
        config: GPTConfig | None = None,
        layer_id: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Part 1: Dual DSSM for 4-directional selective scan
        if not SELECTIVE_SCAN_AVAILABLE:
            raise ImportError(
                "mamba_ssm is required for V10. "
                "Install with: pip install mamba-ssm causal-conv1d"
            )

        self.ln_1 = nn.LayerNorm(hidden_dim)

        # Dual DSSM (split channels for bidirectional modeling)
        self.dssm_1 = DSSM(
            d_model=hidden_dim // 2,
            d_state=d_state,
            expand=expand,
        )
        self.dssm_2 = DSSM(
            d_model=hidden_dim // 2,
            d_state=d_state,
            expand=expand,
        )

        self.drop_path = nn.Identity() if drop_path == 0 else nn.Dropout(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))

        # Part 2: VRSE module (from V9)
        self.ln_2 = nn.LayerNorm(hidden_dim)

        if config is None:
            # Create default config
            config = GPTConfig(
                vocab_size=1,
                ctx_len=4096,
                n_embd=hidden_dim,
                n_layer=1,
                model_type="RWKV",
            )

        # VRSE components
        self.att = BidirectionalRWKVTimeMix(config, layer_id)
        self.ffn = EnhancedChannelMix(config, layer_id)

        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input: Tensor, resolution: tuple[int, int]) -> Tensor:
        """Forward pass through complete RCSS block.

        Args:
            input: Input tokens (B, N, C)
            resolution: Spatial resolution (h, w)

        Returns:
            Output tokens (B, N, C)
        """
        B, L, C = input.shape

        # Assume square spatial grid for reshape
        h = w = int(L ** 0.5)
        input = input.view(B, h, w, C).contiguous()

        # Part 1: Dual DSSM (4-directional selective scan)
        x = self.ln_1(input)

        # Split into two halves for dual DSSM
        x1, x2 = torch.chunk(x, 2, dim=-1)

        # Apply dual DSSM
        dssm_out1 = self.dssm_1(x1)
        dssm_out2 = self.dssm_2(x2)

        # Concatenate and apply residual
        dssm_out = torch.cat((dssm_out1, dssm_out2), dim=-1)
        x = input * self.skip_scale + self.drop_path(dssm_out)

        # Part 2: VRSE module (BidirectionalRWKV + EnhancedChannelMix)
        x = self.ln_2(x)
        x = x.view(B, -1, C).contiguous()  # Reshape back to (B, N, C)

        # Bidirectional RWKV TimeMix
        att_out = self.att(x, resolution)
        x = x + att_out

        # Enhanced ChannelMix
        x = x.view(B, h, w, C).contiguous()  # Reshape to (B, H, W, C) for OSRM
        ffn_out = self.ffn(x, resolution)
        ffn_out = ffn_out.view(B, -1, C).contiguous()  # Reshape back to (B, N, C)

        x = x + ffn_out
        x = x.view(B, -1, C).contiguous()

        return x


class RWKVBottleneckV10(RWKVBottleneckV1):
    """V10: Complete DyRSRNet RCSSblock (DSSM + VRSE + OSRM + Squared ReLU).

    This is the MOST COMPLETE DyRSRNet implementation, combining:

    From DSSM bottleneck:
      ✓ 4-directional selective scan (left→right, top→bottom, right→left, bottom→top)
      ✓ Dual DSSM for bidirectional state space modeling

    From V9:
      ✓ BidirectionalRWKVTimeMix (H ↔ W spatial recurrence)
      ✓ EnhancedChannelMix (OSRM + Squared ReLU activation)

    Combined features:
      ✓ OSRM for multi-scale spatial modeling (both TimeMix and FFN)
      ✓ Squared ReLU for better feature representation
      ✓ Dual spatial modeling (DSSM + BidirectionalRWKV)

    Key features:
    1. Part 1: Dual DSSM with 4-directional selective scan
    2. Part 2: VRSE module with:
       - BidirectionalRWKVTimeMix (H ↔ W) + OSRM
       - EnhancedChannelMix + OSRM + Squared ReLU

    This fully aligns with DyRSRNet's RCSSblock architecture.

    Args:
        input_dim: Input feature dimension from encoder
        hidden_dim: Hidden dimension for blocks
        num_layers: Number of RCSS blocks
        d_state: SSM state dimension for DSSM
        expand: Expansion factor for DSSM
        drop_path: Drop path rate
        drop_rate: Dropout rate

    Example:
        >>> bottleneck = RWKVBottleneckV10(input_dim=768, hidden_dim=384, num_layers=6)
        >>> x = torch.randn(2, 64, 768)  # (B, H*W, C)
        >>> h, w = 8, 8  # Spatial resolution
        >>> y = bottleneck(x, resolution=(h, w))

    Note: Requires mamba_ssm to be installed:
        pip install mamba-ssm causal-conv1d
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        num_layers: int = 6,
        d_state: int = 16,
        expand: float = 2.0,
        drop_path: float = 0.0,
        drop_rate: float = 0.1,
    ) -> None:
        # Initialize parent class (but we'll override blocks)
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            drop_rate=drop_rate,
        )

        # Create RWKV config for VRSE components
        config = GPTConfig(
            vocab_size=1,
            ctx_len=4096,
            n_embd=hidden_dim,
            n_layer=num_layers,
            model_type="RWKV",
        )

        # Replace standard blocks with complete RCSS blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                CompleteRCSSBlock(
                    hidden_dim=hidden_dim,
                    d_state=d_state,
                    expand=expand,
                    drop_path=drop_path,
                    config=config,
                    layer_id=i,
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        """Process token features through complete DyRSRNet bottleneck.

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


__all__ = ["RWKVBottleneckV10", "CompleteRCSSBlock"]

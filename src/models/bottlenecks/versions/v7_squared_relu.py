"""V7: RWKV bottleneck + DyRSRNet-style Squared ReLU FFN + OSRM.

This version integrates the Channel_Receptive_Interaction module from DyRSRNet,
which enhances the standard RWKV ChannelMix with:
1. OSRM (Omni-Scale Receptive Module) for multi-scale spatial modeling
2. Squared ReLU activation (ReLU²) for better feature representation

Reference: DyRSRNet (PRCV'25 Oral)
Source: DyRSRNet/basicsr/archs/DyRSRNet_arch.py:232-273

Problem solved: Standard FFN with single-scale processing has limited
receptive field and suboptimal activation for image restoration.

Solution:
- Add OSRM before FFN for multi-scale spatial feature fusion
- Use Squared ReLU (k² = (ReLU(k))²) instead of standard ReLU
- Maintain RWKV's receptance gating mechanism
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from einops import rearrange

from src.models.bottlenecks.rwkv_bottleneck import (
    GPTConfig,
    RWKV_ChannelMix,
    RWKV_TimeMix,
    RWKVBottleneckV1,
)
from src.models.common.osrm import OSRM


class EnhancedChannelMix(nn.Module):
    """RWKV Channel-Mixing enhanced with OSRM and Squared ReLU.

    Compared to standard RWKV_ChannelMix:
    - Adds OSRM for multi-scale spatial modeling
    - Uses Squared ReLU (DyRSRNet-style) instead of standard ReLU
    - Maintains the same input/output interface

    Args:
        config: RWKV GPTConfig containing n_embd and n_layer
        layer_id: Current layer index for parameter initialization
    """

    def __init__(self, config: GPTConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id

        # Time shift for temporal mixing (from RWKV)
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        # Fancy init of time_mix (from RWKV)
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)
            x = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                x[0, 0, i] = i / config.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        # Hidden dimension expansion
        hidden_sz = 4 * config.n_embd

        # Projections
        self.key = nn.Linear(config.n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, config.n_embd, bias=False)

        # OSRM for multi-scale spatial modeling
        self.osrm = OSRM(dim=config.n_embd)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def forward(self, x: Tensor, resolution: tuple[int, int]) -> Tensor:
        """Forward pass with OSRM and Squared ReLU.

        Args:
            x: Input tokens (B, H*W, C)
            resolution: Spatial resolution (h, w)

        Returns:
            Output tokens (B, H*W, C)
        """
        h, w = resolution
        B, T, C = x.shape

        # Time shift mixing (from RWKV)
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        # Apply OSRM for multi-scale spatial modeling
        xk_2d = rearrange(xk, 'b (h w) c -> b c h w', h=h, w=w)
        xk_2d = self.osrm(xk_2d)
        xk = rearrange(xk_2d, 'b c h w -> b (h w) c')

        # Squared ReLU activation (DyRSRNet-style)
        k = self.key(xk)
        k = torch.square(torch.relu(k))  # Squared ReLU
        kv = self.value(k)

        # Receptance gating
        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv


class EnhancedBlock(nn.Module):
    """RWKV Block with enhanced ChannelMix (OSRM + Squared ReLU)."""

    def __init__(self, config: GPTConfig, layer_id: int, drop_rate: float = 0.1) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)

        # Use standard RWKV TimeMix (keep original)
        self.att = RWKV_TimeMix(config, layer_id)

        # Use enhanced ChannelMix with OSRM and Squared ReLU
        self.ffn = EnhancedChannelMix(config, layer_id)

        self.drop1 = nn.Dropout(drop_rate)
        self.drop2 = nn.Dropout(drop_rate)

    def forward(self, x: Tensor, resolution: tuple[int, int]) -> Tensor:
        """Forward pass with spatial resolution awareness.

        Args:
            x: Input tokens (B, H*W, C)
            resolution: Spatial resolution (h, w)

        Returns:
            Output tokens (B, H*W, C)
        """
        if self.layer_id == 0:
            x = self.ln0(x)
        x = x + self.drop1(self.att(self.ln1(x)))
        x = x + self.drop2(self.ffn(self.ln2(x), resolution))
        return x


class RWKVBottleneckV7(RWKVBottleneckV1):
    """V7: RWKV bottleneck with DyRSRNet-style enhancements.

    Key changes from V1:
    1. ChannelMix enhanced with OSRM for multi-scale spatial modeling
    2. Squared ReLU activation (ReLU²) for better feature representation
    3. Spatial resolution passed through blocks for OSRM

    The TimeMix remains unchanged from the original RWKV implementation.

    Args:
        input_dim: Input feature dimension from encoder
        hidden_dim: Hidden dimension for RWKV blocks
        num_layers: Number of RWKV blocks
        drop_rate: Dropout rate

    Example:
        >>> bottleneck = RWKVBottleneckV7(input_dim=768, hidden_dim=384, num_layers=6)
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

        # Replace standard blocks with enhanced blocks
        config = GPTConfig(
            vocab_size=1,
            ctx_len=4096,
            n_embd=hidden_dim,
            n_layer=num_layers,
            model_type="RWKV",
        )

        self.blocks = nn.ModuleList(
            [EnhancedBlock(config, i, drop_rate=drop_rate) for i in range(num_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Process token features through enhanced RWKV bottleneck.

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


__all__ = ["RWKVBottleneckV7", "EnhancedChannelMix", "EnhancedBlock"]

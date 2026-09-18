"""V8: RWKV bottleneck + DyRSRNet-style Bidirectional Spatial Recurrence.

This version integrates the Spatial_Receptive_Fusion module from DyRSRNet,
which enhances RWKV TimeMix with bidirectional spatial recurrence:
- Horizontal direction: standard H×W scan order
- Vertical direction: transposed W×H scan order

Reference: DyRSRNet (PRCV'25 Oral)
Source: DyRSRNet/basicsr/archs/DyRSRNet_arch.py:149-229

Problem solved: Standard RWKV only processes tokens in 1D raster order,
missing cross-directional spatial dependencies (e.g., vertical structures
in horizontal scan).

Solution:
- Add bidirectional recurrence with independent parameters per direction
- First pass: horizontal (H×W order)
- Second pass: vertical (transposed W×H order)
- Combine OSRM for multi-scale spatial modeling
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.models.bottlenecks.rwkv_bottleneck import (
    GPTConfig,
    RWKV_ChannelMix,
    RWKVBottleneckV1,
    RUN_CUDA_OR_CPU,
)
from src.models.common.osrm import OSRM


class BidirectionalRWKVTimeMix(nn.Module):
    """RWKV Time-Mixing with bidirectional spatial recurrence.

    Compared to standard RWKV_TimeMix:
    - Adds bidirectional recurrence (horizontal + vertical)
    - Uses OSRM for multi-scale spatial modeling
    - Independent spatial_decay and spatial_first per direction
    - Maintains the same input/output interface

    Args:
        config: RWKV GPTConfig containing n_embd and n_layer
        layer_id: Current layer index for parameter initialization
    """

    def __init__(self, config: GPTConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        attn_sz = config.n_embd

        # OSRM for multi-scale spatial modeling
        self.osrm = OSRM(dim=config.n_embd)

        # Projections
        self.key = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.value = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.output = nn.Linear(attn_sz, config.n_embd, bias=False)

        # Bidirectional recurrence parameters (2 directions)
        self.recurrence = 2
        with torch.no_grad():
            # Fancy init for spatial decay and first (similar to RWKV)
            ratio_0_to_1 = (layer_id / (config.n_layer - 1)) if config.n_layer > 1 else 0
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)

            # Spatial decay: one set of parameters per direction
            decay_speed = torch.ones((self.recurrence, attn_sz))
            for j in range(self.recurrence):
                for h in range(attn_sz):
                    # Slight variation per direction
                    direction_factor = 1.0 if j == 0 else 0.9
                    decay_speed[j, h] = direction_factor * (-5 + 8 * (h / (attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1))
            self.spatial_decay = nn.Parameter(decay_speed)

            # Spatial first: one set of parameters per direction
            zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(attn_sz)]) * 0.5)
            spatial_first_base = torch.ones(attn_sz) * math.log(0.3) + zigzag
            self.spatial_first = nn.Parameter(
                torch.stack([spatial_first_base * (0.9 if j == 1 else 1.0) for j in range(self.recurrence)])
            )

        # Fancy init for time_mix (from RWKV)
        x = torch.ones(1, 1, config.n_embd)
        for i in range(config.n_embd):
            x[0, 0, i] = i / config.n_embd
        self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
        self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

    def jit_func(self, x: Tensor, resolution: tuple[int, int]) -> tuple[Tensor, Tensor, Tensor]:
        """Mix x with the previous timestep and apply OSRM.

        Args:
            x: Input tokens (B, H*W, C)
            resolution: Spatial resolution (h, w)

        Returns:
            Tuple of (receptance_gate, key, value)
        """
        h, w = resolution

        # Time shift mixing (from RWKV)
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        # Apply OSRM for multi-scale spatial modeling
        xk_2d = rearrange(xk, 'b (h w) c -> b c h w', h=h, w=w)
        xk_2d = self.osrm(xk_2d)
        xk = rearrange(xk_2d, 'b c h w -> b (h w) c')

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x: Tensor, resolution: tuple[int, int]) -> Tensor:
        """Forward pass with bidirectional spatial recurrence.

        Args:
            x: Input tokens (B, H*W, C)
            resolution: Spatial resolution (h, w)

        Returns:
            Output tokens (B, H*W, C)
        """
        B, T, C = x.size()

        # Get projections with OSRM
        sr, k, v = self.jit_func(x, resolution)

        # Bidirectional recurrence
        for j in range(self.recurrence):
            if j % 2 == 0:
                # Horizontal direction: standard H×W order
                v = RUN_CUDA_OR_CPU(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
            else:
                # Vertical direction: transposed W×H order
                h, w = resolution
                k_t = rearrange(k, 'b (h w) c -> b (w h) c', h=h, w=w)
                v_t = rearrange(v, 'b (h w) c -> b (w h) c', h=h, w=w)
                v_t = RUN_CUDA_OR_CPU(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k_t, v_t)
                # Restore original order
                k = rearrange(k_t, 'b (w h) c -> b (h w) c', h=h, w=w)
                v = rearrange(v_t, 'b (w h) c -> b (h w) c', h=h, w=w)

        # Apply receptance gate and output projection
        x = sr * v
        x = self.output(x)
        return x


class BidirectionalBlock(nn.Module):
    """RWKV Block with bidirectional TimeMix."""

    def __init__(self, config: GPTConfig, layer_id: int, drop_rate: float = 0.1) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)

        # Use bidirectional TimeMix
        self.att = BidirectionalRWKVTimeMix(config, layer_id)

        # Use standard RWKV ChannelMix (could also use EnhancedChannelMix from V7)
        self.ffn = RWKV_ChannelMix(config, layer_id)

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
        x = x + self.drop1(self.att(self.ln1(x), resolution))
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x


class RWKVBottleneckV8(RWKVBottleneckV1):
    """V8: RWKV bottleneck with bidirectional spatial recurrence.

    Key changes from V1:
    1. TimeMix enhanced with bidirectional recurrence (H ↔ W)
    2. OSRM for multi-scale spatial modeling in TimeMix
    3. Independent spatial parameters per direction

    The ChannelMix remains standard RWKV (unlike V7 which also enhances it).

    Args:
        input_dim: Input feature dimension from encoder
        hidden_dim: Hidden dimension for RWKV blocks
        num_layers: Number of RWKV blocks
        drop_rate: Dropout rate

    Example:
        >>> bottleneck = RWKVBottleneckV8(input_dim=768, hidden_dim=384, num_layers=6)
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

        # Replace standard blocks with bidirectional blocks
        config = GPTConfig(
            vocab_size=1,
            ctx_len=4096,
            n_embd=hidden_dim,
            n_layer=num_layers,
            model_type="RWKV",
        )

        self.blocks = nn.ModuleList(
            [BidirectionalBlock(config, i, drop_rate=drop_rate) for i in range(num_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Process token features through bidirectional RWKV bottleneck.

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


__all__ = ["RWKVBottleneckV8", "BidirectionalRWKVTimeMix", "BidirectionalBlock"]

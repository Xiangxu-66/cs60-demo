"""V11: Quad-Directional RWKV Bottleneck (Mamba-free replacement for V10).

This version replaces V10's DSSM (which requires mamba_ssm) with a pure PyTorch
4-directional RWKV implementation.

Architecture Comparison:
  V10: Dual DSSM (4-way selective scan) + VRSE
  V11: Quad-DirectionalRWKV (4-way RWKV recurrence) + VRSE

The 4 directions are:
  1. Forward Horizontal: H×W (left→right, top→bottom)
  2. Forward Vertical: W×H (top→bottom, left→right) transposed
  3. Backward Horizontal: reverse H×W (right→left, bottom→top)
  4. Backward Vertical: reverse W×H (bottom→top, right→left) transposed

This provides the same directional coverage as DSSM without requiring mamba_ssm.

Reference: DyRSRNet (PRCV'25 Oral)
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


class QuadDirectionalRWKVTimeMix(nn.Module):
    """RWKV Time-Mixing with QUAD-directional spatial recurrence.

    Extends BidirectionalRWKVTimeMix with 4 directions:
    - Forward H×W (horizontal scan)
    - Forward W×H (vertical scan)
    - Backward H×W (reverse horizontal scan)
    - Backward W×H (reverse vertical scan)

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

        # Quad-directional recurrence parameters (4 directions)
        self.recurrence = 4
        with torch.no_grad():
            # Fancy init for spatial decay and first (similar to RWKV)
            ratio_0_to_1 = (layer_id / (config.n_layer - 1)) if config.n_layer > 1 else 0
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)

            # Spatial decay: one set of parameters per direction
            decay_speed = torch.ones((self.recurrence, attn_sz))
            for j in range(self.recurrence):
                for h in range(attn_sz):
                    # Slight variation per direction for diversity
                    direction_factor = [1.0, 0.9, 0.95, 0.85][j]
                    decay_speed[j, h] = direction_factor * (-5 + 8 * (h / (attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1))
            self.spatial_decay = nn.Parameter(decay_speed)

            # Spatial first: one set of parameters per direction
            zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(attn_sz)]) * 0.5)
            spatial_first_base = torch.ones(attn_sz) * math.log(0.3) + zigzag
            self.spatial_first = nn.Parameter(
                torch.stack([spatial_first_base * factor for factor in [1.0, 0.9, 0.95, 0.85]])
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
        """Forward pass with quad-directional spatial recurrence.

        Args:
            x: Input tokens (B, H*W, C)
            resolution: Spatial resolution (h, w)

        Returns:
            Output tokens (B, H*W, C)
        """
        B, T, C = x.size()
        h, w = resolution

        # Get projections with OSRM
        sr, k, v = self.jit_func(x, resolution)

        # Quad-directional recurrence
        # We accumulate results from all directions
        v_accum = torch.zeros_like(v)

        for j in range(self.recurrence):
            if j == 0:
                # Direction 1: Forward H×W (horizontal left→right, top→bottom)
                v_dir = RUN_CUDA_OR_CPU(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
                v_accum = v_accum + v_dir

            elif j == 1:
                # Direction 2: Forward W×H (vertical top→bottom, left→right) transposed
                k_t = rearrange(k, 'b (h w) c -> b (w h) c', h=h, w=w)
                v_t = rearrange(v, 'b (h w) c -> b (w h) c', h=h, w=w)
                v_t = RUN_CUDA_OR_CPU(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k_t, v_t)
                # Restore original order and accumulate
                v_dir = rearrange(v_t, 'b (w h) c -> b (h w) c', h=h, w=w)
                v_accum = v_accum + v_dir

            elif j == 2:
                # Direction 3: Backward H×W (horizontal right→left, bottom→top)
                k_rev = torch.flip(k, dims=[1])  # Reverse sequence
                v_rev = torch.flip(v, dims=[1])
                v_rev = RUN_CUDA_OR_CPU(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k_rev, v_rev)
                # Flip back and accumulate
                v_dir = torch.flip(v_rev, dims=[1])
                v_accum = v_accum + v_dir

            else:  # j == 3
                # Direction 4: Backward W×H (vertical bottom→top, right→left) transposed
                k_t = rearrange(k, 'b (h w) c -> b (w h) c', h=h, w=w)
                v_t = rearrange(v, 'b (h w) c -> b (w h) c', h=h, w=w)
                k_t_rev = torch.flip(k_t, dims=[1])  # Reverse transposed sequence
                v_t_rev = torch.flip(v_t, dims=[1])
                v_t_rev = RUN_CUDA_OR_CPU(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k_t_rev, v_t_rev)
                # Flip back, restore order, and accumulate
                v_dir = rearrange(torch.flip(v_t_rev, dims=[1]), 'b (w h) c -> b (h w) c', h=h, w=w)
                v_accum = v_accum + v_dir

        # Average over 4 directions and apply receptance gate
        v_final = v_accum / self.recurrence
        x = sr * v_final
        x = self.output(x)
        return x


class CompleteRCSSBlockV11(nn.Module):
    """Complete DyRSRNet RCSSblock (Quad-Directional RWKV + VRSE).

    This is the Mamba-free version of V10's CompleteRCSSBlock:
    - Part 1: Quad-Directional RWKV (replaces Dual DSSM)
    - Part 2: VRSE module (BidirectionalRWKVTimeMix + EnhancedChannelMix)

    Args:
        hidden_dim: Hidden dimension
        d_state: SSM state dimension (kept for API compatibility, not used)
        expand: Expansion factor (kept for API compatibility, not used)
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

        # Part 1: Quad-Directional RWKV (replaces Dual DSSM)
        self.ln_1 = nn.LayerNorm(hidden_dim)

        # Create RWKV config
        if config is None:
            config = GPTConfig(
                vocab_size=1,
                ctx_len=4096,
                n_embd=hidden_dim,
                n_layer=1,
                model_type="RWKV",
            )

        # Use quad-directional RWKV for the first part
        self.quad_rwkv = QuadDirectionalRWKVTimeMix(config, layer_id)

        self.drop_path = nn.Identity() if drop_path == 0 else nn.Dropout(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))

        # Part 2: VRSE module (from V9)
        self.ln_2 = nn.LayerNorm(hidden_dim)

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
        h = w = int(L ** 0.5)

        # Part 1: Quad-Directional RWKV (replaces Dual DSSM)
        x = self.ln_1(input)
        quad_out = self.quad_rwkv(x, resolution)
        x = input * self.skip_scale + self.drop_path(quad_out)

        # Part 2: VRSE module (BidirectionalRWKV + EnhancedChannelMix)
        x = self.ln_2(x)

        # Bidirectional RWKV TimeMix
        att_out = self.att(x, resolution)
        x = x + att_out

        # Enhanced ChannelMix (expects B, N, C format)
        ffn_out = self.ffn(x, resolution)

        x = x + ffn_out

        return x


class RWKVBottleneckV11(RWKVBottleneckV1):
    """V11: Quad-Directional RWKV Bottleneck (Mamba-free V10 replacement).

    This is the Mamba-free version of V10, replacing DSSM with quad-directional
    RWKV recurrence:

    From V10 (kept):
      ✓ VRSE module (BidirectionalRWKVTimeMix + EnhancedChannelMix)
      ✓ OSRM for multi-scale spatial modeling
      ✓ Squared ReLU activation

    From V10 (replaced):
      ✗ Dual DSSM (required mamba_ssm)
      ✓ Quad-Directional RWKV (pure PyTorch, same directional coverage)

    Key features:
    1. Part 1: Quad-Directional RWKV with 4-way spatial recurrence
    2. Part 2: VRSE module with BidirectionalRWKVTimeMix + EnhancedChannelMix

    The 4 directions provide the same coverage as DSSM's selective scan:
      - Forward horizontal (H×W)
      - Forward vertical (W×H transposed)
      - Backward horizontal (reverse H×W)
      - Backward vertical (reverse W×H transposed)

    Args:
        input_dim: Input feature dimension from encoder
        hidden_dim: Hidden dimension for blocks
        num_layers: Number of RCSS blocks
        drop_path: Drop path rate
        drop_rate: Dropout rate

    Example:
        >>> bottleneck = RWKVBottleneckV11(input_dim=768, hidden_dim=384, num_layers=6)
        >>> x = torch.randn(2, 64, 768)  # (B, H*W, C)
        >>> y = bottleneck(x)

    Note: Pure PyTorch implementation, no mamba_ssm required.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        num_layers: int = 6,
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

        # Replace standard blocks with complete RCSS blocks (V11 version)
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                CompleteRCSSBlockV11(
                    hidden_dim=hidden_dim,
                    drop_path=drop_path,
                    config=config,
                    layer_id=i,
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        """Process token features through quad-directional RWKV bottleneck.

        Args:
            x: Encoder output token features of shape (B, N, C_in).
               N should be h * w for some spatial resolution.

        Returns:
            Processed token features of shape (B, N, hidden_dim).
        """
        B, N, C = x.shape

        # Infer spatial resolution from sequence length
        h = w = int(N ** 0.5)
        if h * w != N:
            resolution = (1, N)
        else:
            resolution = (h, w)

        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x, resolution)
        x = self.norm(x)
        return self.proj_out(x)


__all__ = ["RWKVBottleneckV11", "CompleteRCSSBlockV11", "QuadDirectionalRWKVTimeMix"]

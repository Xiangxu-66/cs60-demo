"""RWKV-7 bottleneck for image enhancement.

This implementation uses the official RWKV-7 training implementation from
``src/models/rwkv_lm/RWKV7Official.py``, which includes:
- Official RWKV7_CLAMPW_CUDA kernel (forward + backward)
- Official TimeMix/ChannelMix with complete initialization
- PyTorch fallback for CPU inference

Preserves the standard bottleneck interface: (B, N, C_in) -> (B, N, C_out).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.bottlenecks.base import BaseBottleneck

# Import official RWKV7 implementation
from src.models.rwkv_lm.RWKV7Official import (
    Args,
    Block as OfficialBlock,
    RWKV_Tmix_x070,
    RWKV_CMix_x070,
    _RWKV7_CLAMPW_AVAILABLE,
    RWKV7_CLAMPW_CUDA,
    RWKV7_PYTORCH,
    HEAD_SIZE,
    CHUNK_LEN,
)


def _factorize_grid(num_tokens: int) -> tuple[int, int]:
    """Infer a plausible 2-D grid for patch tokens."""
    h = int(math.isqrt(num_tokens))
    if h * h == num_tokens:
        return h, h

    for candidate in range(h, 0, -1):
        if num_tokens % candidate == 0:
            return candidate, num_tokens // candidate

    return 1, num_tokens


def _pad_to_chunk_len(x: Tensor, chunk_len: int) -> tuple[Tensor, int]:
    """Pad the token dimension to be divisible by chunk_len."""
    length = x.shape[1]
    remainder = length % chunk_len
    if remainder == 0:
        return x, length

    pad_len = chunk_len - remainder
    pad_shape = list(x.shape)
    pad_shape[1] = pad_len
    padded = torch.cat([x, x.new_zeros(pad_shape)], dim=1)
    return padded, length


########################################################################################################
# Spatial Shift Modules
########################################################################################################

class OmniShift(nn.Module):
    """Multi-branch depthwise spatial shift borrowed from the visual RWKV7 sketch."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(dim, dim, 1, groups=dim, bias=False)
        self.conv3x3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.conv5x5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.alpha = nn.Parameter(torch.randn(4))
        self.conv5x5_reparam = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self._reparam_pending = True

    def forward_train(self, x: Tensor) -> Tensor:
        return (
            self.alpha[0] * x
            + self.alpha[1] * self.conv1x1(x)
            + self.alpha[2] * self.conv3x3(x)
            + self.alpha[3] * self.conv5x5(x)
        )

    def _reparam_5x5(self) -> None:
        w1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2))
        w3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1))
        wi = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2))
        combined = (
            self.alpha[0] * wi
            + self.alpha[1] * w1
            + self.alpha[2] * w3
            + self.alpha[3] * self.conv5x5.weight
        )
        self.conv5x5_reparam.weight = nn.Parameter(
            combined.to(self.conv5x5_reparam.weight.device)
        )
        self._reparam_pending = False

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            self._reparam_pending = True
            return self.forward_train(x)
        if self._reparam_pending:
            self._reparam_5x5()
        return self.conv5x5_reparam(x)


class SimpleShift1D(nn.Module):
    """Official 1-D time shift from RWKV x070."""

    def __init__(self) -> None:
        super().__init__()
        self.shift = nn.ZeroPad2d((0, 0, 1, -1))

    def forward(self, x: Tensor, resolution: tuple[int, int]) -> Tensor:
        del resolution
        return self.shift(x)


class SpatialShift2D(nn.Module):
    """2-D patch-token shift used by the visual RWKV7 sketch."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.omni_shift = OmniShift(dim)

    def forward(self, x: Tensor, resolution: tuple[int, int]) -> Tensor:
        h, w = resolution
        B, T, C = x.shape
        if h * w != T:
            raise ValueError(
                f"SpatialShift2D expected h*w == num_tokens, got {h}*{w} != {T}"
            )
        x_2d = x.transpose(1, 2).reshape(B, C, h, w)
        x_2d = self.omni_shift(x_2d)
        return x_2d.reshape(B, C, T).transpose(1, 2)


########################################################################################################
# RWKV7 Core Operations
########################################################################################################

def run_rwkv7_core(
    r: Tensor,
    w: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    *,
    head_size: int,
    use_cuda_kernel: bool = True,
) -> Tensor:
    """Run the RWKV7 core op with CUDA or PyTorch fallback.

    Args:
        r: (B, T, C) receptance
        w: (B, T, C) decay (before clamping)
        k: (B, T, C) key
        v: (B, T, C) value
        a: (B, T, C) aaa
        b: (B, T, C) aaa * k
        head_size: head size for multi-head
        use_cuda_kernel: whether to use CUDA kernel

    Returns:
        y: (B, T, C) output
    """
    if use_cuda_kernel and _RWKV7_CLAMPW_AVAILABLE and r.is_cuda:
        # Official CUDA kernel requires bfloat16 and chunk padding
        orig_dtype = r.dtype
        orig_len = r.shape[1]

        # Pad to chunk length
        r_pad, _ = _pad_to_chunk_len(r, CHUNK_LEN)
        w_pad, _ = _pad_to_chunk_len(w, CHUNK_LEN)
        k_pad, _ = _pad_to_chunk_len(k, CHUNK_LEN)
        v_pad, _ = _pad_to_chunk_len(v, CHUNK_LEN)
        a_pad, _ = _pad_to_chunk_len(a, CHUNK_LEN)
        b_pad, _ = _pad_to_chunk_len(b, CHUNK_LEN)

        # Convert to bfloat16 for CUDA kernel
        r_pad = r_pad.to(torch.bfloat16)
        w_pad = w_pad.to(torch.bfloat16)
        k_pad = k_pad.to(torch.bfloat16)
        v_pad = v_pad.to(torch.bfloat16)
        a_pad = a_pad.to(torch.bfloat16)
        b_pad = b_pad.to(torch.bfloat16)

        y = RWKV7_CLAMPW_CUDA(r_pad, w_pad, k_pad, v_pad, a_pad, b_pad)
        return y[:, :orig_len, :].to(orig_dtype)

    # PyTorch fallback (supports any dtype and device)
    return RWKV7_PYTORCH(r, w, k, v, a, b, head_size=head_size)


def run_rwkv7_core_bidirectional(
    r: Tensor,
    w: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    *,
    head_size: int,
    use_cuda_kernel: bool = True,
) -> Tensor:
    """Average forward and backward scan directions for image tokens."""
    y_fwd = run_rwkv7_core(
        r, w, k, v, a, b, head_size=head_size, use_cuda_kernel=use_cuda_kernel
    )
    y_bwd = run_rwkv7_core(
        r.flip(1),
        w.flip(1),
        k.flip(1),
        v.flip(1),
        a.flip(1),
        b.flip(1),
        head_size=head_size,
        use_cuda_kernel=use_cuda_kernel,
    ).flip(1)
    return 0.5 * (y_fwd + y_bwd)


########################################################################################################
# Configuration
########################################################################################################

@dataclass(frozen=True)
class RWKV7Config:
    """Configuration for RWKV7 bottleneck.

    Note: LoRA dimensions are calculated dynamically if set to None.
    """
    head_size: int = 64
    ffn_ratio: float = 4.0
    # LoRA dimensions (None = auto-calculate like official)
    decay_lora_dim: int | None = None
    aaa_lora_dim: int | None = None
    mv_lora_dim: int | None = None
    gate_lora_dim: int | None = None
    use_omni_shift: bool = False
    bidirectional: bool = False
    use_cuda_kernel: bool = True


def _calc_lora_dim(dim: int, base: float) -> int:
    """Calculate LoRA dimension using official formula."""
    return max(32, int(round((base * (dim**0.5)) / 32) * 32))


########################################################################################################
# RWKV7 TimeMix (Adapter for Official Implementation)
########################################################################################################

class RWKV7TimeMix(nn.Module):
    """RWKV-7 x070-style TimeMix using official implementation.

    This adapts the official RWKV_Tmix_x070 to work with the bottleneck interface,
    adding support for:
    - Spatial resolution
    - Bidirectional scanning
    - Flexible CUDA/PyTorch selection
    """

    def __init__(self, dim: int, n_layer: int, layer_id: int, cfg: RWKV7Config) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.head_size = cfg.head_size
        self.n_head = dim // cfg.head_size
        self.bidirectional = cfg.bidirectional
        self.use_cuda_kernel = cfg.use_cuda_kernel

        # Create Args object for official implementation
        class BottleneckArgs:
            def __init__(self):
                # The official init divides by (n_layer - 1), so keep a
                # one-block bottleneck on a safe two-layer schedule.
                self.n_layer = max(n_layer, 2)
                self.n_embd = dim
                self.dim_att = dim
                self.head_size = cfg.head_size
                self.my_testing = "x070"

                # Override LoRA dimensions if specified
                if cfg.decay_lora_dim is not None:
                    self.D_DECAY_LORA = cfg.decay_lora_dim
                if cfg.aaa_lora_dim is not None:
                    self.D_AAA_LORA = cfg.aaa_lora_dim
                if cfg.mv_lora_dim is not None:
                    self.D_MV_LORA = cfg.mv_lora_dim
                if cfg.gate_lora_dim is not None:
                    self.D_GATE_LORA = cfg.gate_lora_dim

        args = BottleneckArgs()

        # Use official TimeMix internally
        self._official_tmix = RWKV_Tmix_x070(args, layer_id)

        # Spatial shift for bottleneck interface
        self.spatial_shift: nn.Module
        if cfg.use_omni_shift:
            self.spatial_shift = SpatialShift2D(dim)
        else:
            self.spatial_shift = SimpleShift1D()

    @property
    def receptance(self) -> nn.Linear:
        return self._official_tmix.receptance

    @property
    def key(self) -> nn.Linear:
        return self._official_tmix.key

    @property
    def value(self) -> nn.Linear:
        return self._official_tmix.value

    @property
    def output(self) -> nn.Linear:
        return self._official_tmix.output

    def forward(
        self,
        x: Tensor,
        v_first: Tensor,
        resolution: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        B, T, C = x.shape
        H = self.n_head
        N = self.head_size

        # Apply spatial shift (bottleneck interface requirement)
        xx = self.spatial_shift(x, resolution) - x

        xr = x + xx * self._official_tmix.x_r
        xw = x + xx * self._official_tmix.x_w
        xk = x + xx * self._official_tmix.x_k
        xv = x + xx * self._official_tmix.x_v
        xa = x + xx * self._official_tmix.x_a
        xg = x + xx * self._official_tmix.x_g

        r = self._official_tmix.receptance(xr)
        w = self._official_tmix.w0 + torch.tanh(xw @ self._official_tmix.w1) @ self._official_tmix.w2
        k = self._official_tmix.key(xk)
        v = self._official_tmix.value(xv)

        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(
                self._official_tmix.v0 + (xv @ self._official_tmix.v1) @ self._official_tmix.v2
            )

        a = torch.sigmoid(self._official_tmix.a0 + (xa @ self._official_tmix.a1) @ self._official_tmix.a2)
        g = torch.sigmoid(xg @ self._official_tmix.g1) @ self._official_tmix.g2

        kk = F.normalize((k * self._official_tmix.k_k).view(B, T, H, N), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self._official_tmix.k_a)

        a_wkv = -kk
        b_wkv = kk * a

        if self.bidirectional:
            y = run_rwkv7_core_bidirectional(
                r, w, k, v, a_wkv, b_wkv, head_size=self.head_size, use_cuda_kernel=self.use_cuda_kernel
            )
        else:
            y = run_rwkv7_core(
                r, w, k, v, a_wkv, b_wkv, head_size=self.head_size, use_cuda_kernel=self.use_cuda_kernel
            )

        y = self._official_tmix.ln_x(y.view(B * T, C)).view(B, T, C)
        y = y + (
            (r.view(B, T, H, N) * k.view(B, T, H, N) * self._official_tmix.r_k)
            .sum(dim=-1, keepdim=True)
            * v.view(B, T, H, N)
        ).view(B, T, C)

        return self._official_tmix.output(y * g), v_first


########################################################################################################
# RWKV7 ChannelMix (Adapter for Official Implementation)
########################################################################################################

class RWKV7ChannelMix(nn.Module):
    """RWKV-7 x070-style ChannelMix using official implementation."""

    def __init__(self, dim: int, n_layer: int, layer_id: int, cfg: RWKV7Config) -> None:
        super().__init__()
        dim_ffn = int(dim * cfg.ffn_ratio)

        # Spatial shift
        self.spatial_shift: nn.Module
        if cfg.use_omni_shift:
            self.spatial_shift = SpatialShift2D(dim)
        else:
            self.spatial_shift = SimpleShift1D()

        # Create Args for official implementation
        class BottleneckArgs:
            def __init__(self):
                self.n_layer = n_layer
                self.n_embd = dim
                self.my_testing = "x070"

        # Use official ChannelMix internally
        self._official_cmix = RWKV_CMix_x070(BottleneckArgs(), layer_id)

        # Override with custom FFN ratio if needed
        if dim_ffn != dim * 4:  # Official uses 4x
            self._official_cmix.key = nn.Linear(dim, dim_ffn, bias=False)
            self._official_cmix.value = nn.Linear(dim_ffn, dim, bias=False)

    def forward(self, x: Tensor, resolution: tuple[int, int]) -> Tensor:
        xx = self.spatial_shift(x, resolution) - x
        k = x + xx * self._official_cmix.x_k
        return self._official_cmix.value(F.relu(self._official_cmix.key(k)) ** 2)


########################################################################################################
# RWKV7 Block
########################################################################################################

class RWKV7Block(nn.Module):
    """Pre-norm RWKV7 block with optional dropout for training stability."""

    def __init__(
        self,
        dim: int,
        n_layer: int,
        layer_id: int,
        cfg: RWKV7Config,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln0 = nn.LayerNorm(dim) if layer_id == 0 else None

        self.att = RWKV7TimeMix(dim, n_layer, layer_id, cfg)
        self.ffn = RWKV7ChannelMix(dim, n_layer, layer_id, cfg)

        self.drop1 = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()
        self.drop2 = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

    def forward(
        self,
        x: Tensor,
        v_first: Tensor,
        resolution: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        if self.ln0 is not None:
            x = self.ln0(x)

        att_out, v_first = self.att(self.ln1(x), v_first, resolution)
        x = x + self.drop1(att_out)
        x = x + self.drop2(self.ffn(self.ln2(x), resolution))
        return x, v_first


########################################################################################################
# RWKV7 Bottleneck
########################################################################################################

class RWKV7Bottleneck(BaseBottleneck):
    """RWKV-7 bottleneck with official x070 implementation.

    Uses the official RWKV-7 training implementation from RWKV7Official.py,
    including:
    - Official RWKV7_CLAMPW_CUDA kernel (forward + backward)
    - Official parameter initialization with zigzag/linear modulation
    - Dynamic LoRA dimension calculation
    - PyTorch fallback for CPU inference

    Args:
        input_dim: Input channel dimension
        hidden_dim: Hidden channel dimension (must be divisible by head_size)
        num_layers: Number of RWKV7 blocks
        head_size: Head size for multi-head attention (default: 64)
        ffn_ratio: FFN expansion ratio (default: 4.0)
        decay_lora_dim: Decay LoRA dimension (None = auto)
        aaa_lora_dim: AAA LoRA dimension (None = auto)
        mv_lora_dim: MV LoRA dimension (None = auto)
        gate_lora_dim: Gate LoRA dimension (None = auto)
        use_omni_shift: Use OmniShift instead of SimpleShift1D
        bidirectional: Use bidirectional scanning
        use_cuda_kernel: Use CUDA kernel when available
        drop_rate: Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        num_layers: int = 6,
        head_size: int = 64,
        ffn_ratio: float = 4.0,
        decay_lora_dim: int | None = None,
        aaa_lora_dim: int | None = None,
        mv_lora_dim: int | None = None,
        gate_lora_dim: int | None = None,
        use_omni_shift: bool = False,
        bidirectional: bool = False,
        use_cuda_kernel: bool = True,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % head_size != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by head_size={head_size}"
            )

        self._output_dim = hidden_dim
        self._hidden_dim = hidden_dim

        cfg = RWKV7Config(
            head_size=head_size,
            ffn_ratio=ffn_ratio,
            decay_lora_dim=decay_lora_dim,
            aaa_lora_dim=aaa_lora_dim,
            mv_lora_dim=mv_lora_dim,
            gate_lora_dim=gate_lora_dim,
            use_omni_shift=use_omni_shift,
            bidirectional=bidirectional,
            use_cuda_kernel=use_cuda_kernel,
        )

        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                RWKV7Block(
                    hidden_dim,
                    num_layers,
                    layer_id=i,
                    cfg=cfg,
                    drop_rate=drop_rate,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: Tensor,
        resolution: tuple[int, int] | None = None,
    ) -> Tensor:
        B, N, _ = x.shape
        if resolution is None:
            resolution = _factorize_grid(N)

        x = self.proj_in(x)
        v_first = x.new_zeros((B, N, self._hidden_dim))
        for block in self.blocks:
            x, v_first = block(x, v_first, resolution)
        x = self.norm(x)
        return self.proj_out(x)

    @property
    def output_dim(self) -> int:
        return self._output_dim


__all__ = [
    "RWKV7Bottleneck",
    "RWKV7Block",
    "RWKV7TimeMix",
    "RWKV7ChannelMix",
    "RWKV7Config",
    "run_rwkv7_core",
    "run_rwkv7_core_bidirectional",
    "_RWKV7_CLAMPW_AVAILABLE",
]

"""V6 ★: RWKV bottleneck + Mamba-style selective decay.

Bringing the selective-SSM mechanism from Mamba (Gu & Dao, 2023) into the
RWKV WKV recurrence: the per-channel fixed ``time_decay`` becomes an
input-dependent, per-token decay, computed by a Mamba ``dt`` module.

Baseline architecture is the official RWKV-4 bottleneck (V1). Only the
WKV recurrence inside each TimeMix block is modified:

    Official RWKV-4:   w = -exp(time_decay)                 # (C,), fixed
    V6 (this file):    w_t = dt_t * (-exp(time_decay))      # (B,T,C), selective

where ``dt_t`` follows the official Mamba implementation exactly:
    dt_t = softplus(dt_proj(x_proj(x_t)))
with initialisation ported verbatim from
``state-spaces/mamba/mamba_ssm/modules/mamba_simple.py``.

Because the per-token decay cannot use the stock fp32 WKV CUDA kernel
(which assumes a channel-only decay vector), V6 always runs the pure
PyTorch recurrence. Complexity stays O(N).
"""
from __future__ import annotations

import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.bottlenecks.rwkv_bottleneck import RWKVBottleneckV1


def _wkv_selective(
    time_decay: Tensor,
    time_first: Tensor,
    k: Tensor,
    v: Tensor,
    dt: Tensor,
) -> Tensor:
    """Per-token selective WKV recurrence (pure PyTorch, O(N)).

    Args:
        time_decay: (C,) base log-decay parameter (same as RWKV).
        time_first: (C,) time_first "u" bonus (same as RWKV).
        k, v: (B, T, C) key / value projections.
        dt: (B, T, C) positive input-dependent time step (Mamba selective).

    Returns:
        (B, T, C) WKV output with the same math as the CUDA kernel but
        using ``w_t = dt_t * (-exp(time_decay))`` per step.
    """
    dtype = k.dtype
    k = k.float()
    v = v.float()
    u = time_first.float()
    base_w = -torch.exp(time_decay.float())  # (C,), negative
    dt = dt.float()

    B, T, C = k.shape
    p = torch.zeros(B, C, device=k.device, dtype=torch.float32)
    q = torch.zeros_like(p)
    o = torch.full_like(p, -1e38)

    outputs: list[Tensor] = []
    for t in range(T):
        kt = k[:, t, :]
        vt = v[:, t, :]

        # Current-step output using time_first bonus (identical to RWKV WKV).
        no = torch.maximum(o, u + kt)
        a = torch.exp(o - no)
        b = torch.exp(u + kt - no)
        outputs.append((a * p + b * vt) / (a * q + b).clamp_min(1e-9))

        # State update with PER-TOKEN decay: w_t = dt_t * base_w.
        w_t = dt[:, t, :] * base_w  # (B, C)
        no2 = torch.maximum(w_t + o, kt)
        a2 = torch.exp(w_t + o - no2)
        b2 = torch.exp(kt - no2)
        p = a2 * p + b2 * vt
        q = a2 * q + b2
        o = no2

    return torch.stack(outputs, dim=1).to(dtype=dtype)


class SelectiveTimeMix(nn.Module):
    """RWKV-4 TimeMix with Mamba-style selective decay.

    Every parameter shared with ``RWKV_TimeMix`` keeps the *official RWKV-4*
    initialisation. The new ``x_proj`` / ``dt_proj`` modules follow the
    *official Mamba* initialisation verbatim.
    """

    def __init__(
        self,
        dim: int,
        layer_id: int,
        n_layer: int,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.5,
        dt_max: float = 1.5,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer

        # ----- Standard RWKV-4 TimeMix init (identical to rwkv_bottleneck.py) -----
        attn_sz = dim
        with torch.no_grad():
            denom_layer = max(n_layer - 1, 1)
            ratio_0_to_1 = layer_id / denom_layer
            ratio_1_to_almost0 = 1.0 - (layer_id / max(n_layer, 1))

            decay_speed = torch.empty(attn_sz)
            denom_h = max(attn_sz - 1, 1)
            for h in range(attn_sz):
                decay_speed[h] = -5 + 8 * (h / denom_h) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)

            zigzag = torch.tensor(
                [(i + 1) % 3 - 1 for i in range(attn_sz)], dtype=torch.float32
            ) * 0.5
            self.time_first = nn.Parameter(torch.ones(attn_sz) * math.log(0.3) + zigzag)

            x = torch.arange(dim, dtype=torch.float32).view(1, 1, dim) / dim
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(dim, attn_sz, bias=False)
        self.value = nn.Linear(dim, attn_sz, bias=False)
        self.receptance = nn.Linear(dim, attn_sz, bias=False)
        self.output = nn.Linear(attn_sz, dim, bias=False)

        nn.init.zeros_(self.key.weight)
        nn.init.zeros_(self.receptance.weight)
        nn.init.zeros_(self.output.weight)
        nn.init.orthogonal_(self.value.weight, gain=1.0)

        # ----- Mamba selective delta (ported from mamba_simple.py verbatim) -----
        if dt_rank == "auto":
            dt_rank = math.ceil(dim / 16)
        self.dt_rank = int(dt_rank)

        # x -> low-rank dt; dt -> full channel dim
        self.x_proj = nn.Linear(dim, self.dt_rank, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, dim, bias=True)

        # dt_proj.weight init
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"dt_init={dt_init!r}")

        # dt_proj.bias init: log-uniform sample in [dt_min, dt_max], then inverse-softplus
        dt = torch.exp(
            torch.rand(dim) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse of softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Mirror Mamba's flag so external init routines don't reset this bias.
        self.dt_proj.bias._no_reinit = True

    def forward(self, x: Tensor) -> Tensor:
        # Standard RWKV-4 token-mix (identical to RWKV_TimeMix).
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        # Mamba selective delta: dt_t = softplus(dt_proj(x_proj(x_t))).
        dt = F.softplus(self.dt_proj(self.x_proj(x)))  # (B, T, dim)

        wkv = _wkv_selective(self.time_decay, self.time_first, k, v, dt)
        return self.output(sr * wkv)


class RWKVBottleneckV6(RWKVBottleneckV1):
    """V6 ★: RWKV-4 bottleneck with Mamba-style selective decay.

    Inherits every component from V1 (official RWKV-4) and replaces only
    the ``time_mix`` sub-module inside each block with ``SelectiveTimeMix``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        num_layers: int = 6,
        drop_rate: float = 0.0,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.5,
        dt_max: float = 1.5,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        **_unused,  # absorb legacy V2-V5 config keys
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            drop_rate=drop_rate,
        )
        for i, block in enumerate(self.blocks):
            block.time_mix = SelectiveTimeMix(
                hidden_dim,
                layer_id=i,
                n_layer=num_layers,
                dt_rank=dt_rank,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init=dt_init,
                dt_scale=dt_scale,
                dt_init_floor=dt_init_floor,
            )

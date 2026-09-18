"""V11: Quad-directional RWKV bottleneck adapted from the shixin branch."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.models.bottlenecks.base import BaseBottleneck
from src.models.bottlenecks.rwkv_bottleneck import (
    RWKVBottleneckV1,
    _shift_tokens,
    _wkv_reference,
)
from src.models.common.osrm import OSRM


def _factor_hw(num_tokens: int) -> tuple[int, int]:
    """Recover a plausible 2D grid for token reshaping."""
    h = int(math.isqrt(num_tokens))
    if h * h == num_tokens:
        return h, h
    for h_try in range(h, 0, -1):
        if num_tokens % h_try == 0:
            return h_try, num_tokens // h_try
    return 1, num_tokens


def _build_time_mix(dim: int, layer_id: int, num_layers: int, exponent: float = 1.0) -> nn.Parameter:
    ratio = 1.0 - (layer_id / num_layers)
    x = torch.ones(1, 1, dim)
    for i in range(dim):
        x[0, 0, i] = i / dim
    return nn.Parameter(torch.pow(x, exponent * ratio))


def _run_wkv(k: Tensor, v: Tensor, decay: Tensor, first: Tensor) -> Tensor:
    w = -torch.exp(decay)
    return _wkv_reference(w=w, u=first, k=k, v=v)


class EnhancedChannelMix(nn.Module):
    """DyRSRNet-style ChannelMix with OSRM and squared ReLU."""

    def __init__(self, dim: int, layer_id: int, num_layers: int) -> None:
        super().__init__()
        self.time_mix_k = _build_time_mix(dim, layer_id, num_layers)
        self.time_mix_r = _build_time_mix(dim, layer_id, num_layers)

        hidden_sz = 4 * dim
        self.key = nn.Linear(dim, hidden_sz, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(hidden_sz, dim, bias=False)
        self.osrm = OSRM(dim=dim)

    def forward(self, x: Tensor, resolution: tuple[int, int]) -> Tensor:
        h, w = resolution
        xx = _shift_tokens(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        xk_2d = rearrange(xk, "b (h w) c -> b c h w", h=h, w=w)
        xk_2d = self.osrm(xk_2d)
        xk = rearrange(xk_2d, "b c h w -> b (h w) c")

        k = torch.square(torch.relu(self.key(xk)))
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


class BidirectionalRWKVTimeMix(nn.Module):
    """Bidirectional spatial RWKV time-mixing with OSRM."""

    def __init__(self, dim: int, layer_id: int, num_layers: int) -> None:
        super().__init__()
        self.dim = dim
        self.osrm = OSRM(dim=dim)

        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

        ratio_0_to_1 = (layer_id / (num_layers - 1)) if num_layers > 1 else 0.0
        ratio_1_to_almost0 = 1.0 - (layer_id / num_layers)

        decay_speed = torch.ones((2, dim))
        with torch.no_grad():
            for j in range(2):
                direction_factor = 1.0 if j == 0 else 0.9
                for h in range(dim):
                    decay_speed[j, h] = direction_factor * (
                        -5 + 8 * (h / max(dim - 1, 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                    )
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(dim)], dtype=torch.float32) * 0.5
            spatial_first_base = torch.ones(dim) * math.log(0.3) + zigzag
            spatial_first = torch.stack(
                [spatial_first_base * (0.9 if j == 1 else 1.0) for j in range(2)]
            )
        self.spatial_decay = nn.Parameter(decay_speed)
        self.spatial_first = nn.Parameter(spatial_first)

        x = torch.ones(1, 1, dim)
        for i in range(dim):
            x[0, 0, i] = i / dim
        self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
        self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

    def _project(self, x: Tensor, resolution: tuple[int, int]) -> tuple[Tensor, Tensor, Tensor]:
        h, w = resolution
        xx = _shift_tokens(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        xk_2d = rearrange(xk, "b (h w) c -> b c h w", h=h, w=w)
        xk_2d = self.osrm(xk_2d)
        xk = rearrange(xk_2d, "b c h w -> b (h w) c")

        k = self.key(xk)
        v = self.value(xv)
        sr = torch.sigmoid(self.receptance(xr))
        return sr, k, v

    def forward(self, x: Tensor, resolution: tuple[int, int]) -> Tensor:
        h, w = resolution
        sr, k, v = self._project(x, resolution)

        v_h = _run_wkv(k, v, self.spatial_decay[0], self.spatial_first[0])
        k_t = rearrange(k, "b (h w) c -> b (w h) c", h=h, w=w)
        v_t = rearrange(v_h, "b (h w) c -> b (w h) c", h=h, w=w)
        v_t = _run_wkv(k_t, v_t, self.spatial_decay[1], self.spatial_first[1])
        v_out = rearrange(v_t, "b (w h) c -> b (h w) c", h=h, w=w)

        return self.output(sr * v_out)


class QuadDirectionalRWKVTimeMix(nn.Module):
    """RWKV time-mixing with four spatial scan directions."""

    def __init__(self, dim: int, layer_id: int, num_layers: int) -> None:
        super().__init__()
        self.osrm = OSRM(dim=dim)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

        ratio_0_to_1 = (layer_id / (num_layers - 1)) if num_layers > 1 else 0.0
        ratio_1_to_almost0 = 1.0 - (layer_id / num_layers)
        decay_speed = torch.ones((4, dim))
        with torch.no_grad():
            direction_factors = [1.0, 0.9, 0.95, 0.85]
            for j in range(4):
                for h in range(dim):
                    decay_speed[j, h] = direction_factors[j] * (
                        -5 + 8 * (h / max(dim - 1, 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                    )
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(dim)], dtype=torch.float32) * 0.5
            spatial_first_base = torch.ones(dim) * math.log(0.3) + zigzag
            spatial_first = torch.stack([spatial_first_base * f for f in direction_factors])
        self.spatial_decay = nn.Parameter(decay_speed)
        self.spatial_first = nn.Parameter(spatial_first)

        x = torch.ones(1, 1, dim)
        for i in range(dim):
            x[0, 0, i] = i / dim
        self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
        self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

    def _project(self, x: Tensor, resolution: tuple[int, int]) -> tuple[Tensor, Tensor, Tensor]:
        h, w = resolution
        xx = _shift_tokens(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        xk_2d = rearrange(xk, "b (h w) c -> b c h w", h=h, w=w)
        xk_2d = self.osrm(xk_2d)
        xk = rearrange(xk_2d, "b c h w -> b (h w) c")

        k = self.key(xk)
        v = self.value(xv)
        sr = torch.sigmoid(self.receptance(xr))
        return sr, k, v

    def forward(self, x: Tensor, resolution: tuple[int, int]) -> Tensor:
        h, w = resolution
        sr, k, v = self._project(x, resolution)
        v_accum = torch.zeros_like(v)

        v_dir = _run_wkv(k, v, self.spatial_decay[0], self.spatial_first[0])
        v_accum = v_accum + v_dir

        k_t = rearrange(k, "b (h w) c -> b (w h) c", h=h, w=w)
        v_t = rearrange(v, "b (h w) c -> b (w h) c", h=h, w=w)
        v_t = _run_wkv(k_t, v_t, self.spatial_decay[1], self.spatial_first[1])
        v_accum = v_accum + rearrange(v_t, "b (w h) c -> b (h w) c", h=h, w=w)

        k_rev = torch.flip(k, dims=[1])
        v_rev = torch.flip(v, dims=[1])
        v_rev = _run_wkv(k_rev, v_rev, self.spatial_decay[2], self.spatial_first[2])
        v_accum = v_accum + torch.flip(v_rev, dims=[1])

        k_t_rev = torch.flip(k_t, dims=[1])
        v_t_rev = torch.flip(v_t, dims=[1])
        v_t_rev = _run_wkv(k_t_rev, v_t_rev, self.spatial_decay[3], self.spatial_first[3])
        v_accum = v_accum + rearrange(
            torch.flip(v_t_rev, dims=[1]),
            "b (w h) c -> b (h w) c",
            h=h,
            w=w,
        )

        return self.output(sr * (v_accum / 4.0))


class CompleteRCSSBlockV11(nn.Module):
    """Quad-directional RWKV followed by the V9/VRSE refinement path."""

    def __init__(self, hidden_dim: int, num_layers: int, layer_id: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.quad_rwkv = QuadDirectionalRWKVTimeMix(hidden_dim, layer_id, num_layers)
        self.drop_path = nn.Identity() if drop_path == 0 else nn.Dropout(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.att = BidirectionalRWKVTimeMix(hidden_dim, layer_id, num_layers)
        self.ffn = EnhancedChannelMix(hidden_dim, layer_id, num_layers)

    def forward(self, inputs: Tensor, resolution: tuple[int, int]) -> Tensor:
        x = self.ln_1(inputs)
        quad_out = self.quad_rwkv(x, resolution)
        x = inputs * self.skip_scale + self.drop_path(quad_out)

        x = self.ln_2(x)
        x = x + self.att(x, resolution)
        x = x + self.ffn(x, resolution)
        return x


class RWKVBottleneckV11(RWKVBottleneckV1):
    """Performance-oriented V11 bottleneck from the shixin branch."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        num_layers: int = 6,
        drop_path: float = 0.0,
        drop_rate: float = 0.1,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            drop_rate=drop_rate,
        )
        self.blocks = nn.ModuleList(
            [
                CompleteRCSSBlockV11(
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    layer_id=i,
                    drop_path=drop_path,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        resolution = _factor_hw(x.shape[1])
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x, resolution)
        x = self.norm(x)
        return self.proj_out(x)


__all__ = [
    "RWKVBottleneckV11",
    "CompleteRCSSBlockV11",
    "QuadDirectionalRWKVTimeMix",
    "BidirectionalRWKVTimeMix",
    "EnhancedChannelMix",
]

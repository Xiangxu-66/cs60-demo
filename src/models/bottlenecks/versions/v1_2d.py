"""V1: RWKV4-based 2D bottleneck with VRWKV-inspired spatial inductive bias."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from src.models.bottlenecks.base import BaseBottleneck


def _factor_hw(num_tokens: int) -> tuple[int, int]:
    """Recover a plausible 2D grid for ``num_tokens``."""
    h = int(math.isqrt(num_tokens))
    if h * h == num_tokens:
        return h, h

    for h_try in range(h, 0, -1):
        if num_tokens % h_try == 0:
            return h_try, num_tokens // h_try
    return 1, num_tokens


def q_shift(
    x: Tensor,
    *,
    shift_pixel: int = 1,
    channel_gamma: float = 0.25,
    patch_resolution: tuple[int, int] | None = None,
) -> Tensor:
    """VRWKV Q-Shift over token grids."""
    if shift_pixel <= 0:
        return x

    b, n, c = x.shape
    if patch_resolution is None:
        patch_resolution = _factor_hw(n)
    h, w = patch_resolution
    if h * w != n:
        raise ValueError(f"Invalid patch resolution {patch_resolution} for {n} tokens")

    x_2d = x.transpose(1, 2).reshape(b, c, h, w)
    out = torch.zeros_like(x_2d)

    quarter = int(c * channel_gamma)
    if quarter > 0:
        out[:, 0:quarter, :, shift_pixel:w] = x_2d[:, 0:quarter, :, 0 : w - shift_pixel]
        out[:, quarter : 2 * quarter, :, 0 : w - shift_pixel] = x_2d[
            :, quarter : 2 * quarter, :, shift_pixel:w
        ]
        out[:, 2 * quarter : 3 * quarter, shift_pixel:h, :] = x_2d[
            :, 2 * quarter : 3 * quarter, 0 : h - shift_pixel, :
        ]
        out[:, 3 * quarter : 4 * quarter, 0 : h - shift_pixel, :] = x_2d[
            :, 3 * quarter : 4 * quarter, shift_pixel:h, :
        ]

    if 4 * quarter < c:
        out[:, 4 * quarter :, :, :] = x_2d[:, 4 * quarter :, :, :]

    return out.flatten(2).transpose(1, 2).contiguous()


def _wkv_reference(w: Tensor, u: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Pure PyTorch WKV recurrence matching the project CUDA reference."""
    if k.shape != v.shape:
        raise ValueError(f"k and v must share shape, got {k.shape} vs {v.shape}")

    dtype = k.dtype
    k = k.float()
    v = v.float()
    w = w.float()
    u = u.float()

    batch, timesteps, channels = k.shape
    p = torch.zeros(batch, channels, device=k.device, dtype=k.dtype)
    q = torch.zeros_like(p)
    o = torch.full_like(p, -1e38)
    outputs: list[Tensor] = []

    for t in range(timesteps):
        kt = k[:, t, :]
        vt = v[:, t, :]

        no = torch.maximum(o, u + kt)
        a = torch.exp(o - no)
        b = torch.exp(u + kt - no)
        denom = (a * q + b).clamp_min(1e-9)
        yt = (a * p + b * vt) / denom
        outputs.append(yt)

        no = torch.maximum(w + o, kt)
        a = torch.exp(w + o - no)
        b = torch.exp(kt - no)
        p = a * p + b * vt
        q = a * q + b
        o = no

    return torch.stack(outputs, dim=1).to(dtype=dtype)


def _bi_wkv_reference(w: Tensor, u: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Bidirectional WKV approximation used by the VRWKV-style V1 block."""
    forward = _wkv_reference(w=w, u=u, k=k, v=v)
    backward = torch.flip(
        _wkv_reference(
            w=w,
            u=u,
            k=torch.flip(k, dims=[1]),
            v=torch.flip(v, dims=[1]),
        ),
        dims=[1],
    )
    return 0.5 * (forward + backward)


class RWKVSpatialMixV1(nn.Module):
    """Spatial mix with VRWKV-style init and bidirectional WKV."""

    def __init__(
        self,
        dim: int,
        *,
        num_layers: int,
        layer_id: int,
        shift_pixel: int = 1,
        channel_gamma: float = 0.25,
        key_norm: bool = True,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_layers = num_layers
        self.dim = dim
        self.shift_pixel = shift_pixel
        self.channel_gamma = channel_gamma
        self.bidirectional = bidirectional

        self._fancy_init()

        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.key_norm = nn.LayerNorm(dim) if key_norm else None

        nn.init.xavier_uniform_(self.key.weight, gain=0.5)
        nn.init.xavier_uniform_(self.value.weight, gain=0.5)
        nn.init.xavier_uniform_(self.receptance.weight, gain=0.5)
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-4)

    def _fancy_init(self) -> None:
        with torch.no_grad():
            denom = max(self.num_layers - 1, 1)
            ratio_0_to_1 = self.layer_id / denom
            ratio_1_to_almost0 = 1.0 - (self.layer_id / max(self.num_layers, 1))

            decay_speed = torch.ones(self.dim)
            for i in range(self.dim):
                pos = i / max(self.dim - 1, 1)
                decay_speed[i] = -5.0 + 8.0 * (pos ** (0.7 + 1.3 * ratio_0_to_1))
            self.spatial_decay = nn.Parameter(decay_speed)

            zigzag = torch.tensor(
                [(i + 1) % 3 - 1 for i in range(self.dim)],
                dtype=torch.float32,
            ) * 0.5
            self.spatial_first = nn.Parameter(
                torch.ones(self.dim) * math.log(0.3) + zigzag
            )

            x = torch.linspace(0.0, 1.0, self.dim).view(1, 1, self.dim)
            self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.spatial_mix_v = nn.Parameter(
                torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1
            )
            self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

    def forward(self, x: Tensor, patch_resolution: tuple[int, int]) -> Tensor:
        xx = q_shift(
            x,
            shift_pixel=self.shift_pixel,
            channel_gamma=self.channel_gamma,
            patch_resolution=patch_resolution,
        )
        xk = x * self.spatial_mix_k + xx * (1.0 - self.spatial_mix_k)
        xv = x * self.spatial_mix_v + xx * (1.0 - self.spatial_mix_v)
        xr = x * self.spatial_mix_r + xx * (1.0 - self.spatial_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))

        timesteps = x.shape[1]
        w = self.spatial_decay / timesteps
        u = self.spatial_first / timesteps
        y = (
            _bi_wkv_reference(w=w, u=u, k=k, v=v)
            if self.bidirectional
            else _wkv_reference(w=w, u=u, k=k, v=v)
        )
        if self.key_norm is not None:
            y = self.key_norm(y)
        return self.output(r * y)


class RWKVChannelMixV1(nn.Module):
    """Channel mix with the same Q-Shift inductive bias."""

    def __init__(
        self,
        dim: int,
        *,
        num_layers: int,
        layer_id: int,
        shift_pixel: int = 1,
        channel_gamma: float = 0.25,
        hidden_rate: int = 4,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_layers = num_layers
        self.dim = dim
        self.shift_pixel = shift_pixel
        self.channel_gamma = channel_gamma

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (self.layer_id / max(self.num_layers, 1))
            x = torch.linspace(0.0, 1.0, self.dim).view(1, 1, self.dim)
            self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        hidden_dim = dim * hidden_rate
        self.key = nn.Linear(dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)

        nn.init.xavier_uniform_(self.key.weight, gain=0.5)
        nn.init.normal_(self.value.weight, mean=0.0, std=1e-4)
        nn.init.normal_(self.receptance.weight, mean=0.0, std=1e-4)

    def forward(self, x: Tensor, patch_resolution: tuple[int, int]) -> Tensor:
        xx = q_shift(
            x,
            shift_pixel=self.shift_pixel,
            channel_gamma=self.channel_gamma,
            patch_resolution=patch_resolution,
        )
        xk = x * self.spatial_mix_k + xx * (1.0 - self.spatial_mix_k)
        xr = x * self.spatial_mix_r + xx * (1.0 - self.spatial_mix_r)

        k = torch.square(torch.relu(self.key(xk)))
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


class RWKVBlockV1(nn.Module):
    """RWKV4 residual block with small initial residual branches."""

    def __init__(
        self,
        dim: int,
        *,
        num_layers: int,
        layer_id: int,
        drop_rate: float = 0.0,
        shift_pixel: int = 1,
        channel_gamma: float = 0.25,
        hidden_rate: int = 4,
        key_norm: bool = True,
        bidirectional: bool = True,
        layer_scale_init: float | None = 1e-4,
    ) -> None:
        super().__init__()
        self.ln0 = nn.LayerNorm(dim) if layer_id == 0 else None
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.spatial_mix = RWKVSpatialMixV1(
            dim,
            num_layers=num_layers,
            layer_id=layer_id,
            shift_pixel=shift_pixel,
            channel_gamma=channel_gamma,
            key_norm=key_norm,
            bidirectional=bidirectional,
        )
        self.channel_mix = RWKVChannelMixV1(
            dim,
            num_layers=num_layers,
            layer_id=layer_id,
            shift_pixel=shift_pixel,
            channel_gamma=channel_gamma,
            hidden_rate=hidden_rate,
        )
        self.drop1 = nn.Dropout(drop_rate)
        self.drop2 = nn.Dropout(drop_rate)

        if layer_scale_init is None:
            self.gamma1 = None
            self.gamma2 = None
        else:
            self.gamma1 = nn.Parameter(torch.full((dim,), layer_scale_init))
            self.gamma2 = nn.Parameter(torch.full((dim,), layer_scale_init))

    def forward(self, x: Tensor, patch_resolution: tuple[int, int]) -> Tensor:
        if self.ln0 is not None:
            x = self.ln0(x)

        y = self.spatial_mix(self.ln1(x), patch_resolution)
        if self.gamma1 is not None:
            y = y * self.gamma1
        x = x + self.drop1(y)

        y = self.channel_mix(self.ln2(x), patch_resolution)
        if self.gamma2 is not None:
            y = y * self.gamma2
        x = x + self.drop2(y)
        return x


class RWKVBottleneckV1(BaseBottleneck):
    """Stable V1 bottleneck kept in the versioned module path."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        num_layers: int = 6,
        drop_rate: float = 0.0,
        shift_pixel: int = 1,
        channel_gamma: float = 0.25,
        hidden_rate: int = 4,
        key_norm: bool = True,
        bidirectional: bool = True,
        layer_scale_init: float | None = 1e-4,
    ) -> None:
        super().__init__()
        self._output_dim = hidden_dim

        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                RWKVBlockV1(
                    hidden_dim,
                    num_layers=num_layers,
                    layer_id=i,
                    drop_rate=drop_rate,
                    shift_pixel=shift_pixel,
                    channel_gamma=channel_gamma,
                    hidden_rate=hidden_rate,
                    key_norm=key_norm,
                    bidirectional=bidirectional,
                    layer_scale_init=layer_scale_init,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)

        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.normal_(self.proj_out.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: Tensor) -> Tensor:
        patch_resolution = _factor_hw(x.shape[1])
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x, patch_resolution)
        x = self.norm(x)
        return self.proj_out(x)

    @property
    def output_dim(self) -> int:
        return self._output_dim


__all__ = ["RWKVBottleneckV1"]

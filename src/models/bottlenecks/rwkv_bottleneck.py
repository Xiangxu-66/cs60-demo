"""RWKV bottleneck built from the WKV recurrence used by the CUDA kernel."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.bottlenecks.base import BaseBottleneck


def _shift_tokens(x: Tensor) -> Tensor:
    """Shift the token sequence right by one step, padding the first token."""
    return torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)


def _wkv_reference(w: Tensor, u: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Pure PyTorch WKV recurrence matching the logic in ``cuda/wkv_cuda.cu``.

    This keeps the RWKV bottleneck trainable on CPU, MPS, and CUDA without
    requiring the custom extension to be compiled first. The CUDA kernel can
    still be wired in later as a performance optimization.
    """
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


class RWKVTimeMix(nn.Module):
    """Simple single-direction RWKV time-mixing block."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        mix = torch.linspace(0.0, 1.0, dim).view(1, 1, dim)
        self.time_mix_k = nn.Parameter(mix.clone())
        self.time_mix_v = nn.Parameter(mix.clone())
        self.time_mix_r = nn.Parameter(mix.clone())

        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

        # The CUDA kernel expects the effective decay to be negative:
        # w = -exp(time_decay). Mild negative initialisation keeps the
        # recurrence stable while still allowing useful temporal mixing.
        self.time_decay = nn.Parameter(torch.linspace(-3.0, -1.0, dim))
        self.time_first = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        xx = _shift_tokens(x)
        xk = x * self.time_mix_k + xx * (1.0 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1.0 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1.0 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))

        w = -torch.exp(self.time_decay)
        wkv = _wkv_reference(w=w, u=self.time_first, k=k, v=v)
        return self.output(r * wkv)


class RWKVChannelMix(nn.Module):
    """RWKV feed-forward channel mixer."""

    def __init__(self, dim: int, expansion_factor: int = 4) -> None:
        super().__init__()
        mix = torch.linspace(0.0, 1.0, dim).view(1, 1, dim)
        self.time_mix_k = nn.Parameter(mix.clone())
        self.time_mix_r = nn.Parameter(mix.clone())

        hidden_dim = dim * expansion_factor
        self.key = nn.Linear(dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        xx = _shift_tokens(x)
        xk = x * self.time_mix_k + xx * (1.0 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1.0 - self.time_mix_r)

        k = torch.relu(self.key(xk)).square()
        v = self.value(k)
        r = torch.sigmoid(self.receptance(xr))
        return r * v


class RWKVBlock(nn.Module):
    """Minimal RWKV block: TimeMix + ChannelMix with residual connections."""

    def __init__(self, dim: int, drop_rate: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.time_mix = RWKVTimeMix(dim)
        self.drop1 = nn.Dropout(drop_rate)

        self.ln2 = nn.LayerNorm(dim)
        self.channel_mix = RWKVChannelMix(dim)
        self.drop2 = nn.Dropout(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop1(self.time_mix(self.ln1(x)))
        x = x + self.drop2(self.channel_mix(self.ln2(x)))
        return x


class RWKVBottleneckV1(BaseBottleneck):
    """Simple RWKV bottleneck.

    This is a pragmatic V1 that keeps the project trainable today:
    - single-direction RWKV recurrence
    - no Q-Shift
    - no bidirectional scan
    - sequence-only mixing, so it works for any token count ``N``

    It follows the same high-level contract as the planned bottleneck:
    ``(B, N, C_in) -> (B, N, hidden_dim)``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        num_layers: int = 6,
        drop_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self._output_dim = hidden_dim
        self._hidden_dim = hidden_dim

        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [RWKVBlock(hidden_dim, drop_rate=drop_rate) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.proj_out(x)

    @property
    def output_dim(self) -> int:
        return self._output_dim

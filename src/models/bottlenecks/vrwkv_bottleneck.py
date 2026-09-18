"""vRWKV bottleneck in token-in/token-out format.

This module adapts a 2D vRWKV-style spatial mixer to the project bottleneck
interface:
    input : (B, N, C_in)
    output: (B, N, C_out)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange 
from torch import Tensor

from src.models.bottlenecks.base import BaseBottleneck


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for 2D maps (B, C, H, W)."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        # Keep LN stat-only (no affine params) to avoid unstable gamma/beta grads.
        self.norm = nn.LayerNorm(channels, eps=eps, elementwise_affine=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def q_shift(x: Tensor) -> Tensor:
    """Shift channels in four directions to inject local spatial bias."""
    _, c, _, _ = x.shape
    quarter = c // 4
    out = x.clone()

    if quarter <= 0:
        return out

    # left
    out[:, :quarter, :, 1:] = x[:, :quarter, :, :-1]
    # right
    out[:, quarter : 2 * quarter, :, :-1] = x[:, quarter : 2 * quarter, :, 1:]
    # up
    out[:, 2 * quarter : 3 * quarter, 1:, :] = x[:, 2 * quarter : 3 * quarter, :-1, :]
    # down
    out[:, 3 * quarter : 4 * quarter, :-1, :] = x[:, 3 * quarter : 4 * quarter, 1:, :]
    return out


def directional_scan_1d(x: Tensor, decay: Tensor) -> Tensor:
    """Simple recurrent 1D scan along token order."""
    orig_dtype = x.dtype
    # Keep recurrence on the original device to preserve stable autograd.
    x = x.float()
    decay = decay.float()

    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    b, c, t = x.shape
    y = torch.zeros_like(x)
    prev = torch.zeros(b, c, device=x.device, dtype=x.dtype)
    decay = torch.clamp(decay.squeeze(-1), 0.1, 0.99)  # (1, C)

    for i in range(t):
        prev = decay * prev + (1.0 - decay) * x[:, :, i]
        prev = torch.clamp(prev, -1e4, 1e4)
        y[:, :, i] = prev
    y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
    return y.to(dtype=orig_dtype)


class SpatialMix(nn.Module):
    """vRWKV spatial mixing on 2D feature maps."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.receptance = nn.Conv2d(channels, channels, kernel_size=1)
        self.output = nn.Conv2d(channels, channels, kernel_size=1)

        # Fixed decay for stability on MPS recurrent backprop.
        self.register_buffer("decay_h", torch.full((1, channels, 1), 0.8))
        self.register_buffer("decay_v", torch.full((1, channels, 1), 0.8))

        # Start close to identity in residual blocks.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.normal_(self.key.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.key.bias)
        nn.init.normal_(self.value.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.value.bias)
        nn.init.normal_(self.receptance.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.receptance.bias)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        xx = q_shift(torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4))

        k = self.key(xx)
        v = self.value(xx)
        r = torch.sigmoid(self.receptance(xx))

        # Horizontal recurrent scan (along W)
        kh = k.permute(0, 2, 1, 3).reshape(b * h, c, w)
        vh = v.permute(0, 2, 1, 3).reshape(b * h, c, w)
        khv = torch.tanh(kh) * torch.tanh(vh)
        yh = directional_scan_1d(khv, self.decay_h)
        yh = yh.reshape(b, h, c, w).permute(0, 2, 1, 3)

        # Vertical recurrent scan (along H)
        kv = k.permute(0, 3, 1, 2).reshape(b * w, c, h)
        vv = v.permute(0, 3, 1, 2).reshape(b * w, c, h)
        kvv = torch.tanh(kv) * torch.tanh(vv)
        yv = directional_scan_1d(kvv, self.decay_v)
        yv = yv.reshape(b, w, c, h).permute(0, 2, 3, 1)

        y = 0.5 * (yh + yv)
        y = r * y
        y = self.output(y)
        return torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)


class ChannelMix(nn.Module):
    """vRWKV channel mixer."""

    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = channels * expansion
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.receptance = nn.Conv2d(channels, channels, kernel_size=1)

        # Start close to identity in residual blocks.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.receptance.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.receptance.bias)

    def forward(self, x: Tensor) -> Tensor:
        xx = q_shift(torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4))
        k = F.gelu(self.fc1(xx))
        k = self.fc2(k)
        r = torch.sigmoid(self.receptance(xx))
        y = r * k
        return torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)


class vRWKVBlock(nn.Module):
    """Residual vRWKV block with spatial and channel mixing."""

    def __init__(self, channels: int, drop_rate: float = 0.1) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.spatial_mix = SpatialMix(channels)
        self.channel_mix = ChannelMix(channels)
        self.drop = nn.Dropout2d(drop_rate)
        # Fixed layer-scale for stability on MPS recurrent backprop.
        self.register_buffer("gamma1", torch.full((1, channels, 1, 1), 1e-3))
        self.register_buffer("gamma2", torch.full((1, channels, 1, 1), 1e-3))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.gamma1 * self.drop(self.spatial_mix(self.norm1(x)))
        x = x + self.gamma2 * self.drop(self.channel_mix(self.norm2(x)))
        return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


class VRWKVBottleneck(BaseBottleneck):
    """vRWKV bottleneck compatible with project pipeline."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        drop_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self._output_dim = hidden_dim

        self.proj_in = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            *[vRWKVBlock(hidden_dim, drop_rate=drop_rate) for _ in range(num_layers)]
        )
        self.proj_out = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        nn.init.normal_(self.proj_in.weight, mean=0.0, std=1e-3)
        if self.proj_in.bias is not None:
            nn.init.zeros_(self.proj_in.bias)
        nn.init.normal_(self.proj_out.weight, mean=0.0, std=1e-3)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)
        self._register_proj_in_grad_guard()
        self._register_all_grad_guard()

    def _register_proj_in_grad_guard(self) -> None:
        """Guard only proj_in grads against non-finite values."""
        clip = 100.0

        def _sanitize(g: Tensor) -> Tensor:
            g = torch.nan_to_num(g, nan=0.0, posinf=clip, neginf=-clip)
            return torch.clamp(g, -clip, clip)

        if self.proj_in.weight.requires_grad:
            self.proj_in.weight.register_hook(_sanitize)
        if self.proj_in.bias is not None and self.proj_in.bias.requires_grad:
            self.proj_in.bias.register_hook(_sanitize)

    def _register_all_grad_guard(self) -> None:
        """Guard all trainable grads in this bottleneck."""
        clip = 100.0

        def _sanitize(g: Tensor) -> Tensor:
            g = torch.nan_to_num(g, nan=0.0, posinf=clip, neginf=-clip)
            return torch.clamp(g, -clip, clip)

        for p in self.parameters():
            if p.requires_grad:
                p.register_hook(_sanitize)

    @staticmethod
    def _factor_hw(num_tokens: int) -> tuple[int, int]:
        h = int(math.isqrt(num_tokens))
        if h * h == num_tokens:
            return h, h

        # Fallback for non-square token counts.
        for h_try in range(int(num_tokens**0.5), 0, -1):
            if num_tokens % h_try == 0:
                return h_try, num_tokens // h_try
        return 1, num_tokens

    def forward(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        h, w = self._factor_hw(n)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        x = self.proj_in(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        x = torch.clamp(x, -100.0, 100.0)

        for block in self.blocks:
            x = block(x)

        x = torch.nan_to_num(self.proj_out(x), nan=0.0, posinf=1e4, neginf=-1e4)
        x = rearrange(x, "b c h w -> b (h w) c")
        return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

    @property
    def output_dim(self) -> int:
        return self._output_dim

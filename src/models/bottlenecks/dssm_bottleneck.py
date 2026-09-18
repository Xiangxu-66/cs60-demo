"""DSSM Bottleneck with Selective Scan from DyRSRNet.

This module implements the Dynamic State Space Model (DSSM) with 4-directional
selective scan, providing efficient long-range dependency modeling for
spatial data.

Reference: DyRSRNet (PRCV'25 Oral)
Source: DyRSRNet/basicsr/archs/DyRSRNet_arch.py:460-795

Key components:
  - DSSM: 4-directional selective scan (left→right, top→bottom, right→left, bottom→top)
  - RCSSblock: Combines dual DSSM with VRSE (Visual Recurrent Shift Encoding)

Note: Requires mamba_ssm package for selective_scan_fn.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from torch import Tensor

# Try to import selective_scan_fn from mamba_ssm
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    SELECTIVE_SCAN_AVAILABLE = True
except ImportError:
    SELECTIVE_SCAN_AVAILABLE = False
    selective_scan_fn = None

from src.models.bottlenecks.base import BaseBottleneck


class DSSM(nn.Module):
    """Dynamic State Space Model with 4-directional selective scan.

    This module implements selective state space modeling with 4 scan directions:
    1. Left→Right, Top→Bottom (forward)
    2. Transposed forward (W×H instead of H×W)
    3. Right→Left, Bottom→Top (backward)
    4. Transposed backward

    Args:
        d_model: Input/output dimension
        d_state: SSM state dimension (default: 16)
        d_conv: Depthwise convolution kernel size (default: 3)
        expand: Expansion factor for inner dimension (default: 2.0)
        dt_rank: Rank of delta projection (default: "auto" → ceil(d_model/16))
        dt_min: Minimum delta value (default: 0.001)
        dt_max: Maximum delta value (default: 0.1)
        dt_init: Delta initialization mode (default: "random")
        dt_scale: Delta initialization scale (default: 1.0)
        dt_init_floor: Floor for delta initialization (default: 1e-4)
        dropout: Dropout rate (default: 0.0)
        conv_bias: Whether to use bias in conv (default: True)
        bias: Whether to use bias in linear layers (default: False)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: float = 2.0,
        dt_rank: str | int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # Input projection: splits into x (branch) and z (gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Depthwise convolution for local context
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()

        # 4-directional projections (K=4 directions)
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )  # (K=4, N, inner)
        del self.x_proj

        # 4-directional delta projections
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )  # (K=4, inner)
        del self.dt_projs

        # SSM parameters (A and D)
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D)

        # Selective scan function
        if not SELECTIVE_SCAN_AVAILABLE:
            raise ImportError(
                "mamba_ssm is required for DSSM. "
                "Install with: pip install mamba-ssm causal-conv1d"
            )
        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    @staticmethod
    def dt_init(
        dt_rank: int,
        d_inner: int,
        dt_scale: float = 1.0,
        dt_init: str = "random",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ):
        """Initialize delta projection parameters."""
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unknown dt_init: {dt_init}")

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state: int, d_inner: int, copies: int = 1, device=None, merge=True):
        """Initialize A parameter (S4D real initialization)."""
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner: int, copies: int = 1, device=None, merge=True):
        """Initialize D skip parameter."""
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: Tensor) -> tuple:
        """Core selective scan computation with 4 directions.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Tuple of 4 directional outputs: (y1, y2, y3, y4)
        """
        B, C, H, W = x.shape
        L = H * W
        K = 4  # 4 directions

        # Prepare 4 directional scans
        # [0]: H×W (left→right, top→bottom)
        # [1]: W×H transposed
        # [2]: H×W reversed (right→left, bottom→top)
        # [3]: W×H reversed transposed
        x_hwwh = torch.stack([
            x.view(B, -1, L),
            torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
        ], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (B, 4, C, L)

        # Project to get dt, B, C for each direction
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        # Project deltas
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        # Prepare for selective scan
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)

        # SSM parameters
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        # Selective scan for all directions
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # Extract and restore directional outputs
        # y1: forward H×W
        # y2: transposed forward
        # y3: backward H×W (reversed)
        # y4: transposed backward (reversed)
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with 4-directional selective scan.

        Args:
            x: Input tensor (B, H, W, C)

        Returns:
            Output tensor (B, H, W, C)
        """
        B, H, W, C = x.shape

        # Split into x (branch) and z (gate)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Conv2d + activation
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, H, W, C) → (B, C, H, W)
        x = self.act(self.conv2d(x))

        # 4-directional selective scan
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32

        # Combine all directions
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)

        return out


class RCSSBlock(nn.Module):
    """Recursive Context-Aware State Space Block from DyRSRNet.

    Combines dual DSSM (bidirectional state space modeling) with
    VRSE (Visual Recurrent Shift Encoding) module.

    Args:
        hidden_dim: Hidden dimension
        drop_path: Drop path rate (default: 0)
        d_state: SSM state dimension (default: 16)
        expand: Expansion factor for DSSM (default: 2.0)
    """

    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        d_state: int = 16,
        expand: float = 2.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ln_1 = nn.LayerNorm(hidden_dim)

        # Dual DSSM for bidirectional state space modeling
        self.self_attention_1 = DSSM(
            d_model=hidden_dim // 2,
            d_state=d_state,
            expand=expand,
        )
        self.self_attention_2 = DSSM(
            d_model=hidden_dim // 2,
            d_state=d_state,
            expand=expand,
        )

        self.drop_path = nn.Identity() if drop_path == 0 else nn.Dropout(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

        self.ln_2 = nn.LayerNorm(hidden_dim)

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass through RCSS block.

        Args:
            input: Input tokens (B, N, C)

        Returns:
            Output tokens (B, N, C)
        """
        B, L, C = input.shape

        # Assume square spatial grid for reshape
        H = W = int(L ** 0.5)
        input = input.view(B, H, W, C).contiguous()

        x = self.ln_1(input)

        # Split into two halves for dual DSSM
        x1, x2 = torch.chunk(x, 2, dim=-1)

        # Apply dual DSSM
        sa_out1 = self.self_attention_1(x1)
        sa_out2 = self.self_attention_2(x2)

        # Concatenate and apply residual
        sa_out = torch.cat((sa_out1, sa_out2), dim=-1)
        x = input * self.skip_scale + self.drop_path(sa_out)

        # Note: DyRSRNet also applies VRSE (Block with OSRM) here
        # For simplicity, we skip it in this initial implementation
        x = x * self.skip_scale2
        x = x.view(B, -1, C).contiguous()

        return x


class DSSMBottleneck(BaseBottleneck):
    """DSSM Bottleneck for color enhancement.

    Uses RCSS blocks with 4-directional selective scan for efficient
    long-range dependency modeling.

    Args:
        input_dim: Input feature dimension from encoder
        hidden_dim: Hidden dimension for DSSM blocks
        num_layers: Number of RCSS blocks
        d_state: SSM state dimension
        expand: Expansion factor for DSSM
        drop_path: Drop path rate
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        num_layers: int = 6,
        d_state: int = 16,
        expand: float = 2.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self._output_dim = hidden_dim

        # Project input to hidden dimension
        self.proj_in = nn.Linear(input_dim, hidden_dim)

        # Create RCSS blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                RCSSBlock(
                    hidden_dim=hidden_dim,
                    drop_path=drop_path,
                    d_state=d_state,
                    expand=expand,
                )
            )

        # Output projection and normalization
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Process token features through DSSM bottleneck.

        Args:
            x: Encoder output token features (B, N, C_in)

        Returns:
            Processed token features (B, N, hidden_dim)
        """
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.proj_out(x)

    @property
    def output_dim(self) -> int:
        return self._output_dim


__all__ = [
    "DSSM",
    "RCSSBlock",
    "DSSMBottleneck",
    "SELECTIVE_SCAN_AVAILABLE",
]

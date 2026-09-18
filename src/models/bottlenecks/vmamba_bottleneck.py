"""B-4: VMamba bottleneck — official SS2D architecture (MzeroMiko/VMamba).

Faithfully reproduces the official VMamba SS2D module with:
  - 4-direction Cross-Scan / Cross-Merge (row, col, row-flip, col-flip)
  - DWConv2d local spatial prior
  - SiLU gated branch
  - S4D-real A initialization, learned dt/D parameters
  - VSSBlock = pre-norm SS2D + pre-norm MLP with DropPath

Reference: https://github.com/MzeroMiko/VMamba
Paper: "VMamba: Visual State Space Model" (NeurIPS 2024)
"""
from __future__ import annotations

import math
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.bottlenecks.base import BaseBottleneck

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


# =====================================================
# Cross-Scan / Cross-Merge  (pure PyTorch, from official csm_triton.py)
# =====================================================

def cross_scan_fwd(x: Tensor) -> Tensor:
    """4-direction cross scan on a 2-D feature map.

    Args:
        x: (B, C, H, W) channel-first feature map.

    Returns:
        (B, 4, C, L) where L = H*W.  Directions:
          0 — row-major              (left→right, top→bottom)
          1 — column-major           (top→bottom, left→right)
          2 — row-major reversed     (right→left, bottom→top)
          3 — column-major reversed  (bottom→top, right→left)
    """
    B, C, H, W = x.shape
    L = H * W
    y = x.new_empty((B, 4, C, L))
    y[:, 0, :, :] = x.flatten(2, 3)
    y[:, 1, :, :] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
    y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])
    return y


def cross_merge_fwd(y: Tensor, H: int, W: int) -> Tensor:
    """Merge 4-direction scan outputs back to a single feature map.

    Args:
        y: (B, 4, C, L) with L = H*W — output from selective scan.
        H, W: spatial dimensions.

    Returns:
        (B, C, L) merged features (sum of 4 directions).
    """
    B, K, C, L = y.shape
    # Reverse the flipped directions, then sum all 4
    inv_y = torch.flip(y[:, 2:4], dims=[-1]).view(B, 2, C, L)
    wh_y = y[:, 1].view(B, C, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, C, L)
    invwh_y = inv_y[:, 1].view(B, C, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, C, L)
    return y[:, 0] + inv_y[:, 0] + wh_y + invwh_y


class CrossScan(torch.autograd.Function):
    """Autograd wrapper for cross scan (forward) / cross merge (backward)."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        return cross_scan_fwd(x)

    @staticmethod
    def backward(ctx, ys: Tensor) -> Tensor:
        B, C, H, W = ctx.shape
        return cross_merge_fwd(ys, H, W).view(B, C, H, W)


class CrossMerge(torch.autograd.Function):
    """Autograd wrapper for cross merge (forward) / cross scan (backward)."""

    @staticmethod
    def forward(ctx, ys: Tensor, H: int, W: int) -> Tensor:
        B, K, C, L = ys.shape
        ctx.shape = (B, C, H, W)
        return cross_merge_fwd(ys, H, W)

    @staticmethod
    def backward(ctx, x: Tensor) -> tuple:
        B, C, H, W = ctx.shape
        return cross_scan_fwd(x.view(B, C, H, W)), None, None


# =====================================================
# SS2D — official Selective Scan 2D (based on SS2Dv0)
# =====================================================

class SS2D(nn.Module):
    """Official SS2D module — the core of VMamba.

    Reproduces SS2Dv0 from MzeroMiko/VMamba exactly:
      in_proj → split(x, z) → DWConv2d + SiLU(x) → 4-dir cross scan
      → selective_scan (S6) → cross merge → LayerNorm → × SiLU(z) → out_proj

    Args:
        d_model: Input/output feature dimension.
        d_state: SSM state expansion factor (N).
        ssm_ratio: Inner dimension expansion d_inner = ssm_ratio * d_model.
        dt_rank: Rank of delta projection ("auto" → ceil(d_model/16)).
        d_conv: Depthwise convolution kernel size.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 96,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        dt_rank: str | int = "auto",
        d_conv: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if selective_scan_fn is None:
            raise ImportError(
                "mamba-ssm is required for VMambaBottleneck. "
                "Install with: pip install mamba-ssm"
            )

        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank
        k_group = 4  # 4 scan directions

        # ── in_proj: project to 2*d_inner, split into x and gate z ──
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.act = nn.SiLU()

        # ── DWConv2d: local spatial prior ──
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )

        # ── x_proj: project each direction to (dt_rank + 2*d_state) ──
        x_proj_list = [
            nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in x_proj_list], dim=0)  # (K, dt_rank+2N, d_inner)
        )
        del x_proj_list

        # ── dt_proj, A, D initialization (official S4D-real init) ──
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = (
            self._init_dt_A_D(d_state, dt_rank, d_inner, k_group)
        )

        # ── out_proj ──
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    # ----------------------------------------------------------------
    # Official initialization (from mamba_init in MzeroMiko/VMamba)
    # ----------------------------------------------------------------

    @staticmethod
    def _init_dt_A_D(
        d_state: int,
        dt_rank: int,
        d_inner: int,
        k_group: int = 4,
        dt_scale: float = 1.0,
        dt_init: str = "random",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ):
        # ── dt_proj ──
        dt_projs = []
        for _ in range(k_group):
            dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
            dt_init_std = dt_rank ** -0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            # Initialize dt bias so that F.softplus(dt_bias) ∈ [dt_min, dt_max]
            dt = torch.exp(
                torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            dt_projs.append(dt_proj)

        dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in dt_projs], dim=0)  # (K, d_inner, dt_rank)
        )
        dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in dt_projs], dim=0)  # (K, d_inner)
        )
        del dt_projs

        # ── A: S4D real initialization ──
        A = torch.arange(1, d_state + 1, dtype=torch.float32).view(1, -1).repeat(d_inner, 1)
        A_logs = torch.log(A)  # (d_inner, d_state)
        A_logs = A_logs[None].repeat(k_group, 1, 1).flatten(0, 1)  # (K*d_inner, d_state)
        A_logs = nn.Parameter(A_logs)
        A_logs._no_weight_decay = True

        # ── D: skip parameter ──
        D = torch.ones(d_inner)
        D = D[None].repeat(k_group, 1).flatten(0, 1)  # (K*d_inner,)
        D = nn.Parameter(D)
        D._no_weight_decay = True

        return A_logs, D, dt_projs_weight, dt_projs_bias

    # ----------------------------------------------------------------
    # Forward  (official forwardv0 from SS2Dv0)
    # ----------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """SS2D forward pass.

        Args:
            x: (B, H, W, C) channel-last feature map.

        Returns:
            (B, H, W, C) processed features.
        """
        # ── in_proj → split into content x and gate z ──
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)  # (B, H, W, d_inner) each
        z = self.act(z)

        # ── DWConv + SiLU on content ──
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, d_inner, H, W)
        x = self.act(self.conv2d(x))

        # ── 4-direction cross scan ──
        B, D, H, W = x.shape
        N = self.d_state
        K = 4
        R = self.dt_rank
        L = H * W

        xs = CrossScan.apply(x)  # (B, 4, D, L)

        # ── project each direction to dt, B, C ──
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        # ── flatten for batched selective scan ──
        xs = xs.view(B, -1, L)         # (B, K*D, L)
        dts = dts.contiguous().view(B, -1, L)  # (B, K*D, L)
        Bs = Bs.contiguous()            # (B, K, N, L)
        Cs = Cs.contiguous()            # (B, K, N, L)

        As = -self.A_logs.float().exp()              # (K*D, N)
        Ds = self.Ds.float()                         # (K*D,)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (K*D,)

        # ── force fp32 for numerical stability ──
        xs, dts, Bs, Cs = (t.to(torch.float32) for t in (xs, dts, Bs, Cs))

        # ── selective scan (S6) ──
        out_y = selective_scan_fn(
            xs, dts, As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)  # (B, 4, D, L)
        assert out_y.dtype == torch.float32

        # ── cross merge: sum 4 directions ──
        y = CrossMerge.apply(out_y, H, W)  # (B, D, L)

        # ── output projection with gating ──
        y = y.transpose(1, 2).contiguous()  # (B, L, D)
        y = self.out_norm(y).view(B, H, W, -1)
        y = y * z
        return self.dropout(self.out_proj(y))


# =====================================================
# VSSBlock — official Visual State Space Block
# =====================================================

class Mlp(nn.Module):
    """Standard MLP as used in official VMamba VSSBlock."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VSSBlock(nn.Module):
    """Official VMamba VSSBlock: pre-norm SS2D + pre-norm MLP with DropPath.

    Structure:
        x = x + DropPath(SS2D(Norm(x)))
        x = x + DropPath(MLP(Norm(x)))
    """

    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        norm_layer: type = nn.LayerNorm,
        # SS2D params
        ssm_d_state: int = 16,
        ssm_ratio: float = 2.0,
        ssm_dt_rank: str | int = "auto",
        ssm_conv: int = 3,
        ssm_drop_rate: float = 0.0,
        # MLP params
        mlp_ratio: float = 4.0,
        mlp_act_layer: type = nn.GELU,
        mlp_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        # SSM branch
        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            d_conv=ssm_conv,
            dropout=ssm_drop_rate,
        )

        from timm.models.layers import DropPath as TimmDropPath
        self.drop_path = TimmDropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # MLP branch
        self.norm2 = norm_layer(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=mlp_act_layer,
            drop=mlp_drop_rate,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Args: x: (B, H, W, C) channel-last."""
        x = x + self.drop_path(self.op(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# =====================================================
# VMambaBottleneck — bottleneck adapter for our pipeline
# =====================================================

class VMambaBottleneck(BaseBottleneck):
    """B-4: VMamba selective SSM bottleneck (official architecture).

    Adapts the official VMamba VSSBlock stack to work as a bottleneck module
    in our Encoder → Bottleneck → Decoder pipeline.

    Input:  (B, N, C_in)  flat token sequence from encoder
    Output: (B, N, C_out) processed tokens for decoder

    Internally reshapes tokens to (B, H, W, C) 2D spatial layout for the
    official SS2D 4-direction cross scan, then flattens back.

    Args:
        input_dim: Encoder output dimension (injected by pipeline).
        hidden_dim: Internal feature dimension.
        num_layers: Number of VSSBlock layers.
        drop_rate: Dropout rate for SSM and MLP.
        drop_path_rate: Stochastic depth rate (linearly increases).
        d_state: SSM state expansion factor.
        ssm_ratio: Inner dimension expansion for SS2D.
        d_conv: Depthwise convolution kernel size in SS2D.
        mlp_ratio: MLP expansion ratio in VSSBlock.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        num_layers: int = 6,
        drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        d_conv: int = 3,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self._output_dim = hidden_dim

        self.proj_in = nn.Linear(input_dim, hidden_dim)

        # Stochastic depth decay (official VMamba uses linear schedule)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=hidden_dim,
                drop_path=dpr[i],
                ssm_d_state=d_state,
                ssm_ratio=ssm_ratio,
                ssm_conv=d_conv,
                ssm_drop_rate=drop_rate,
                mlp_ratio=mlp_ratio,
                mlp_drop_rate=drop_rate,
            )
            for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Official VMamba weight initialization."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        """Process token features through official VMamba VSSBlocks.

        Args:
            x: Encoder output token features of shape (B, N, C_in).

        Returns:
            Processed token features of shape (B, N, hidden_dim).
        """
        B, N, _ = x.shape
        x = self.proj_in(x)

        # Infer spatial dims from token count (encoder produces h*w tokens)
        h = w = int(math.sqrt(N))
        assert h * w == N, (
            f"Token count {N} is not a perfect square. "
            f"VMamba SS2D requires 2D spatial layout."
        )

        # Reshape to 2D spatial (channel-last for SS2D)
        x = x.view(B, h, w, -1)  # (B, H, W, C)

        for block in self.blocks:
            x = block(x)

        # Flatten back to token sequence
        x = x.view(B, N, -1)  # (B, N, C)
        x = self.norm(x)
        return self.proj_out(x)

    @property
    def output_dim(self) -> int:
        return self._output_dim

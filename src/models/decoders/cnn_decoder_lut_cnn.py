"""LUT Decoder with Shallow CNN gated fusion.

This decoder combines:
1. Global Branch: LUT-based global color adjustment (from RWKV features)
2. Spatial Branch: Local modulation from Shallow CNN features (gated fusion)

Architecture:
                      Input Image
                           │
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
   DINOv3 → RWKV                        Shallow CNN
        │                                     │
        ▼                                     ▼
  rwkv_tokens                         cnn_features
  (B, N, C)                            [f0, f1, f2, f3]
        │                                     │
        │    ┌────────────────────────────────┘
        │    ▼
        │ Global Branch                Spatial Branch
        │ LUT(img)                     M, R (gated fusion)
        │                                     │
        └──────────────┬──────────────────────┘
                       ▼
    enhanced = (1-M)·LUT(img) + M·(img + R)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.decoders.base import BaseDecoder


class ChanLayerNorm(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps (B, C, H, W)."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).to(x.dtype).permute(0, 3, 1, 2)


class GatedSkipFusion(nn.Module):
    """Lightweight gated skip fusion.

    Fuses decoder features with skip features using a learned gate.
    Simpler and more stable than cross-attention.

    Args:
        decoder_channels: Channels from the decoder.
        skip_channels: Channels from the skip connection.
        gate_reduction: Channel reduction ratio for gate network.
    """

    def __init__(
        self,
        decoder_channels: int,
        skip_channels: int,
        gate_reduction: int = 4,
    ) -> None:
        super().__init__()

        # Project skip features to decoder channels
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, decoder_channels, 1, bias=True),
            ChanLayerNorm(decoder_channels),
            nn.GELU(),
        )

        # Gate network
        hidden_ch = decoder_channels * 2 // gate_reduction
        self.gate = nn.Sequential(
            nn.Conv2d(decoder_channels * 2, hidden_ch, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_ch, 1, 1, bias=True),
            nn.Sigmoid(),
        )

        # Output projection
        self.out_proj = nn.Sequential(
            ChanLayerNorm(decoder_channels),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=True),
            nn.GELU(),
        )

    def forward(self, decoder_feat: Tensor, skip_feat: Tensor) -> Tensor:
        """Gated skip fusion.

        Args:
            decoder_feat: Decoder features (B, C, H, W).
            skip_feat: Skip features (B, C_skip, H_skip, W_skip).

        Returns:
            Fused features (B, C, H, W).
        """
        # Align spatial sizes
        if skip_feat.shape[-2:] != decoder_feat.shape[-2:]:
            skip_feat = F.interpolate(
                skip_feat,
                size=decoder_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # Project skip features
        skip_proj = self.skip_proj(skip_feat)

        # Compute gate
        gate = self.gate(torch.cat([decoder_feat, skip_proj], dim=1))

        # Gated fusion
        fused = decoder_feat + gate * skip_proj

        return self.out_proj(fused)


class ResidualRefineBlock(nn.Module):
    """Residual refinement block with learnable gain."""

    def __init__(
        self,
        channels: int,
        residual_gain_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ChanLayerNorm(channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
        )
        self.gain = nn.Parameter(torch.full((1, channels, 1, 1), residual_gain_init))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.gain.to(dtype=x.dtype) * self.net(x)


class UpsampleBlock(nn.Module):
    """Upsampling block with gated skip fusion."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int | None,
        num_refine_blocks: int = 1,
        residual_gain_init: float = 0.1,
    ) -> None:
        super().__init__()

        # Upsampling with PixelShuffle
        self.up = nn.Sequential(
            ChanLayerNorm(in_channels),
            nn.Conv2d(in_channels, out_channels * 4, 3, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.GELU(),
        )

        # Gated skip fusion
        self.gated_fusion = (
            GatedSkipFusion(
                decoder_channels=out_channels,
                skip_channels=skip_channels,
            )
            if skip_channels is not None
            else None
        )

        # Refinement blocks
        self.refine_blocks = nn.ModuleList([
            ResidualRefineBlock(out_channels, residual_gain_init)
            for _ in range(num_refine_blocks)
        ])

    def forward(
        self,
        decoder_feat: Tensor,
        skip_feat: Tensor | None = None,
    ) -> Tensor:
        """Upsample and optionally fuse with skip features."""
        x = self.up(decoder_feat)

        if skip_feat is not None and self.gated_fusion is not None:
            x = self.gated_fusion(x, skip_feat)

        for block in self.refine_blocks:
            x = block(x)

        return x


class SpatialBranchWithCNN(nn.Module):
    """Spatial branch that uses Shallow CNN features.

    Unlike the original LUT decoder which uses RWKV tokens,
    this branch directly processes multi-scale CNN features.

    Args:
        cnn_feature_channels: List of channel counts for CNN features.
        base_channels: Base channel count for upsampling.
        num_upsample_blocks: Number of upsampling blocks.
        num_refine_blocks: Number of refinement blocks per stage.
    """

    def __init__(
        self,
        cnn_feature_channels: list[int],
        base_channels: int = 64,
        num_upsample_blocks: int = 4,
        num_refine_blocks: int = 1,
    ) -> None:
        super().__init__()

        self.cnn_feature_channels = cnn_feature_channels
        self.num_upsample_blocks = num_upsample_blocks
        if len(cnn_feature_channels) != num_upsample_blocks:
            raise ValueError(
                "SpatialBranchWithCNN expects one decoder stage per CNN feature level: "
                f"got {len(cnn_feature_channels)=} and {num_upsample_blocks=}"
            )

        # Channel schedule
        channels = self._build_channel_schedule(base_channels, num_upsample_blocks)
        self.channels = channels

        # Start from the coarsest feature map, then restore spatial detail
        # coarse-to-fine. This avoids blowing up H/2 features to 4x input size.
        self.init_proj = nn.Sequential(
            nn.Conv2d(cnn_feature_channels[-1], channels[0], 3, padding=1, bias=True),
            ChanLayerNorm(channels[0]),
            nn.GELU(),
        )

        # Fuse the second-coarsest feature at the same H/8 resolution before
        # any spatial upsampling. This keeps the whole branch coarse-to-fine.
        self.coarse_fusion = GatedSkipFusion(
            decoder_channels=channels[0],
            skip_channels=cnn_feature_channels[-2],
        )

        # Upsample H/8 -> H/4 -> H/2 -> H. We use f1 as the semantic-aligned
        # skip at H/4, then inject the high-resolution f0 detail map with
        # dedicated fusion blocks at H/2 and again before the output heads.
        num_pixelshuffle = len(cnn_feature_channels) - 1
        skip_channel_schedule = [cnn_feature_channels[1]] + [None] * (num_pixelshuffle - 1)
        self.upsample_blocks = nn.ModuleList()
        for i in range(num_pixelshuffle):
            block = UpsampleBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                skip_channels=skip_channel_schedule[i],
                num_refine_blocks=num_refine_blocks,
            )
            self.upsample_blocks.append(block)

        # Dedicated f0 detail fusion after restoring the decoder to H/2.
        self.halfres_detail_fusion = GatedSkipFusion(
            decoder_channels=channels[2],
            skip_channels=cnn_feature_channels[0],
        )

        # Final projection block
        self.final_proj = nn.Sequential(
            ChanLayerNorm(channels[num_pixelshuffle]),
            nn.Conv2d(channels[num_pixelshuffle], channels[-1], 3, padding=1, bias=True),
            nn.GELU(),
        )

        # Final refine blocks
        self.final_refine = nn.ModuleList([
            ResidualRefineBlock(channels[-1])
            for _ in range(2)
        ])

        # Final detail injection at full resolution before predicting M and R.
        self.output_detail_fusion = GatedSkipFusion(
            decoder_channels=channels[-1],
            skip_channels=cnn_feature_channels[0],
        )

        # Output heads
        self.weight_head = nn.Sequential(
            ChanLayerNorm(channels[-1]),
            nn.Conv2d(channels[-1], channels[-1] // 2, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels[-1] // 2, 1, 1, bias=True),
            nn.Sigmoid(),
        )

        self.residual_head = nn.Sequential(
            ChanLayerNorm(channels[-1]),
            nn.Conv2d(channels[-1], channels[-1], 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels[-1], 3, 1, bias=True),
        )
        # Zero initialize residual head
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    @staticmethod
    def _build_channel_schedule(base: int, num_blocks: int) -> list[int]:
        """Build channel schedule."""
        channels = [base]
        for i in range(1, num_blocks + 1):
            ratio = 1.0 - (0.25 * i / num_blocks)
            ch = max(round(base * ratio), 32)
            channels.append(ch)
        return channels

    @staticmethod
    def _downsample_size(size: tuple[int, int], repeats: int) -> tuple[int, int]:
        """Mirror the encoder's stride-2 convolutions using ceil division."""
        h, w = size
        for _ in range(repeats):
            h = (h + 1) // 2
            w = (w + 1) // 2
        return h, w

    @classmethod
    def _expected_feature_sizes(cls, target_size: tuple[int, int]) -> list[tuple[int, int]]:
        f0 = cls._downsample_size(target_size, 1)
        f1 = cls._downsample_size(target_size, 2)
        f2 = cls._downsample_size(target_size, 3)
        return [f0, f1, f2, f2]

    def _validate_cnn_features(
        self,
        cnn_features: list[Tensor],
        target_size: tuple[int, int],
    ) -> None:
        if len(cnn_features) != len(self.cnn_feature_channels):
            raise ValueError(
                f"Expected {len(self.cnn_feature_channels)} CNN features, got {len(cnn_features)}"
            )

        expected_sizes = self._expected_feature_sizes(target_size)
        for idx, (feat, expected_size, channels) in enumerate(
            zip(cnn_features, expected_sizes, self.cnn_feature_channels)
        ):
            if feat.dim() != 4:
                raise ValueError(f"cnn_features[{idx}] must be 4D, got shape {tuple(feat.shape)}")
            if feat.shape[1] != channels:
                raise ValueError(
                    f"cnn_features[{idx}] channel mismatch: expected {channels}, got {feat.shape[1]}"
                )
            if feat.shape[-2:] != expected_size:
                raise ValueError(
                    f"cnn_features[{idx}] size mismatch: expected {expected_size}, got {feat.shape[-2:]}"
                )

    def forward(
        self,
        cnn_features: list[Tensor],
        target_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        """Predict spatial weight map and residual.

        Args:
            cnn_features: List of CNN feature maps [f0, f1, f2, f3].
            target_size: Target output size (H, W).

        Returns:
            M: (B, 1, H, W) spatial weight map
            R: (B, 3, H, W) local residual
        """
        self._validate_cnn_features(cnn_features, target_size)
        f0, f1, f2, f3 = cnn_features

        # Coarse-to-fine decoding:
        #   start at H/8 (f3), fuse f2 at the same scale,
        #   restore H/4 with f1, inject strong f0 detail at H/2,
        #   then inject f0 once more before the output heads.
        x = self.init_proj(f3)
        x = self.coarse_fusion(x, f2)

        skip_schedule = [f1, None, None]
        for block, skip_feat in zip(self.upsample_blocks, skip_schedule):
            x = block(x, skip_feat)
            if x.shape[-2:] == f0.shape[-2:]:
                x = self.halfres_detail_fusion(x, f0)

        # Final projection
        x = self.final_proj(x)

        # Upsample to target size
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

        # Final refinement
        for block in self.final_refine:
            x = block(x)

        # Re-inject the finest detail map at full resolution before the heads.
        x = self.output_detail_fusion(x, f0)

        # Predict M and R
        M = self.weight_head(x)
        R = self.residual_head(x)

        return M, R


# Import LUT components from the original decoder
class FastLUTSampling(nn.Module):
    """Fast 1D LUT sampling (copied from cnn_decoder_lut for self-containment)."""

    def __init__(self, lut_size: int = 33) -> None:
        super().__init__()
        self.lut_size = lut_size

    def forward(self, rgb: Tensor, lut: Tensor) -> Tensor:
        B, C, H, W = rgb.shape

        if lut.dim() == 3:  # 1D LUT
            return self._sample_1d_lut(rgb, lut)

        return self._sample_3d_lut_approx(rgb, lut)

    def _sample_1d_lut(self, rgb: Tensor, lut: Tensor) -> Tensor:
        B, C, H, W = rgb.shape
        D = lut.shape[-1]

        idx = rgb * (D - 1)
        idx_floor = torch.floor(idx).long().clamp(0, D - 1)
        idx_ceil = torch.ceil(idx).long().clamp(0, D - 1)

        weight = idx - idx_floor.float()

        output = []
        for c in range(C):
            lut_c = lut[:, c]
            idx_floor_c = idx_floor[:, c].reshape(B, -1)
            idx_ceil_c = idx_ceil[:, c].reshape(B, -1)

            val_floor = torch.gather(lut_c, 1, idx_floor_c)
            val_ceil = torch.gather(lut_c, 1, idx_ceil_c)

            weight_c = weight[:, c].reshape(B, -1)
            val = val_floor * (1 - weight_c) + val_ceil * weight_c
            output.append(val.reshape(B, 1, H, W))

        return torch.cat(output, dim=1)

    def _sample_3d_lut_approx(self, rgb: Tensor, lut: Tensor) -> Tensor:
        # Simplified 3D LUT sampling
        B, C, H, W = rgb.shape
        D = self.lut_size

        if lut.shape[1] == 3:
            lut = lut.permute(0, 2, 3, 4, 1)

        coords = rgb * (D - 1)
        coords_floor = torch.floor(coords).long().clamp(0, D - 1)
        coords_ceil = torch.ceil(coords).long().clamp(0, D - 1)

        frac = coords - coords_floor.float()
        HW = H * W

        coords_floor = coords_floor.reshape(B, 3, HW)
        coords_ceil = coords_ceil.reshape(B, 3, HW)
        r_frac, g_frac, b_frac = frac[:, 0], frac[:, 1], frac[:, 2]
        r_frac = r_frac.reshape(B, 1, HW)
        g_frac = g_frac.reshape(B, 1, HW)
        b_frac = b_frac.reshape(B, 1, HW)

        output = torch.zeros(B, 3, HW, device=rgb.device, dtype=rgb.dtype)

        for c in range(3):
            lut_c = lut[:, :, :, :, c]
            r0, g0, b0 = coords_floor[:, 0], coords_floor[:, 1], coords_floor[:, 2]
            r1, g1, b1 = coords_ceil[:, 0], coords_ceil[:, 1], coords_ceil[:, 2]

            D2 = D * D
            idx000 = r0 * D2 + g0 * D + b0
            idx111 = r1 * D2 + g1 * D + b1

            lut_flat = lut_c.reshape(B, -1)
            v000 = torch.gather(lut_flat, 1, idx000.clamp(0, D * D * D - 1))
            v111 = torch.gather(lut_flat, 1, idx111.clamp(0, D * D * D - 1))

            r_frac = r_frac.unsqueeze(1) if r_frac.dim() == 2 else r_frac
            g_frac = g_frac.unsqueeze(1) if g_frac.dim() == 2 else g_frac
            b_frac = b_frac.unsqueeze(1) if b_frac.dim() == 2 else b_frac

            v000 = v000.unsqueeze(1) if v000.dim() == 2 else v000
            v111 = v111.unsqueeze(1) if v111.dim() == 2 else v111

            v = v000 * (1 - r_frac) * (1 - g_frac) * (1 - b_frac) + \
                v111 * r_frac * g_frac * b_frac
            output[:, c, :] = v.squeeze(1)

        return output.reshape(B, 3, H, W)


class GlobalLUTBranch(nn.Module):
    """Global LUT branch for color adjustment (from RWKV features)."""

    def __init__(
        self,
        input_dim: int,
        num_luts: int = 5,
        lut_size: int = 33,
        hidden_dim: int = 128,
        use_1d_lut: bool = True,
    ) -> None:
        super().__init__()
        self.num_luts = num_luts
        self.lut_size = lut_size
        self.use_1d_lut = use_1d_lut

        # RWKV features → LUT weights
        self.weight_predictor = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_luts),
            nn.Softmax(dim=-1),
        )

        # Initialize for near-uniform weights
        nn.init.zeros_(self.weight_predictor[-2].weight)
        nn.init.zeros_(self.weight_predictor[-2].bias)

        # Learnable base LUTs
        if use_1d_lut:
            base_luts = torch.zeros(num_luts, 3, lut_size)
            coords = torch.linspace(0, 1, lut_size)

            for i in range(num_luts):
                style_factor = i / max(num_luts - 1, 1)
                for c in range(3):
                    if c == 0:
                        base_luts[i, c] = coords + 0.1 * torch.sin(coords * 3.14159)
                    elif c == 1:
                        base_luts[i, c] = coords ** (0.8 + 0.4 * style_factor)
                    else:
                        base_luts[i, c] = 1 - (1 - coords) ** (0.7 + 0.6 * style_factor)

            base_luts += torch.randn_like(base_luts) * 0.03
            base_luts = base_luts.clamp(0, 1)
            self.base_luts = nn.Parameter(base_luts)
        else:
            coords = torch.linspace(0, 1, lut_size)
            grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing='ij'), dim=-1)
            base_luts = torch.zeros(num_luts, lut_size, lut_size, lut_size, 3)

            for i in range(num_luts):
                if i == 0:
                    base_luts[i] = grid.clone()
                else:
                    base_luts[i] = grid.clone()
                    style_factor = i / max(num_luts - 1, 1)
                    s_curve = torch.sin(coords * 3.14159 / 2) ** 2
                    for c in range(3):
                        strength = 0.05 * style_factor * (1 if c == 0 else -1 if c == 2 else 0.5)
                        base_luts[i, :, :, :, c] += strength * s_curve.unsqueeze(-1)
                    base_luts[i] += torch.randn_like(base_luts[i]) * 0.005

            base_luts = base_luts.clamp(0, 1)
            self.base_luts = nn.Parameter(base_luts)
            self.register_buffer('identity_grid', grid)

        self.sampler = FastLUTSampling(lut_size)

    def forward(self, bottleneck_feat: Tensor, img: Tensor) -> Tensor:
        """Apply LUT-based color adjustment.

        Args:
            bottleneck_feat: (B, N, C) RWKV output features.
            img: (B, 3, H, W) input image.

        Returns:
            (B, 3, H, W) LUT-adjusted image.
        """
        B = bottleneck_feat.shape[0]

        # Predict LUT weights (using first token)
        weights = self.weight_predictor(bottleneck_feat[:, 0])

        # Fuse base LUTs
        if self.use_1d_lut:
            weights_expanded = weights.view(B, self.num_luts, 1, 1)
            fused_lut = (self.base_luts.unsqueeze(0) * weights_expanded).sum(dim=1)
        else:
            weights_expanded = weights.view(B, self.num_luts, 1, 1, 1, 1)
            fused_lut = (self.base_luts.unsqueeze(0) * weights_expanded).sum(dim=1)

        # Apply LUT
        return self.sampler(img, fused_lut)


class CNNDecoderLUTCNN(BaseDecoder):
    """LUT Decoder with Shallow CNN gated fusion.

    Architecture:
        1. Global Branch: RWKV features → LUT weights → global color adjustment
        2. Spatial Branch: CNN features → gated upsampling → M, R
        3. Fusion: enhanced = (1-M) * LUT(img) + M * (img + R)

    Args:
        rwkv_dim: RWKV bottleneck output dimension.
        cnn_feature_channels: Channel counts for CNN features.
        patch_size: Encoder patch size.
        num_upsample_blocks: Spatial branch upsampling stages.
        base_channels: Spatial branch base channels.
        residual_scale: Residual output scaling.
        num_luts: Number of base LUTs.
        lut_size: LUT grid size.
        use_1d_lut: Use 1D LUT (recommended).
    """

    def __init__(
        self,
        input_dim: int,
        cnn_feature_channels: list[int] | None = None,
        patch_size: int = 16,
        num_upsample_blocks: int = 4,
        base_channels: int = 64,
        residual_scale: float = 0.5,
        num_luts: int = 5,
        lut_size: int = 33,
        use_1d_lut: bool = True,
    ) -> None:
        super().__init__()

        if cnn_feature_channels is None:
            cnn_feature_channels = [64, 64, 64, 64]

        self.patch_size = patch_size
        self.residual_scale = residual_scale
        self.predict_residual = True

        # Global LUT branch (from RWKV features)
        self.global_branch = GlobalLUTBranch(
            input_dim=input_dim,
            num_luts=num_luts,
            lut_size=lut_size,
            use_1d_lut=use_1d_lut,
        )

        # Spatial branch (from CNN features)
        self.spatial_branch = SpatialBranchWithCNN(
            cnn_feature_channels=cnn_feature_channels,
            base_channels=base_channels,
            num_upsample_blocks=num_upsample_blocks,
        )

    def forward(
        self,
        x: Tensor,
        h: int,
        w: int,
        img: Tensor | None = None,
        cnn_features: list[Tensor] | None = None,
        **kwargs,
    ) -> Tensor:
        """Decode with LUT global adjustment and CNN spatial modulation.

        Args:
            x: (B, N, C) token features from RWKV.
            h, w: Token grid size.
            img: (B, 3, H, W) original input image (required).
            cnn_features: List of CNN feature maps.

        Returns:
            (B, 3, H, W) residual delta (enhanced - img).
        """
        B = x.shape[0]

        if img is None:
            img = x.new_zeros(B, 3, h * self.patch_size, w * self.patch_size)
        _, _, H, W = img.shape

        if cnn_features is None:
            # Fallback dummy features matching ShallowCNNEncoder's scale pyramid:
            # f0=H/2, f1=H/4, f2=H/8, f3=H/8.
            expected_sizes = self.spatial_branch._expected_feature_sizes((H, W))
            cnn_features = [
                torch.zeros(
                    B,
                    channels,
                    feat_h,
                    feat_w,
                    device=x.device,
                    dtype=x.dtype,
                )
                for channels, (feat_h, feat_w) in zip(
                    self.spatial_branch.cnn_feature_channels,
                    expected_sizes,
                )
            ]

        # Global LUT color adjustment
        lut_output = self.global_branch(x, img)

        # Spatial modulation from CNN features
        M, R = self.spatial_branch(cnn_features, target_size=(H, W))

        # Fusion
        R_scaled = torch.tanh(R) * self.residual_scale
        enhanced = (1 - M) * lut_output + M * (img + R_scaled)
        enhanced = enhanced.clamp(0, 1)

        # Return residual format
        return enhanced - img


__all__ = [
    "ChanLayerNorm",
    "GatedSkipFusion",
    "ResidualRefineBlock",
    "UpsampleBlock",
    "SpatialBranchWithCNN",
    "GlobalLUTBranch",
    "FastLUTSampling",
    "CNNDecoderLUTCNN",
]

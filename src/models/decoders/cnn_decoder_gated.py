"""Residual Gated CNN Decoder with Soft-Gate Skip mechanism.

This decoder implements Scheme A from the architectural improvements:
- Soft-Gate Skip: Decoder generates gate signals to dynamically modulate encoder skips
- Residual Blocks: Deep residual refinement with residual design
- Channel-wise LayerNorm: No batch dependency for image enhancement

Key differences from other decoders:
- SoftGateSkip uses decoder features to gate encoder skips (not simple concat)
- Strong residual design for stable training
- Progressive upsampling with PixelShuffle + skip fusion
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.decoders.base import BaseDecoder


class ChanLayerNorm(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps (B, C, H, W)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1).float()).to(x.dtype).permute(0, 3, 1, 2)


class SoftGateSkip(nn.Module):
    """Gated skip connection fusion.

    Decoder features generate gate signals to dynamically modulate encoder skip
    features. This allows the decoder to selectively use skip information based
    on its current state.

    Design philosophy:
    - Decoder decides how much skip information to use at each position
    - Gate is sigmoid-bounded to [0, 1], providing smooth modulation
    - Residual design: skip is modulated, not replaced
    """

    def __init__(
        self,
        decoder_channels: int,
        skip_channels: int,
        use_bn: bool = False,
    ):
        super().__init__()
        self.use_bn = use_bn

        # Generate gate from decoder features
        self.gate_gen = nn.Conv2d(decoder_channels, skip_channels, 1, bias=True)

        # Optional batch norm for stability
        if use_bn:
            self.gate_bn = nn.BatchNorm2d(skip_channels)

        self.gate_act = nn.Sigmoid()

        # Fusion projection
        self.fuse = nn.Sequential(
            ChanLayerNorm(decoder_channels + skip_channels),
            nn.Conv2d(decoder_channels + skip_channels, decoder_channels, 1, bias=True),
        )

    def forward(self, decoder_feat: Tensor, skip_feat: Tensor) -> Tensor:
        """Fuse decoder and skip features with gating.

        Args:
            decoder_feat: Decoder upsampled features (B, C_dec, H, W)
            skip_feat: Encoder skip features (B, C_skip, H_skip, W_skip)

        Returns:
            Fused features (B, C_dec, H, W)
        """
        # Align spatial size
        if skip_feat.shape[-2:] != decoder_feat.shape[-2:]:
            skip_feat = F.interpolate(
                skip_feat,
                size=decoder_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # Generate gate from decoder features
        gate = self.gate_gen(decoder_feat)
        if self.use_bn:
            gate = self.gate_bn(gate)
        gate = self.gate_act(gate)

        # Modulate skip with gate
        modulated_skip = skip_feat * gate

        # Fuse decoder and modulated skip
        fused = torch.cat([decoder_feat, modulated_skip], dim=1)
        return self.fuse(fused)


class ResidualBlock(nn.Module):
    """Residual block with deep residual design.

    Uses two 3x3 convolutions with LayerNorm and GELU activation.
    Residual connection with final activation (pre-activation style).
    """

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [
            ChanLayerNorm(channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.extend([
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
        ])
        self.net = nn.Sequential(*layers)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.net(x))


class UpsampleBlock(nn.Module):
    """Upsample block with Soft-Gate Skip fusion.

    1. PixelShuffle upsampling (2x)
    2. Soft-Gate Skip fusion (if skip_channels provided)
    3. Residual refinement blocks
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int | None = None,
        num_residual_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_skip = skip_channels is not None

        # Upsampling via PixelShuffle
        self.up = nn.Sequential(
            ChanLayerNorm(in_channels),
            nn.Conv2d(in_channels, out_channels * 4, 3, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.GELU(),
        )

        # Skip fusion
        if self.use_skip:
            self.skip_fuse = SoftGateSkip(
                decoder_channels=out_channels,
                skip_channels=skip_channels,
                use_bn=False,
            )

        # Residual refinement
        self.res_blocks = nn.ModuleList([
            ResidualBlock(out_channels, dropout=dropout)
            for _ in range(num_residual_blocks)
        ])

    def forward(self, x: Tensor, skip_feat: Tensor | None = None) -> Tensor:
        x = self.up(x)

        if self.use_skip and skip_feat is not None:
            x = self.skip_fuse(x, skip_feat)

        for block in self.res_blocks:
            x = block(x)

        return x


class FinalUpsampleBlock(nn.Module):
    """Final upsampling block with precise size alignment.

    Uses bilinear interpolation to reach exact target size, then applies
    skip fusion and residual refinement.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int | None = None,
        num_residual_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_skip = skip_channels is not None

        # Initial projection
        self.proj = nn.Sequential(
            ChanLayerNorm(in_channels),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.GELU(),
        )

        # Skip fusion
        if self.use_skip:
            self.skip_fuse = SoftGateSkip(
                decoder_channels=out_channels,
                skip_channels=skip_channels,
                use_bn=False,
            )

        # Residual refinement
        self.res_blocks = nn.ModuleList([
            ResidualBlock(out_channels, dropout=dropout)
            for _ in range(num_residual_blocks)
        ])

    def forward(
        self,
        x: Tensor,
        target_size: tuple[int, int],
        skip_feat: Tensor | None = None,
    ) -> Tensor:
        x = self.proj(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

        if self.use_skip and skip_feat is not None:
            x = self.skip_fuse(x, skip_feat)

        for block in self.res_blocks:
            x = block(x)

        return x


class GatedResidualDecoder(BaseDecoder):
    """Residual Gated CNN Decoder with Soft-Gate Skip mechanism.

    This decoder implements Scheme A:
    1. Soft-Gate Skip: Decoder generates gates to modulate encoder skips
    2. Residual Blocks: Deep residual refinement for stable training
    3. Progressive Upsampling: PixelShuffle + skip fusion at each stage

    Architecture:
        Token features → 2D projection → Progressive upsampling blocks → Output head

    Each upsampling block:
        1. PixelShuffle (2x) or bilinear (final block)
        2. Soft-Gate Skip fusion (modulates encoder skip with decoder gate)
        3. Residual blocks for refinement

    Args:
        input_dim: Bottleneck output dimension
        patch_size: Encoder patch size
        num_upsample_blocks: Number of upsampling stages
        base_channels: Channel count after initial projection
        skip_channels: List of channel counts for each encoder skip.
                       If None, uses default [768, 768, ...]
        residual_scale: Maximum absolute residual magnitude
        num_skip_fusion_blocks: Number of residual blocks after skip fusion
        num_final_blocks: Number of residual blocks in final stage
        dropout: Dropout rate for residual blocks
        use_final_align: Whether to use alignment refinement

    Example:
        >>> decoder = GatedResidualDecoder(
        ...     input_dim=384,
        ...     skip_channels=[768, 768, 768, 768],
        ...     num_upsample_blocks=4,
        ... )
        >>> output = decoder(tokens, h=12, w=12, vit_skips=skips)
    """

    def __init__(
        self,
        input_dim: int,
        patch_size: int = 14,
        num_upsample_blocks: int = 4,
        base_channels: int = 64,
        skip_channels: list[int] | None = None,
        residual_scale: float = 0.5,
        predict_residual: bool = True,
        num_skip_fusion_blocks: int = 2,
        num_final_blocks: int = 3,
        dropout: float = 0.0,
        use_final_align: bool = True,
    ) -> None:
        super().__init__()
        if not predict_residual:
            raise ValueError("GatedResidualDecoder only supports residual output mode")

        self.patch_size = patch_size
        self.residual_scale = residual_scale
        self.predict_residual = True
        self.num_upsample_blocks = num_upsample_blocks

        # Token projection
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, base_channels),
            nn.GELU(),
        )

        # Build channel schedule
        self.channels = self._build_channel_schedule(base_channels, num_upsample_blocks)

        # Default skip channels
        if skip_channels is None:
            skip_channels = [768] * num_upsample_blocks

        if len(skip_channels) != num_upsample_blocks:
            raise ValueError(
                f"skip_channels length ({len(skip_channels)}) must match "
                f"num_upsample_blocks ({num_upsample_blocks})"
            )

        self.skip_channels = skip_channels

        # Build upsampling blocks
        num_pixelshuffle = num_upsample_blocks - 1

        self.pixelshuffle_blocks = nn.ModuleList()
        for i in range(num_pixelshuffle):
            self.pixelshuffle_blocks.append(
                UpsampleBlock(
                    in_channels=self.channels[i],
                    out_channels=self.channels[i + 1],
                    skip_channels=skip_channels[i],
                    num_residual_blocks=num_skip_fusion_blocks,
                    dropout=dropout,
                )
            )

        # Final block
        self.final_block = FinalUpsampleBlock(
            in_channels=self.channels[num_pixelshuffle],
            out_channels=self.channels[-1],
            skip_channels=skip_channels[-1] if num_upsample_blocks > 0 else None,
            num_residual_blocks=num_final_blocks,
            dropout=dropout,
        )

        # Optional alignment refinement
        if use_final_align:
            self.align_refine = nn.Conv2d(self.channels[-1], self.channels[-1], 3, padding=1)
            nn.init.zeros_(self.align_refine.weight)
            nn.init.zeros_(self.align_refine.bias)
        else:
            self.align_refine = None

        # Output head
        self.head = nn.Sequential(
            ChanLayerNorm(self.channels[-1]),
            nn.Conv2d(self.channels[-1], self.channels[-1], 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(self.channels[-1], 3, 1, bias=True),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _build_channel_schedule(base_channels: int, num_blocks: int) -> list[int]:
        """Build channel schedule for decoder stages.

        Keeps higher channels in deeper stages for better representation.
        """
        channels = [base_channels]
        for i in range(1, num_blocks + 1):
            # Gradual decrease: keep high capacity
            ratio = 1.0 - (0.2 * i / num_blocks)
            ch = max(round(base_channels * ratio), 32)
            channels.append(ch)
        return channels

    def forward(
        self,
        x: Tensor,
        h: int,
        w: int,
        vit_skips: list[Tensor] | None = None,
        vit_h: int | None = None,
        vit_w: int | None = None,
        img: Tensor | None = None,
    ) -> Tensor:
        """Decode token features to residual delta.

        Args:
            x: Token features (B, N, input_dim)
            h: Token grid height
            w: Token grid width
            vit_skips: List of encoder skip features
            vit_h: Original encoder feature height
            vit_w: Original encoder feature width
            img: Ignored (for interface compatibility)

        Returns:
            Residual delta (B, 3, H, W) in [-residual_scale, residual_scale]
        """
        B = x.shape[0]
        vit_h = vit_h or h
        vit_w = vit_w or w

        # Project tokens and reshape to 2D
        x = self.proj(x)
        x = x.transpose(1, 2).contiguous().reshape(B, -1, h, w)

        # Process skip features
        if vit_skips is None:
            vit_skips = [None] * self.num_upsample_blocks

        # Adapt to actual skip count
        if len(vit_skips) < self.num_upsample_blocks:
            vit_skips = list(vit_skips) + [None] * (self.num_upsample_blocks - len(vit_skips))
        elif len(vit_skips) > self.num_upsample_blocks:
            vit_skips = vit_skips[:self.num_upsample_blocks]

        # Normalize skip features to 2D format
        normalized_skips = []
        for skip in vit_skips:
            if skip is None:
                normalized_skips.append(None)
            elif skip.dim() == 3:
                # (B, N, C) -> (B, C, H, W)
                C_skip = skip.shape[-1]
                N_skip = skip.shape[1]
                h_skip = w_skip = int(N_skip ** 0.5)
                if h_skip * w_skip != N_skip:
                    h_skip, w_skip = vit_h, vit_w
                normalized_skips.append(
                    skip.transpose(1, 2).contiguous().reshape(B, -1, h_skip, w_skip)
                )
            else:
                normalized_skips.append(skip)

        # Apply PixelShuffle upsampling blocks
        for i, block in enumerate(self.pixelshuffle_blocks):
            skip_feat = normalized_skips[i]
            x = block(x, skip_feat)

        # Final upsampling block
        target_h = h * (2 ** self.num_upsample_blocks)
        target_w = w * (2 ** self.num_upsample_blocks)
        skip_feat = normalized_skips[-1] if self.num_upsample_blocks > 0 else None
        x = self.final_block(x, (target_h, target_w), skip_feat)

        # Optional alignment refinement
        if self.align_refine is not None:
            x = x + self.align_refine(x)

        return torch.tanh(self.head(x)) * self.residual_scale


__all__ = ["GatedResidualDecoder", "SoftGateSkip", "ResidualBlock"]

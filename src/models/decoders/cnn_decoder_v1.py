"""PSNR-oriented CNN decoder with precise final upsampling and stronger image guidance.

This module keeps the original decoder contract:
token features -> residual delta with shape (B, 3, H, W).

Compared with ``cnn_decoder.py``, the main changes are:
- the final upsampling stage can resize directly to the target image size before
  the output head, so geometry is aligned in feature space instead of repaired
  after producing RGB;
- high-resolution stages keep more channels and use more refinement blocks;
- image-guided skips use local 3x3 convolutions plus gated residual fusion;
- the full-resolution tail is deeper, while the final RGB projection remains
  zero-initialized for identity-friendly training.
"""
from __future__ import annotations

from collections.abc import Sequence

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


class ResidualRefineBlock(nn.Module):
    """Stable local residual refinement block.

    The learnable residual gain starts small. This lets us add capacity at
    high resolution for PSNR without making early training overwrite the
    identity-like residual path too aggressively.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int | None = None,
        residual_gain_init: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_channels = hidden_channels or channels
        self.net = nn.Sequential(
            ChanLayerNorm(channels),
            nn.Conv2d(channels, hidden_channels, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, 3, padding=1, bias=True),
        )
        self.gain = nn.Parameter(torch.full((1, channels, 1, 1), residual_gain_init))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.gain.to(dtype=x.dtype) * self.net(x)


class StrongImgProj(nn.Module):
    """Extract local image guidance features before decoder fusion.

    A 1x1-only projection mostly passes color channels through. The two 3x3
    layers here expose local edges, texture, and color transitions, which are
    exactly the cues that help reduce paired pixel reconstruction error.
    """

    def __init__(self, out_channels: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden_channels = hidden_channels or max(out_channels, 16)
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, 1, bias=True),
            ChanLayerNorm(out_channels),
        )

    def forward(self, img: Tensor) -> Tensor:
        return self.net(img)


class GatedFuse(nn.Module):
    """Residual gated fusion between decoder features and image guidance.

    The decoder feature remains the main signal. Image guidance is injected as
    a learned residual delta with a spatial/channel gate, which is more stable
    for restoration than replacing features with a raw concat + 1x1 projection.
    """

    def __init__(
        self,
        feat_channels: int,
        img_channels: int,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        in_channels = feat_channels + img_channels
        hidden_channels = hidden_channels or feat_channels
        self.delta = nn.Sequential(
            ChanLayerNorm(in_channels),
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, feat_channels, 3, padding=1, bias=True),
        )
        self.gate = nn.Sequential(
            ChanLayerNorm(in_channels),
            nn.Conv2d(in_channels, feat_channels, 1, bias=True),
            nn.Sigmoid(),
        )

        # Start from neutral fusion. The model learns where image guidance helps
        # instead of perturbing every scale at initialization.
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, x: Tensor, img_feat: Tensor) -> Tensor:
        fused = torch.cat([x, img_feat], dim=1)
        return x + self.gate(fused) * self.delta(fused)


def _make_refine_stack(
    channels: int,
    num_blocks: int,
    residual_gain_init: float,
) -> nn.Sequential:
    return nn.Sequential(
        *[
            ResidualRefineBlock(
                channels=channels,
                residual_gain_init=residual_gain_init,
            )
            for _ in range(num_blocks)
        ]
    )


class PixelShuffleUpsampleBlock(nn.Module):
    """Learned 2x upsampling block for stages that are geometrically exact."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_channels: int,
        num_refine_blocks: int,
        residual_gain_init: float,
    ) -> None:
        super().__init__()
        self.up = nn.Sequential(
            ChanLayerNorm(in_channels),
            nn.Conv2d(in_channels, out_channels * 4, 3, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.GELU(),
        )
        self.fuse = GatedFuse(out_channels, img_channels)
        self.refine = _make_refine_stack(out_channels, num_refine_blocks, residual_gain_init)

    def forward(self, x: Tensor, img_feat: Tensor) -> Tensor:
        x = self.up(x)
        x = self.fuse(x, img_feat)
        return self.refine(x)


class PreciseUpsampleBlock(nn.Module):
    """Resize to an explicit target size, then refine in feature space.

    This block is used for the last stage by default. Even when DINOv3's
    patch_size=16 makes four 2x stages naturally align, this block keeps the
    geometry explicit and avoids a separate RGB-space alignment repair path.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_channels: int,
        num_refine_blocks: int,
        residual_gain_init: float,
        mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.pre = nn.Sequential(
            ChanLayerNorm(in_channels),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.GELU(),
        )
        self.fuse = GatedFuse(out_channels, img_channels)
        self.refine = _make_refine_stack(out_channels, num_refine_blocks, residual_gain_init)

    def forward(self, x: Tensor, target_size: tuple[int, int], img_feat: Tensor) -> Tensor:
        x = self.pre(x)
        align_corners = False if self.mode in {"bilinear", "bicubic"} else None
        x = F.interpolate(x, size=target_size, mode=self.mode, align_corners=align_corners)
        x = self.fuse(x, img_feat)
        return self.refine(x)


class RefineTail(nn.Module):
    """Full-resolution refinement tail before residual RGB prediction."""

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        residual_gain_init: float,
    ) -> None:
        super().__init__()
        self.blocks = _make_refine_stack(channels, num_blocks, residual_gain_init)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class CNNDecoderV1(BaseDecoder):
    """PSNR-oriented image-guided CNN decoder.

    Args:
        input_dim: Bottleneck output dimension.
        patch_size: Encoder patch size. DINOv3 ViT-B uses 16.
        num_upsample_blocks: Total number of upsampling stages.
        base_channels: Channel count after token projection.
        residual_scale: Maximum absolute residual magnitude after tanh.
        img_proj_channels: Image guidance feature channels at each scale.
        predict_residual: Kept for pipeline compatibility; must be True.
        precise_upsample_stages: Number of final stages that resize to explicit
            target sizes instead of using fixed PixelShuffle. The default of 1
            makes the full-res geometry explicit before the output head.
        stage_depths: Number of refinement blocks per upsampling stage. Later
            stages should be thicker because PSNR depends on full-res detail.
        tail_blocks: Number of full-resolution refinement blocks before RGB.
        channel_multipliers: Optional channel schedule relative to base_channels.
            Defaults to [1, 1, 1, 0.75, 0.75] for four stages, avoiding the thin
            high-res tail of the original decoder.
        min_channels: Lower bound for every stage channel count.
        img_hidden_channels: Hidden width for StrongImgProj.
        residual_gain_init: Initial gain inside refinement residual blocks.
        precise_mode: Interpolation mode for PreciseUpsampleBlock.
    """

    def __init__(
        self,
        input_dim: int,
        patch_size: int = 16,
        num_upsample_blocks: int = 4,
        base_channels: int = 128,
        residual_scale: float = 1.0,
        img_proj_channels: int = 32,
        predict_residual: bool = True,
        precise_upsample_stages: int = 1,
        stage_depths: Sequence[int] | None = None,
        tail_blocks: int = 3,
        channel_multipliers: Sequence[float] | None = None,
        min_channels: int = 48,
        img_hidden_channels: int | None = None,
        residual_gain_init: float = 0.1,
        precise_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        if not predict_residual:
            raise ValueError("CNNDecoderV1 only supports residual output mode")
        if num_upsample_blocks < 1:
            raise ValueError("num_upsample_blocks must be >= 1")
        if precise_upsample_stages < 0:
            raise ValueError("precise_upsample_stages must be >= 0")

        self.patch_size = patch_size
        self.residual_scale = residual_scale
        self.predict_residual = True
        self.num_upsample_blocks = num_upsample_blocks
        self.precise_upsample_stages = min(precise_upsample_stages, num_upsample_blocks)

        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, base_channels),
            nn.GELU(),
        )

        channels = self._build_channel_schedule(
            base_channels=base_channels,
            num_upsample_blocks=num_upsample_blocks,
            channel_multipliers=channel_multipliers,
            min_channels=min_channels,
        )
        depths = self._build_stage_depths(num_upsample_blocks, stage_depths)

        # Shared image branch. It is evaluated on resized copies of the input
        # image at each decoder scale, preserving the original img-guided design
        # while making the guidance locally aware.
        self.img_proj = StrongImgProj(
            out_channels=img_proj_channels,
            hidden_channels=img_hidden_channels,
        )

        num_pixelshuffle = num_upsample_blocks - self.precise_upsample_stages
        self.pixelshuffle_blocks = nn.ModuleList(
            [
                PixelShuffleUpsampleBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    img_channels=img_proj_channels,
                    num_refine_blocks=depths[i],
                    residual_gain_init=residual_gain_init,
                )
                for i in range(num_pixelshuffle)
            ]
        )
        self.precise_blocks = nn.ModuleList(
            [
                PreciseUpsampleBlock(
                    in_channels=channels[num_pixelshuffle + i],
                    out_channels=channels[num_pixelshuffle + i + 1],
                    img_channels=img_proj_channels,
                    num_refine_blocks=depths[num_pixelshuffle + i],
                    residual_gain_init=residual_gain_init,
                    mode=precise_mode,
                )
                for i in range(self.precise_upsample_stages)
            ]
        )

        final_channels = channels[-1]
        self.align_refine = ResidualRefineBlock(
            final_channels,
            residual_gain_init=min(residual_gain_init, 0.05),
        )
        self.tail = RefineTail(
            channels=final_channels,
            num_blocks=tail_blocks,
            residual_gain_init=residual_gain_init,
        )
        self.head = nn.Sequential(
            ChanLayerNorm(final_channels),
            nn.Conv2d(final_channels, final_channels, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(final_channels, final_channels, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(final_channels, 3, 1, bias=True),
        )

        # Identity-friendly residual learning: initial delta is exactly zero.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _build_channel_schedule(
        base_channels: int,
        num_upsample_blocks: int,
        channel_multipliers: Sequence[float] | None,
        min_channels: int,
    ) -> list[int]:
        if channel_multipliers is None:
            # For four stages this gives [base, base, base, 0.75b, 0.75b].
            # The last stages stay deliberately wider than the original decoder
            # because edge, texture, and color refinement happen at high res.
            channel_multipliers = (1.0, 1.0, 1.0, 0.75, 0.75)
        if len(channel_multipliers) == 0:
            raise ValueError("channel_multipliers must not be empty")

        channels: list[int] = []
        for i in range(num_upsample_blocks + 1):
            ratio = channel_multipliers[min(i, len(channel_multipliers) - 1)]
            channels.append(max(round(base_channels * ratio), min_channels))
        return channels

    @staticmethod
    def _build_stage_depths(
        num_upsample_blocks: int,
        stage_depths: Sequence[int] | None,
    ) -> list[int]:
        if stage_depths is not None:
            if len(stage_depths) != num_upsample_blocks:
                raise ValueError("stage_depths length must equal num_upsample_blocks")
            return [int(v) for v in stage_depths]

        depths = [1 for _ in range(num_upsample_blocks)]
        if num_upsample_blocks >= 2:
            depths[-2] = 2
        depths[-1] = 2
        return depths

    @staticmethod
    def _interp(
        img: Tensor,
        size: tuple[int, int],
        mode: str = "bilinear",
    ) -> Tensor:
        align_corners = False if mode in {"bilinear", "bicubic"} else None
        return F.interpolate(img, size=size, mode=mode, align_corners=align_corners)

    def _next_precise_target(
        self,
        current_size: tuple[int, int],
        final_size: tuple[int, int],
        remaining_blocks: int,
    ) -> tuple[int, int]:
        if remaining_blocks <= 1:
            return final_size

        cur_h, cur_w = current_size
        final_h, final_w = final_size
        return (
            min(final_h, max(cur_h + 1, cur_h * 2)),
            min(final_w, max(cur_w + 1, cur_w * 2)),
        )

    def forward(self, x: Tensor, h: int, w: int, img: Tensor | None = None) -> Tensor:
        """Decode token features to a bounded residual delta.

        Args:
            x: Token features (B, N, C), where N should be h * w.
            h: Token grid height.
            w: Token grid width.
            img: Original input image (B, 3, H, W). If omitted, a zero image
                with size (h * patch_size, w * patch_size) is used.

        Returns:
            Residual delta (B, 3, H, W) in [-residual_scale, residual_scale].
        """
        b, n, _ = x.shape
        if n != h * w:
            raise ValueError(f"Expected token count h*w={h * w}, got {n}")

        if img is None:
            img = x.new_zeros(b, 3, h * self.patch_size, w * self.patch_size)
        _, _, target_h, target_w = img.shape
        final_size = (target_h, target_w)

        x = self.proj(x)
        x = x.transpose(1, 2).contiguous().reshape(b, -1, h, w)

        for block in self.pixelshuffle_blocks:
            out_size = (x.shape[-2] * 2, x.shape[-1] * 2)
            img_feat = self.img_proj(self._interp(img, out_size))
            x = block(x, img_feat)

        for i, block in enumerate(self.precise_blocks):
            remaining = len(self.precise_blocks) - i
            out_size = self._next_precise_target(x.shape[-2:], final_size, remaining)
            img_feat = self.img_proj(self._interp(img, out_size))
            x = block(x, out_size, img_feat)

        # Fallback only. With DINOv3 patch_size=16 and default stages, x should
        # already be full-res here. If a config changes scales, align in feature
        # space and use a very small residual correction before the RGB head.
        if x.shape[-2:] != final_size:
            x = self._interp(x, final_size)
            x = self.align_refine(x)

        x = self.tail(x)
        return torch.tanh(self.head(x)) * self.residual_scale


__all__ = ["CNNDecoderV1"]

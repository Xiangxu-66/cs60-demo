"""CNN decoder with progressive encoder skip connections."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Literal
import math

from src.models.decoders.base import BaseDecoder
from src.models.decoders.cnn_decoder import (
    ChanLayerNorm,
    ResidualBlock,
    UpsampleBlock,
)
from src.models.modules.fusion import MultiScaleFusionBlock


class SkipFusionBlock(MultiScaleFusionBlock):
    """Legacy alias for MultiScaleFusionBlock."""
    pass


class FiLMCondition(nn.Module):
    """Feature-wise Linear Modulation conditioned on a global token.

    Zero-initialized so it is transparent at the start of training:
    gamma=0, beta=0 → output = input. This ensures no disruption to
    already-converged skip features when FiLM is first introduced.

    Formula: out = (1 + gamma) * x + beta
    """

    def __init__(self, feat_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(cond_dim, 2 * feat_ch)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        # x: (B, C, H, W)   cond: (B, cond_dim)
        gamma, beta = self.fc(cond).chunk(2, dim=-1)
        return (1 + gamma[:, :, None, None]) * x + beta[:, :, None, None]


class SimpleUpsampleBlock(nn.Module):
    """2× upsample via PixelShuffle without image skip fusion.

    Simpler version of UpsampleBlock that does not fuse image features.
    Used when img_skip_enabled is False.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm = ChanLayerNorm(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels * 4, 3, padding=1, bias=True)
        self.ps = nn.PixelShuffle(2)
        self.act = nn.GELU()
        self.res = ResidualBlock(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.ps(self.conv(self.norm(x))))
        return self.res(x)


class CNNSkipDecoder(BaseDecoder):
    """CNN decoder with progressive encoder skip connections.

    Extends CNNDecoder with multi-scale skip features from the encoder.
    Supports dual-path skip connections:
    - color_naf: Multi-scale texture/color features from ColorNAF encoder
    - dino: Semantic features from DINOv3 encoder
    - legacy: local/colour features from v2_dinov3 encoder

    Skip connection points:
    - Stage 1 (2h→4h): color_naf[stage1] or local[128ch] + colour[64ch]
    - Stage 2 (4h→8h): color_naf[stage2] or local[64ch] + colour[32ch]
    - Stage 3 (8h→16h): color_naf (if available) or local_raw[3ch] + colour_raw[6ch]

    Args:
        input_dim: Input token dimension.
        patch_size: Encoder patch size.
        num_upsample_blocks: Number of 2× upsample stages.
        base_channels: Base channel count.
        residual_scale: Output residual scaling.
        img_proj_channels: Image skip projection channels (used if img_skip_enabled).
        predict_residual: Whether to predict residual or full image.
        skip_enabled: Enable/disable encoder skip connections.
        skip_fusion_mode: Fusion mode for encoder skip connections.
            "concat": Concatenate and project.
            "add": Project and add.
            "gate": Simple gated fusion.
            "gated_residual": Gated residual fusion.
            "attention_gate": Attention gate fusion.
        skip_stages: Which stages to apply encoder skip (0-indexed).
        img_skip_enabled: Enable/disable original image skip connections.
        img_skip_fusion_mode: Fusion mode for the per-stage image guidance skip.
            Only takes effect when img_skip_enabled=True. "concat" preserves
            the legacy 1×1 conv path; other modes route through
            MultiScaleFusionBlock.
        color_naf_channels: Channel config for ColorNAF skips [stage0, stage1, stage2].
    """

    def __init__(
        self,
        input_dim: int,
        patch_size: int = 16,
        num_upsample_blocks: int = 4,
        base_channels: int = 64,
        residual_scale: float = 0.5,
        img_proj_channels: int = 3,
        predict_residual: bool = True,
        skip_enabled: bool = True,
        skip_fusion_mode: Literal["concat", "add", "gate", "gated_residual", "attention_gate"] = "gated_residual",
        skip_stages: list[int] = [1, 2, 3],
        img_skip_enabled: bool = True,
        img_skip_fusion_mode: Literal["concat", "add", "gate", "gated_residual", "attention_gate"] = "concat",
        color_naf_channels: list[int] = [64, 64, 128],
        film_stages: list[int] = [],
        film_cond_dim: int = 384,
    ) -> None:
        super().__init__()
        if not predict_residual:
            raise ValueError("CNNSkipDecoder only supports residual output mode")

        self.patch_size = patch_size
        self.residual_scale = residual_scale
        self.predict_residual = True
        self.skip_enabled = skip_enabled
        self.skip_fusion_mode = skip_fusion_mode
        self.skip_stages = skip_stages
        self.img_skip_enabled = img_skip_enabled
        self.img_skip_fusion_mode = img_skip_fusion_mode
        self.color_naf_channels = color_naf_channels
        self.film_stages = list(film_stages)

        # Token projection (same as CNNDecoder)
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, base_channels),
            nn.GELU(),
        )

        # Image skip projection (only used if img_skip_enabled)
        if self.img_skip_enabled:
            self.img_proj = nn.Sequential(
                nn.Conv2d(3, img_proj_channels, 1, bias=True),
                ChanLayerNorm(img_proj_channels),
            )

        # Channel schedule
        _ratios = (1.0, 0.75, 0.5, 0.375)
        ch = [base_channels] + [
            max(round(base_channels * _ratios[min(i - 1, len(_ratios) - 1)]), 24)
            for i in range(1, num_upsample_blocks + 1)
        ]

        # Upsample blocks - choose type based on img_skip_enabled
        if self.img_skip_enabled:
            self.upsample_blocks = nn.ModuleList([
                UpsampleBlock(
                    ch[i],
                    ch[i + 1],
                    img_proj_channels,
                    img_fusion_mode=img_skip_fusion_mode,
                )
                for i in range(num_upsample_blocks)
            ])
        else:
            self.upsample_blocks = nn.ModuleList([
                SimpleUpsampleBlock(ch[i], ch[i + 1])
                for i in range(num_upsample_blocks)
            ])

        # ===== FiLM conditioning blocks =====
        # Keyed by str(stage_idx) to satisfy nn.ModuleDict requirements.
        # Empty when film_stages=[] so existing configs are unaffected.
        self.film_blocks = nn.ModuleDict()
        for stage_idx in self.film_stages:
            target_ch = ch[stage_idx + 1]
            self.film_blocks[str(stage_idx)] = FiLMCondition(target_ch, film_cond_dim)

        # ===== Skip fusion blocks =====
        # Support multiple skip sources: color_naf, legacy (local/colour), dino
        # Only built when skip_enabled=True so that skip_enabled=False gives a
        # parameter-matched baseline (no dead weights with zero gradients).
        self.skip_fusion_blocks = nn.ModuleDict()

        if self.skip_enabled:
            # Legacy skip config (v2_dinov3 format)
            stage_skip_config = {
                1: {"local": 128, "colour": 64},   # After first upsample
                2: {"local": 64, "colour": 32},    # After second upsample
                3: {"local": 3, "colour": 6},      # After third upsample
            }

            # ColorNAF skip config
            # stage0 (H/2) -> upsample stage 0
            # stage1 (H/4) -> upsample stage 1
            # stage2 (H/8) -> upsample stage 2
            color_naf_skip_config = {
                0: color_naf_channels[0],  # stage0
                1: color_naf_channels[1],  # stage1
                2: color_naf_channels[2],  # stage2
            }

            for stage_idx in skip_stages:
                target_ch = ch[stage_idx + 1]  # Output channels of this stage

                # Check if we have ColorNAF skip for this stage
                if stage_idx in color_naf_skip_config:
                    skip_ch_list = [color_naf_skip_config[stage_idx]]
                    self.skip_fusion_blocks[f"color_naf_{stage_idx}"] = SkipFusionBlock(
                        main_ch=target_ch,
                        skip_channels=skip_ch_list,
                        mode=skip_fusion_mode,
                    )

                # Legacy skip support
                if stage_idx in stage_skip_config:
                    skip_ch_list = [
                        stage_skip_config[stage_idx]["local"],
                        stage_skip_config[stage_idx]["colour"],
                    ]
                    self.skip_fusion_blocks[f"legacy_{stage_idx}"] = SkipFusionBlock(
                        main_ch=target_ch,
                        skip_channels=skip_ch_list,
                        mode=skip_fusion_mode,
                    )

        # Output head
        self.head = nn.Sequential(
            ChanLayerNorm(ch[-1]),
            nn.Conv2d(ch[-1], ch[-1], 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(ch[-1], 3, 1, bias=True),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        # Alignment refinement
        self.align_refine = nn.Conv2d(ch[-1], ch[-1], 3, padding=1, bias=True)
        nn.init.zeros_(self.align_refine.weight)
        nn.init.zeros_(self.align_refine.bias)

    def forward(
        self,
        x: Tensor,
        h: int,
        w: int,
        img: Tensor | None = None,
        skip_dict: dict | None = None,
        hist_token: Tensor | None = None,
    ) -> Tensor:
        """Decode token features to residual delta.

        Args:
            x: Token features (B, N, C).
            h: Spatial grid height.
            w: Spatial grid width.
            img: Original input image (B, 3, H, W). Required if img_skip_enabled.
            skip_dict: Skip features from encoder. Supports multiple formats:
                - color_naf: {"color_naf": {"stage0": ..., "stage1": ..., "stage2": ...}}
                - legacy: {"local": {...}, "colour": {...}, "raw_input": {...}}
                - dino: {"dino": {"layer2": ..., "layer5": ..., "layer8": ..., "layer11": ...}}
            hist_token: Global color statistics token from ColorNAF, shape (B, cond_dim).
                Used for FiLM conditioning when film_stages is non-empty.

        Returns:
            Residual delta (B, 3, H, W).
        """
        B = x.shape[0]
        scale = 2 ** len(self.upsample_blocks)
        if img is None:
            img = x.new_zeros(B, 3, h * scale, w * scale)
        _, _, H, W = img.shape

        # Project tokens and reshape
        x = self.proj(x)
        x = x.transpose(1, 2).contiguous().reshape(B, -1, h, w)

        # Progressive upsampling with skip fusion
        for i, block in enumerate(self.upsample_blocks):
            out_h = h * (2 ** (i + 1))
            out_w = w * (2 ** (i + 1))

            if self.img_skip_enabled:
                # Standard image skip
                img_resized = F.interpolate(img, size=(out_h, out_w), mode="bilinear", align_corners=False)
                img_feat = self.img_proj(img_resized)
                x = block(x, img_feat)
            else:
                # No image skip
                x = block(x)

            # ===== Encoder skip fusion =====
            if self.skip_enabled and skip_dict is not None and i in self.skip_stages:
                # Try ColorNAF skips first
                if "color_naf" in skip_dict:
                    color_naf_skips = skip_dict["color_naf"]
                    stage_key = f"stage{i}"
                    fusion_key = f"color_naf_{i}"

                    if stage_key in color_naf_skips and fusion_key in self.skip_fusion_blocks:
                        skip_feat = color_naf_skips[stage_key]
                        # Resize if needed
                        if skip_feat.shape[-2:] != (out_h, out_w):
                            skip_feat = F.interpolate(
                                skip_feat, size=(out_h, out_w), mode="bilinear", align_corners=False
                            )
                        x = self.skip_fusion_blocks[fusion_key](x, [skip_feat], (out_h, out_w))

                # Try legacy skips (v2_dinov3 format)
                elif "local" in skip_dict or "colour" in skip_dict:
                    fusion_key = f"legacy_{i}"
                    if fusion_key in self.skip_fusion_blocks:
                        # Collect skip features for this stage
                        skip_feats = []
                        if "local" in skip_dict:
                            local_dict = skip_dict["local"]
                            if i == 1 and "layer1" in local_dict:
                                skip_feats.append(local_dict["layer1"])
                            elif i == 2 and "layer0" in local_dict:
                                skip_feats.append(local_dict["layer0"])
                            elif i == 3 and "raw_input" in skip_dict:
                                skip_feats.append(skip_dict["raw_input"]["local"])

                        if "colour" in skip_dict:
                            colour_dict = skip_dict["colour"]
                            if i == 1 and "layer1" in colour_dict:
                                skip_feats.append(colour_dict["layer1"])
                            elif i == 2 and "layer0" in colour_dict:
                                skip_feats.append(colour_dict["layer0"])
                            elif i == 3 and "raw_input" in skip_dict:
                                skip_feats.append(skip_dict["raw_input"]["colour"])

                        # Apply skip fusion if we have features
                        if len(skip_feats) >= 1:
                            x = self.skip_fusion_blocks[fusion_key](x, skip_feats, (out_h, out_w))

            # ===== FiLM conditioning =====
            if hist_token is not None and str(i) in self.film_blocks:
                x = self.film_blocks[str(i)](x, hist_token)

        # Align to original size
        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
            x = x + self.align_refine(x)

        return torch.tanh(self.head(x)) * self.residual_scale

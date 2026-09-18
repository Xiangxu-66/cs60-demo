"""方案B: 双支路空间感知版 CNN Decoder.

核心思想：
    DINOv3 深层特征丢失低层纹理细节，传统的 skip 连接难以有效恢复。
    方案B 显式分离：
    - 语义支路: DINOv3 + RWKV → 高层语义理解
    - 细节支路: Shallow CNN → 低层纹理提取
    - Cross-Gated Attention: 语义调制细节的双向交互

Architecture:
                      ┌─────────────────────────────────────┐
                      │    Input Image (B, 3, H, W)         │
                      └─────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
        ┌───────────▼───────────┐       ┌──────────────▼──────────────┐
        │  Semantic Branch      │       │   Detail Branch             │
        │  (DINOv3 + RWKV)      │       │   (Shallow CNN)             │
        │  ─────────────────    │       │   ────────────────          │
        │  Global semantics,    │       │   Local textures,           │
        │  color understanding  │       │   fine edges                 │
        └───────────┬───────────┘       └──────────────┬──────────────┘
                    │                                   │
                    │    Bottleneck tokens (B,N,C)      │
                    │    Shallow features (B,Cd,H',W')  │
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │   Cross-Gated Attention Module    │
                    │   ────────────────────────────    │
                    │   - Semantic → Detail: Modulation │
                    │   - Detail → Semantic: Refinement │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │   Progressive Upsampling          │
                    │   (PixelShuffle + Refinement)     │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │   Output Head → Residual Delta     │
                    └───────────────────────────────────┘
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
        # Preserve input dtype to avoid memory spike from float() conversion
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ShallowCNNBranch(nn.Module):
    """细节支路: 从输入图像提取低层纹理特征.

    Design principles:
    - 保持浅层，避免过度抽象
    - 使用小卷积核捕获局部细节
    - 多尺度特征融合，保留不同粒度的纹理信息

    Args:
        in_channels: Input channels (3 for RGB).
        base_channels: Base channel count for the branch.
        num_stages: Number of downsampling stages.
        out_channels: Output channels for cross-gated attention.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_stages: int = 2,
        out_channels: int = 64,
    ) -> None:
        super().__init__()
        self.num_stages = num_stages
        self.out_channels = out_channels
        self.base_channels = base_channels

        # Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=True),
            ChanLayerNorm(base_channels),
            nn.GELU(),
        )

        # Project stem output to out_channels
        self.proj_stem = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 1, bias=True),
            ChanLayerNorm(out_channels),
            nn.GELU(),
        )

        # Multi-scale feature extraction and projection
        self.stages = nn.ModuleList()
        self.projs = nn.ModuleList()
        current_channels = base_channels
        for i in range(num_stages):
            out_ch = base_channels * (2 ** i)
            stage = nn.Sequential(
                nn.Conv2d(current_channels, out_ch, 3, stride=2, padding=1, bias=True),
                ChanLayerNorm(out_ch),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
                ChanLayerNorm(out_ch),
                nn.GELU(),
            )
            proj = nn.Sequential(
                nn.Conv2d(out_ch, out_channels, 1, bias=True),
                ChanLayerNorm(out_channels),
                nn.GELU(),
            )
            self.stages.append(stage)
            self.projs.append(proj)
            current_channels = out_ch

    def forward(self, x: Tensor) -> list[Tensor]:
        """Extract multi-scale detail features.

        Args:
            x: Input image (B, 3, H, W).

        Returns:
            List of feature maps at different scales, from fine to coarse.
            All features have output_channels for consistent cross attention.
        """
        features = []
        x = self.stem(x)
        # Save projected stem output
        features.append(self.proj_stem(x))

        for stage, proj in zip(self.stages, self.projs):
            x = stage(x)
            # Save projected stage output
            features.append(proj(x))

        return features


class CrossGatedAttention(nn.Module):
    """Cross-Gated Attention: 语义与细节的双向交互.

    核心机制：
    1. Semantic Gate (语义 → 细节): 高层语义调制细节特征的权重
    2. Detail Refinement (细节 → 语义): 细节特征通过门控增强语义表示

    Args:
        semantic_channels: Channels from semantic branch (bottleneck output).
        detail_channels: Channels from detail branch (shallow CNN output).
        hidden_channels: Hidden channels for gate computation.
        reduction: Channel reduction ratio for efficiency.
    """

    def __init__(
        self,
        semantic_channels: int,
        detail_channels: int,
        hidden_channels: int | None = None,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        hidden_channels = hidden_channels or max(semantic_channels, detail_channels) // 2

        # Semantic → Detail: 语义调制细节
        self.semantic_to_detail_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(semantic_channels, semantic_channels // reduction, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(semantic_channels // reduction, detail_channels, 1, bias=True),
            nn.Sigmoid(),
        )

        # Detail → Semantic: 细节增强语义
        self.detail_to_semantic = nn.Sequential(
            ChanLayerNorm(detail_channels),
            nn.Conv2d(detail_channels, hidden_channels, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, semantic_channels, 3, padding=1, bias=True),
        )

        self.detail_gate = nn.Sequential(
            ChanLayerNorm(semantic_channels + detail_channels),
            nn.Conv2d(semantic_channels + detail_channels, semantic_channels, 1, bias=True),
            nn.Sigmoid(),
        )

        # Output projections
        self.semantic_proj = nn.Conv2d(semantic_channels * 2, semantic_channels, 1, bias=True)
        self.detail_proj = nn.Conv2d(detail_channels * 2, detail_channels, 1, bias=True)

    def forward(
        self,
        semantic_feat: Tensor,
        detail_feat: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply cross-gated attention.

        Args:
            semantic_feat: Semantic features (B, C_sem, H, W).
            detail_feat: Detail features (B, C_det, H, W).

        Returns:
            enhanced_semantic: Semantic features enhanced by detail.
            enhanced_detail: Detail features modulated by semantic.
        """
        # Align spatial sizes
        if semantic_feat.shape[-2:] != detail_feat.shape[-2:]:
            detail_feat = F.interpolate(
                detail_feat,
                size=semantic_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # Semantic → Detail: 语义调制细节权重
        sem_gate = self.semantic_to_detail_gate(semantic_feat)
        modulated_detail = detail_feat * sem_gate

        # Detail → Semantic: 细节增强语义
        detail_enhancement = self.detail_to_semantic(modulated_detail)

        # Compute fusion gate
        concat = torch.cat([semantic_feat, modulated_detail], dim=1)
        gate = self.detail_gate(concat)

        # Apply gated refinement
        enhanced_semantic_raw = semantic_feat + gate * detail_enhancement
        enhanced_semantic = self.semantic_proj(
            torch.cat([semantic_feat, enhanced_semantic_raw], dim=1)
        )

        # Enhanced detail: original + modulated
        enhanced_detail = self.detail_proj(
            torch.cat([detail_feat, modulated_detail], dim=1)
        )

        return enhanced_semantic, enhanced_detail


class ResidualRefineBlock(nn.Module):
    """Stable local residual refinement block."""

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


class DualBranchUpsampleBlock(nn.Module):
    """Upsampling block with dual branch fusion.

    Combines semantic and detail features during progressive upsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        detail_channels: int,
        num_refine_blocks: int = 1,
        residual_gain_init: float = 0.1,
        use_cross_attention: bool = True,
    ) -> None:
        super().__init__()
        self.use_cross_attention = use_cross_attention

        # Upsampling
        self.up = nn.Sequential(
            ChanLayerNorm(in_channels),
            nn.Conv2d(in_channels, out_channels * 4, 3, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.GELU(),
        )

        # Cross-attention at this scale
        if use_cross_attention:
            self.cross_attn = CrossGatedAttention(
                semantic_channels=out_channels,
                detail_channels=detail_channels,
            )

        # Detail projection
        self.detail_proj = nn.Sequential(
            nn.Conv2d(detail_channels, out_channels // 2, 1, bias=True),
            ChanLayerNorm(out_channels // 2),
            nn.GELU(),
        )

        # Fusion
        self.fuse = nn.Sequential(
            ChanLayerNorm(out_channels + out_channels // 2),
            nn.Conv2d(out_channels + out_channels // 2, out_channels, 1, bias=True),
            nn.GELU(),
        )

        # Refinement
        self.refine_blocks = nn.ModuleList([
            ResidualRefineBlock(out_channels, residual_gain_init=residual_gain_init)
            for _ in range(num_refine_blocks)
        ])

    def forward(
        self,
        semantic_feat: Tensor,
        detail_feat: Tensor | None = None,
    ) -> Tensor:
        """Upsample and fuse semantic and detail features."""
        x = self.up(semantic_feat)

        if detail_feat is not None:
            # Align detail feature size
            if detail_feat.shape[-2:] != x.shape[-2:]:
                detail_feat = F.interpolate(
                    detail_feat,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            if self.use_cross_attention:
                # Cross-gated attention
                x, _ = self.cross_attn(x, detail_feat)
            else:
                # Simple fusion
                detail_proj = self.detail_proj(detail_feat)
                x = self.fuse(torch.cat([x, detail_proj], dim=1))

        for block in self.refine_blocks:
            x = block(x)

        return x


class PreciseUpsampleBlock(nn.Module):
    """Final upsampling block with precise size alignment."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        detail_channels: int,
        num_refine_blocks: int = 2,
        residual_gain_init: float = 0.1,
        use_cross_attention: bool = True,
    ) -> None:
        super().__init__()
        self.use_cross_attention = use_cross_attention

        # Projection
        self.pre = nn.Sequential(
            ChanLayerNorm(in_channels),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.GELU(),
        )

        # Cross-attention
        if use_cross_attention:
            self.cross_attn = CrossGatedAttention(
                semantic_channels=out_channels,
                detail_channels=detail_channels,
            )

        # Detail projection
        self.detail_proj = nn.Sequential(
            nn.Conv2d(detail_channels, out_channels // 2, 1, bias=True),
            ChanLayerNorm(out_channels // 2),
            nn.GELU(),
        )

        # Fusion
        self.fuse = nn.Sequential(
            ChanLayerNorm(out_channels + out_channels // 2),
            nn.Conv2d(out_channels + out_channels // 2, out_channels, 1, bias=True),
            nn.GELU(),
        )

        # Refinement
        self.refine_blocks = nn.ModuleList([
            ResidualRefineBlock(out_channels, residual_gain_init=residual_gain_init)
            for _ in range(num_refine_blocks)
        ])

    def forward(
        self,
        semantic_feat: Tensor,
        target_size: tuple[int, int],
        detail_feat: Tensor | None = None,
    ) -> Tensor:
        """Project and resize to target size."""
        x = self.pre(semantic_feat)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

        if detail_feat is not None:
            if detail_feat.shape[-2:] != x.shape[-2:]:
                detail_feat = F.interpolate(
                    detail_feat,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            if self.use_cross_attention:
                x, _ = self.cross_attn(x, detail_feat)
            else:
                detail_proj = self.detail_proj(detail_feat)
                x = self.fuse(torch.cat([x, detail_proj], dim=1))

        for block in self.refine_blocks:
            x = block(x)

        return x


class CNNDecoderWithDualBranch(BaseDecoder):
    """方案B: 双支路空间感知版 CNN Decoder.

    通过显式分离语义和细节支路，解决 DINOv3 深层特征丢失纹理细节的问题。

    Architecture:
        1. 语义支路: 接收 DINOv3 + RWKV bottleneck tokens
        2. 细节支路: Shallow CNN 从输入图像提取多尺度纹理
        3. Cross-Gated Attention: 双向交互融合
        4. 渐进式上采样: PixelShuffle + 细节融合
        5. 输出头: Residual delta 预测

    Args:
        input_dim: Bottleneck output dimension (semantic branch).
        patch_size: Encoder patch size (16 for DINOv3).
        num_upsample_blocks: Number of upsampling stages.
        base_channels: Base channel count for decoder.
        detail_base_channels: Base channels for detail branch.
        detail_num_stages: Number of downsampling stages in detail branch.
        detail_out_channels: Output channels for detail branch.
        residual_scale: Maximum absolute residual magnitude.
        use_cross_attention: Whether to use cross-gated attention.
        stage_depths: Number of refinement blocks per stage.
        tail_blocks: Number of final refinement blocks.
        residual_gain_init: Initial gain for residual blocks.
    """

    def __init__(
        self,
        input_dim: int,
        patch_size: int = 16,
        num_upsample_blocks: int = 4,
        base_channels: int = 128,
        detail_base_channels: int = 32,
        detail_num_stages: int = 2,
        detail_out_channels: int = 64,
        residual_scale: float = 1.0,
        use_cross_attention: bool = True,
        stage_depths: Sequence[int] | None = None,
        tail_blocks: int = 3,
        residual_gain_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.residual_scale = residual_scale
        self.predict_residual = True
        self.num_upsample_blocks = num_upsample_blocks
        self.use_cross_attention = use_cross_attention

        # Token projection (semantic branch)
        self.semantic_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, base_channels),
            nn.GELU(),
        )

        # Detail branch (shallow CNN)
        self.detail_branch = ShallowCNNBranch(
            in_channels=3,
            base_channels=detail_base_channels,
            num_stages=detail_num_stages,
            out_channels=detail_out_channels,
        )

        # Channel schedule
        channels = self._build_channel_schedule(base_channels, num_upsample_blocks)
        depths = self._build_stage_depths(num_upsample_blocks, stage_depths)

        # PixelShuffle upsampling blocks
        num_pixelshuffle = num_upsample_blocks - 1
        self.pixelshuffle_blocks = nn.ModuleList()
        for i in range(num_pixelshuffle):
            self.pixelshuffle_blocks.append(
                DualBranchUpsampleBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    detail_channels=detail_out_channels,
                    num_refine_blocks=depths[i],
                    residual_gain_init=residual_gain_init,
                    use_cross_attention=use_cross_attention,
                )
            )

        # Final precise upsampling block
        self.final_block = PreciseUpsampleBlock(
            in_channels=channels[num_pixelshuffle],
            out_channels=channels[-1],
            detail_channels=detail_out_channels,
            num_refine_blocks=depths[-1] if depths else 2,
            residual_gain_init=residual_gain_init,
            use_cross_attention=use_cross_attention,
        )

        # Alignment refinement
        self.align_refine = ResidualRefineBlock(
            channels[-1],
            residual_gain_init=min(residual_gain_init, 0.05),
        )

        # Tail refinement
        self.tail_blocks = nn.ModuleList([
            ResidualRefineBlock(channels[-1], residual_gain_init=residual_gain_init)
            for _ in range(tail_blocks)
        ])

        # Output head
        self.head = nn.Sequential(
            ChanLayerNorm(channels[-1]),
            nn.Conv2d(channels[-1], channels[-1], 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels[-1], channels[-1], 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels[-1], 3, 1, bias=True),
        )

        # Zero-init output for residual learning
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _build_channel_schedule(base_channels: int, num_blocks: int) -> list[int]:
        """Build channel schedule for decoder stages."""
        channels = [base_channels]
        for i in range(1, num_blocks + 1):
            ratio = 1.0 - (0.25 * i / num_blocks)
            ch = max(round(base_channels * ratio), 48)
            channels.append(ch)
        return channels

    @staticmethod
    def _build_stage_depths(
        num_upsample_blocks: int,
        stage_depths: Sequence[int] | None,
    ) -> list[int]:
        """Build refinement block counts per stage."""
        if stage_depths is not None:
            if len(stage_depths) != num_upsample_blocks:
                raise ValueError("stage_depths length must equal num_upsample_blocks")
            return [int(v) for v in stage_depths]

        depths = [1 for _ in range(num_upsample_blocks)]
        if num_upsample_blocks >= 2:
            depths[-2] = 2
        depths[-1] = 2
        return depths

    def forward(self, x: Tensor, h: int, w: int, img: Tensor | None = None) -> Tensor:
        """Decode token features with dual branch architecture.

        Args:
            x: Token features from semantic branch (B, N, input_dim).
            h: Token grid height.
            w: Token grid width.
            img: Original input image (B, 3, H, W) for detail branch.

        Returns:
            Residual delta (B, 3, H, W) in [-residual_scale, residual_scale].
        """
        B = x.shape[0]

        if img is None:
            img = x.new_zeros(B, 3, h * self.patch_size, w * self.patch_size)
        _, _, target_h, target_w = img.shape

        # Semantic branch: project tokens to 2D feature map
        x = self.semantic_proj(x)
        x = x.transpose(1, 2).contiguous().reshape(B, -1, h, w)

        # Detail branch: extract multi-scale features from image
        detail_features = self.detail_branch(img)

        # Progressive upsampling with dual branch fusion
        # Use detail features from coarse to fine
        for i, block in enumerate(self.pixelshuffle_blocks):
            # Select appropriate detail feature scale
            detail_idx = min(len(detail_features) - 1 - i, len(detail_features) - 1)
            detail_feat = detail_features[max(0, detail_idx)]
            x = block(x, detail_feat)

        # Final upsampling with finest detail features
        x = self.final_block(x, (target_h, target_w), detail_features[0])

        # Alignment refinement
        if x.shape[-2:] != (target_h, target_w):
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        x = self.align_refine(x)

        # Tail refinement
        for block in self.tail_blocks:
            x = block(x)

        return torch.tanh(self.head(x)) * self.residual_scale


__all__ = ["CNNDecoderWithDualBranch"]

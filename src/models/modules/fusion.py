"""Feature fusion modules for multi-branch and skip connection fusion."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Literal

from src.models.decoders.cnn_decoder import ChanLayerNorm


class GatedResidualFusion(nn.Module):
    """Gated residual fusion: output = main + gate(main, branch) * branch.

    The gate learns to selectively combine features from the main branch and
    additional branches based on their joint information.

    Args:
        main_ch: Main feature channel count.
        branch_channels: List of branch feature channel counts.
        use_scale: Use learnable scale parameters (like encoder's colour_scale).
        skip_projection: If True, branch features are already projected to main_ch.
    """

    def __init__(
        self,
        main_ch: int,
        branch_channels: list[int],
        use_scale: bool = False,
        skip_projection: bool = False,
    ) -> None:
        super().__init__()
        self.main_ch = main_ch
        self.branch_channels = branch_channels
        self.use_scale = use_scale
        self.skip_projection = skip_projection

        # Branch projection to main channels (if needed)
        if not skip_projection:
            self.branch_proj = nn.ModuleList([
                nn.Conv2d(c, main_ch, 1, bias=True)
                for c in branch_channels
            ])

        # Gate projection: after all branches are projected to main_ch
        # Input is concat([main, proj_branch1, proj_branch2, ...])
        gate_in_ch = main_ch * (1 + len(branch_channels))
        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_in_ch, main_ch, 1, bias=True),
            nn.Sigmoid(),
        )

        # Learnable scales (optional)
        if use_scale:
            self.scales = nn.ParameterList([
                nn.Parameter(torch.tensor(0.1))
                for _ in branch_channels
            ])
        else:
            self.scales = None

    def forward(
        self,
        main_feat: Tensor,
        branch_feats: list[Tensor],
        target_size: tuple[int, int] | None = None,
    ) -> Tensor:
        """Gated residual fusion.

        Args:
            main_feat: Main feature (B, main_ch, H, W).
            branch_feats: List of branch features to fuse.
            target_size: Target spatial size (H, W) for upsampling.

        Returns:
            Fused features (B, main_ch, H, W).
        """
        output = main_feat

        # Project and upsample branch features
        proj_branches = []
        for i, branch in enumerate(branch_feats):
            if target_size is not None and branch.shape[-2:] != target_size:
                branch = F.interpolate(branch, size=target_size, mode="bilinear", align_corners=False)

            if not self.skip_projection:
                branch = self.branch_proj[i](branch)

            if self.scales is not None:
                branch = self.scales[i] * branch
            proj_branches.append(branch)

        # Compute gate from concatenated features (all projected to main_ch)
        gate_input = torch.cat([main_feat, *proj_branches], dim=1)
        gate = self.gate_proj(gate_input)

        # Apply gated residual: main + gate * sum(branches)
        branch_sum = sum(proj_branches)
        return output + gate * branch_sum


class AttentionGateFusion(nn.Module):
    """Attention Gate fusion for skip connections.

    Based on "Attention U-Net: Learning Where to Look for the Pancreas"
    (Oktay et al., 2018). The attention mechanism learns to weight skip
    features based on the current decoder state.

    Args:
        gate_ch: Gate signal channels (usually from decoder).
        skip_ch: Skip feature channels.
        inner_ch: Intermediate channels for attention computation.
    """

    def __init__(
        self,
        main_ch: int,
        skip_channels: list[int],
        inner_ch: int = 64,
    ) -> None:
        super().__init__()
        self.main_ch = main_ch
        self.skip_channels = skip_channels
        self.inner_ch = inner_ch

        # Attention components for each skip branch
        self.attention_modules = nn.ModuleList()
        for skip_ch in skip_channels:
            self.attention_modules.append(
                self._make_attention_block(main_ch, skip_ch, inner_ch)
            )

        # Skip projection to main channels
        self.skip_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, main_ch, 1, bias=True),
                ChanLayerNorm(main_ch),
            )
            for c in skip_channels
        ])

    def _make_attention_block(self, gate_ch: int, skip_ch: int, inner_ch: int) -> nn.Module:
        """Create attention block for one skip branch."""
        return nn.ModuleDict({
            "W_g": nn.Conv2d(gate_ch, inner_ch, 1, bias=False),  # Gate signal
            "W_s": nn.Conv2d(skip_ch, inner_ch, 1, bias=False),  # Skip signal
            "psi": nn.Sequential(
                nn.Conv2d(inner_ch, 1, 1, bias=True),
                nn.Sigmoid(),
            ),
            "relu": nn.ReLU(inplace=True),
        })

    def forward(
        self,
        main_feat: Tensor,
        skip_feats: list[Tensor],
        target_size: tuple[int, int] | None = None,
    ) -> Tensor:
        """Attention gate fusion.

        Args:
            main_feat: Main (gate) feature (B, main_ch, H, W).
            skip_feats: List of skip features.
            target_size: Target spatial size (H, W) for upsampling.

        Returns:
            Fused features (B, main_ch, H, W).
        """
        output = main_feat

        for skip_feat, attn, proj in zip(skip_feats, self.attention_modules, self.skip_proj):
            # Upsample if needed
            if target_size is not None:
                g_size = main_feat.shape[-2:]
                s_size = skip_feat.shape[-2:]
                if g_size != target_size:
                    main_resized = F.interpolate(main_feat, size=target_size, mode="bilinear", align_corners=False)
                else:
                    main_resized = main_feat
                if s_size != target_size:
                    skip_resized = F.interpolate(skip_feat, size=target_size, mode="bilinear", align_corners=False)
                else:
                    skip_resized = skip_feat
            else:
                main_resized = main_feat
                skip_resized = skip_feat

            # Compute attention weights
            # g1 = ReLU(W_g * gate + W_s * skip)
            g1 = attn["W_g"](main_resized)
            s1 = attn["W_s"](skip_resized)

            # Handle size mismatch (different spatial sizes)
            if g1.shape[-2:] != s1.shape[-2:]:
                s1 = F.interpolate(s1, size=g1.shape[-2:], mode="bilinear", align_corners=False)

            attention = attn["psi"](attn["relu"](g1 + s1))

            # Apply attention to skip and project
            skip_weighted = skip_resized * attention
            skip_proj = proj(skip_weighted)

            # Upsample projection if needed
            if skip_proj.shape[-2:] != output.shape[-2:]:
                skip_proj = F.interpolate(skip_proj, size=output.shape[-2:], mode="bilinear", align_corners=False)

            output = output + skip_proj

        return output


class MultiScaleFusionBlock(nn.Module):
    """Unified fusion block supporting multiple fusion modes.

    Args:
        main_ch: Main feature channel count.
        skip_channels: List of skip feature channel counts.
        mode: Fusion mode - "concat", "add", "gate", "gated_residual", "attention_gate".
    """

    def __init__(
        self,
        main_ch: int,
        skip_channels: list[int],
        mode: Literal["concat", "add", "gate", "gated_residual", "attention_gate"] = "gate",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.skip_channels = skip_channels
        self.main_ch = main_ch

        if mode == "concat":
            total_ch = main_ch + sum(skip_channels)
            self.fusion = nn.Sequential(
                ChanLayerNorm(total_ch),
                nn.Conv2d(total_ch, main_ch, 1, bias=True),
                nn.GELU(),
            )
        elif mode == "add":
            self.proj_skip = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(c, main_ch, 1, bias=True),
                    ChanLayerNorm(main_ch),
                )
                for c in skip_channels
            ])
        elif mode == "gate":
            self.proj_skip = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(c, main_ch, 1, bias=True),
                    ChanLayerNorm(main_ch),
                    nn.GELU(),
                )
                for c in skip_channels
            ])
            gate_ch = main_ch * (1 + len(skip_channels))
            self.gate = nn.Sequential(
                nn.Conv2d(gate_ch, main_ch, 1, bias=True),
                nn.Sigmoid(),
            )
        elif mode == "gated_residual":
            self.fusion = GatedResidualFusion(
                main_ch=main_ch,
                branch_channels=skip_channels,
                use_scale=True,
                skip_projection=False,  # GatedResidualFusion will handle projection
            )
        elif mode == "attention_gate":
            self.fusion = AttentionGateFusion(
                main_ch=main_ch,
                skip_channels=skip_channels,
                inner_ch=max(main_ch // 4, 32),
            )
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")

    def forward(
        self,
        main_feat: Tensor,
        skip_feats: list[Tensor],
        target_size: tuple[int, int] | None = None,
    ) -> Tensor:
        """Fuse main features with skip features.

        Args:
            main_feat: Main decoder features (B, main_ch, H, W).
            skip_feats: List of skip features to fuse.
            target_size: Target spatial size (H, W).

        Returns:
            Fused features (B, main_ch, H, W).
        """
        if self.mode == "concat":
            skip_upsampled = [
                F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
                if f.shape[-2:] != target_size else f
                for f in skip_feats
            ]
            fused = torch.cat([main_feat, *skip_upsampled], dim=1)
            return self.fusion(fused)

        elif self.mode == "add":
            output = main_feat
            for skip, proj in zip(skip_feats, self.proj_skip):
                skip_up = F.interpolate(skip, size=target_size, mode="bilinear", align_corners=False) \
                    if skip.shape[-2:] != target_size else skip
                output = output + proj(skip_up)
            return output

        elif self.mode == "gate":
            skip_proj_list = []
            for skip, proj in zip(skip_feats, self.proj_skip):
                skip_up = F.interpolate(skip, size=target_size, mode="bilinear", align_corners=False) \
                    if skip.shape[-2:] != target_size else skip
                skip_proj_list.append(proj(skip_up))

            gate_input = torch.cat([main_feat, *skip_proj_list], dim=1)
            gate = self.gate(gate_input)

            skip_sum = sum(skip_proj_list)
            return main_feat + gate * skip_sum

        elif self.mode in ("gated_residual", "attention_gate"):
            return self.fusion(main_feat, skip_feats, target_size)

        return main_feat

"""Frozen DINOv3 ViT-B encoder wrapper with PEFT LoRA adaptation."""
from __future__ import annotations

from typing import Optional, Literal
import math

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

try:
    from transformers import AutoModel
except ModuleNotFoundError:
    AutoModel = None  # type: ignore[assignment]

try:
    from peft import LoraConfig, inject_adapter_in_model
except ModuleNotFoundError:
    LoraConfig = None  # type: ignore[assignment]
    inject_adapter_in_model = None  # type: ignore[assignment]

from src.models.encoders.base import BaseEncoder
from src.models.modules.fusion import MultiScaleFusionBlock

class GhostDWBlock(nn.Module):
    """Ghost convolution block following the official GhostNet implementation."""

    def __init__(self, in_channels: int, out_channels: int, ratio: int = 2, dw_size: int = 3) -> None:
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, init_channels),
            nn.GELU(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, kernel_size=dw_size,
                      padding=dw_size // 2, groups=init_channels, bias=False),
            nn.GroupNorm(8, new_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)[:, :self.out_channels, :, :]

class ResidualChannelSpatialGate(nn.Module):
    """Residual channel-spatial attention for fused encoder features."""

    def __init__(self, dim: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_dim = max(dim // reduction, 16)

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        channel_weight = self.channel_gate(x)

        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        spatial_weight = self.spatial_gate(torch.cat([avg_map, max_map], dim=1))

        return x + x * channel_weight * spatial_weight

class ColourContextBranch(nn.Module):
    """Extracts global colour statistics for aesthetic enhancement conditioning.

    Encodes luminance, colour opponency, saturation, and cross-channel correlations.

    Input:  6-channel colour representation (no learned params in preprocessing)
    Output: (B, embed_dim, H, W)

    Args:
        embed_dim: Output embedding dimension.
        return_intermediate: Return intermediate layer outputs for skip connections.
    """

    def __init__(self, embed_dim: int, return_intermediate: bool = False) -> None:
        super().__init__()
        self.return_intermediate = return_intermediate

        self.local_colour = nn.ModuleDict({
            "layer0": GhostDWBlock(6, 32),
            "layer1": GhostDWBlock(32, 64),
            "layer2": nn.Conv2d(64, embed_dim, kernel_size=1),  # project late
        })

        self.global_scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, embed_dim // 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 8, embed_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        x0 = self.local_colour["layer0"](x)   # (B, 32, H, W)
        x1 = self.local_colour["layer1"](x0)  # (B, 64, H, W)
        x2 = self.local_colour["layer2"](x1)  # (B, D, H, W)

        local_feats = x2
        global_scale = self.global_scale(local_feats)  # (B, D, 1, 1)
        output = local_feats * global_scale

        if self.return_intermediate:
            intermediates = {
                "layer0": x0,  # (B, 32, H, W)
                "layer1": x1,  # (B, 64, H, W)
            }
            return output, intermediates
        return output

class DINOv3LoRAEncoder(BaseEncoder):
    """DINOv3 ViT-B encoder with PEFT LoRA adapters.

    The backbone weights are fully frozen. Only LoRA A/B matrices are trainable.

    Args:
        model_name:         HuggingFace model ID.
        frozen:             Freeze backbone weights (keep True).
        patch_size:         Patch size of the backbone (16 for ViT-B).
        embed_dim:          Output feature dimension (768 for ViT-B).
        lora_rank:          Intrinsic rank r for every LoRA matrix pair.
        lora_alpha:         LoRA scaling factor.
        lora_dropout:       Dropout probability on the low-rank branch.
        enable_dino:        Enable DINO backbone features.
        enable_local:       Enable local conv branch features.
        enable_colour:      Enable colour context branch features.
        fusion_mode:        Fusion mode for multi-branch combination.
                            "gate": Original gated fusion (default).
                            "concat": Simple concatenation + projection.
                            "add": Direct addition after projection.
                            "gated_residual": Gated residual fusion.
                            "attention_gate": Attention gate fusion.
        skip_enabled:       Enable skip connections (return intermediate features).
        return_skip:        Return skip features even when skip_enabled is False.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        frozen: bool = True,
        patch_size: int = 16,
        embed_dim: int = 768,
        lora_rank: int = 4,
        lora_alpha: Optional[float] = None,
        lora_dropout: float = 0.0,
        enable_dino: bool = True,
        enable_local: bool = True,
        enable_colour: bool = True,
        fusion_mode: Literal["gate", "concat", "add", "gated_residual", "attention_gate"] = "gate",
        skip_enabled: bool = True,
        return_skip: bool = True,
    ) -> None:
        super().__init__()
        if AutoModel is None or LoraConfig is None or inject_adapter_in_model is None:
            raise ModuleNotFoundError(
                "DINOv3LoRAEncoder requires `transformers` and `peft` to be installed."
            )

        self._patch_size = patch_size
        self._embed_dim = embed_dim
        self.skip_enabled = skip_enabled
        self._return_skip = return_skip

        # Branch enable flags
        self.enable_dino = enable_dino
        self.enable_local = enable_local
        self.enable_colour = enable_colour
        self.fusion_mode = fusion_mode

        lora_alpha = lora_alpha if lora_alpha is not None else float(lora_rank)

        self.backbone = AutoModel.from_pretrained(model_name)
        self.num_register_tokens = self.backbone.config.num_register_tokens

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )

        self.backbone = inject_adapter_in_model(lora_config, self.backbone)

        # Freeze all backbone parameters except LoRA matrices
        for name, param in self.backbone.named_parameters():
            param.requires_grad = "lora" in name

        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        # ===== [ADD] Local image branch for color, contrast, and local structure =====
        self.local_conv = nn.ModuleDict({
            "layer0": GhostDWBlock(3, 64),
            "layer1": GhostDWBlock(64, 128),
            "layer2": nn.Conv2d(128, self._embed_dim, kernel_size=1),
        })

        # ===== [ADD] Project DINO features before fusion =====
        self.dino_proj = nn.Sequential(
            nn.Conv2d(self._embed_dim, self._embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(16, self._embed_dim),
            nn.GELU(),
        )

        # ===== [FUSION] Branch fusion modules =====
        # Determine which branches are enabled for fusion
        self._enabled_branch_ch = []
        if enable_local:
            self._enabled_branch_ch.append(self._embed_dim)  # local_feat
        if enable_colour:
            self._enabled_branch_ch.append(self._embed_dim)  # colour_feat
        if enable_dino:
            self._dino_ch_for_fusion = self._embed_dim
        else:
            self._dino_ch_for_fusion = None

        # Create fusion modules based on mode
        if fusion_mode in ("gated_residual", "attention_gate"):
            # Use new MultiScaleFusionBlock
            if enable_dino:
                # DINO is main branch, local/colour are skip branches
                self.branch_fusion = MultiScaleFusionBlock(
                    main_ch=self._embed_dim,
                    skip_channels=self._enabled_branch_ch,
                    mode=fusion_mode,
                )
            else:
                # No DINO, first branch is main, rest are skips
                if len(self._enabled_branch_ch) > 1:
                    self.branch_fusion = MultiScaleFusionBlock(
                        main_ch=self._enabled_branch_ch[0],
                        skip_channels=self._enabled_branch_ch[1:],
                        mode=fusion_mode,
                    )
                else:
                    self.branch_fusion = None
            self.fusion = None  # Not needed for new modes
            self.fusion_gate = None
        else:
            # Use original fusion modules for backward compatibility
            self.branch_fusion = None

            # Original gated fusion (for "gate" mode)
            self.fusion_gate = nn.Sequential(
                nn.Conv2d(self._embed_dim * 3, self._embed_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(self._embed_dim, self._embed_dim * 2, kernel_size=1),
                nn.Sigmoid(),
            )

            # Learnable residual scales
            self.colour_scale = nn.Parameter(torch.tensor(0.1))
            self.local_scale = nn.Parameter(torch.tensor(0.1))

            # Full fusion module (for all modes)
            if fusion_mode == "add":
                # Simple projection for add mode
                self.fusion = nn.Conv2d(self._embed_dim * 3, self._embed_dim, kernel_size=1, bias=False)
            else:
                self.fusion = nn.Sequential(
                    nn.Conv2d(self._embed_dim * 3, self._embed_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(16, self._embed_dim),
                    nn.GELU(),
                    GhostDWBlock(self._embed_dim, self._embed_dim),
                )

        # ===== [ADD] Colour context branch =====
        self.colour_branch = ColourContextBranch(
            self._embed_dim,
            return_intermediate=self.skip_enabled or self._return_skip
        )

        # ===== [ADD] Attention reweighting after fusion =====
        self.feature_attention = ResidualChannelSpatialGate(self._embed_dim)

        if frozen:
            self.freeze()

    def freeze(self) -> None:
        # First, freeze everything
        for param in self.parameters():
            param.requires_grad_(False)

        # Then selectively unfreeze what should train
        for name, param in self.backbone.named_parameters():
            if "lora" in name.lower():
                param.requires_grad_(True)

        # [ADD] Keep newly added encoder adaptation modules trainable.
        modules_to_train = [
            self.colour_branch,
            *self.local_conv.values(),
            self.dino_proj,
            self.feature_attention,
        ]
        # Add fusion modules if they exist (not None)
        if hasattr(self, 'fusion_gate') and self.fusion_gate is not None:
            modules_to_train.append(self.fusion_gate)
        if hasattr(self, 'fusion') and self.fusion is not None:
            modules_to_train.append(self.fusion)
        if hasattr(self, 'branch_fusion') and self.branch_fusion is not None:
            modules_to_train.append(self.branch_fusion)

        for module in modules_to_train:
            for param in module.parameters():
                param.requires_grad = True

        # [ADD] Keep learnable fusion scales trainable (if they exist).
        if hasattr(self, 'colour_scale'):
            self.colour_scale.requires_grad = True
        if hasattr(self, 'local_scale'):
            self.local_scale.requires_grad = True

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, dict]:
        """Extract patch token features (no CLS or register tokens).

        Args:
            x: Input image (B, 3, H, W), values in [0, 1].

        Returns:
            If skip_enabled and return_skip:
                (fused_tokens, skip_features) where skip_features is a dict.
            Otherwise:
                fused_tokens (B, N, C).
        """
        x = x.clamp(0.0, 1.0)
        mean = self.pixel_mean.to(device=x.device, dtype=x.dtype)
        std = self.pixel_std.to(device=x.device, dtype=x.dtype)
        x_norm = (x - mean) / std

        B, C, H, W = x.shape
        h = H // self._patch_size
        w = W // self._patch_size

        # ===== [DINO] Extract DINO features if enabled =====
        dino_feat = None
        if self.enable_dino:
            outputs = self.backbone(pixel_values=x_norm)
            R = self.num_register_tokens
            dino_tokens = outputs.last_hidden_state[:, 1 + R:, :]
            dino_feat = dino_tokens.transpose(1, 2).reshape(B, self._embed_dim, h, w)
            dino_feat = self.dino_proj(dino_feat)

        # ===== [PREP] Prepare low-resolution image for CNN branches =====
        x_small = F.interpolate(x_norm, size=(h, w), mode="bilinear", align_corners=False)

        # ===== [COLOUR] Extract colour features if enabled =====
        colour_feat = None
        colour_intermediates = {}
        if self.enable_colour:
            # Un-normalize for colour operations
            x_raw = x_norm * self.pixel_std.to(x_norm) + self.pixel_mean.to(x_norm)
            x_raw = x_raw.clamp(0.0, 1.0)
            r, g, b = x_raw[:, 0:1], x_raw[:, 1:2], x_raw[:, 2:3]
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            rg = r - g
            yb = 0.5 * (r + g) - b
            max_c = x_raw.max(dim=1, keepdim=True).values
            min_c = x_raw.min(dim=1, keepdim=True).values
            sat = (max_c - min_c) / (max_c + 1e-6)
            hue_proxy = rg / (x_raw.norm(dim=1, keepdim=True) + 1e-6)
            colour_input = torch.cat([lum, rg, yb, sat, hue_proxy, max_c - min_c], dim=1)
            colour_input = F.interpolate(colour_input, size=(h, w), mode="bilinear", align_corners=False)

            colour_result = self.colour_branch(colour_input)
            if self.skip_enabled or self._return_skip:
                colour_feat, colour_intermediates = colour_result
            else:
                colour_feat = colour_result

        # ===== [LOCAL] Extract local conv features if enabled =====
        local_feat0, local_feat1, local_feat = None, None, None
        if self.enable_local:
            local_feat0 = self.local_conv["layer0"](x_small)  # (B, 64, h, w)
            local_feat1 = self.local_conv["layer1"](local_feat0)  # (B, 128, h, w)
            local_feat = self.local_conv["layer2"](local_feat1)  # (B, 768, h, w)

        # ===== [FUSION] Combine enabled branches =====
        # Collect enabled features
        feat_dict = {}
        if self.enable_dino and dino_feat is not None:
            feat_dict["dino"] = dino_feat
        if self.enable_local and local_feat is not None:
            feat_dict["local"] = local_feat
        if self.enable_colour and colour_feat is not None:
            feat_dict["colour"] = colour_feat

        if len(feat_dict) == 0:
            raise ValueError("At least one branch must be enabled")

        # Single branch case
        if len(feat_dict) == 1:
            fused_feat = list(feat_dict.values())[0]
        # Multi-branch fusion
        elif self.fusion_mode in ("gated_residual", "attention_gate"):
            # Use new MultiScaleFusionBlock
            if self.enable_dino and self.branch_fusion is not None:
                # DINO is main branch
                skip_feats = []
                if self.enable_local:
                    skip_feats.append(local_feat)
                if self.enable_colour:
                    skip_feats.append(colour_feat)
                fused_feat = self.branch_fusion(dino_feat, skip_feats, (h, w))
            elif self.branch_fusion is not None:
                # No DINO, first enabled branch is main
                feat_list = list(feat_dict.values())
                fused_feat = self.branch_fusion(feat_list[0], feat_list[1:], (h, w))
            else:
                # Fallback to concat
                fused_feat = torch.cat(list(feat_dict.values()), dim=1)
        elif self.fusion_mode == "gate":
            # Original gated fusion
            if self.enable_dino:
                fusion_input = torch.cat([
                    dino_feat if self.enable_dino else torch.zeros_like(dino_feat),
                    colour_feat if self.enable_colour else torch.zeros_like(dino_feat),
                    local_feat if self.enable_local else torch.zeros_like(dino_feat),
                ], dim=1)
                freq_gate, local_gate = self.fusion_gate(fusion_input).chunk(2, dim=1)

                enhanced_feat = dino_feat
                if self.enable_colour:
                    enhanced_feat = enhanced_feat + self.colour_scale * freq_gate * colour_feat
                if self.enable_local:
                    enhanced_feat = enhanced_feat + self.local_scale * local_gate * local_feat

                # Full fusion
                fusion_cat = [enhanced_feat]
                if self.enable_colour:
                    fusion_cat.append(colour_feat)
                if self.enable_local:
                    fusion_cat.append(local_feat)
                fused_feat = torch.cat(fusion_cat, dim=1)
                fused_feat = self.fusion(fused_feat)
                fused_feat = fused_feat + dino_feat
            else:
                # DINO disabled, use concat fallback
                fused_feat = torch.cat(list(feat_dict.values()), dim=1)
                if self.fusion is not None:
                    fused_feat = self.fusion(fused_feat)
        elif self.fusion_mode == "add":
            # Direct addition after projection
            # Simple version: just concat then use fusion to project
            fused_feat = torch.cat(list(feat_dict.values()), dim=1)
            if self.fusion is not None:
                fused_feat = self.fusion(fused_feat)
        else:  # concat
            # Simple concat fusion
            fused_feat = torch.cat(list(feat_dict.values()), dim=1)
            if self.fusion is not None:
                fused_feat = self.fusion(fused_feat)

        # Apply attention reweighting (only if feature has expected channels)
        if fused_feat.shape[1] == self._embed_dim:
            fused_feat = self.feature_attention(fused_feat)

        # ===== [OUTPUT] Convert back to token format =====
        fused_tokens = fused_feat.flatten(2).transpose(1, 2)  # (B, N, C)

        # ===== [SKIP] Collect skip features for decoder =====
        # Only return skip_features if skip_enabled is True
        if self.skip_enabled and (self.enable_local or self.enable_colour):
            skip_features = {
                "local": {},
                "colour": {},
                "raw_input": {},
            }
            if self.enable_local:
                skip_features["local"] = {
                    "layer0": local_feat0,  # (B, 64, h, w)
                    "layer1": local_feat1,  # (B, 128, h, w)
                }
                skip_features["raw_input"]["local"] = x_small
            if self.enable_colour:
                skip_features["colour"] = {
                    "layer0": colour_intermediates.get("layer0"),  # (B, 32, h, w)
                    "layer1": colour_intermediates.get("layer1"),  # (B, 64, h, w)
                }
                # Re-compute colour_input for raw_input skip
                if self._return_skip and colour_feat is not None:
                    x_raw = x_norm * self.pixel_std.to(x_norm) + self.pixel_mean.to(x_norm)
                    x_raw = x_raw.clamp(0.0, 1.0)
                    r, g, b = x_raw[:, 0:1], x_raw[:, 1:2], x_raw[:, 2:3]
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    rg = r - g
                    yb = 0.5 * (r + g) - b
                    max_c = x_raw.max(dim=1, keepdim=True).values
                    min_c = x_raw.min(dim=1, keepdim=True).values
                    sat = (max_c - min_c) / (max_c + 1e-6)
                    hue_proxy = rg / (x_raw.norm(dim=1, keepdim=True) + 1e-6)
                    colour_input = torch.cat([lum, rg, yb, sat, hue_proxy, max_c - min_c], dim=1)
                    colour_input = F.interpolate(colour_input, size=(h, w), mode="bilinear", align_corners=False)
                    skip_features["raw_input"]["colour"] = colour_input

            return fused_tokens, skip_features

        return fused_tokens

    def train(self, mode: bool = True) -> "DINOv3LoRAEncoder":
        super(BaseEncoder, self).train(mode)
        self.backbone.train(False)
        return self

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size

    def __repr__(self) -> str:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"{self.__class__.__name__}("
            f"embed_dim={self._embed_dim}, patch_size={self._patch_size}, "
            f"params={total:,}, lora_trainable={trainable:,} "
            f"[{100 * trainable / max(total, 1):.2f}%])"
        )

"""ImageEnhancementPipeline: assembles Encoder → Bottleneck → Decoder."""
from __future__ import annotations

import inspect
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.encoders.base import BaseEncoder
from src.models.bottlenecks.base import BaseBottleneck
from src.models.decoders.base import BaseDecoder
from src.models.modules.fusion import MultiScaleFusionBlock


class ImageEnhancementPipeline(nn.Module):
    """Full image color enhancement pipeline.

    Connects components in order:
        [Frozen DINOv3 Encoder] ────┐
        [ColorNAF Encoder] ─────────┼─→ [Fusion] → [RWKV7 Bottleneck] → [Decoder]
        [Histogram Token] ───────────┘                    ↑
                                                        Skip Connections

    The main encoder runs under torch.no_grad() and stays frozen. An optional
    CNN encoder (for example ColorNAF) may remain partially trainable
    according to its own ``freeze()`` policy.

    Skip Connections:
        - DINOv3: Optional intermediate layer features (semantic skips)
        - ColorNAF: Multi-scale stage outputs (texture/color skips)

    Args:
        encoder: Pretrained, frozen encoder (e.g., DINOv3). Must implement BaseEncoder.
        bottleneck: Trainable context-modeling bottleneck. Must implement BaseBottleneck.
        decoder: Trainable image reconstruction decoder. Must implement BaseDecoder.
        cnn_encoder: Optional second encoder for local texture features (e.g., ColorNAF).
        encoder_skip_layers: Optional list of layer indices for DINOv3 skip connections.
        fusion_mode: How to fuse DINOv3 and ColorNAF tokens.
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        bottleneck: BaseBottleneck,
        decoder: BaseDecoder,
        cnn_encoder: BaseEncoder | None = None,
        encoder_skip_layers: list[int] | None = None,
        fusion_mode: str = "concat",
        main_encoder_proj_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.cnn_encoder = cnn_encoder
        self.encoder_skip_layers = encoder_skip_layers or []
        self.fusion_mode = fusion_mode
        self.main_encoder_proj_dim = main_encoder_proj_dim
        bottleneck_sig = inspect.signature(self.bottleneck.forward)
        self._bottleneck_accepts_resolution = "resolution" in bottleneck_sig.parameters

        # Apply each encoder's freeze policy at construction time.
        self.encoder.freeze()
        if self.cnn_encoder is not None:
            self.cnn_encoder.freeze()

        self.cnn_proj = None
        self.cnn_fusion_proj = None
        self.cnn_fusion_block = None   # MultiScaleFusionBlock for gate/gated_residual/attention_gate
        self.dino_skip_proj = None

        # Pre-initialize so the projection is registered before configure_optimizers runs.
        # Lazy init via _get_or_init_linear would miss the optimizer build step in Lightning.
        if main_encoder_proj_dim is not None and encoder.embed_dim != main_encoder_proj_dim:
            self.main_encoder_proj: nn.Linear | None = nn.Linear(encoder.embed_dim, main_encoder_proj_dim)
        else:
            self.main_encoder_proj = None

    def _get_or_init_linear(
        self,
        attr_name: str,
        in_features: int,
        out_features: int,
        device: torch.device,
    ) -> nn.Linear:
        module = getattr(self, attr_name)
        if (
            module is None
            or module.in_features != in_features
            or module.out_features != out_features
        ):
            module = nn.Linear(in_features, out_features).to(device)
            setattr(self, attr_name, module)
        return module

    def _align_cnn_tokens(
        self,
        cnn_tokens: Tensor,
        *,
        input_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor | None]:
        """Align CNN tokens to the main encoder grid and split global tokens."""
        if self.cnn_encoder is None:
            return cnn_tokens, None

        H, W = input_size
        target_h, target_w = target_size
        cnn_patch = self.cnn_encoder.patch_size
        cnn_h = H // cnn_patch
        cnn_w = W // cnn_patch
        expected_spatial = cnn_h * cnn_w

        hist_token = None
        if cnn_tokens.shape[1] == expected_spatial + 1:
            hist_token = cnn_tokens[:, :1, :]
            cnn_tokens = cnn_tokens[:, 1:, :]
        elif cnn_tokens.shape[1] != expected_spatial:
            raise ValueError(
                "CNN encoder token count does not match its spatial grid: "
                f"expected {expected_spatial} or {expected_spatial + 1}, "
                f"got {cnn_tokens.shape[1]}"
            )

        B, _, C = cnn_tokens.shape
        cnn_feat = cnn_tokens.transpose(1, 2).reshape(B, C, cnn_h, cnn_w)
        if (cnn_h, cnn_w) != (target_h, target_w):
            cnn_feat = F.interpolate(
                cnn_feat,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
        cnn_tokens = cnn_feat.flatten(2).transpose(1, 2)
        return cnn_tokens, hist_token

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _project_main_encoder_tokens(self, features: Tensor) -> Tensor:
        """Optionally project the frozen main encoder to a fusion-friendly width."""
        if self.main_encoder_proj is None:
            return features
        return self.main_encoder_proj(features)

    def forward(self, x: Tensor) -> Tensor:
        """Enhance input image.

        Args:
            x: Input image of shape (B, 3, H, W), values in [0, 1].

        Returns:
            Enhanced image of shape (B, 3, H, W), values in [0, 1].
        """
        _, _, H, W = x.shape

        # Spatial grid dimensions used by the decoder
        h = H // self.encoder.patch_size
        w = W // self.encoder.patch_size

        # 如果 encoder 里面有可训练参数，例如 LoRA，则不能用 torch.no_grad()
        encoder_trainable = any(p.requires_grad for p in self.encoder.parameters())

        if self.training and encoder_trainable:
            if self.encoder_skip_layers:
                features, dino_skips = self.encoder(
                    x,
                    return_layer_indices=self.encoder_skip_layers,
                )
                skip_dict = None
            elif hasattr(self.encoder, "skip_enabled") and self.encoder.skip_enabled:
                features, skip_dict = self.encoder(x)
                dino_skips = None
            else:
                features = self.encoder(x)
                dino_skips = None
                skip_dict = None
        else:
            with torch.no_grad():
                if self.encoder_skip_layers:
                    features, dino_skips = self.encoder(
                        x,
                        return_layer_indices=self.encoder_skip_layers,
                    )
                    skip_dict = None
                elif hasattr(self.encoder, "skip_enabled") and self.encoder.skip_enabled:
                    features, skip_dict = self.encoder(x)
                    dino_skips = None
                else:
                    features = self.encoder(x)
                    dino_skips = None
                    skip_dict = None
        features = self._project_main_encoder_tokens(features)

        # ── CNN/ColorNAF Encoder ──
        color_naf_tokens = None
        color_naf_skips = None
        saved_hist_token: Tensor | None = None

        if self.cnn_encoder is not None:
            # Check if cnn_encoder returns tuple (tokens, skip_dict)
            cnn_output = self.cnn_encoder(x)

            if isinstance(cnn_output, tuple):
                # ColorNAF-style encoder: returns (tokens, skip_dict)
                cnn_tokens, cnn_skip_dict = cnn_output
                color_naf_tokens = cnn_tokens
                color_naf_skips = cnn_skip_dict
            else:
                # Legacy encoder: returns list of features
                cnn_features = cnn_output
                cnn_feat = cnn_features[-1]

                B, C, Hc, Wc = cnn_feat.shape
                cnn_token = cnn_feat.flatten(2).transpose(1, 2)

                B, Nd, Cd = features.shape

                if cnn_token.shape[1] != Nd:
                    cnn_feat_resized = F.interpolate(
                        cnn_feat,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    cnn_token = cnn_feat_resized.flatten(2).transpose(1, 2)

                if C != Cd:
                    if self.cnn_proj is None:
                        self.cnn_proj = nn.Linear(C, Cd).to(cnn_token.device)
                    cnn_token = self.cnn_proj(cnn_token)

                features = features + 0.1 * cnn_token

        # ── Fusion: DINOv3 + ColorNAF tokens ──
        if color_naf_tokens is not None:
            color_naf_tokens, hist_token = self._align_cnn_tokens(
                color_naf_tokens,
                input_size=(H, W),
                target_size=(h, w),
            )
            if hist_token is not None:
                saved_hist_token = hist_token.squeeze(1)   # (B, 1, C) → (B, C)
                color_naf_tokens = color_naf_tokens + hist_token

            if self.fusion_mode == "concat":
                # Concatenate features per aligned spatial token, then project
                # back to the main encoder dimension expected by the bottleneck.
                dino_dim = features.shape[-1]
                naf_dim = color_naf_tokens.shape[-1]
                fusion_proj = self._get_or_init_linear(
                    "cnn_fusion_proj",
                    dino_dim + naf_dim,
                    dino_dim,
                    features.device,
                )
                features = fusion_proj(torch.cat([features, color_naf_tokens], dim=-1))

            elif self.fusion_mode == "add":
                # Add after projection
                naf_dim = color_naf_tokens.shape[-1]
                dino_dim = features.shape[-1]

                if naf_dim != dino_dim:
                    proj = self._get_or_init_linear(
                        "cnn_proj",
                        naf_dim,
                        dino_dim,
                        features.device,
                    )
                    color_naf_tokens = proj(color_naf_tokens)
                features = features + color_naf_tokens

            elif self.fusion_mode in ("gate", "gated_residual", "attention_gate"):
                # Reuse MultiScaleFusionBlock from src.models.modules.fusion.
                # Tokens (B, N, C) are reshaped to 2D feature maps (B, C, h, w)
                # so the existing Conv2d-based fusion blocks apply directly.
                dino_dim = features.shape[-1]
                naf_dim = color_naf_tokens.shape[-1]
                if self.cnn_fusion_block is None or \
                        self.cnn_fusion_block.main_ch != dino_dim or \
                        self.cnn_fusion_block.skip_channels != [naf_dim]:
                    self.cnn_fusion_block = MultiScaleFusionBlock(
                        main_ch=dino_dim,
                        skip_channels=[naf_dim],
                        mode=self.fusion_mode,
                    ).to(features.device)
                # (B, N, C) → (B, C, h, w)
                main_2d = features.transpose(1, 2).reshape(-1, dino_dim, h, w)
                naf_2d = color_naf_tokens.transpose(1, 2).reshape(-1, naf_dim, h, w)
                fused_2d = self.cnn_fusion_block(main_2d, [naf_2d], target_size=(h, w))
                # (B, C, h, w) → (B, N, C)
                features = fused_2d.flatten(2).transpose(1, 2)

            else:
                raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")

        # ── Combine skip dictionaries ──
        combined_skip_dict = {}
        if skip_dict is not None:
            combined_skip_dict.update(skip_dict)
        if color_naf_skips is not None:
            combined_skip_dict.update(color_naf_skips)

        # Add DINOv3 skips if available
        if dino_skips is not None and self.encoder_skip_layers:
            # Convert list of tensors to dict format
            dino_skip_dict = {}
            for i, skip_feat in enumerate(dino_skips):
                layer_idx = self.encoder_skip_layers[i]
                B, N, C = skip_feat.shape
                # Reshape tokens to 2D feature map
                skip_2d = skip_feat.transpose(1, 2).reshape(B, C, h, w)
                dino_skip_dict[f"layer{layer_idx}"] = skip_2d
            combined_skip_dict["dino"] = dino_skip_dict

        # ── Bottleneck (trainable) ──
        bottleneck_kwargs = {}
        if self._bottleneck_accepts_resolution:
            bottleneck_kwargs["resolution"] = (h, w)
        features = self.bottleneck(features, **bottleneck_kwargs)    # (B, N, C_bot)

        # ── Decoder (trainable) ──
        sig = inspect.signature(self.decoder.forward)
        decoder_kwargs = {}
        if "img" in sig.parameters:
            decoder_kwargs["img"] = x
        if "vit_skips" in sig.parameters and dino_skips is not None:
            decoder_kwargs.update({"vit_skips": dino_skips, "vit_h": h, "vit_w": w})
        if "skip_dict" in sig.parameters and combined_skip_dict:
            decoder_kwargs["skip_dict"] = combined_skip_dict
        if "cnn_features" in sig.parameters and color_naf_skips is not None:
            decoder_kwargs["cnn_features"] = color_naf_skips
        if "hist_token" in sig.parameters:
            decoder_kwargs["hist_token"] = saved_hist_token
        output = self.decoder(features, h, w, **decoder_kwargs)

        if output.shape[-2:] != (H, W):
            output = F.interpolate(output, size=(H, W), mode="bilinear", align_corners=False)

        # Decoders may emit either a residual delta (default) or a full image.
        # Detected via the `predict_residual` flag, defaulting to True for
        # backward compatibility with the existing CNN decoders.
        if getattr(self.decoder, "predict_residual", True):
            return (x + output).clamp(0.0, 1.0)
        return output.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield all parameters that should receive optimizer updates.

        The main encoder is frozen and must never appear in the optimizer.
        cnn_encoder (e.g., ColorNAF) can be semi-frozen with trainable stages.

        Use this method to build the optimizer parameter group:

            optimizer = AdamW(pipeline.trainable_parameters(), lr=1e-4)
        """
        for param in self.encoder.parameters():
            if param.requires_grad:
                yield param

        if self.cnn_encoder is not None:
            for param in self.cnn_encoder.parameters():
                if param.requires_grad:
                    yield param

        if self.cnn_proj is not None:
            for param in self.cnn_proj.parameters():
                if param.requires_grad:
                    yield param

        if self.cnn_fusion_proj is not None:
            for param in self.cnn_fusion_proj.parameters():
                if param.requires_grad:
                    yield param

        if self.cnn_fusion_block is not None:
            for param in self.cnn_fusion_block.parameters():
                if param.requires_grad:
                    yield param

        if self.dino_skip_proj is not None:
            for param in self.dino_skip_proj.parameters():
                if param.requires_grad:
                    yield param

        if self.main_encoder_proj is not None:
            for param in self.main_encoder_proj.parameters():
                if param.requires_grad:
                    yield param

        for param in self.bottleneck.parameters():
            if param.requires_grad:
                yield param

        for param in self.decoder.parameters():
            if param.requires_grad:
                yield param

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts for each component.

        Returns:
            Dict with keys 'encoder', 'cnn_encoder', 'bottleneck', 'decoder', 'trainable'.
        """
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        result = {
            "encoder":    _count(self.encoder),
            "bottleneck": _count(self.bottleneck),
            "decoder":    _count(self.decoder),
            "trainable":  trainable,
        }
        if self.cnn_encoder is not None:
            result["cnn_encoder"] = _count(self.cnn_encoder)
        return result

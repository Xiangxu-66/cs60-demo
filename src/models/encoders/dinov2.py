"""Frozen DINOv2 encoder wrapper (E-2, default encoder)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.encoders.base import BaseEncoder


class DINOv2Encoder(BaseEncoder):
    """Frozen DINOv2 ViT encoder.

    Loads a pretrained DINOv2 model via timm and extracts patch token
    features (CLS token is discarded). Always kept in eval mode.

    Args:
        name: timm model name, e.g. 'vit_base_patch14_dinov2'.
        pretrained: Whether to load pretrained ImageNet weights.
        frozen: Whether to freeze all parameters (should always be True).
        patch_size: Patch size of the ViT backbone (14 for DINOv2-B).
        embed_dim: Output feature dimension (768 for DINOv2-B).
    """

    def __init__(
        self,
        name: str = "vit_base_patch14_dinov2",
        pretrained: bool = True,
        frozen: bool = True,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self._patch_size = patch_size
        self._embed_dim = embed_dim

        self.backbone = self._load_backbone(name, pretrained, img_size)

        # [MOD] DINOv2 expects ImageNet-style normalized inputs.
        # Keep buffers non-persistent to avoid polluting checkpoints.
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        
        if frozen:
            self.freeze()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_backbone(name: str, pretrained: bool, img_size: int) -> nn.Module:
        """Load DINOv2 backbone, trying timm first then torch.hub."""
        timm_name = {
            "dinov2_vitb14": "vit_base_patch14_dinov2",
            "dinov2_vitl14": "vit_large_patch14_dinov2",
            "dinov2_vitg14": "vit_giant_patch14_dinov2",
            "dinov2_vits14": "vit_small_patch14_dinov2",
        }.get(name, name)

        try:
            import timm
            # timm >= 0.9 ships dinov2 models
            model = timm.create_model(
                timm_name,
                pretrained=pretrained,
                num_classes=0,
                img_size=img_size,
                dynamic_img_size=True,
            )
            return model
        except Exception:
            pass

        # Fallback: Facebook's official torch.hub release
        hub_name = {
            "vit_base_patch14_dinov2": "dinov2_vitb14",
            "vit_large_patch14_dinov2": "dinov2_vitl14",
            "vit_giant_patch14_dinov2": "dinov2_vitg14",
            "vit_small_patch14_dinov2": "dinov2_vits14",
        }.get(timm_name, timm_name)
        model = torch.hub.load(
            "facebookresearch/dinov2",
            hub_name,
            pretrained=pretrained,
            verbose=False,
        )
        return model

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor, return_layer_indices: list[int] | None = None) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Extract patch token features (no CLS token).

        The forward pass is always wrapped in torch.no_grad() because
        this encoder is frozen and should never accumulate gradients.

        Args:
            x: Input image (B, 3, H, W), values in [0, 1].
            return_layer_indices: Optional list of layer indices to return.
                                  If None, only returns final layer output.

        Returns:
            If return_layer_indices is None:
                Patch token features (B, N, C) where N = (H/patch_size)*(W/patch_size).
            Else:
                Tuple of (final_output, skip_features).
        """
        # [MOD] Convert to the exact input distribution expected by DINOv2.
        x = x.clamp(0.0, 1.0)
        mean = self.pixel_mean.to(device=x.device, dtype=x.dtype)
        std = self.pixel_std.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / std

        output_hidden_states = return_layer_indices is not None

        with torch.no_grad():
            if output_hidden_states:
                # Use forward_intermediate to get all layer outputs
                features = self.backbone.forward_intermediate(x)
            else:
                features = self.backbone.forward_features(x)

        if return_layer_indices is None:
            # Return only final layer
            if isinstance(features, dict):
                if "x_norm_patchtokens" in features:
                    return features["x_norm_patchtokens"]
                if "patch_tokens" in features:
                    return features["patch_tokens"]
                tokens = next(v for v in features.values() if isinstance(v, Tensor))
                return tokens[:, 1:, :] if tokens.dim() == 3 else tokens
            else:
                return features[:, 1:, :]
        else:
            # Return final output + intermediate layers
            # features is now the output from forward_intermediate
            if isinstance(features, dict) and "x_norm_clstoken" in features:
                # Newer timm format: has intermediate outputs
                skip_features = []
                for idx in return_layer_indices:
                    layer_key = f"layer{idx}"
                    if layer_key in features:
                        layer_tokens = features[layer_key]  # (B, N+1, C)
                        skip_features.append(layer_tokens[:, 1:, :])

                # Final output
                final = features.get("x_norm_patchtokens")
                if final is None:
                    final = features.get("patch_tokens")
                if final is None:
                    tokens = next(v for v in features.values() if isinstance(v, Tensor) and v is not features.get("x_norm_clstoken"))
                    final = tokens[:, 1:, :] if tokens.dim() == 3 else tokens

                return final, skip_features
            else:
                # Fallback: can't get intermediate outputs, return only final
                if isinstance(features, dict):
                    if "x_norm_patchtokens" in features:
                        return features["x_norm_patchtokens"], []
                    if "patch_tokens" in features:
                        return features["patch_tokens"], []
                    tokens = next(v for v in features.values() if isinstance(v, Tensor))
                    final = tokens[:, 1:, :] if tokens.dim() == 3 else tokens
                else:
                    final = features[:, 1:, :]
                return final, []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size

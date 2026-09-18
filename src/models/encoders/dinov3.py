"""Frozen DINOv3 ViT-B encoder wrapper."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel

from src.models.encoders.base import BaseEncoder


class DINOv3Encoder(BaseEncoder):
    """Frozen DINOv3 ViT-B image encoder.

    Loads a pretrained DINOv3 ViT-B model via HuggingFace Transformers and
    extracts patch token features. CLS token and register tokens are discarded.
    Always kept in eval mode. The current HuggingFace
    ``facebook/dinov3-vitb16-pretrain-lvd1689m`` checkpoint exposes 12
    transformer blocks. Drop-in replacement inside ImageEnhancementPipeline.

    Args:
        model_name: HuggingFace model ID.
        frozen:     Whether to freeze all parameters (default True).
        patch_size: Patch size of the backbone (16 for DINOv3 ViT-B).
        embed_dim:  Output feature dimension (768 for DINOv3 ViT-B).
    """

    DEFAULT_SKIP_LAYERS: tuple[int, int, int, int] = (2, 5, 8, 11)
    """Standard DINOv3 ViT-B skip set used across this repository."""

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        frozen: bool = True,
        patch_size: int = 16,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self._patch_size = patch_size
        self._embed_dim  = embed_dim

        self.backbone = AutoModel.from_pretrained(model_name)
        self.num_hidden_layers = int(self.backbone.config.num_hidden_layers)
        self.num_register_tokens = self.backbone.config.num_register_tokens
        
        # [MOD] DINOv3 expects ImageNet-style normalized inputs.
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

    def forward(self, x: Tensor, return_layer_indices: list[int] | None = None) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Extract patch token features (no CLS or register tokens).

        Args:
            x: Input image (B, 3, H, W), values in [0, 1].
               Note: DINOv3 expects ImageNet-normalised input. If your
               pipeline already normalises, set normalise=False in the
               data transforms.
            return_layer_indices: Optional list of layer indices to return.
                                  If None, only returns final layer output.
                                  Layer indices are 0-indexed from the first
                                  transformer block (0 = first,
                                  11 = last for the current ViT-B/12 checkpoint).
                                  The repo-standard skip set is [2, 5, 8, 11].

        Returns:
            If return_layer_indices is None:
                Patch token features (B, N, C) where N = (H/patch_size)*(W/patch_size).
            Else:
                Tuple of (final_output, skip_features) where skip_features is a list
                of patch token features from the specified layers.
        """
        # [MOD] Convert to the exact input distribution expected by DINOv3.
        x = x.clamp(0.0, 1.0)
        mean = self.pixel_mean.to(device=x.device, dtype=x.dtype)
        std = self.pixel_std.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / std

        output_hidden_states = return_layer_indices is not None
        if return_layer_indices is not None:
            invalid = [
                idx for idx in return_layer_indices
                if idx < 0 or idx >= self.num_hidden_layers
            ]
            if invalid:
                raise ValueError(
                    "DINOv3 return_layer_indices out of range for "
                    f"{self.num_hidden_layers}-layer backbone: {invalid}"
                )

        with torch.no_grad():
            outputs = self.backbone(pixel_values=x, output_hidden_states=output_hidden_states)

        R = self.num_register_tokens
        slice_idx = 1 + R  # Skip CLS and register tokens

        if return_layer_indices is None:
            # Return only final layer
            return outputs.last_hidden_state[:, slice_idx:, :]  # (B, N, C)
        else:
            # Return final output + intermediate layers
            hidden_states = outputs.hidden_states  # Tuple of (embeddings + all layer outputs)
            # hidden_states[0] = embedding output
            # hidden_states[1] = after transformer layer 0
            # hidden_states[2] = after transformer layer 1
            # ...
            # hidden_states[12] = after transformer layer 11 (final) for the
            # current 12-layer ViT-B checkpoint.

            # Adjust indices: layer 0 -> hidden_states[1], layer L -> hidden_states[L+1]
            skip_features = []
            for idx in return_layer_indices:
                # idx=0 means first transformer layer -> hidden_states[1]
                hidden_idx = idx + 1
                if hidden_idx < len(hidden_states):
                    layer_output = hidden_states[hidden_idx][:, slice_idx:, :]
                    skip_features.append(layer_output)

            final_output = outputs.last_hidden_state[:, slice_idx:, :]
            return final_output, skip_features

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size

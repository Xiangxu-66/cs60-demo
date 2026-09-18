"""Frozen DINOv3 ViT-B encoder wrapper."""
from __future__ import annotations

import torch
from torch import Tensor
from transformers import AutoModel

from src.models.encoders.base import BaseEncoder


class DINOv3Encoder(BaseEncoder):
    """Frozen DINOv3 ViT-B image encoder."""

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        frozen: bool = True,
        patch_size: int = 16,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self._patch_size = patch_size
        self._embed_dim = embed_dim

        self.backbone = AutoModel.from_pretrained(model_name)
        self.num_register_tokens = self.backbone.config.num_register_tokens

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

    def forward(self, x: Tensor) -> Tensor:
        """Extract patch token features (no CLS or register tokens)."""
        x = x.clamp(0.0, 1.0)
        mean = self.pixel_mean.to(device=x.device, dtype=x.dtype)
        std = self.pixel_std.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / std

        with torch.no_grad():
            outputs = self.backbone(pixel_values=x)

        register_count = self.num_register_tokens
        return outputs.last_hidden_state[:, 1 + register_count :, :]

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size

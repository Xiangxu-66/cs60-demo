"""Frozen ViT encoder wrapper (E-1)."""
from __future__ import annotations

# TODO: Implement ViTEncoder using timm.create_model('vit_base_patch16_224', ...)
#       Follow the same pattern as DINOv2Encoder:
#         1. Load via timm, freeze, keep in eval mode
#         2. forward() returns (B, N, C) patch tokens (no CLS)
#         3. embed_dim = 768, patch_size = 16

from src.models.encoders.base import BaseEncoder
from torch import Tensor


class ViTEncoder(BaseEncoder):
    """Frozen ViT-B/16 encoder (E-1 in encoder comparison).

    TODO: Full implementation pending.
    """

    def __init__(
        self,
        name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        frozen: bool = True,
        patch_size: int = 16,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self._patch_size = patch_size
        self._embed_dim = embed_dim
        # TODO: self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)
        # TODO: if frozen: self.freeze()
        raise NotImplementedError("ViTEncoder not yet implemented. See TODO in vit.py")

    def forward(self, x: Tensor) -> Tensor:
        # TODO: implement
        raise NotImplementedError

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size

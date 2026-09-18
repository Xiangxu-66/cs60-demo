"""VRWKV-based image encoder (E-3)."""
from __future__ import annotations

# TODO: Implement VRWKVEncoder
#       Architecture reference: VRWKV (ICLR 2025)
#       Key components:
#         1. PatchEmbed: Conv2d(3, embed_dim, patch_size, stride=patch_size)
#         2. Stack of VRWKVBlock (Bi-WKV + Q-Shift + FFN)
#         3. LayerNorm output
#       Output: (B, N, C) patch tokens, same interface as DINOv2Encoder
#       Note: pretrained=False until we have checkpoint weights

from src.models.encoders.base import BaseEncoder
from torch import Tensor


class VRWKVEncoder(BaseEncoder):
    """VRWKV-based image encoder (E-3 in encoder comparison).

    TODO: Full implementation pending.
    """

    def __init__(
        self,
        name: str = "vrwkv_base",
        pretrained: bool = False,
        frozen: bool = True,
        patch_size: int = 16,
        embed_dim: int = 512,
        n_layer: int = 12,
    ) -> None:
        super().__init__()
        self._patch_size = patch_size
        self._embed_dim = embed_dim
        # TODO: build PatchEmbed + VRWKVBlock stack
        # TODO: if frozen: self.freeze()
        raise NotImplementedError("VRWKVEncoder not yet implemented. See TODO in vrwkv_encoder.py")

    def forward(self, x: Tensor) -> Tensor:
        # TODO: implement
        raise NotImplementedError

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size

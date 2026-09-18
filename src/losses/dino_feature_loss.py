"""Patch-token perceptual loss using a frozen DINO encoder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel


class DINOFeatureLoss(nn.Module):
    """Perceptual loss on frozen DINOv3 patch tokens.

    DINOv3 is reported to produce strong dense features, so this loss uses
    patch-token distances instead of VGG activations. Parameters are frozen,
    but gradients still flow to the predicted image through the backbone.
    """

    def __init__(
        self,
        weight: float = 0.01,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        patch_size: int = 16,
        normalize_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.patch_size = patch_size
        self.normalize_inputs = normalize_inputs

        self.backbone = AutoModel.from_pretrained(model_name)
        self.num_register_tokens = self.backbone.config.num_register_tokens

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

    def _prepare(self, x: Tensor) -> Tensor:
        x = x.clamp(0.0, 1.0)
        if self.normalize_inputs:
            mean = self.mean.to(device=x.device, dtype=x.dtype)
            std = self.std.to(device=x.device, dtype=x.dtype)
            x = (x - mean) / std
        return x

    def _forward_tokens(self, x: Tensor) -> Tensor:
        outputs = self.backbone(pixel_values=x)
        register_count = self.num_register_tokens
        return outputs.last_hidden_state[:, 1 + register_count :, :]

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_tokens = self._forward_tokens(self._prepare(pred))
        with torch.no_grad():
            target_tokens = self._forward_tokens(self._prepare(target))
        return self.weight * F.l1_loss(pred_tokens, target_tokens)

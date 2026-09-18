from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class CLIPLoss(nn.Module):
    """CLIP perceptual loss (official-style implementation).

    Uses CLIP's official preprocessing and image encoder.
    Computes cosine similarity loss between pred and target.
    """

    def __init__(self, weight: float = 0.1, device: str = "mps"):
        super().__init__()
        self.weight = weight
        self.device = device
        self.model, _ = clip.load("ViT-B/32", device=device)

        for p in self.model.parameters():
            p.requires_grad = False

        self.register_buffer(
            "mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """CLIP official preprocessing."""
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (B, 3, H, W) in [0, 1]
        """

        pred = self.preprocess(pred)
        target = self.preprocess(target)


        pred_feat = self.model.encode_image(pred)

        with torch.no_grad():  
            target_feat = self.model.encode_image(target)


        pred_feat = F.normalize(pred_feat, dim=-1)
        target_feat = F.normalize(target_feat, dim=-1)


        similarity = (pred_feat * target_feat).sum(dim=-1)

        loss = 1 - similarity.mean()

        return self.weight * loss
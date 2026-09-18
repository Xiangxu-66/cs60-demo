"""NIMA aesthetic quality loss — Stage 2 placeholder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.models as models


class NIMALoss(nn.Module):
    """
    Full NIMA Loss (official-style + ranking + stable)

    Features:
    - MobileNetV2 backbone (NIMA-style)
    - 10-bin distribution
    - EMD loss
    - Ranking loss
    - Score maximization
    - Auto device (Lightning compatible)
    """

    def __init__(
        self,
        weight: float = 0.1,
        alpha: float = 1.0,   # EMD
        beta: float = 0.5,    # ranking
        gamma: float = 0.2,   # score boost
        margin: float = 0.1,
    ) -> None:
        super().__init__()

        self.weight = weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.margin = margin

        # -------- Backbone --------
        backbone = models.mobilenet_v2(pretrained=True)

        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, 10)

        self.nima = backbone

        for p in self.nima.parameters():
            p.requires_grad = False

        self.nima.eval()

        self.register_buffer(
            "score_weights",
            torch.arange(1, 11).float().view(1, -1)
        )

    # -------- utils --------

    def _predict_distribution(self, x: Tensor) -> Tensor:
        device = next(self.nima.parameters()).device
        x = x.to(device)

        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        x = (x - 0.5) / 0.5

        logits = self.nima(x)
        return F.softmax(logits, dim=1)

    def _emd_loss(self, p: Tensor, q: Tensor) -> Tensor:
        prev = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)

        cdf_p = torch.cumsum(p, dim=1)
        cdf_q = torch.cumsum(q, dim=1)

        torch.use_deterministic_algorithms(prev)
 
        return torch.mean((cdf_p - cdf_q) ** 2)

    def _mean_score(self, dist: Tensor) -> Tensor:
        return (dist * self.score_weights.to(dist.device)).sum(dim=1)

    # -------- forward --------

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        pred: enhanced image
        target: original image
        """

        with torch.no_grad():
            target_dist = self._predict_distribution(target)

        pred_dist = self._predict_distribution(pred)

        # -------- 1. EMD --------
        emd = self._emd_loss(pred_dist, target_dist)

        # -------- 2. Ranking --------
        pred_score = self._mean_score(pred_dist)
        target_score = self._mean_score(target_dist)

        ranking = torch.relu(self.margin - (pred_score - target_score)).mean()

        # -------- 3. Score maximization --------
        score_loss = -pred_score.mean()

        # -------- Final --------
        loss = (
            self.alpha * emd
            + self.beta * ranking
            + self.gamma * score_loss
        )

        return self.weight * loss
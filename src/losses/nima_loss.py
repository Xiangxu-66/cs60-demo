"""NIMA aesthetic quality loss — Stage 2 placeholder."""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class NIMALoss(nn.Module):
    """NIMA-based aesthetic quality loss (Stage 2).

    TODO: Implement using a pretrained NIMA model (MobileNet + EMD loss).
          Reference: NIMA: Neural Image Assessment (TPAMI 2018).

    Args:
        weight: Scalar multiplier for this loss component.
    """

    def __init__(self, weight: float = 0.05) -> None:
        super().__init__()
        self.weight = weight
        # TODO: load pretrained NIMA backbone (MobileNetV2 + rating head)
        # TODO: freeze NIMA backbone weights

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute NIMA-based aesthetic loss.

        Args:
            pred:   Predicted image (B, 3, H, W).
            target: Ground truth image (B, 3, H, W) — unused for NIMA.

        Returns:
            Scalar loss (currently 0 — not yet implemented).
        """
        # TODO: score = nima_model(pred)           # (B, 10) rating distribution
        # TODO: return weight * emd_loss(score)    # Earth Mover's Distance
        return pred.new_zeros(1).squeeze()

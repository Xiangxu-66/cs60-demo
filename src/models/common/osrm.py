"""OSRM: Omni-Scale Receptive Module from DyRSRNet."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OSRM(nn.Module):
    """Depthwise multi-branch receptive-field mixer with reparameterized eval path."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=False)
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.conv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)

        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True)
        self.conv5x5_reparam = nn.Conv2d(
            dim,
            dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=False,
        )
        self._reparam_done = False

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)
        return (
            self.alpha[0] * x
            + self.alpha[1] * out1x1
            + self.alpha[2] * out3x3
            + self.alpha[3] * out5x5
        )

    def _reparam_5x5(self) -> None:
        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2))
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1))
        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2))

        combined_weight = (
            self.alpha[0] * identity_weight
            + self.alpha[1] * padded_weight_1x1
            + self.alpha[2] * padded_weight_3x3
            + self.alpha[3] * self.conv5x5.weight
        )
        combined_weight = combined_weight.to(self.conv5x5_reparam.weight.device)
        self.conv5x5_reparam.weight = nn.Parameter(combined_weight)
        self._reparam_done = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._reparam_done = False
            return self.forward_train(x)
        if not self._reparam_done:
            self._reparam_5x5()
        return self.conv5x5_reparam(x)


__all__ = ["OSRM"]

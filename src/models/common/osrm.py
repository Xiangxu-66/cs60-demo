"""OSRM: Omni-Scale Receptive Module from DyRSRNet.

This module provides multi-scale receptive field fusion with reparameterization:
- Training mode: Uses multi-branch fusion (1×1, 3×3, 5×5 + identity)
- Inference mode: Reparameterized to single 5×5 conv (zero overhead)

Reference: DyRSRNet (PRCV'25 Oral)
Source: https://github.com/...
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OSRM(nn.Module):
    """Omni-Scale Receptive Module (OSRM).

    Provides multi-scale receptive field fusion through learnable weighted
    combination of identity, 1×1, 3×3, and 5×5 depthwise convolutions.

    During training, uses all branches with learnable alpha weights.
    During inference, reparameterizes to a single 5×5 depthwise convolution.

    Args:
        dim: Number of input/output channels (depthwise groups = dim).

    Shape:
        Input: (B, C, H, W)
        Output: (B, C, H, W)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()

        # Training branches: multi-scale depthwise convolutions
        self.conv1x1 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1,
            groups=dim, bias=False
        )
        self.conv3x3 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=3, padding=1,
            groups=dim, bias=False
        )
        self.conv5x5 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=5, padding=2,
            groups=dim, bias=False
        )

        # Learnable weights for multi-branch fusion
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True)

        # Inference branch: reparameterized 5×5 convolution
        self.conv5x5_reparam = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=5, padding=2,
            groups=dim, bias=False
        )

        # Flag to track whether reparameterization has been performed
        self._reparam_done = False

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass in training mode with multi-branch fusion.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Fused output tensor (B, C, H, W)
        """
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)

        # Learnable weighted fusion
        out = (
            self.alpha[0] * x +
            self.alpha[1] * out1x1 +
            self.alpha[2] * out3x3 +
            self.alpha[3] * out5x5
        )
        return out

    def _reparam_5x5(self) -> None:
        """Fuse multi-branch weights into single 5×5 convolution.

        Pads smaller kernels and combines with learned alpha weights.
        """
        # Pad 1×1 and 3×3 weights to 5×5
        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2))
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1))

        # Create identity kernel padded to 5×5
        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2))

        # Combine all weights with alpha scaling
        combined_weight = (
            self.alpha[0] * identity_weight +
            self.alpha[1] * padded_weight_1x1 +
            self.alpha[2] * padded_weight_3x3 +
            self.alpha[3] * self.conv5x5.weight
        )

        device = self.conv5x5_reparam.weight.device
        combined_weight = combined_weight.to(device)

        # Update reparameterized convolution weight
        self.conv5x5_reparam.weight = nn.Parameter(combined_weight)
        self._reparam_done = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic training/inference mode handling.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, C, H, W)
        """
        if self.training:
            # Reset reparam flag when returning to training
            self._reparam_done = False
            return self.forward_train(x)
        elif not self._reparam_done:
            # First eval call: perform reparameterization
            self._reparam_5x5()
            return self.conv5x5_reparam(x)
        else:
            # Subsequent eval calls: use reparameterized conv
            return self.conv5x5_reparam(x)


__all__ = ["OSRM"]

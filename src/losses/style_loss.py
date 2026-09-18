from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch import Tensor


class StyleLoss(nn.Module):
    """Gram Matrix based Style Loss (Johnson et al. 2016).

    Args:
        weight: loss weight
        layers: VGG layers to extract features
    """

    def __init__(
        self,
        weight: float = 1.0,
        layers: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.weight = weight

        if layers is None:
            layers = ["relu1_2", "relu2_2", "relu3_3"]

        self.layers = layers

        # Load pretrained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg = vgg.eval()

        for p in self.vgg.parameters():
            p.requires_grad = False

        # layer mapping
        self.layer_name_mapping = {
            "0": "conv1_1",
            "1": "relu1_1",
            "2": "conv1_2",
            "3": "relu1_2",
            "5": "conv2_1",
            "6": "relu2_1",
            "7": "conv2_2",
            "8": "relu2_2",
            "10": "conv3_1",
            "11": "relu3_1",
            "12": "conv3_2",
            "13": "relu3_2",
            "14": "conv3_3",
            "15": "relu3_3",
            "19": "conv4_1",
            "20": "relu4_1",
        }

    def gram_matrix(self, x: Tensor) -> Tensor:
        B, C, H, W = x.size()
        features = x.view(B, C, H * W)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (C * H * W)

    def extract_features(self, x: Tensor) -> dict[str, Tensor]:
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layer_name_mapping:
                layer_name = self.layer_name_mapping[name]
                if layer_name in self.layers:
                    features[layer_name] = x
        return features

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        pred, target: (B, 3, H, W) in [0,1] or [-1,1]
        """

   
        if pred.min() < 0:
            pred = (pred + 1) / 2
            target = (target + 1) / 2

        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)

        loss = 0.0

        for layer in self.layers:
            gram_pred = self.gram_matrix(pred_features[layer])
            gram_target = self.gram_matrix(target_features[layer])

            loss += F.l1_loss(gram_pred, gram_target)

        return self.weight * loss
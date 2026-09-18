import torch
import torch.nn as nn
import lpips


class LPIPSLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.loss_fn = lpips.LPIPS(net='alex')

        for p in self.loss_fn.parameters():
            p.requires_grad = False

        self.loss_fn.eval()

    def forward(self, pred, target):


        loss = self.loss_fn(pred, target).mean()
        return self.weight * loss


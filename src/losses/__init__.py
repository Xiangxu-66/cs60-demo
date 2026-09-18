from src.losses.l1_loss import L1Loss
from src.losses.ssim_loss import SSIMLoss
from src.losses.perceptual_loss import PerceptualLoss
from src.losses.dino_feature_loss import DINOFeatureLoss
from src.losses.combined_loss import CombinedLoss

__all__ = ["L1Loss", "SSIMLoss", "PerceptualLoss", "DINOFeatureLoss", "CombinedLoss"]

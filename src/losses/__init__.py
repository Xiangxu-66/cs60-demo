"""Loss package exports."""
from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "L1Loss": "src.losses.l1_loss",
    "SSIMLoss": "src.losses.ssim_loss",
    "PerceptualLoss": "src.losses.perceptual_loss",
    "ColorAngleLoss": "src.losses.color_angle_loss",
    "CombinedLoss": "src.losses.combined_loss",
    "CoVWeightedLoss": "src.losses.cov_weighted_loss",
    "SelectableAdaptiveLoss": "src.losses.selectable_adaptive_loss",
    "ContrastLoss": "src.losses.contrast_loss",
    "SaturationLoss": "src.losses.saturation_loss",
    "DarkEnhanceLoss": "src.losses.dark_enhance_loss",
    "MSSWCLoss": "src.losses.ms_swc_loss",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value

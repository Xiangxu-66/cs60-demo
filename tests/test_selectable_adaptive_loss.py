from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.losses.selectable_adaptive_loss import SelectableAdaptiveLoss


def test_fixed_selectable_loss_uses_only_active_components() -> None:
    cfg = OmegaConf.create(
        {
            "components": {
                "l1": {
                    "enabled": True,
                    "_target_": "src.losses.l1_loss.L1Loss",
                    "weight": 9.0,
                },
                "ssim": {
                    "enabled": False,
                    "_target_": "src.losses.ssim_loss.SSIMLoss",
                    "weight": 7.0,
                    "window_size": 11,
                },
                "hist": {
                    "enabled": True,
                    "_target_": "src.losses.color_histogram_loss.ColorHistogramLoss",
                    "weight": 5.0,
                    "num_bins": 8,
                },
            },
            "loss_weights": {"l1": 2.0, "hist": 0.5},
            "use_adaptive_loss": False,
            "adaptive": {},
        }
    )
    loss_mod = SelectableAdaptiveLoss(**cfg)

    pred = torch.rand(2, 3, 16, 16)
    target = torch.rand(2, 3, 16, 16)
    total, raw_losses, weighted_losses, weights = loss_mod.compute(pred, target)

    assert sorted(raw_losses) == ["hist", "l1"]
    assert sorted(loss_mod.loss_fns.keys()) == ["hist", "l1"]
    assert sorted(weighted_losses) == ["hist", "l1"]
    assert sorted(weights) == ["hist", "l1"]
    assert torch.isclose(weights["l1"], torch.tensor(2.0))
    assert torch.isclose(weights["hist"], torch.tensor(0.5))
    assert torch.allclose(weighted_losses["l1"], raw_losses["l1"] * 2.0)
    assert torch.allclose(weighted_losses["hist"], raw_losses["hist"] * 0.5)
    assert torch.allclose(total, weighted_losses["l1"] + weighted_losses["hist"])



def test_adaptive_selectable_loss_updates_weights_after_warmup() -> None:
    cfg = OmegaConf.create(
        {
            "components": {
                "l1": {
                    "enabled": True,
                    "_target_": "src.losses.l1_loss.L1Loss",
                    "weight": 3.0,
                },
                "ssim": {
                    "enabled": True,
                    "_target_": "src.losses.ssim_loss.SSIMLoss",
                    "weight": 4.0,
                    "window_size": 11,
                },
                "hist": {
                    "enabled": True,
                    "_target_": "src.losses.color_histogram_loss.ColorHistogramLoss",
                    "weight": 5.0,
                    "num_bins": 8,
                },
            },
            "loss_weights": {"l1": 1.0, "ssim": 0.1, "hist": 0.01},
            "use_adaptive_loss": True,
            "adaptive": {
                "warmup_steps": 2,
                "ema_decay": 0.9,
                "epsilon": 1e-8,
                "weight_min": 1e-3,
                "weight_max": 10.0,
            },
        }
    )
    loss_mod = SelectableAdaptiveLoss(**cfg)

    pred = torch.rand(2, 3, 16, 16)
    target = torch.rand(2, 3, 16, 16)
    _, _, _, weights_first = loss_mod.compute(pred, target, update_stats=True)
    _, _, _, weights_second = loss_mod.compute(pred * 0.85, target, update_stats=True)

    assert all(torch.isclose(v, torch.tensor(1.0, dtype=v.dtype)) for v in weights_first.values())
    total_weight = sum(float(v) for v in weights_second.values())
    assert abs(total_weight - 3.0) < 1e-4
    assert any(abs(float(v) - 1.0) > 1e-3 for v in weights_second.values())

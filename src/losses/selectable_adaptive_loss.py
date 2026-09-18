"""Selectable loss wrapper with fixed or CoV adaptive weighting."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import hydra
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.losses.cov_weighted_loss import CoVWeightedLoss


class SelectableAdaptiveLoss(nn.Module):
    """Compute active losses and aggregate them with fixed or adaptive weights."""

    _FORCE_UNIT_WEIGHT_TARGETS = {
        "src.losses.l1_loss.L1Loss",
        "src.losses.ssim_loss.SSIMLoss",
        "src.losses.color_histogram_loss.ColorHistogramLoss",
        "src.losses.color_angle_loss.ColorAngleLoss",
        "src.losses.perceptual_loss.PerceptualLoss",
    }

    def __init__(
        self,
        components: DictConfig | Mapping[str, Any],
        loss_weights: DictConfig | Mapping[str, float] | None = None,
        use_adaptive_loss: bool = False,
        adaptive: DictConfig | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.use_adaptive_loss = bool(use_adaptive_loss)
        self.loss_fns = nn.ModuleDict()
        self.fixed_weights: dict[str, float] = {}
        self._last_raw_losses: dict[str, Tensor] = {}
        self._last_weighted_losses: dict[str, Tensor] = {}
        self._last_weights: dict[str, Tensor] = {}

        weights_cfg = self._to_plain_dict(loss_weights)
        component_map = self._to_plain_dict(components)
        for name, raw_cfg in component_map.items():
            if raw_cfg is None or str(name).startswith("_"):
                continue

            if isinstance(raw_cfg, nn.Module):
                self.loss_fns[name] = raw_cfg
                self.fixed_weights[name] = float(
                    weights_cfg.get(name, getattr(raw_cfg, "weight", 1.0))
                )
                continue

            cfg = dict(raw_cfg)
            enabled = bool(cfg.pop("enabled", True))
            if not enabled:
                continue

            target = str(cfg.get("_target_", ""))
            component_weight = float(cfg.get("weight", 1.0))
            if target in self._FORCE_UNIT_WEIGHT_TARGETS:
                cfg["weight"] = 1.0

            module = hydra.utils.instantiate(OmegaConf.create(cfg))
            self.loss_fns[name] = module
            self.fixed_weights[name] = float(
                weights_cfg.get(name, component_weight)
            )

        if not self.loss_fns:
            raise ValueError(
                "SelectableAdaptiveLoss requires at least one active component."
            )

        adaptive_cfg = self._to_plain_dict(adaptive)
        self.adaptive_weighter = (
            CoVWeightedLoss(list(self.loss_fns.keys()), **adaptive_cfg)
            if self.use_adaptive_loss
            else None
        )

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Return the aggregated total loss."""
        total_loss, _, _, _ = self.compute(pred, target, update_stats=True)
        return total_loss

    def compute(
        self,
        pred: Tensor,
        target: Tensor,
        update_stats: bool = True,
        **_: Any,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        """Return total loss, raw losses, weighted losses, and weights."""
        raw_losses = self._compute_raw_losses(pred, target)

        if self.adaptive_weighter is not None:
            total_loss, weighted_losses, weights = self.adaptive_weighter(
                raw_losses,
                update_stats=update_stats,
            )
        else:
            total_loss = pred.new_zeros(())
            weights = {
                name: pred.new_tensor(weight)
                for name, weight in self.fixed_weights.items()
                if name in raw_losses
            }
            weighted_losses = {
                name: raw_losses[name] * weights[name]
                for name in raw_losses
            }
            for value in weighted_losses.values():
                total_loss = total_loss + value

        self._last_raw_losses = {
            name: value.detach() for name, value in raw_losses.items()
        }
        self._last_weighted_losses = {
            name: value.detach() for name, value in weighted_losses.items()
        }
        self._last_weights = {
            name: value.detach() for name, value in weights.items()
        }
        return total_loss, raw_losses, weighted_losses, weights

    def raw_component_losses(self) -> dict[str, Tensor]:
        """Return the most recent raw losses."""
        return dict(self._last_raw_losses)

    def weighted_component_losses(self) -> dict[str, Tensor]:
        """Return the most recent weighted losses."""
        return dict(self._last_weighted_losses)

    def component_weights(self) -> dict[str, Tensor]:
        """Return the most recent active weights."""
        return dict(self._last_weights)

    def _compute_raw_losses(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> dict[str, Tensor]:
        raw_losses: dict[str, Tensor] = {}
        for name, fn in self.loss_fns.items():
            loss = fn(pred, target)
            if loss.ndim != 0:
                loss = loss.mean()
            if not bool(loss.isfinite().item()):
                raise ValueError(
                    f"Raw loss '{name}' is not finite: {float(loss.detach().cpu())}"
                )
            raw_losses[name] = loss
        return raw_losses

    @staticmethod
    def _to_plain_dict(
        value: DictConfig | Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, DictConfig):
            container = OmegaConf.to_container(value, resolve=False)
            return dict(container) if isinstance(container, dict) else {}
        if isinstance(value, Mapping):
            return dict(value)
        raise TypeError(f"Unsupported config mapping type: {type(value)!r}")

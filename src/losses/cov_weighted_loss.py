"""Coefficient-of-Variation adaptive weighting for multiple losses."""
from __future__ import annotations

import warnings
from typing import Mapping

import torch
import torch.nn as nn
from torch import Tensor


class CoVWeightedLoss(nn.Module):
    """Adaptive loss weighter based on per-loss coefficient of variation."""

    def __init__(
        self,
        loss_names: list[str],
        *,
        warmup_steps: int = 20,
        ema_decay: float = 0.9,
        epsilon: float = 1e-8,
        weight_min: float = 1e-3,
        weight_max: float = 10.0,
    ) -> None:
        super().__init__()
        self.loss_names = list(loss_names)
        self.warmup_steps = int(warmup_steps)
        self.ema_decay = float(ema_decay)
        self.epsilon = float(epsilon)
        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)

        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))
        self._state_keys: dict[str, str] = {}
        for name in self.loss_names:
            state_key = self._sanitize_name(name)
            self._state_keys[name] = state_key
            self.register_buffer(f"{state_key}_mean", torch.zeros(()))
            self.register_buffer(f"{state_key}_mean_sq", torch.zeros(()))
            self.register_buffer(
                f"{state_key}_initialized",
                torch.tensor(False, dtype=torch.bool),
            )

    def forward(
        self,
        loss_dict: Mapping[str, Tensor],
        *,
        update_stats: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        """Weight active losses and return total, weighted losses, and weights."""
        if not loss_dict:
            raise ValueError("CoVWeightedLoss requires at least one active loss.")

        self._validate_loss_dict(loss_dict)

        if update_stats:
            self._update_statistics(loss_dict)

        weights = self._compute_weights(loss_dict)
        weighted_losses = {
            name: loss * weights[name]
            for name, loss in loss_dict.items()
        }
        total_loss = next(iter(loss_dict.values())).new_zeros(())
        for value in weighted_losses.values():
            total_loss = total_loss + value
        return total_loss, weighted_losses, weights

    def _update_statistics(self, loss_dict: Mapping[str, Tensor]) -> None:
        for name, loss in loss_dict.items():
            state_key = self._state_keys[name]
            mean = getattr(self, f"{state_key}_mean")
            mean_sq = getattr(self, f"{state_key}_mean_sq")
            initialized = getattr(self, f"{state_key}_initialized")
            value = loss.detach()

            if not bool(initialized.item()):
                mean.copy_(value)
                mean_sq.copy_(value.square())
                initialized.fill_(True)
                continue

            mean.mul_(self.ema_decay).add_(value * (1.0 - self.ema_decay))
            mean_sq.mul_(self.ema_decay).add_(
                value.square() * (1.0 - self.ema_decay)
            )

        self.num_updates.add_(1)

    def _compute_weights(self, loss_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
        loss_names = list(loss_dict.keys())
        device = next(iter(loss_dict.values())).device
        dtype = next(iter(loss_dict.values())).dtype
        num_losses = len(loss_names)

        if num_losses == 1:
            only_name = loss_names[0]
            return {only_name: torch.ones((), device=device, dtype=dtype)}

        if int(self.num_updates.item()) < self.warmup_steps:
            return self._equal_weights(loss_names, device, dtype)

        cvs = []
        for name in loss_names:
            state_key = self._state_keys[name]
            mean = getattr(self, f"{state_key}_mean").to(device=device, dtype=dtype)
            mean_sq = getattr(self, f"{state_key}_mean_sq").to(device=device, dtype=dtype)
            variance = torch.clamp(mean_sq - mean.square(), min=0.0)
            std = torch.sqrt(variance + self.epsilon)
            cv = std / mean.abs().clamp_min(self.epsilon)
            cvs.append(cv)

        cv_tensor = torch.stack(cvs)
        if not torch.isfinite(cv_tensor).all():
            warnings.warn(
                "Non-finite CoV weights detected; falling back to equal weights.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._equal_weights(loss_names, device, dtype)

        if float(cv_tensor.sum().item()) <= self.epsilon:
            return self._equal_weights(loss_names, device, dtype)

        try:
            weights = cv_tensor / cv_tensor.sum().clamp_min(self.epsilon)
            weights = weights * float(num_losses)
            weights = torch.clamp(weights, min=self.weight_min, max=self.weight_max)
            weights = weights / weights.sum().clamp_min(self.epsilon)
            weights = weights * float(num_losses)
        except RuntimeError:
            warnings.warn(
                "CoV weight normalization failed; falling back to equal weights.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._equal_weights(loss_names, device, dtype)

        if not torch.isfinite(weights).all():
            warnings.warn(
                "CoV weight normalization produced non-finite values; "
                "falling back to equal weights.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._equal_weights(loss_names, device, dtype)

        return {
            name: weights[idx].detach()
            for idx, name in enumerate(loss_names)
        }

    def _equal_weights(
        self,
        loss_names: list[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Tensor]:
        return {
            name: torch.ones((), device=device, dtype=dtype)
            for name in loss_names
        }

    def _validate_loss_dict(self, loss_dict: Mapping[str, Tensor]) -> None:
        unknown = set(loss_dict) - set(self.loss_names)
        if unknown:
            raise KeyError(
                f"Unknown loss names for CoVWeightedLoss: {sorted(unknown)}"
            )

        for name, loss in loss_dict.items():
            if loss.ndim != 0:
                raise ValueError(
                    f"Loss '{name}' must be a scalar tensor, got shape {tuple(loss.shape)}."
                )
            if not torch.isfinite(loss).item():
                raise ValueError(
                    f"Raw loss '{name}' is not finite: {float(loss.detach().cpu())}"
                )

    @staticmethod
    def _sanitize_name(name: str) -> str:
        chars = [
            ch if ch.isalnum() or ch == "_" else "_"
            for ch in name
        ]
        safe = "".join(chars).strip("_")
        return safe or "loss"

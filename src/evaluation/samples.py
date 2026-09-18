"""Sample index selection strategies for model evaluation.

Provides utilities to select test samples based on different strategies:
- Equidistant: uniform sampling across the test set
- Difficulty: based on PSNR quantiles (requires baseline model)
- Scene: manual annotation by scene type
- Custom: explicit index list
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from collections.abc import Sequence


class SampleIndexSelector:
    """Selects sample indices from test set based on configured strategy.

    Supports multiple sampling strategies for reproducible model evaluation:
    - equidistant: uniform spacing across test set
    - difficulty: quantile-based on PSNR scores
    - scene: manually annotated scene categories
    - custom: explicit index list

    Example:
        selector = SampleIndexSelector.from_config("configs/eval_samples.yaml")
        indices = selector.get_indices(total_test_size=500)
    """

    STRATEGIES = ("equidistant", "difficulty", "scene", "custom")

    def __init__(
        self,
        strategy: str = "equidistant",
        num_samples: int = 30,
        start_idx: int = 0,
        custom_indices: Sequence[int] | None = None,
        scene_indices: dict[str, Sequence[int]] | None = None,
        difficulty_config: dict | None = None,
    ) -> None:
        """Initialize sample selector.

        Args:
            strategy: Sampling strategy name.
            num_samples: Number of samples for equidistant/difficulty strategies.
            start_idx: Starting index for equidistant sampling.
            custom_indices: Explicit index list for custom strategy.
            scene_indices: Scene category -> indices mapping.
            difficulty_config: Config for difficulty-based sampling.
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Must be one of {self.STRATEGIES}"
            )
        self.strategy = strategy
        self.num_samples = num_samples
        self.start_idx = start_idx
        self.custom_indices = list(custom_indices) if custom_indices else []
        self.scene_indices = {k: list(v) for k, v in (scene_indices or {}).items()}
        self.difficulty_config = difficulty_config or {}

    @classmethod
    def from_config(cls, config_path: str | Path) -> "SampleIndexSelector":
        """Load selector configuration from YAML file.

        Args:
            config_path: Path to eval_samples.yaml config file.

        Returns:
            Configured SampleIndexSelector instance.
        """
        cfg = OmegaConf.load(config_path)
        strategy = cfg.get("active_strategy", "equidistant")

        if strategy == "equidistant":
            return cls(
                strategy=strategy,
                num_samples=cfg.equidistant.get("num_samples", 30),
                start_idx=cfg.equidistant.get("start_idx", 0),
            )
        elif strategy == "difficulty":
            return cls(
                strategy=strategy,
                num_samples=cfg.difficulty.get("num_per_level", 10) * 3,
                difficulty_config=dict(cfg.difficulty),
            )
        elif strategy == "scene":
            return cls(
                strategy=strategy,
                scene_indices=dict(cfg.scene),
            )
        else:  # custom
            return cls(
                strategy=strategy,
                custom_indices=list(cfg.custom.get("indices", [])),
            )

    def get_indices(
        self,
        total_test_size: int,
        scene_category: str | None = None,
        psnr_scores: dict[int, float] | None = None,
    ) -> list[int]:
        """Get sample indices based on configured strategy.

        Args:
            total_test_size: Total number of samples in test set.
            scene_category: Specific scene category (for scene strategy).
            psnr_scores: Index -> PSNR mapping (for difficulty strategy).

        Returns:
            List of selected sample indices.
        """
        if self.strategy == "equidistant":
            return self._equidistant(total_test_size)
        elif self.strategy == "difficulty":
            return self._by_difficulty(total_test_size, psnr_scores)
        elif self.strategy == "scene":
            return self._by_scene(scene_category)
        else:  # custom
            return self._custom(total_test_size)

    def _equidistant(self, total_size: int) -> list[int]:
        """Select samples at uniform intervals."""
        if self.num_samples >= total_size:
            return list(range(self.start_idx, total_size))

        step = (total_size - self.start_idx) / self.num_samples
        indices = []
        for i in range(self.num_samples):
            idx = int(self.start_idx + i * step)
            if idx < total_size:
                indices.append(idx)
        return indices

    def _by_difficulty(
        self,
        total_size: int,
        psnr_scores: dict[int, float] | None,
    ) -> list[int]:
        """Select samples based on PSNR quantiles."""
        if psnr_scores is None:
            raise ValueError(
                "psnr_scores required for difficulty strategy. "
                "Run baseline model first."
            )

        valid_scores = {k: v for k, v in psnr_scores.items() if k < total_size}
        if not valid_scores:
            raise ValueError("No valid PSNR scores provided")

        # Sort by PSNR and split into quantiles
        sorted_items = sorted(valid_scores.items(), key=lambda x: x[1])
        indices = [k for k, _ in sorted_items]

        num_per_level = self.num_samples // 3
        quantile1 = len(indices) // 3
        quantile2 = 2 * len(indices) // 3

        # Low PSNR = hard, High PSNR = easy
        easy = indices[-quantile1:][-num_per_level:]
        medium = indices[quantile1:quantile2][:num_per_level]
        hard = indices[:quantile1][:num_per_level]

        return hard + medium + easy  # Return in order: hard -> medium -> easy

    def _by_scene(self, category: str | None) -> list[int]:
        """Select samples by scene category."""
        if not self.scene_indices:
            raise ValueError("No scene indices configured in config file")

        if category is None:
            # Return all scenes concatenated
            all_indices = []
            for indices in self.scene_indices.values():
                all_indices.extend(indices)
            return all_indices

        if category not in self.scene_indices:
            available = list(self.scene_indices.keys())
            raise ValueError(
                f"Unknown scene category: {category}. "
                f"Available: {available}"
            )
        return list(self.scene_indices[category])

    def _custom(self, total_size: int) -> list[int]:
        """Return custom indices, clamped to valid range."""
        if not self.custom_indices:
            return list(range(min(self.num_samples, total_size)))

        return [i for i in self.custom_indices if 0 <= i < total_size]


def load_sample_indices(
    config_path: str | Path,
    total_test_size: int,
    **kwargs,
) -> list[int]:
    """Convenience function to load sample indices from config.

    Args:
        config_path: Path to eval_samples.yaml.
        total_test_size: Total test set size.
        **kwargs: Additional args passed to get_indices().

    Returns:
        List of sample indices.
    """
    selector = SampleIndexSelector.from_config(config_path)
    return selector.get_indices(total_test_size, **kwargs)


def compute_psnr_scores(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[int, float]:
    """Compute PSNR scores for all test samples.

    Args:
        model: Model to evaluate.
        dataloader: Test dataloader (shuffle=False).
        device: Torch device.

    Returns:
        Mapping from global test index to PSNR score.
    """
    from src.evaluation.metrics import compute_psnr

    model.eval()
    psnr_scores = {}
    global_idx = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            inputs, targets = batch[0], batch[1]
            inputs = inputs.to(device)
            targets = targets.to(device)

            batch_size = inputs.shape[0]
            preds = model(inputs)

            # Compute per-sample PSNR
            for i in range(batch_size):
                pred_i = preds[i : i + 1]
                target_i = targets[i : i + 1]
                score = compute_psnr(pred_i, target_i)
                psnr_scores[global_idx] = score.item()
                global_idx += 1

    return psnr_scores

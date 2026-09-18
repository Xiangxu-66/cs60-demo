"""Unified experiment metadata capture.

Creates one artefact per experiment:
  outputs/logs/{name}/experiment_record.json  — full metadata + results

The global comparison table (experiments/results_summary.csv) is built
on demand by scripts/rebuild_summary.py, which scans all experiment directories.

Usage (called automatically from ColorEnhanceLitModule.on_test_end):

    record = ExperimentRecord(cfg)
    # ... training happens ...
    record.finalize(
        logged_metrics=trainer.logged_metrics,
        trainer=trainer,
        datamodule=dm,
    )
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Summary CSV column order (all experiments share this schema)
# ---------------------------------------------------------------------------
SUMMARY_FIELDS: list[str] = [
    # Identity
    "experiment_name", "timestamp", "git_commit", "git_branch",
    # Architecture
    "encoder", "bottleneck", "decoder", "fusion_mode",
    "skip_connection", "loss",
    # Data
    "dataset", "image_size", "crop_size", "train_crop_mode", "eval_resize_mode",
    "train_size", "val_size", "test_size",
    # Training
    "batch_size", "grad_accum", "effective_batch",
    "lr", "weight_decay", "scheduler", "max_epochs", "precision",
    # Efficiency (auto-computed, no profiler needed)
    "total_params", "trainable_params", "gflops", "training_time_min", "peak_vram_mb",
    # Results
    "best_epoch",
    "val_psnr", "val_ssim", "val_lpips",
    "test_psnr", "test_ssim", "test_lpips", "test_delta_e", "test_nima",
    # Artefacts
    "checkpoint_path", "log_dir", "experiment_dir", "notes",
]

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git_info() -> tuple[str, str]:
    """Return (short commit hash, branch name), or ('unknown', 'unknown')."""
    def _run(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                args, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return "unknown"

    commit = _run(["git", "rev-parse", "--short", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return commit, branch


# ---------------------------------------------------------------------------
# ExperimentRecord
# ---------------------------------------------------------------------------

class ExperimentRecord:
    """Captures experiment metadata and writes JSON + CSV artefacts on finalize().

    Args:
        cfg: Fully-resolved Hydra DictConfig (all overrides already applied).
        notes: Free-text annotation shown in the summary CSV.
    """

    def __init__(self, cfg: DictConfig, notes: str = "") -> None:
        self._cfg = cfg
        self._data: dict[str, Any] = {}
        self._notes = notes
        self._capture_static(cfg)

    # ------------------------------------------------------------------
    # Static metadata (captured once at init from cfg)
    # ------------------------------------------------------------------

    def _capture_static(self, cfg: DictConfig) -> None:
        commit, branch = _git_info()

        # Identity
        self._data["experiment_name"] = cfg.experiment_name
        self._data["timestamp"] = datetime.now().isoformat(timespec="seconds")
        self._data["git_commit"] = commit
        self._data["git_branch"] = branch

        # Architecture
        self._data["encoder"] = self._leaf(cfg.encoder.get("_target_", "unknown"))
        self._data["bottleneck"] = self._leaf(cfg.bottleneck.get("_target_", "unknown"))
        self._data["decoder"] = self._leaf(cfg.decoder.get("_target_", "unknown"))
        self._data["fusion_mode"] = str(cfg.get("fusion_mode", "concat"))

        skip = cfg.model.get("encoder_skip_layers", None)
        self._data["skip_connection"] = str(skip) if skip else "none"

        loss_target = cfg.loss.get("_target_", "unknown")
        self._data["loss"] = self._leaf(loss_target)

        # Data
        self._data["dataset"] = str(cfg.data.get("_target_", "fivek")).split(".")[-1]
        self._data["image_size"] = cfg.data.get("image_size", None)
        self._data["crop_size"] = cfg.data.get("crop_size", None)
        self._data["train_crop_mode"] = cfg.data.get("train_crop_mode", "none")
        self._data["eval_resize_mode"] = cfg.data.get("eval_resize_mode", "none")

        # Training hyperparameters
        t = cfg.training
        self._data["batch_size"] = t.batch_size_per_device
        accum = int(t.get("accumulate_grad_batches", 1))
        self._data["grad_accum"] = accum
        self._data["effective_batch"] = t.batch_size_per_device * accum
        self._data["lr"] = t.learning_rate
        self._data["weight_decay"] = t.get("weight_decay", 0.0)
        self._data["scheduler"] = t.get("scheduler", "cosine")
        self._data["max_epochs"] = t.max_epochs

        # Artefacts
        log_dir = Path(cfg.logging.log_dir) / cfg.experiment_name
        self._data["log_dir"] = str(log_dir)
        self._data["notes"] = self._notes

        # Placeholders filled in finalize() or build_experiment_dir()
        for key in [
            "precision", "train_size", "val_size", "test_size",
            "total_params", "trainable_params", "gflops",
            "training_time_min", "peak_vram_mb",
            "best_epoch", "val_psnr", "val_ssim", "val_lpips",
            "test_psnr", "test_ssim", "test_lpips", "test_delta_e", "test_nima",
            "checkpoint_path", "experiment_dir",
        ]:
            self._data[key] = ""

    # ------------------------------------------------------------------
    # Finalize: fill results, persist JSON, append CSV
    # ------------------------------------------------------------------

    def finalize(
        self,
        logged_metrics: dict[str, Any],
        trainer: Any = None,
        datamodule: Any = None,
        pipeline: Any = None,
        train_start_time: float | None = None,
    ) -> None:
        """Fill in training results and write all artefacts.

        Args:
            logged_metrics:   trainer.logged_metrics at on_test_end time.
            trainer:          pl.Trainer (precision + checkpoint path).
            datamodule:       LightningDataModule (split sizes).
            pipeline:         ImageEnhancementPipeline (param counts + GFLOPs).
            train_start_time: time.time() recorded at on_train_start.
        """
        import time

        # Precision
        if trainer is not None:
            self._data["precision"] = str(getattr(trainer, "precision", "32"))

        # Split sizes
        if datamodule is not None:
            self._data["train_size"] = len(datamodule._train) if datamodule._train else 0
            self._data["val_size"] = len(datamodule._val) if datamodule._val else 0
            self._data["test_size"] = len(datamodule._test) if datamodule._test else 0

        # Best epoch + checkpoint path from ModelCheckpoint callback
        if trainer is not None:
            ckpt_cb = self._find_checkpoint_callback(trainer)
            if ckpt_cb is not None:
                self._data["best_epoch"] = int(getattr(ckpt_cb, "best_model_epoch", -1) or -1)
                self._data["checkpoint_path"] = str(getattr(ckpt_cb, "best_model_path", ""))

        # Efficiency metrics
        if pipeline is not None:
            self._capture_efficiency(pipeline)

        if train_start_time is not None:
            elapsed_min = (time.time() - train_start_time) / 60.0
            self._data["training_time_min"] = round(elapsed_min, 1)

        try:
            import torch
            if torch.cuda.is_available():
                self._data["peak_vram_mb"] = round(
                    torch.cuda.max_memory_allocated() / 1e6, 1
                )
        except Exception:
            pass

        # Quality metrics from logged_metrics
        for phase, short in [("val", "val"), ("test", "test")]:
            for metric in ["psnr", "ssim", "lpips", "delta_e", "nima"]:
                key = f"{phase}/{metric}"
                col = f"{short}_{metric}"
                val = logged_metrics.get(key, "")
                self._data[col] = float(val) if val != "" else ""

        # Persist
        self._save_json()

    def _capture_efficiency(self, pipeline: Any) -> None:
        """Count parameters and estimate GFLOPs from the pipeline."""
        import torch
        import torch.nn as nn

        total = sum(p.numel() for p in pipeline.parameters())
        trainable = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
        self._data["total_params"] = total
        self._data["trainable_params"] = trainable

        # GFLOPs: one forward pass at a representative 480p landscape resolution
        try:
            from fvcore.nn import FlopCountAnalysis
            image_size = int(self._data.get("image_size") or 480)
            dummy = torch.zeros(1, 3, image_size, round(image_size * 1.5),
                                device=next(pipeline.parameters()).device)
            flops = FlopCountAnalysis(pipeline, dummy)
            flops.unsupported_ops_warnings(False)
            self._data["gflops"] = round(flops.total() / 1e9, 2)
        except Exception:
            self._data["gflops"] = ""

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_json(self) -> None:
        path = Path(self._data["log_dir"]) / "experiment_record.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _leaf(dotted: str) -> str:
        return dotted.split(".")[-1]

    @staticmethod
    def _find_checkpoint_callback(trainer: Any) -> Any:
        for cb in getattr(trainer, "callbacks", []):
            if hasattr(cb, "best_model_path"):
                return cb
        return None

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def data(self) -> dict[str, Any]:
        return dict(self._data)

    def set_notes(self, notes: str) -> None:
        self._data["notes"] = notes
        self._notes = notes

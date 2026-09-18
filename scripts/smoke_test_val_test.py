"""Quick smoke test: compare val and test metrics on small dataset.

This script runs full validation and test on the same model with small dataset
to check if metrics are similar.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.loggers import CSVLogger

from src.lit_module import ColorEnhanceLitModule
from src.utils.checkpoint import allow_trusted_checkpoint_loading
from src.utils.device import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test val vs test metrics")
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument("--config", required=True, help="Path to config.yaml snapshot")
    p.add_argument("--data-dir", default=None, help="Override data directory")
    p.add_argument("--val-samples", type=int, default=20, help="Number of val samples")
    p.add_argument("--test-samples", type=int, default=20, help="Number of test samples")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    allow_trusted_checkpoint_loading()

    # Load config
    cfg = OmegaConf.load(args.config)
    if args.data_dir:
        cfg.data.data_dir = args.data_dir

    # Use small subsets for quick smoke test
    cfg.data.val_subset_size = args.val_samples
    cfg.data.test_subset_size = args.test_samples
    cfg.data.batch_size = 4  # Small batch for faster test
    cfg.data.num_workers = 0  # Avoid multiprocessing issues

    # Disable problematic optional metrics
    if cfg.get("evaluation") is None:
        cfg.evaluation = {}
    cfg.evaluation.enable_clip = False
    cfg.evaluation.enable_nima = False

    # Device
    device_type = (
        get_device() if args.device == "auto" else args.device
    )
    # Map 'mps' to 'cpu' for smoke test to avoid potential issues
    if device_type == "mps":
        device_type = "cpu"

    # Load model
    model = ColorEnhanceLitModule.load_from_checkpoint(
        args.checkpoint, cfg=cfg, map_location=device_type
    )

    # Setup data
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    # Trainer for validation
    val_logger = CSVLogger("outputs/smoke_test", name="val")
    val_trainer = pl.Trainer(
        accelerator=device_type,
        devices=1,
        logger=val_logger,
        callbacks=[ModelSummary(max_depth=0)],
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        max_epochs=1,  # Only run validation
        limit_val_batches=None,  # Run all val batches
    )

    # Run validation
    print(f"\n{'='*70}")
    print("Running VALIDATION on {} samples...".format(args.val_samples))
    print(f"{'='*70}\n")
    val_trainer.validate(model, datamodule=datamodule)

    # Get val metrics
    val_metrics = val_trainer.callback_metrics
    print(f"\n{'='*70}")
    print("Validation Metrics:")
    print(f"{'='*70}")
    for k, v in sorted(val_metrics.items()):
        if k.startswith("val/"):
            print(f"  {k:<20} {v:.6f}")

    # Trainer for test
    test_logger = CSVLogger("outputs/smoke_test", name="test")
    test_trainer = pl.Trainer(
        accelerator=device_type,
        devices=1,
        logger=test_logger,
        callbacks=[ModelSummary(max_depth=0)],
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        max_epochs=1,  # Only run test
        limit_test_batches=None,  # Run all test batches
    )

    # Run test
    print(f"\n{'='*70}")
    print("Running TEST on {} samples...".format(args.test_samples))
    print(f"{'='*70}\n")
    test_trainer.test(model, datamodule=datamodule)

    # Get test metrics
    test_metrics = test_trainer.callback_metrics
    print(f"\n{'='*70}")
    print("Test Metrics:")
    print(f"{'='*70}")
    for k, v in sorted(test_metrics.items()):
        if k.startswith("test/"):
            print(f"  {k:<20} {v:.6f}")

    # Compare val and test metrics
    print(f"\n{'='*70}")
    print("Val vs Test Comparison:")
    print(f"{'='*70}")

    # Map val metric names to test metric names
    metric_pairs = [
        ("val/psnr", "test/psnr"),
        ("val/ssim", "test/ssim"),
        ("val/lpips", "test/lpips"),
        ("val/vgg_perceptual", "test/vgg_perceptual"),
        ("val/loss", "test/loss"),
    ]

    print(f"\n{'Metric':<25} {'Val':<12} {'Test':<12} {'Diff':<15} {'% Diff':<10}")
    print("-" * 80)

    for val_key, test_key in metric_pairs:
        if val_key in val_metrics and test_key in test_metrics:
            val_val = val_metrics[val_key].item()
            test_val = test_metrics[test_key].item()
            diff = test_val - val_val
            pct_diff = (diff / abs(val_val) * 100) if val_val != 0 else 0

            # Color code the difference
            diff_str = f"{diff:+.6f}"
            pct_str = f"{pct_diff:+.2f}%"

            print(f"{val_key.replace('val/', ''):<25} {val_val:<12.6f} {test_val:<12.6f} {diff_str:<15} {pct_str:<10}")

    print("-" * 80)
    print(f"\nDataset sizes: Val={args.val_samples}, Test={args.test_samples}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

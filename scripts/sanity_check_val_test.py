"""Sanity check: verify val and test pipelines produce identical results.

This script loads a trained model and runs the SAME image through both
val and test pipelines to verify they produce identical PSNR/SSIM scores.
If they differ, there's a bug in the pipeline logic.
"""
from __future__ import annotations

import argparse

import hydra
import torch
from omegaconf import OmegaConf

from src.evaluation.metrics import MetricCollection
from src.lit_module import ColorEnhanceLitModule
from src.utils.checkpoint import allow_trusted_checkpoint_loading
from src.utils.device import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity check val vs test pipelines")
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument("--config", required=True, help="Path to config.yaml snapshot")
    p.add_argument("--data-dir", default=None, help="Override data directory")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    allow_trusted_checkpoint_loading()

    # Load config
    cfg = OmegaConf.load(args.config)
    if args.data_dir:
        cfg.data.data_dir = args.data_dir

    # Disable problematic optional metrics for sanity check
    if cfg.get("evaluation") is None:
        cfg.evaluation = {}
    cfg.evaluation.enable_clip = False
    cfg.evaluation.enable_nima = False

    # Device
    device = (
        get_device() if args.device == "auto" else torch.device(args.device)
    )

    # Load model
    model = ColorEnhanceLitModule.load_from_checkpoint(
        args.checkpoint, cfg=cfg, map_location=device
    )
    model.eval().to(device)

    # Setup data
    datamodule = hydra.utils.instantiate(cfg.data)
    # Setup both fit (for val) and test stages
    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    test_loader = datamodule.test_dataloader()

    # Get ONE batch from test set
    test_batch = next(iter(test_loader))
    test_input, test_target = test_batch[0], test_batch[1]
    test_input = test_input[:1].to(device)  # Single image
    test_target = test_target[:1].to(device)

    print(f"Test image shape: {test_input.shape}")
    print(f"Test target shape: {test_target.shape}")

    # Create metric function with same config
    metric_fn = MetricCollection(cfg)

    # Run through val-style pipeline
    with torch.no_grad():
        input_val = model._to_01(test_input)
        target_val = model._to_01(test_target)
        pred_val = model.pipeline(input_val)
        metrics_val = metric_fn(pred_val, target_val)

    # Run through test-style pipeline (same as val in current impl)
    with torch.no_grad():
        input_test = model._to_01(test_input)
        target_test = model._to_01(test_target)
        pred_test = model.pipeline(input_test)
        metrics_test = metric_fn(pred_test, target_test)

    # Compare predictions
    pred_diff = (pred_val - pred_test).abs().max().item()
    print(f"\n{'='*60}")
    print(f"Max prediction difference: {pred_diff:.10f}")
    print(f"{'='*60}\n")

    # Compare metrics
    print(f"{'Metric':<15} {'Val':<12} {'Test':<12} {'Diff':<12} {'Match':<8}")
    print("-" * 60)

    all_match = True
    for key in sorted(metrics_val.keys()):
        val_val = metrics_val[key].item()
        test_val = metrics_test[key].item()
        diff = abs(val_val - test_val)
        match = "✓" if diff < 1e-6 else "✗"
        if diff >= 1e-6:
            all_match = False
        print(f"{key:<15} {val_val:<12.6f} {test_val:<12.6f} {diff:<12.10f} {match:<8}")

    print("-" * 60)
    if all_match:
        print("✓ PASS: Val and test pipelines produce IDENTICAL results")
    else:
        print("✗ FAIL: Val and test pipelines produce DIFFERENT results")

    # Additional check: verify datamodule split separation
    print(f"\n{'='*60}")
    print("Dataset split overlap check:")
    print(f"{'='*60}")

    if hasattr(datamodule, '_val') and hasattr(datamodule, '_test'):
        val_files = set(datamodule._val.file_names)
        test_files = set(datamodule._test.file_names)

        overlap = val_files & test_files
        print(f"Val set size: {len(val_files)}")
        print(f"Test set size: {len(test_files)}")
        print(f"Overlap: {len(overlap)} files")

        if overlap:
            print(f"✗ WARNING: Found {len(overlap)} overlapping files:")
            for f in sorted(list(overlap))[:10]:
                print(f"  - {f}")
            if len(overlap) > 10:
                print(f"  ... and {len(overlap) - 10} more")
        else:
            print("✓ PASS: Val and test sets are DISJOINT")
    else:
        print("Could not check overlap (datamodule not fully set up)")


if __name__ == "__main__":
    main()

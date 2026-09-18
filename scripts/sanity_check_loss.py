"""Sanity check: verify val and test produce identical loss on the same data.

This script verifies that loss computation is identical between val and test modes
by running the same batch through both paths.
"""
from __future__ import annotations

import argparse

import hydra
import torch
from omegaconf import OmegaConf

from src.lit_module import ColorEnhanceLitModule
from src.utils.checkpoint import allow_trusted_checkpoint_loading
from src.utils.device import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity check val vs test loss")
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

    # Disable problematic optional metrics
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
    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    test_loader = datamodule.test_dataloader()

    # Get ONE batch from test set
    test_batch = next(iter(test_loader))
    test_input, test_target = test_batch[0], test_batch[1]
    test_input = test_input[:1].to(device)
    test_target = test_target[:1].to(device)

    print(f"Test image shape: {test_input.shape}")
    print(f"Model training flag: {model.training}")

    # Simulate validation_step loss computation
    model.eval()  # Set to eval mode (training=False)
    with torch.no_grad():
        loss_val, pred_val, target_val, _, _, raw_losses_val, weighted_losses_val, weights_val = model._shared_step(
            (test_input, test_target)
        )

    # Simulate test_step loss computation
    model.eval()  # Ensure still in eval mode
    with torch.no_grad():
        loss_test, pred_test, target_test, _, _, raw_losses_test, weighted_losses_test, weights_test = model._shared_step(
            (test_input, test_target)
        )

    # Compare results
    print(f"\n{'='*70}")
    print("Loss Comparison (Val vs Test):")
    print(f"{'='*70}")

    # Total loss
    loss_diff = abs(loss_val.item() - loss_test.item())
    print("\nTotal Loss:")
    print(f"  Val:  {loss_val.item():.6f}")
    print(f"  Test: {loss_test.item():.6f}")
    print(f"  Diff: {loss_diff:.10f}")
    print(f"  Match: {'✓' if loss_diff < 1e-6 else '✗'}")

    # Raw losses breakdown
    print(f"\n{'='*70}")
    print("Raw Loss Breakdown:")
    print(f"{'='*70}")

    all_loss_keys = set(raw_losses_val.keys()) | set(raw_losses_test.keys())
    all_match = True

    for key in sorted(all_loss_keys):
        val_val = raw_losses_val.get(key, torch.tensor(0.0)).item()
        test_val = raw_losses_test.get(key, torch.tensor(0.0)).item()
        diff = abs(val_val - test_val)
        match = "✓" if diff < 1e-6 else "✗"
        if diff >= 1e-6:
            all_match = False
        print(f"  {key:<20} Val: {val_val:<12.6f} Test: {test_val:<12.6f} Diff: {diff:<12.10f} {match}")

    # Weighted losses breakdown
    print(f"\n{'='*70}")
    print("Weighted Loss Breakdown:")
    print(f"{'='*70}")

    all_weighted_keys = set(weighted_losses_val.keys()) | set(weighted_losses_test.keys())

    for key in sorted(all_weighted_keys):
        val_val = weighted_losses_val.get(key, torch.tensor(0.0)).item()
        test_val = weighted_losses_test.get(key, torch.tensor(0.0)).item()
        diff = abs(val_val - test_val)
        match = "✓" if diff < 1e-6 else "✗"
        if diff >= 1e-6:
            all_match = False
        print(f"  {key:<20} Val: {val_val:<12.6f} Test: {test_val:<12.6f} Diff: {diff:<12.10f} {match}")

    # Adaptive weights (if any)
    if weights_val and weights_test:
        print(f"\n{'='*70}")
        print("Adaptive Weights:")
        print(f"{'='*70}")

        all_weight_keys = set(weights_val.keys()) | set(weights_test.keys())

        for key in sorted(all_weight_keys):
            val_val = weights_val.get(key, torch.tensor(0.0)).item()
            test_val = weights_test.get(key, torch.tensor(0.0)).item()
            diff = abs(val_val - test_val)
            match = "✓" if diff < 1e-6 else "✗"
            if diff >= 1e-6:
                all_match = False
            print(f"  {key:<20} Val: {val_val:<12.6f} Test: {test_val:<12.6f} Diff: {diff:<12.10f} {match}")

    # Summary
    print(f"\n{'='*70}")
    if loss_diff < 1e-6 and all_match:
        print("✓ PASS: Val and Test loss computation is IDENTICAL")
    else:
        print("✗ FAIL: Val and Test loss computation DIFFERS")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

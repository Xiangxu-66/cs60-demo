"""Check dataset splits and verify they are disjoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import hydra
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check dataset splits")
    p.add_argument("--config", required=True, help="Path to config.yaml snapshot")
    p.add_argument("--data-dir", default=None, help="Override data directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)
    if args.data_dir:
        cfg.data.data_dir = args.data_dir

    # Setup data
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    print(f"{'='*70}")
    print(f"Dataset: {cfg.data.data_dir}")
    print(f"Expert: {cfg.data.expert}")
    print(f"{'='*70}")

    # Get file names from each split
    train_names = set(datamodule._train.file_names) if datamodule._train else set()
    val_names = set(datamodule._val.file_names) if datamodule._val else set()
    test_names = set(datamodule._test.file_names) if datamodule._test else set()

    print(f"\nTrain set: {len(train_names)} images")
    print(f"Val set:   {len(val_names)} images")
    print(f"Test set:  {len(test_names)} images")

    # Check for overlaps
    print(f"\n{'='*70}")
    print("Overlap Analysis:")
    print(f"{'='*70}")

    train_val_overlap = train_names & val_names
    train_test_overlap = train_names & test_names
    val_test_overlap = val_names & test_names
    all_three_overlap = train_names & val_names & test_names

    print(f"Train ∩ Val:   {len(train_val_overlap)} images")
    print(f"Train ∩ Test:  {len(train_test_overlap)} images")
    print(f"Val ∩ Test:    {len(val_test_overlap)} images")
    print(f"Train ∩ Val ∩ Test: {len(all_three_overlap)} images")

    if train_val_overlap:
        print(f"\n✗ WARNING: {len(train_val_overlap)} images appear in BOTH train and val:")
        for f in sorted(list(train_val_overlap))[:10]:
            print(f"  - {f}")
        if len(train_val_overlap) > 10:
            print(f"  ... and {len(train_val_overlap) - 10} more")
    else:
        print(f"\n✓ Train and Val are DISJOINT")

    if train_test_overlap:
        print(f"\n✗ WARNING: {len(train_test_overlap)} images appear in BOTH train and test:")
        for f in sorted(list(train_test_overlap))[:10]:
            print(f"  - {f}")
        if len(train_test_overlap) > 10:
            print(f"  ... and {len(train_test_overlap) - 10} more")
    else:
        print(f"✓ Train and Test are DISJOINT")

    if val_test_overlap:
        print(f"\n✗ WARNING: {len(val_test_overlap)} images appear in BOTH val and test:")
        for f in sorted(list(val_test_overlap))[:10]:
            print(f"  - {f}")
        if len(val_test_overlap) > 10:
            print(f"  ... and {len(val_test_overlap) - 10} more")
    else:
        print(f"✓ Val and Test are DISJOINT")

    # Show sample images from each split
    print(f"\n{'='*70}")
    print("Sample Images (first 10 from each split):")
    print(f"{'='*70}")

    print(f"\nTrain samples:")
    for f in sorted(list(train_names))[:10]:
        print(f"  {f}")

    print(f"\nVal samples:")
    for f in sorted(list(val_names))[:10]:
        print(f"  {f}")

    print(f"\nTest samples:")
    for f in sorted(list(test_names))[:10]:
        print(f"  {f}")

    # Summary
    print(f"\n{'='*70}")
    if not train_val_overlap and not train_test_overlap and not val_test_overlap:
        print("✓ ALL SPLITS ARE DISJOINT - No data leakage detected")
    else:
        print("✗ DATA LEAKAGE DETECTED - Some images appear in multiple splits!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

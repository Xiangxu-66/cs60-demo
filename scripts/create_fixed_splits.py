"""Create reproducible FiveK split files.

Supports two workflows:

1. Seeded ratio split:
   shuffle all images with a fixed seed, then split by train/val/test ratios.
2. Paper-aligned fixed test split:
   keep a provided test list untouched, then sample a fixed-size validation set
   from the remaining images with a fixed seed.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Sequence

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create fixed dataset splits")
    p.add_argument("--data-dir", required=True, help="Path to dataset root")
    p.add_argument("--output-dir", default=None, help="Output directory for split files")
    p.add_argument("--expert", default="c", help="Expert letter (a-e)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    # Seeded ratio mode (default fallback).
    p.add_argument("--train-ratio", type=float, default=0.8, help="Training set ratio")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Validation set ratio")
    p.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio")

    # Paper-aligned mode: fixed test list + validation sampled from remaining.
    p.add_argument(
        "--test-list",
        default=None,
        help=(
            "Path to an explicit test split file. When provided, the script "
            "keeps those images as test and samples validation images from the "
            "remaining pool with --seed."
        ),
    )
    p.add_argument(
        "--val-size",
        type=int,
        default=None,
        help=(
            "Validation set size when --test-list is used. Defaults to the same "
            "size as the explicit test split."
        ),
    )
    return p.parse_args()


def load_name_list(path: str | Path) -> list[str]:
    """Load sample names from a text file, normalizing any extensions away."""
    file_path = Path(path)
    with open(file_path) as f:
        return [Path(line.strip()).stem for line in f if line.strip()]


def _ensure_unique_names(names: Sequence[str], source: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        dup_preview = ", ".join(sorted(duplicates)[:5])
        raise ValueError(f"Duplicate sample names in {source}: {dup_preview}")


def find_image_files(data_dir: Path, expert: str) -> list[str]:
    """Find image names that exist in both the input and target directories."""
    input_candidates = ["input", "raw"]
    input_dir = None
    for name in input_candidates:
        path = data_dir / name
        if path.is_dir():
            input_dir = path
            break

    if input_dir is None:
        raise FileNotFoundError(f"Could not find input directory in {data_dir}")

    file_names = sorted(
        p.stem
        for p in input_dir.glob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    expert_upper = expert.upper()
    expert_lower = expert.lower()
    target_candidates = [
        f"expert{expert_lower}",
        expert_lower,
        f"expert{expert_upper}",
        expert_upper,
    ]
    target_dir = None
    for name in target_candidates:
        path = data_dir / name
        if path.is_dir():
            target_dir = path
            break

    if target_dir is None:
        raise FileNotFoundError(
            f"Could not find target directory for expert {expert_upper}"
        )

    valid_files = []
    for name in file_names:
        found = False
        for ext in [".png", ".jpg", ".jpeg"]:
            if (target_dir / f"{name}{ext}").exists():
                found = True
                break
        if found:
            valid_files.append(name)

    print(f"Found {len(valid_files)} valid images in {data_dir}")
    return valid_files


def create_seeded_ratio_splits(
    all_files: Sequence[str],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
    """Shuffle the full dataset once with a fixed seed and split by ratios."""
    ratios = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }
    if any(value < 0 for value in ratios.values()):
        raise ValueError("Split ratios must be non-negative")

    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("train_ratio + val_ratio + test_ratio must be > 0")

    shuffled = list(all_files)
    random.Random(seed).shuffle(shuffled)

    train_norm = train_ratio / total_ratio
    val_norm = val_ratio / total_ratio
    n_train = int(len(shuffled) * train_norm)
    n_val = int(len(shuffled) * val_norm)

    train_files = shuffled[:n_train]
    val_files = shuffled[n_train:n_train + n_val]
    test_files = shuffled[n_train + n_val:]

    metadata = {
        "split_strategy": "seeded_ratios",
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "normalized_train_ratio": train_norm,
        "normalized_val_ratio": val_norm,
        "normalized_test_ratio": test_ratio / total_ratio,
    }
    return train_files, val_files, test_files, metadata


def create_fixed_test_splits(
    all_files: Sequence[str],
    *,
    fixed_test_names: Sequence[str],
    seed: int,
    val_size: int | None = None,
) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
    """Keep an explicit test split and sample validation from the remainder."""
    explicit_test = [Path(name).stem for name in fixed_test_names]
    _ensure_unique_names(explicit_test, "explicit test list")

    dataset_names = list(all_files)
    dataset_name_set = set(dataset_names)
    missing = sorted(name for name in explicit_test if name not in dataset_name_set)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Explicit test split contains unknown samples: {preview}")

    resolved_val_size = len(explicit_test) if val_size is None else val_size
    if resolved_val_size < 0:
        raise ValueError("val_size must be >= 0")

    explicit_test_set = set(explicit_test)
    remaining = [name for name in dataset_names if name not in explicit_test_set]
    if resolved_val_size > len(remaining):
        raise ValueError(
            f"val_size={resolved_val_size} exceeds remaining pool size={len(remaining)}"
        )

    shuffled_remaining = list(remaining)
    random.Random(seed).shuffle(shuffled_remaining)
    val_files = shuffled_remaining[:resolved_val_size]
    train_files = shuffled_remaining[resolved_val_size:]

    metadata = {
        "split_strategy": "fixed_test_from_file",
        "seed": seed,
        "explicit_test_size": len(explicit_test),
        "val_size": resolved_val_size,
    }
    return train_files, val_files, explicit_test, metadata


def write_split_artifacts(
    splits_dir: Path,
    *,
    train_files: Sequence[str],
    val_files: Sequence[str],
    test_files: Sequence[str],
    metadata: dict[str, Any],
) -> Path:
    """Write split text files plus metadata.yaml."""
    splits_dir.mkdir(parents=True, exist_ok=True)

    split_files = {
        "train.txt": list(train_files),
        "val.txt": list(val_files),
        "test.txt": list(test_files),
    }
    for file_name, names in split_files.items():
        with open(splits_dir / file_name, "w") as f:
            f.write("\n".join(names))

    full_metadata = {
        "total_images": len(train_files) + len(val_files) + len(test_files),
        "train_size": len(train_files),
        "val_size": len(val_files),
        "test_size": len(test_files),
        "train_samples": list(train_files)[:5],
        "val_samples": list(val_files)[:5],
        "test_samples": list(test_files)[:5],
        **metadata,
    }

    metadata_file = splits_dir / "metadata.yaml"
    with open(metadata_file, "w") as f:
        yaml.safe_dump(full_metadata, f, default_flow_style=False, sort_keys=False)

    return metadata_file


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    splits_dir = data_dir / "splits" if args.output_dir is None else Path(args.output_dir)

    all_files = find_image_files(data_dir, args.expert)

    if args.test_list is not None:
        explicit_test_names = load_name_list(args.test_list)
        train_files, val_files, test_files, metadata = create_fixed_test_splits(
            all_files,
            fixed_test_names=explicit_test_names,
            seed=args.seed,
            val_size=args.val_size,
        )
        metadata["test_list_path"] = str(Path(args.test_list).resolve())
    else:
        train_files, val_files, test_files, metadata = create_seeded_ratio_splits(
            all_files,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    print("\nSplit sizes:")
    print(f"  Train: {len(train_files)} ({len(train_files)/len(all_files)*100:.1f}%)")
    print(f"  Val:   {len(val_files)} ({len(val_files)/len(all_files)*100:.1f}%)")
    print(f"  Test:  {len(test_files)} ({len(test_files)/len(all_files)*100:.1f}%)")

    metadata_file = write_split_artifacts(
        splits_dir,
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        metadata=metadata,
    )

    print("\nSplit files written to:")
    print(f"  {splits_dir / 'train.txt'}")
    print(f"  {splits_dir / 'val.txt'}")
    print(f"  {splits_dir / 'test.txt'}")
    print(f"  {metadata_file}")


if __name__ == "__main__":
    main()

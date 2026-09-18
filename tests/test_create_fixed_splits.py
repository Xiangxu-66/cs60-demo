from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from PIL import Image

from scripts.create_fixed_splits import (
    create_fixed_test_splits,
    create_seeded_ratio_splits,
    find_image_files,
    load_name_list,
    write_split_artifacts,
)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def _build_fivek_dirs(root: Path, *, count: int) -> list[str]:
    names = []
    for idx in range(1, count + 1):
        name = f"a{idx:04d}"
        names.append(name)
        _write_image(root / "raw" / f"{name}.jpg", (idx, idx, idx))
        _write_image(root / "c" / f"{name}.jpg", (255 - idx, 0, idx))
    return names


def test_seeded_ratio_splits_are_reproducible_and_disjoint(tmp_path: Path) -> None:
    _build_fivek_dirs(tmp_path, count=10)
    all_files = find_image_files(tmp_path, "c")

    split_a = create_seeded_ratio_splits(
        all_files,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=7,
    )
    split_b = create_seeded_ratio_splits(
        all_files,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=7,
    )

    train_files, val_files, test_files, metadata = split_a
    assert split_a == split_b
    assert len(train_files) == 6
    assert len(val_files) == 2
    assert len(test_files) == 2
    assert set(train_files).isdisjoint(val_files)
    assert set(train_files).isdisjoint(test_files)
    assert set(val_files).isdisjoint(test_files)
    assert set(train_files) | set(val_files) | set(test_files) == set(all_files)
    assert metadata["split_strategy"] == "seeded_ratios"


def test_fixed_test_splits_preserve_test_list_and_default_val_size(tmp_path: Path) -> None:
    names = _build_fivek_dirs(tmp_path, count=12)
    test_list = tmp_path / "paper_test.txt"
    test_list.write_text("\n".join([f"{names[1]}.jpg", names[4], names[7]]))

    all_files = find_image_files(tmp_path, "c")
    train_files, val_files, test_files, metadata = create_fixed_test_splits(
        all_files,
        fixed_test_names=load_name_list(test_list),
        seed=11,
        val_size=None,
    )

    assert test_files == [names[1], names[4], names[7]]
    assert len(val_files) == len(test_files)
    assert len(train_files) == len(names) - len(val_files) - len(test_files)
    assert set(train_files).isdisjoint(val_files)
    assert set(train_files).isdisjoint(test_files)
    assert set(val_files).isdisjoint(test_files)
    assert set(train_files) | set(val_files) | set(test_files) == set(names)
    assert metadata["split_strategy"] == "fixed_test_from_file"
    assert metadata["explicit_test_size"] == 3
    assert metadata["val_size"] == 3


def test_fixed_test_splits_reject_invalid_explicit_lists(tmp_path: Path) -> None:
    names = _build_fivek_dirs(tmp_path, count=8)
    all_files = find_image_files(tmp_path, "c")

    with pytest.raises(ValueError, match="unknown samples"):
        create_fixed_test_splits(
            all_files,
            fixed_test_names=[names[0], "missing_sample"],
            seed=0,
            val_size=2,
        )

    with pytest.raises(ValueError, match="Duplicate sample names"):
        create_fixed_test_splits(
            all_files,
            fixed_test_names=[names[0], names[0]],
            seed=0,
            val_size=2,
        )


def test_write_split_artifacts_records_metadata(tmp_path: Path) -> None:
    splits_dir = tmp_path / "splits"
    metadata_file = write_split_artifacts(
        splits_dir,
        train_files=["a0001", "a0002"],
        val_files=["a0003"],
        test_files=["a0004"],
        metadata={
            "split_strategy": "fixed_test_from_file",
            "seed": 42,
            "test_list_path": "/tmp/paper_test.txt",
        },
    )

    assert (splits_dir / "train.txt").read_text().splitlines() == ["a0001", "a0002"]
    assert (splits_dir / "val.txt").read_text().splitlines() == ["a0003"]
    assert (splits_dir / "test.txt").read_text().splitlines() == ["a0004"]

    metadata = yaml.safe_load(metadata_file.read_text())
    assert metadata["total_images"] == 4
    assert metadata["train_size"] == 2
    assert metadata["val_size"] == 1
    assert metadata["test_size"] == 1
    assert metadata["split_strategy"] == "fixed_test_from_file"
    assert metadata["seed"] == 42
    assert metadata["test_list_path"] == "/tmp/paper_test.txt"

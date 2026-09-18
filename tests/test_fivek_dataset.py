from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch
from PIL import Image

pytest.importorskip("torchvision")

import src.data.fivek_dataset as fivek_dataset
from src.data.fivek_dataset import (
    FiveKDataModule,
    FiveKDataset,
    _pad_image_tensor,
    build_padded_collate_fn,
)
from src.data.transforms import (
    RandomRatioCrop,
    get_train_transforms,
    get_val_transforms,
    to_image_range,
)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def _build_fivek_dirs(
    root: Path,
    *,
    input_dir: str,
    target_dir: str,
    count: int,
) -> None:
    for idx in range(count):
        name = f"a{idx:04d}.jpg"
        _write_image(root / input_dir / name, (idx, idx, idx))
        _write_image(root / target_dir / name, (255 - idx, 0, idx))


def _write_three_way_splits(
    splits_dir: Path,
    *,
    train: list[str],
    val: list[str],
    test: list[str],
) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.txt").write_text("\n".join(train) + "\n")
    (splits_dir / "val.txt").write_text("\n".join(val) + "\n")
    (splits_dir / "test.txt").write_text("\n".join(test) + "\n")


def test_fivek_dataset_supports_raw_and_letter_expert_dirs(tmp_path: Path) -> None:
    _build_fivek_dirs(tmp_path, input_dir="raw", target_dir="c", count=6)

    dataset = FiveKDataset(tmp_path, split="train", expert="C", max_items=3)

    assert dataset.input_dir.name == "raw"
    assert dataset.target_dir.name == "c"
    assert len(dataset) == 3

    inputs, targets = dataset[0]
    assert tuple(inputs.shape) == (3, 8, 8)
    assert tuple(targets.shape) == (3, 8, 8)


def test_fivek_dataset_supports_original_and_expertc_dirs(tmp_path: Path) -> None:
    _build_fivek_dirs(tmp_path, input_dir="Original", target_dir="expertC", count=6)

    dataset = FiveKDataset(tmp_path, split="train", expert="C", max_items=3)

    assert dataset.input_dir.name.lower() == "original"
    assert dataset.target_dir.name.lower() == "expertc"
    assert len(dataset) == 3

    inputs, targets = dataset[0]
    assert tuple(inputs.shape) == (3, 8, 8)
    assert tuple(targets.shape) == (3, 8, 8)


def test_fivek_datamodule_respects_split_limits(tmp_path: Path) -> None:
    _build_fivek_dirs(tmp_path, input_dir="input", target_dir="expertC", count=30)

    datamodule = FiveKDataModule(
        data_dir=str(tmp_path),
        train_subset_size=5,
        val_subset_size=2,
        test_subset_size=2,
        image_size=8,
        crop_size=8,
        use_augmentation=False,
        batch_size=2,
        num_workers=0,
    )

    datamodule.setup(stage="fit")
    assert len(datamodule._train) == 5
    assert len(datamodule._val) == 2

    datamodule.setup(stage="test")
    assert len(datamodule._test) == 2


def test_fivek_datamodule_keeps_legacy_subset_args(tmp_path: Path) -> None:
    _build_fivek_dirs(tmp_path, input_dir="input", target_dir="expertC", count=50)

    datamodule = FiveKDataModule(
        data_dir=str(tmp_path),
        train_split=4,
        test_split=3,
        image_size=8,
        crop_size=8,
        use_augmentation=False,
        batch_size=2,
        num_workers=0,
    )

    datamodule.setup(stage="fit")
    assert len(datamodule._train) == 4
    assert len(datamodule._val) == 3

    datamodule.setup(stage="test")
    assert len(datamodule._test) == 3


def test_fivek_datamodule_uses_three_way_ratio_split(tmp_path: Path) -> None:
    _build_fivek_dirs(tmp_path, input_dir="input", target_dir="expertC", count=20)

    datamodule = FiveKDataModule(
        data_dir=str(tmp_path),
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        image_size=8,
        crop_size=8,
        use_augmentation=False,
        batch_size=2,
        num_workers=0,
    )

    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    assert len(datamodule._train) == 16
    assert len(datamodule._val) == 2
    assert len(datamodule._test) == 2


def test_fivek_datamodule_prefers_repo_config_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_fivek_dirs(tmp_path, input_dir="input", target_dir="expertC", count=6)
    repo_splits = tmp_path / "repo" / "configs" / "splits" / "five"
    _write_three_way_splits(
        repo_splits,
        train=["a0003", "a0001"],
        val=["a0004"],
        test=["a0005"],
    )
    monkeypatch.setattr(fivek_dataset, "_CONFIG_SPLITS_DIR", repo_splits)

    datamodule = FiveKDataModule(
        data_dir=str(tmp_path),
        image_size=8,
        crop_size=8,
        use_augmentation=False,
        batch_size=2,
        num_workers=0,
    )

    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    assert datamodule._train.file_names == ["a0003", "a0001"]
    assert datamodule._val.file_names == ["a0004"]
    assert datamodule._test.file_names == ["a0005"]


def test_to_image_range_uses_explicit_normalization_flag() -> None:
    x = torch.tensor([[[[-1.0, 0.0, 1.0]]]])

    converted = to_image_range(x, normalized_to_neg_one_one=True)
    assert torch.allclose(converted, torch.tensor([[[[0.0, 0.5, 1.0]]]]))

    untouched = to_image_range(converted, normalized_to_neg_one_one=False)
    assert torch.allclose(untouched, converted)


def test_fivek_datamodule_can_disable_minus1_to_1_normalization(tmp_path: Path) -> None:
    _build_fivek_dirs(tmp_path, input_dir="input", target_dir="expertC", count=10)

    datamodule = FiveKDataModule(
        data_dir=str(tmp_path),
        train_subset_size=2,
        val_subset_size=1,
        test_subset_size=1,
        image_size=8,
        crop_size=8,
        normalize_to_neg_one_one=False,
        use_augmentation=False,
        batch_size=1,
        num_workers=0,
    )

    datamodule.setup(stage="fit")
    inputs, targets = datamodule._train[0]

    assert 0.0 <= float(inputs.min()) <= 1.0
    assert 0.0 <= float(targets.min()) <= 1.0
    assert 0.0 <= float(inputs.max()) <= 1.0
    assert 0.0 <= float(targets.max()) <= 1.0


def test_fivek_datamodule_can_overfit_on_train_subset(tmp_path: Path) -> None:
    _build_fivek_dirs(tmp_path, input_dir="input", target_dir="expertC", count=20)

    datamodule = FiveKDataModule(
        data_dir=str(tmp_path),
        train_subset_size=4,
        val_subset_size=2,
        test_subset_size=2,
        image_size=8,
        crop_size=8,
        normalize_to_neg_one_one=False,
        overfit_on_train_subset=True,
        use_augmentation=False,
        batch_size=2,
        num_workers=0,
    )

    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    assert len(datamodule._train) == 4
    assert len(datamodule._val) == 4
    assert len(datamodule._test) == 4
    assert datamodule._train.file_names == datamodule._val.file_names
    assert datamodule._train.file_names == datamodule._test.file_names


def test_fivek_datamodule_can_merge_train_val_and_disable_validation(
    tmp_path: Path,
) -> None:
    _build_fivek_dirs(tmp_path, input_dir="input", target_dir="expertC", count=20)

    datamodule = FiveKDataModule(
        data_dir=str(tmp_path),
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        image_size=8,
        crop_size=8,
        normalize_to_neg_one_one=False,
        merge_train_val_for_final_fit=True,
        use_val_split=False,
        use_augmentation=False,
        batch_size=2,
        num_workers=0,
    )

    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    assert len(datamodule._train) == 18
    assert datamodule._val is None
    assert datamodule.val_dataloader() is None
    assert len(datamodule._test) == 2


def test_padded_collate_fn_preserves_valid_sizes_and_rounds_up() -> None:
    collate = build_padded_collate_fn(pad_to_multiple=8)
    a = torch.zeros(3, 8, 10)
    b = torch.zeros(3, 6, 12)

    inputs, targets, sizes = collate([(a, a), (b, b)])

    assert tuple(inputs.shape) == (2, 3, 8, 16)
    assert tuple(targets.shape) == (2, 3, 8, 16)
    assert sizes.tolist() == [[8, 10], [6, 12]]


def test_padded_collate_fn_preserves_sample_names() -> None:
    collate = build_padded_collate_fn(pad_to_multiple=8)
    a = torch.zeros(3, 8, 10)
    b = torch.zeros(3, 6, 12)

    inputs, targets, sizes, names = collate([(a, a, "a0001"), (b, b, "a0002")])

    assert tuple(inputs.shape) == (2, 3, 8, 16)
    assert tuple(targets.shape) == (2, 3, 8, 16)
    assert sizes.tolist() == [[8, 10], [6, 12]]
    assert names == ["a0001", "a0002"]


def test_padded_collate_fn_is_picklable_for_spawn_workers() -> None:
    collate = build_padded_collate_fn(pad_to_multiple=8)

    restored = pickle.loads(pickle.dumps(collate))

    assert restored.pad_to_multiple == 8


def test_val_transforms_can_keep_original_resolution() -> None:
    img = Image.new("RGB", (10, 6), color=(128, 64, 32))
    tfm = get_val_transforms(
        image_size=None,
        normalize_to_neg_one_one=False,
        resize_mode="none",
    )

    tensor = tfm(img)

    assert tuple(tensor.shape) == (3, 6, 10)


def test_train_transforms_can_use_ratio_crop_mode() -> None:
    img = Image.new("RGB", (10, 6), color=(128, 64, 32))
    tfm = get_train_transforms(
        image_size=8,
        crop_mode="ratio",
        ratio_crop_min_scale=0.5,
        ratio_crop_max_scale=1.0,
        normalize_to_neg_one_one=False,
        enable_horizontal_flip=False,
    )

    torch.manual_seed(0)
    tensor = tfm(img)

    assert 4 <= tensor.shape[1] <= 8
    assert 6 <= tensor.shape[2] <= 13


# ---------------------------------------------------------------------------
# P1 – ratio crop wired through DataModule
# ---------------------------------------------------------------------------

def test_datamodule_ratio_crop_mode_injects_random_ratio_crop(tmp_path: Path) -> None:
    _build_fivek_dirs(tmp_path, input_dir="input", target_dir="expertC", count=10)

    datamodule = FiveKDataModule(
        data_dir=str(tmp_path),
        image_size=8,
        train_crop_mode="ratio",
        ratio_crop_min_scale=0.5,
        ratio_crop_max_scale=1.0,
        use_augmentation=False,
        batch_size=2,
        num_workers=0,
    )
    datamodule.setup(stage="fit")

    train_transforms = datamodule._train.transform.transforms
    crop_ops = [t for t in train_transforms if isinstance(t, RandomRatioCrop)]
    assert len(crop_ops) == 1, "Expected exactly one RandomRatioCrop in train transform"
    assert crop_ops[0].min_scale == 0.5
    assert crop_ops[0].max_scale == 1.0


def test_ratio_crop_paired_seeding_gives_identical_spatial_shape(tmp_path: Path) -> None:
    # Input and target must receive the same random crop (same seed applied twice).
    _build_fivek_dirs(tmp_path, input_dir="input", target_dir="expertC", count=10)

    datamodule = FiveKDataModule(
        data_dir=str(tmp_path),
        image_size=8,
        train_crop_mode="ratio",
        ratio_crop_min_scale=0.5,
        ratio_crop_max_scale=1.0,
        use_augmentation=False,
        batch_size=2,
        num_workers=0,
    )
    datamodule.setup(stage="fit")

    for idx in range(len(datamodule._train)):
        inp, tgt = datamodule._train[idx]
        assert inp.shape == tgt.shape, (
            f"Sample {idx}: input shape {inp.shape} != target shape {tgt.shape}"
        )


# ---------------------------------------------------------------------------
# P2 – train/val batches carry sizes; test also carries file names
# ---------------------------------------------------------------------------

def test_dataloaders_return_sizes_and_test_names(tmp_path: Path) -> None:
    _build_fivek_dirs(tmp_path, input_dir="input", target_dir="expertC", count=20)

    datamodule = FiveKDataModule(
        data_dir=str(tmp_path),
        image_size=8,
        train_crop_mode="none",
        use_augmentation=False,
        batch_size=4,
        num_workers=0,
    )
    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    train_batch = next(iter(datamodule.train_dataloader()))
    assert len(train_batch) == 3, "train_dataloader must return (inputs, targets, sizes)"
    inputs, targets, sizes = train_batch
    assert sizes.shape == (inputs.shape[0], 2)

    val_batch = next(iter(datamodule.val_dataloader()))
    assert len(val_batch) == 3, "val_dataloader must return (inputs, targets, sizes)"

    test_batch = next(iter(datamodule.test_dataloader()))
    assert len(test_batch) == 4, "test_dataloader must return (inputs, targets, sizes, names)"
    test_inputs, _test_targets, test_sizes, names = test_batch
    assert test_sizes.shape == (test_inputs.shape[0], 2)
    assert len(names) == test_inputs.shape[0]
    assert all(isinstance(name, str) for name in names)


# ---------------------------------------------------------------------------
# P2 – _pad_image_tensor falls back to replicate when pad >= image dimension
# ---------------------------------------------------------------------------

def test_pad_image_tensor_uses_replicate_when_pad_equals_or_exceeds_dimension() -> None:
    # pad_h == h  → reflect would require pad < h, so replicate must be used
    x = torch.zeros(3, 4, 6)
    padded = _pad_image_tensor(x, target_h=8, target_w=12)  # pad_h=4==h, pad_w=6==w
    assert tuple(padded.shape) == (3, 8, 12)

    # pad_h > h
    x2 = torch.zeros(3, 2, 4)
    padded2 = _pad_image_tensor(x2, target_h=8, target_w=8)  # pad_h=6>h=2
    assert tuple(padded2.shape) == (3, 8, 8)


def test_pad_image_tensor_uses_reflect_when_pad_is_small() -> None:
    # Build a non-uniform tensor so reflect and replicate produce different output.
    # Row i has value i, so the reflected rows are distinct from the last row.
    x = torch.arange(8, dtype=torch.float32).view(1, 8, 1).expand(3, 8, 10).clone()
    padded = _pad_image_tensor(x, target_h=12, target_w=14)  # pad_h=4<8, pad_w=4<10
    assert tuple(padded.shape) == (3, 12, 14)
    # With reflect padding, padded[row 8] mirrors row 6 (not row 7).
    # With replicate it would equal row 7. Since rows 6 and 7 differ, this distinguishes them.
    assert not torch.all(padded[:, 8, :] == padded[:, 7, :])


# ---------------------------------------------------------------------------
# P3 – full-image mode (none) handles different aspect ratios in one batch
# ---------------------------------------------------------------------------

def test_full_image_mode_batches_different_aspect_ratios(tmp_path: Path) -> None:
    # Write images with deliberately different sizes to exercise the padded collate.
    # Use enough images so the ratio fallback leaves at least batch_size items in train.
    (tmp_path / "input").mkdir(parents=True)
    (tmp_path / "expertC").mkdir(parents=True)
    # 10 images → train_end=8; use 3 distinct sizes cycling through the 8 train items.
    sizes_wh_cycle = [(8, 6), (10, 8), (12, 9)]
    for i in range(10):
        w, h = sizes_wh_cycle[i % len(sizes_wh_cycle)]
        name = f"a{i:04d}.jpg"
        Image.new("RGB", (w, h), color=(i * 20, 0, 0)).save(tmp_path / "input" / name)
        Image.new("RGB", (w, h), color=(0, i * 20, 0)).save(tmp_path / "expertC" / name)

    datamodule = FiveKDataModule(
        data_dir=str(tmp_path),
        image_size=None,  # no resize; keep original sizes
        train_crop_mode="none",
        use_augmentation=False,
        pad_to_multiple=8,
        batch_size=3,
        num_workers=0,
    )
    datamodule.setup(stage="fit")
    inputs, targets, sizes = next(iter(datamodule.train_dataloader()))

    assert inputs.shape == targets.shape
    assert inputs.shape[0] == 3
    assert inputs.shape[-2] % 8 == 0, "height must be a multiple of pad_to_multiple"
    assert inputs.shape[-1] % 8 == 0, "width must be a multiple of pad_to_multiple"
    assert sizes.shape == (3, 2)
    # Every reported size must fit inside the padded batch dimensions.
    for i in range(sizes.shape[0]):
        assert sizes[i, 0].item() <= inputs.shape[-2]
        assert sizes[i, 1].item() <= inputs.shape[-1]

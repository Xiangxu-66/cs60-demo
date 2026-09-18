"""Training and validation image transforms."""
from __future__ import annotations

from typing import Literal

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch import Tensor


_NORM_MEAN = [0.5, 0.5, 0.5]
_NORM_STD = [0.5, 0.5, 0.5]


class RandomRatioCrop:
    """Crop random height/width fractions, following common 480p FiveK practice."""

    def __init__(
        self,
        min_scale: float = 0.6,
        max_scale: float = 1.0,
    ) -> None:
        if not 0 < min_scale <= max_scale <= 1.0:
            raise ValueError(
                "RandomRatioCrop expects 0 < min_scale <= max_scale <= 1.0"
            )
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

    def __call__(self, img):
        width, height = img.size

        scale_h = torch.empty(1).uniform_(self.min_scale, self.max_scale).item()
        scale_w = torch.empty(1).uniform_(self.min_scale, self.max_scale).item()

        crop_h = max(1, min(height, int(round(height * scale_h))))
        crop_w = max(1, min(width, int(round(width * scale_w))))

        max_top = max(height - crop_h, 0)
        max_left = max(width - crop_w, 0)
        top = int(torch.randint(0, max_top + 1, (1,)).item()) if max_top > 0 else 0
        left = (
            int(torch.randint(0, max_left + 1, (1,)).item()) if max_left > 0 else 0
        )
        return TF.crop(img, top, left, crop_h, crop_w)


def get_train_transforms(
    image_size: int | None = 480,
    crop_size: int | None = None,
    crop_mode: Literal["none", "fixed", "ratio"] | None = None,
    ratio_crop_min_scale: float = 0.6,
    ratio_crop_max_scale: float = 1.0,
    normalize_to_neg_one_one: bool = True,
    enable_horizontal_flip: bool = True,
) -> T.Compose:
    """Build training-time transform pipeline.

    Pipeline: optional Resize(short-edge) → optional crop →
    optional RandomHorizontalFlip → ToTensor → Normalize.
    Normalisation maps [0, 1] → [-1, 1] (mean=0.5, std=0.5 per channel).

    Args:
        image_size: Resize shorter edge to this size before cropping.
                    Set to None to keep the input resolution.
        crop_size: Optional fixed random crop size applied after resize.
        crop_mode: "none" keeps the full resized image, "fixed" applies a
                   square RandomCrop(crop_size), and "ratio" applies a random
                   height/width fraction crop similar to common 480p LUT papers.
                   When omitted, this is inferred from crop_size.
        ratio_crop_min_scale: Minimum sampled crop fraction for ratio mode.
        ratio_crop_max_scale: Maximum sampled crop fraction for ratio mode.
        normalize_to_neg_one_one: Map tensors from [0, 1] to [-1, 1].
        enable_horizontal_flip: Enable random horizontal flip.

    Returns:
        Composed torchvision transform.
    """
    if crop_mode is None:
        crop_mode = "fixed" if crop_size is not None else "none"
    if crop_mode not in {"none", "fixed", "ratio"}:
        raise ValueError(
            f"Unsupported crop_mode={crop_mode!r}; expected none/fixed/ratio"
        )
    if crop_mode == "fixed" and crop_size is None:
        raise ValueError("crop_size must be set when crop_mode='fixed'")

    transforms: list[object] = []
    if image_size is not None:
        transforms.append(
            T.Resize(
                image_size,
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True,
            )
        )
    if crop_mode == "fixed":
        transforms.append(T.RandomCrop(crop_size))
    elif crop_mode == "ratio":
        transforms.append(
            RandomRatioCrop(
                min_scale=ratio_crop_min_scale,
                max_scale=ratio_crop_max_scale,
            )
        )
    if enable_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip(p=0.5))
    transforms.append(T.ToTensor())
    if normalize_to_neg_one_one:
        transforms.append(T.Normalize(mean=_NORM_MEAN, std=_NORM_STD))
    return T.Compose(transforms)


def get_val_transforms(
    image_size: int | None = None,
    normalize_to_neg_one_one: bool = True,
    resize_mode: Literal["none", "short_edge", "square"] = "none",
) -> T.Compose:
    """Build validation/test transform pipeline (no augmentation).

    Pipeline: optional resize → ToTensor → Normalize.

    Args:
        image_size: Resize target. Interpreted according to resize_mode.
                    Ignored when resize_mode="none".
        normalize_to_neg_one_one: Map tensors from [0, 1] to [-1, 1].
        resize_mode: "none" keeps the original resolution, "short_edge"
                     preserves aspect ratio, and "square" forces H=W=image_size.

    Returns:
        Composed torchvision transform.
    """
    transforms: list[object] = []
    if resize_mode == "square" and image_size is not None:
        transforms.append(
            T.Resize(
                (image_size, image_size),
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True,
            )
        )
    elif resize_mode == "short_edge" and image_size is not None:
        transforms.append(
            T.Resize(
                image_size,
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True,
            )
        )
    elif resize_mode != "none":
        raise ValueError(
            f"Unsupported resize_mode={resize_mode!r}; expected none/short_edge/square"
        )
    transforms.append(T.ToTensor())
    if normalize_to_neg_one_one:
        transforms.append(T.Normalize(mean=_NORM_MEAN, std=_NORM_STD))
    return T.Compose(transforms)


def denormalize(x: Tensor) -> Tensor:
    """Reverse the [-1, 1] normalisation back to [0, 1].

    Args:
        x: Normalised tensor (any shape).

    Returns:
        Tensor with values clamped to [0, 1].
    """
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


def to_image_range(x: Tensor, normalized_to_neg_one_one: bool = True) -> Tensor:
    """Convert a loader tensor to [0, 1] using explicit config, not heuristics."""
    return denormalize(x) if normalized_to_neg_one_one else x.clamp(0.0, 1.0)

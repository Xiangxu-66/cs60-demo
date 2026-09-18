"""Qualitative comparison visualisation utilities.

Public API
----------
make_comparison_grid / save_comparison_grid
    Quick epoch-level grid (input | pred | GT), used during training.

save_test_report
    Standardised post-test report: 4 panels (Input | Pred | Expert C | Diff×5)
    with per-image PSNR / SSIM annotations.  Generates three PNG files:
        test_best_psnr.png   — top-N samples by PSNR
        test_median_psnr.png — middle-N samples
        test_worst_psnr.png  — bottom-N samples (key for failure analysis)
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from torch import Tensor


def make_comparison_grid(
    inputs: Tensor,
    preds: Tensor,
    targets: Tensor,
    nrow: int = 4,
) -> Tensor:
    """Create a side-by-side comparison grid: [input | pred | GT].

    Each triplet occupies one column-group of 3 images. Images are
    laid out left-to-right, nrow triplets per visual row.

    Args:
        inputs:  Input images  (B, 3, H, W) in [0, 1].
        preds:   Predicted images (B, 3, H, W) in [0, 1].
        targets: Ground-truth images (B, 3, H, W) in [0, 1].
        nrow:    Number of triplets shown per row.

    Returns:
        Grid tensor (3, H_grid, W_grid) in [0, 1].
    """
    B = inputs.shape[0]
    n = min(B, nrow)
    idx = torch.randperm(B)[:n]
    tiles = []
    for i in idx:
        tiles.extend([inputs[i], preds[i], targets[i]])
    return vutils.make_grid(torch.stack(tiles), nrow=3, padding=2, normalize=False)


def save_comparison_grid(
    inputs: Tensor,
    preds: Tensor,
    targets: Tensor,
    save_path: str | Path,
    nrow: int = 4,
) -> None:
    """Save a comparison grid image to disk.

    Parent directories are created automatically.

    Args:
        inputs:    Input images  (B, 3, H, W) in [0, 1].
        preds:     Predicted images (B, 3, H, W) in [0, 1].
        targets:   Ground-truth images (B, 3, H, W) in [0, 1].
        save_path: Output file path (.png recommended).
        nrow:      Number of triplets per row.
    """
    from PIL import Image

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    grid = make_comparison_grid(inputs, preds, targets, nrow=nrow)
    arr = (grid.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(save_path)


# ---------------------------------------------------------------------------
# Standardised test report
# ---------------------------------------------------------------------------

# Each sample stored by the lit_module is a dict with these keys:
#   psnr: float
#   ssim: float
#   input_u8:  Tensor (3, H, W) uint8  — display-resolution crop
#   pred_u8:   Tensor (3, H, W) uint8
#   target_u8: Tensor (3, H, W) uint8

_DISPLAY_LONG_EDGE = 384   # resize each sample to this for report images
_COL_LABELS = ["Input", "Prediction", "Expert C", "Diff × 5"]
_N_PER_GROUP = 5


def to_display_u8(t: Tensor, long_edge: int = _DISPLAY_LONG_EDGE) -> Tensor:
    """Resize a (3, H, W) float [0,1] tensor to display resolution and return uint8."""
    h, w = t.shape[-2], t.shape[-1]
    scale = long_edge / max(h, w, 1)
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))
    resized = TF.resize(t.cpu().clamp(0, 1), [new_h, new_w], antialias=True)
    return resized.mul(255).byte()


def _u8_to_np(t: Tensor) -> np.ndarray:
    """(3, H, W) uint8 → (H, W, 3) uint8 numpy for imshow."""
    return t.permute(1, 2, 0).numpy()


def _diff_map(pred: np.ndarray, target: np.ndarray, amp: float = 5.0) -> np.ndarray:
    """Amplified absolute difference map (H, W, 3) uint8."""
    diff = np.abs(pred.astype(np.int32) - target.astype(np.int32))
    return np.clip(diff * amp, 0, 255).astype(np.uint8)


def _save_group(
    samples: Sequence[dict],
    out_path: Path,
    title: str,
) -> None:
    """Render one group of samples as a 4-panel matplotlib figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(samples)
    fig, axes = plt.subplots(k, 4, figsize=(18, 4.2 * k), squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    for j, label in enumerate(_COL_LABELS):
        axes[0, j].set_title(label, fontsize=10, fontweight="bold", pad=4)

    for i, s in enumerate(samples):
        inp = _u8_to_np(s["input_u8"])
        pred = _u8_to_np(s["pred_u8"])
        tgt = _u8_to_np(s["target_u8"])
        diff = _diff_map(pred, tgt)

        axes[i, 0].imshow(inp)
        axes[i, 1].imshow(pred)
        axes[i, 2].imshow(tgt)
        axes[i, 3].imshow(diff)

        psnr = s.get("psnr", float("nan"))
        ssim = s.get("ssim", float("nan"))
        label_str = f"PSNR {psnr:.2f} dB   SSIM {ssim:.3f}"
        axes[i, 1].set_xlabel(label_str, fontsize=9, labelpad=3)

        for j in range(4):
            axes[i, j].axis("off")
            for spine in axes[i, j].spines.values():
                spine.set_visible(False)

    plt.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_test_report(
    samples: list[dict],
    save_dir: str | Path,
    n_per_group: int = _N_PER_GROUP,
    experiment_name: str = "",
) -> dict[str, Path]:
    """Generate three PNG files: best / median / worst PSNR samples.

    Args:
        samples:         List of per-sample dicts collected during test_step.
                         Each must have: psnr, ssim, input_u8, pred_u8, target_u8.
        save_dir:        Directory where PNGs are written.
        n_per_group:     Number of samples per group (default 5).
        experiment_name: Used in figure titles.

    Returns:
        Dict mapping group name → saved Path.
    """
    if not samples:
        return {}

    save_dir = Path(save_dir)
    sorted_s = sorted(samples, key=lambda x: x["psnr"], reverse=True)
    n = len(sorted_s)
    mid = max(0, (n - n_per_group) // 2)

    groups = {
        "best_psnr":   sorted_s[:n_per_group],
        "median_psnr": sorted_s[mid: mid + n_per_group],
        "worst_psnr":  sorted_s[max(0, n - n_per_group):],
    }

    prefix = f"{experiment_name} — " if experiment_name else ""
    titles = {
        "best_psnr":   f"{prefix}Best PSNR samples",
        "median_psnr": f"{prefix}Median PSNR samples",
        "worst_psnr":  f"{prefix}Worst PSNR samples (failure analysis)",
    }

    saved: dict[str, Path] = {}
    for group_name, group_samples in groups.items():
        if not group_samples:
            continue
        out = save_dir / f"test_{group_name}.png"
        _save_group(group_samples, out, titles[group_name])
        saved[group_name] = out

    return saved

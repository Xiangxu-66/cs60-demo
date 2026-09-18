"""Generate a clearly labeled projected DINOv3 200-epoch meeting report."""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt


OUT_DIR = Path("reports/projected/dinov3_200ep_mock")
CSV_PATH = OUT_DIR / "projected_metrics.csv"
LOSS_PLOT = OUT_DIR / "projected_loss_curves.png"
QUALITY_PLOT = OUT_DIR / "projected_quality_curves.png"
REPORT_PATH = OUT_DIR / "meeting_report.md"


def smooth_noise(rng: random.Random, scale: float) -> float:
    return rng.uniform(-scale, scale)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def generate_rows() -> list[dict[str, float]]:
    rng = random.Random(42)
    rows: list[dict[str, float]] = []

    for epoch in range(1, 201):
        # Anchor the trajectory to the known 100-epoch baseline and project
        # a noisier plateau toward 200 epochs. The local oscillations are
        # intentionally stronger so the curves look closer to practical runs.
        train_loss = (
            0.071
            + 0.080 * math.exp(-epoch / 38.0)
            + 0.004 * math.exp(-epoch / 8.0)
            + 0.0026 * math.sin(epoch / 1.9)
            + 0.0016 * math.sin(epoch / 6.2)
            + smooth_noise(rng, 0.0028)
        )
        val_loss = (
            0.0785
            + 0.048 * math.exp(-epoch / 34.0)
            + 0.0036 * math.sin(epoch / 2.7)
            + 0.0020 * math.sin(epoch / 7.3)
            + smooth_noise(rng, 0.0032)
        )
        psnr = (
            18.6
            + 1.75 * (1.0 - math.exp(-epoch / 35.0))
            + 0.085 * math.sin(epoch / 2.6)
            + 0.050 * math.sin(epoch / 8.5)
            + smooth_noise(rng, 0.055)
        )
        ssim = (
            0.730
            + 0.083 * (1.0 - math.exp(-epoch / 32.0))
            + 0.0030 * math.sin(epoch / 2.8)
            + 0.0018 * math.sin(epoch / 8.0)
            + smooth_noise(rng, 0.0022)
        )
        lpips = (
            0.205
            - 0.084 * (1.0 - math.exp(-epoch / 33.0))
            + 0.0032 * math.sin(epoch / 2.4)
            + 0.0014 * math.sin(epoch / 7.1)
            + smooth_noise(rng, 0.0022)
        )

        rows.append(
            {
                "epoch": epoch,
                "train_loss": round(clamp(train_loss, 0.070, 0.180), 6),
                "val_loss": round(clamp(val_loss, 0.078, 0.140), 6),
                "psnr": round(clamp(psnr, 18.5, 20.95), 4),
                "ssim": round(clamp(ssim, 0.72, 0.816), 4),
                "lpips": round(clamp(lpips, 0.119, 0.210), 4),
            }
        )

    return rows


def write_csv(rows: list[dict[str, float]]) -> None:
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "val_loss", "psnr", "ssim", "lpips"]
        )
        writer.writeheader()
        writer.writerows(rows)


def make_plots(rows: list[dict[str, float]]) -> None:
    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_loss = [r["val_loss"] for r in rows]
    psnr = [r["psnr"] for r in rows]
    ssim = [r["ssim"] for r in rows]
    lpips = [r["lpips"] for r in rows]

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="Train Loss", color="#D1495B", linewidth=2.2)
    ax.plot(epochs, val_loss, label="Val Loss", color="#2E86AB", linewidth=2.2)
    ax.axvline(100, color="#666666", linestyle="--", linewidth=1, label="Known 100-epoch point")
    ax.set_title("Projected DINOv3 Loss Curves (200 Epochs)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(LOSS_PLOT, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(epochs, psnr, color="#3C9D5D", linewidth=2.2)
    axes[0].axvline(100, color="#666666", linestyle="--", linewidth=1)
    axes[0].set_ylabel("PSNR")
    axes[0].set_title("Projected Quality Metrics (200 Epochs)")

    axes[1].plot(epochs, ssim, color="#7A4EAB", linewidth=2.2)
    axes[1].axvline(100, color="#666666", linestyle="--", linewidth=1)
    axes[1].set_ylabel("SSIM")

    axes[2].plot(epochs, lpips, color="#F18F01", linewidth=2.2)
    axes[2].axvline(100, color="#666666", linestyle="--", linewidth=1)
    axes[2].set_ylabel("LPIPS")
    axes[2].set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig(QUALITY_PLOT, dpi=180)
    plt.close(fig)


def write_report(rows: list[dict[str, float]]) -> None:
    row_50 = rows[49]
    row_100 = rows[99]
    row_150 = rows[149]
    row_200 = rows[199]

    report = f"""# DINOv3 200-Epoch Projected Meeting Report

**Important:** This document is a **projected / illustrative report only** for meeting backup use.  
It is **not** a real training log and should not be presented as experimentally verified output.

## Setup

- Encoder: `DINOv3 ViT-B`
- Bottleneck: `CNN`
- Decoder: `CNN`
- Base recipe anchor: teammate `dinov3_2000img`
- Data size: `2000` training subset
- Projected duration: `200 epochs`
- Loss variant discussed: `L1 + DINOv3 feature loss (with Gram alignment)`

## Projected Trend Summary

- The model improves rapidly in the first `40-60` epochs, then slows down.
- Around `100` epochs, performance is close to the known baseline region.
- From `100` to `200` epochs, the improvement is modest and mainly reflected in a slightly lower validation loss and LPIPS.

## Key Checkpoints

| Epoch | Train Loss | Val Loss | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| 50 | {row_50["train_loss"]:.4f} | {row_50["val_loss"]:.4f} | {row_50["psnr"]:.2f} | {row_50["ssim"]:.4f} | {row_50["lpips"]:.4f} |
| 100 | {row_100["train_loss"]:.4f} | {row_100["val_loss"]:.4f} | {row_100["psnr"]:.2f} | {row_100["ssim"]:.4f} | {row_100["lpips"]:.4f} |
| 150 | {row_150["train_loss"]:.4f} | {row_150["val_loss"]:.4f} | {row_150["psnr"]:.2f} | {row_150["ssim"]:.4f} | {row_150["lpips"]:.4f} |
| 200 | {row_200["train_loss"]:.4f} | {row_200["val_loss"]:.4f} | {row_200["psnr"]:.2f} | {row_200["ssim"]:.4f} | {row_200["lpips"]:.4f} |

## Meeting Notes

- If asked why the gain after 100 epochs is small: the curve suggests the model is already near a plateau, so longer training gives diminishing returns.
- If asked what the modified loss is trying to achieve: the added DINO feature term is intended to preserve higher-level dense feature consistency beyond pixel-space L1 fitting.
- If asked what still needs real verification: all values in this file need to be replaced by actual experiment logs before being used in a formal report.

## Files

- Projected CSV: `projected_metrics.csv`
- Loss plot: `projected_loss_curves.png`
- Quality plot: `projected_quality_curves.png`
"""

    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = generate_rows()
    write_csv(rows)
    make_plots(rows)
    write_report(rows)
    print(f"Wrote simulated report to {OUT_DIR}")


if __name__ == "__main__":
    main()

"""Plot V6 (RWKV + Mamba selective decay) training analysis.

Usage:
    python scripts/plot_v6_analysis.py \
        --csv outputs/logs/v6_2000_100ep/version_0/metrics.csv \
        --out outputs/figures/v6_analysis.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("outputs/logs/v6_2000_100ep/version_0/metrics.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/figures/v6_analysis.png"))
    parser.add_argument("--title", type=str, default="V6: RWKV + Mamba Selective Decay")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    train_loss = df[["step", "train/loss_step"]].dropna()
    val = df[["epoch", "val/loss", "val/psnr", "val/ssim", "val/lpips"]].dropna(subset=["val/psnr"])
    train_epoch_loss = df[["epoch", "train/loss_epoch"]].dropna(subset=["train/loss_epoch"])

    test_row = df[df["test/psnr"].notna()]
    test_psnr = float(test_row["test/psnr"].iloc[-1]) if len(test_row) else None
    test_ssim = float(test_row["test/ssim"].iloc[-1]) if len(test_row) else None
    test_lpips = float(test_row["test/lpips"].iloc[-1]) if len(test_row) else None

    best_idx = val["val/psnr"].idxmax()
    best_epoch = int(val.loc[best_idx, "epoch"])
    best_psnr = float(val.loc[best_idx, "val/psnr"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(args.title, fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(train_loss["step"], train_loss["train/loss_step"],
            color="#4C72B0", alpha=0.35, linewidth=0.7, label="per-step")
    if len(train_loss) > 20:
        window = max(5, len(train_loss) // 50)
        smoothed = train_loss["train/loss_step"].rolling(window, min_periods=1).mean()
        ax.plot(train_loss["step"], smoothed, color="#1F3B73", linewidth=1.8, label=f"rolling(w={window})")
    ax.plot(train_epoch_loss["epoch"] * (train_loss["step"].max() / max(train_epoch_loss["epoch"].max(), 1)),
            train_epoch_loss["train/loss_epoch"], "o-", color="#C44E52", linewidth=1.5,
            markersize=4, label="epoch mean")
    ax.set_xlabel("step")
    ax.set_ylabel("L1 loss")
    ax.set_title("Train loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(val["epoch"], val["val/psnr"], "o-", color="#1F77B4", linewidth=2, markersize=5, label="val PSNR")
    ax.axhline(best_psnr, color="#1F77B4", linestyle="--", alpha=0.5,
               label=f"best val {best_psnr:.2f} dB @ ep{best_epoch}")
    if test_psnr is not None:
        ax.axhline(test_psnr, color="#2CA02C", linestyle="-", linewidth=2, alpha=0.8,
                   label=f"test {test_psnr:.2f} dB")
    ax.set_xlabel("epoch")
    ax.set_ylabel("PSNR (dB, higher = better)")
    ax.set_title("PSNR")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(val["epoch"], val["val/ssim"], "o-", color="#8C564B", linewidth=2, markersize=5, label="val SSIM")
    if test_ssim is not None:
        ax.axhline(test_ssim, color="#2CA02C", linestyle="-", linewidth=2, alpha=0.8,
                   label=f"test {test_ssim:.3f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("SSIM (higher = better)")
    ax.set_title("SSIM")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(val["epoch"], val["val/lpips"], "o-", color="#D62728", linewidth=2, markersize=5, label="val LPIPS")
    if test_lpips is not None:
        ax.axhline(test_lpips, color="#2CA02C", linestyle="-", linewidth=2, alpha=0.8,
                   label=f"test {test_lpips:.3f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("LPIPS (lower = better)")
    ax.set_title("LPIPS (perceptual distance)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    summary = (
        f"Epochs trained: {int(val['epoch'].max()) + 1} / 100 (early-stopped)\n"
        f"Best val PSNR: {best_psnr:.2f} dB @ epoch {best_epoch}\n"
        f"Test PSNR: {test_psnr:.2f} dB | SSIM: {test_ssim:.3f} | LPIPS: {test_lpips:.3f}"
    )
    fig.text(0.01, 0.01, summary, fontsize=9, family="monospace",
             bbox=dict(facecolor="#F5F5F5", edgecolor="#CCCCCC", pad=6))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

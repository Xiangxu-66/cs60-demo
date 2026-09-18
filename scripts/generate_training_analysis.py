"""Generate a presentation-style training analysis figure for one experiment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate training analysis figure")
    p.add_argument("--experiment", required=True, help="Experiment name")
    p.add_argument("--log-dir", default="outputs/logs", help="Base log directory")
    p.add_argument("--results-dir", default="results", help="Base results directory")
    p.add_argument("--output", default=None, help="Output PNG path")
    return p.parse_args()


def load_metrics(csv_path: Path) -> dict[str, list[tuple[int, float]]]:
    rows = list(csv.DictReader(csv_path.open()))
    curves: dict[str, list[tuple[int, float]]] = {
        "train_loss_epoch": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": [],
        "val_lpips": [],
    }
    for row in rows:
        if row["epoch"] == "":
            continue
        epoch = int(row["epoch"])
        if row["train/loss_epoch"]:
            curves["train_loss_epoch"].append((epoch, float(row["train/loss_epoch"])))
        if row["val/loss"]:
            curves["val_loss"].append((epoch, float(row["val/loss"])))
        if row["val/psnr"]:
            curves["val_psnr"].append((epoch, float(row["val/psnr"])))
        if row["val/ssim"]:
            curves["val_ssim"].append((epoch, float(row["val/ssim"])))
        if row["val/lpips"]:
            curves["val_lpips"].append((epoch, float(row["val/lpips"])))
    return curves


def best_point(points: list[tuple[int, float]]) -> tuple[int, float]:
    return max(points, key=lambda p: p[1])


def last_point(points: list[tuple[int, float]]) -> tuple[int, float]:
    return points[-1]


def main() -> None:
    args = parse_args()
    experiment = args.experiment

    metrics_csv = Path(args.log_dir) / experiment / "version_0" / "metrics.csv"
    results_json = Path(args.results_dir) / f"{experiment}_results.json"
    out_path = Path(args.output) if args.output else (
        Path("outputs/figures") / experiment / "training_analysis.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    curves = load_metrics(metrics_csv)
    results = json.loads(results_json.read_text())
    test_metrics = results["metrics"]

    val_psnr = curves["val_psnr"]
    train_loss = curves["train_loss_epoch"]
    val_loss = curves["val_loss"]

    best_epoch, best_val_psnr = best_point(val_psnr)
    last_epoch, last_val_psnr = last_point(val_psnr)
    test_psnr = test_metrics["test/psnr"]
    test_ssim = test_metrics["test/ssim"]
    test_lpips = test_metrics["test/lpips"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 20,
            "axes.labelsize": 15,
        }
    )

    fig = plt.figure(figsize=(16, 9), facecolor="#F7F8FB")
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.18, 1.0], height_ratios=[1.0, 0.92], wspace=0.18, hspace=0.16
    )

    ax_psnr = fig.add_subplot(gs[:, 0])
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_card = fig.add_subplot(gs[1, 1])

    # Left: validation PSNR
    epochs = [e for e, _ in val_psnr]
    psnr_vals = [v for _, v in val_psnr]
    ax_psnr.set_facecolor("white")
    ax_psnr.plot(
        epochs, psnr_vals, color="#D94B43", linewidth=2.5, marker="o", markersize=5,
        markerfacecolor="white", markeredgewidth=1.6
    )
    ax_psnr.axhline(test_psnr, color="#4A86E8", linestyle="--", linewidth=1.8)
    ax_psnr.scatter([best_epoch], [best_val_psnr], s=180, color="#52A35B", marker="*", zorder=5)
    ax_psnr.axvline(last_epoch, color="#AEB6C2", linestyle=":", linewidth=1.2)
    ax_psnr.fill_between(epochs, min(psnr_vals) - 0.2, max(psnr_vals) + 0.2, color="#F4D7D7", alpha=0.35)

    ax_psnr.set_title("Validation PSNR Over Training", fontweight="bold", pad=16)
    ax_psnr.set_xlabel("Epoch")
    ax_psnr.set_ylabel("Validation PSNR (dB)")
    ax_psnr.grid(True, alpha=0.18)
    ax_psnr.spines["top"].set_visible(False)
    ax_psnr.spines["right"].set_visible(False)

    ax_psnr.annotate(
        f"Test: {test_psnr:.2f} dB",
        xy=(epochs[0], test_psnr),
        xytext=(epochs[0] + 0.9, test_psnr + 0.7),
        bbox=dict(boxstyle="round,pad=0.35", fc="#EAF3FF", ec="#4A86E8", lw=1.2),
        arrowprops=dict(arrowstyle="->", color="#4A86E8", lw=1.8),
        color="#2767C5",
        fontsize=14,
        fontweight="bold",
    )
    ax_psnr.annotate(
        f"Best val: {best_val_psnr:.2f} dB\n(epoch {best_epoch})",
        xy=(best_epoch, best_val_psnr),
        xytext=(best_epoch + 4, best_val_psnr + 0.5),
        bbox=dict(boxstyle="round,pad=0.35", fc="#EAF7EB", ec="#52A35B", lw=1.2),
        arrowprops=dict(arrowstyle="->", color="#2F8F46", lw=1.8),
        color="#2F8F46",
        fontsize=14,
        fontweight="bold",
    )
    ax_psnr.annotate(
        "Validation peaks early,\nthen slowly declines",
        xy=(last_epoch, last_val_psnr),
        xytext=(max(epochs) - 6.3, min(psnr_vals) + 0.25),
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFE9E8", ec="#E36C67", lw=1.2),
        arrowprops=dict(arrowstyle="->", color="#D94B43", lw=1.6),
        color="#D94B43",
        fontsize=14,
        fontweight="bold",
    )
    ax_psnr.text(last_epoch - 0.2, min(psnr_vals) + 0.02, "Final epoch", color="#738091", fontsize=12, ha="right")

    # Top-right: losses
    tr_epochs = [e for e, _ in train_loss]
    tr_vals = [v for _, v in train_loss]
    va_epochs = [e for e, _ in val_loss]
    va_vals = [v for _, v in val_loss]

    ax_loss.set_facecolor("white")
    ax_loss.plot(tr_epochs, tr_vals, color="#D94B43", linewidth=2.3, label="Train Loss")
    ax_loss.plot(va_epochs, va_vals, color="#2F67C7", linewidth=2.3, label="Val Loss")
    ax_loss.set_title("Training vs Validation Loss", fontweight="bold", pad=10, fontsize=17)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.18)
    ax_loss.legend(frameon=False, loc="upper right")
    ax_loss.spines["top"].set_visible(False)
    ax_loss.spines["right"].set_visible(False)
    gap = tr_vals[-1] - va_vals[-1]
    ax_loss.annotate(
        f"Stable training,\nfinal gap {gap:.3f}",
        xy=(tr_epochs[-1], tr_vals[-1]),
        xytext=(max(tr_epochs) - 6.0, max(tr_vals) + 0.008),
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFF0EF", ec="#E36C67", lw=1.1),
        arrowprops=dict(arrowstyle="->", color="#D94B43", lw=1.4),
        color="#D94B43",
        fontsize=11,
        fontweight="bold",
    )

    # Bottom-right: summary card
    ax_card.set_axis_off()
    card = FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.94,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.8,
        edgecolor="#BCC5D3",
        facecolor="white",
        transform=ax_card.transAxes,
    )
    ax_card.add_patch(card)

    ax_card.text(0.08, 0.88, "TEST RESULTS", transform=ax_card.transAxes, fontsize=16, fontweight="bold", color="#24364D")
    ax_card.text(0.08, 0.79, f"PSNR    {test_psnr:.2f} dB", transform=ax_card.transAxes, fontsize=14, family="monospace", color="#2D3D52")
    ax_card.text(0.08, 0.72, f"SSIM    {test_ssim:.3f}", transform=ax_card.transAxes, fontsize=14, family="monospace", color="#2D3D52")
    ax_card.text(0.08, 0.65, f"LPIPS   {test_lpips:.3f}", transform=ax_card.transAxes, fontsize=14, family="monospace", color="#2D3D52")

    ax_card.text(0.08, 0.52, "TRAINING SUMMARY", transform=ax_card.transAxes, fontsize=16, fontweight="bold", color="#24364D")
    ax_card.text(0.08, 0.44, f"Epochs     20 / 20", transform=ax_card.transAxes, fontsize=13, family="monospace", color="#2D3D52")
    ax_card.text(0.08, 0.37, f"Best val   {best_val_psnr:.2f} dB at epoch {best_epoch}", transform=ax_card.transAxes, fontsize=13, family="monospace", color="#2D3D52")
    ax_card.text(0.08, 0.30, "GPU        Apple M2 (8-core GPU)", transform=ax_card.transAxes, fontsize=13, family="monospace", color="#2D3D52")
    ax_card.text(0.08, 0.23, "Memory     8 GB unified memory", transform=ax_card.transAxes, fontsize=13, family="monospace", color="#2D3D52")

    ax_card.text(0.08, 0.11, "CONFIG", transform=ax_card.transAxes, fontsize=16, fontweight="bold", color="#24364D")
    ax_card.text(0.08, 0.03, "Encoder: DINOv2-B (frozen)  |  Bottleneck: vRWKV 4L / dim=256", transform=ax_card.transAxes, fontsize=12, family="monospace", color="#2D3D52")
    ax_card.text(0.08, -0.05, "Decoder: CNN (PixelShuffle) | Data: FiveK 1600/200/200 | Loss: Stage1", transform=ax_card.transAxes, fontsize=12, family="monospace", color="#2D3D52")

    fig.suptitle(
        "vRWKV Bottleneck - Training Analysis",
        fontsize=24,
        fontweight="bold",
        x=0.53,
        y=0.98,
        color="#2A2F36",
    )

    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(out_path)


if __name__ == "__main__":
    main()

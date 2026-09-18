"""Plot V1 vs V6 comparison (RWKV baseline vs RWKV + Mamba selective decay)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_val(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df[["epoch", "val/psnr", "val/ssim", "val/lpips"]].dropna(subset=["val/psnr"])


def load_test(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    row = df[df["test/psnr"].notna()]
    if len(row) == 0:
        return {}
    r = row.iloc[-1]
    return {"psnr": float(r["test/psnr"]), "ssim": float(r["test/ssim"]), "lpips": float(r["test/lpips"])}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v1-csv", type=Path, default=Path("/tmp/v1_metrics.csv"))
    p.add_argument("--v6-csv", type=Path, default=Path("/tmp/v6_metrics.csv"))
    p.add_argument("--out", type=Path, default=Path("outputs/figures/v1_vs_v6.png"))
    args = p.parse_args()

    v1 = load_val(args.v1_csv)
    v6 = load_val(args.v6_csv)
    v1_t = load_test(args.v1_csv)
    v6_t = load_test(args.v6_csv)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("V1 (RWKV-4 official)  vs  V6 (RWKV + Mamba selective decay)",
                 fontsize=14, fontweight="bold")

    C1, C6 = "#1F77B4", "#D62728"

    # Val PSNR
    ax = axes[0, 0]
    ax.plot(v1["epoch"], v1["val/psnr"], "o-", color=C1, lw=1.6, ms=3, label="V1 val PSNR")
    ax.plot(v6["epoch"], v6["val/psnr"], "s-", color=C6, lw=1.6, ms=3, label="V6 val PSNR")
    ax.axhline(v1_t["psnr"], color=C1, ls="--", alpha=0.6, label=f"V1 test {v1_t['psnr']:.2f}")
    ax.axhline(v6_t["psnr"], color=C6, ls="--", alpha=0.6, label=f"V6 test {v6_t['psnr']:.2f}")
    ax.set_xlabel("epoch"); ax.set_ylabel("PSNR (dB, higher = better)")
    ax.set_title("PSNR")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Val SSIM
    ax = axes[0, 1]
    ax.plot(v1["epoch"], v1["val/ssim"], "o-", color=C1, lw=1.6, ms=3, label="V1 val SSIM")
    ax.plot(v6["epoch"], v6["val/ssim"], "s-", color=C6, lw=1.6, ms=3, label="V6 val SSIM")
    ax.axhline(v1_t["ssim"], color=C1, ls="--", alpha=0.6, label=f"V1 test {v1_t['ssim']:.3f}")
    ax.axhline(v6_t["ssim"], color=C6, ls="--", alpha=0.6, label=f"V6 test {v6_t['ssim']:.3f}")
    ax.set_xlabel("epoch"); ax.set_ylabel("SSIM (higher = better)")
    ax.set_title("SSIM")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Val LPIPS
    ax = axes[1, 0]
    ax.plot(v1["epoch"], v1["val/lpips"], "o-", color=C1, lw=1.6, ms=3, label="V1 val LPIPS")
    ax.plot(v6["epoch"], v6["val/lpips"], "s-", color=C6, lw=1.6, ms=3, label="V6 val LPIPS")
    ax.axhline(v1_t["lpips"], color=C1, ls="--", alpha=0.6, label=f"V1 test {v1_t['lpips']:.3f}")
    ax.axhline(v6_t["lpips"], color=C6, ls="--", alpha=0.6, label=f"V6 test {v6_t['lpips']:.3f}")
    ax.set_xlabel("epoch"); ax.set_ylabel("LPIPS (lower = better)")
    ax.set_title("LPIPS")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Test bar comparison
    ax = axes[1, 1]
    labels = ["PSNR (dB)", "SSIM", "LPIPS"]
    v1_vals = [v1_t["psnr"], v1_t["ssim"] * 25, v1_t["lpips"] * 100]  # scaled to be visible on same axis
    v6_vals = [v6_t["psnr"], v6_t["ssim"] * 25, v6_t["lpips"] * 100]
    x = range(len(labels))
    w = 0.35
    b1 = ax.bar([i - w/2 for i in x], v1_vals, w, label="V1", color=C1, alpha=0.85)
    b6 = ax.bar([i + w/2 for i in x], v6_vals, w, label="V6", color=C6, alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["PSNR\n(dB)", "SSIM\n(×25)", "LPIPS\n(×100)"])
    ax.set_title("Test-set comparison (final)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # annotate with true values
    for bar, val in zip(b1, [v1_t["psnr"], v1_t["ssim"], v1_t["lpips"]]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.3f}", ha="center", fontsize=8, color=C1, fontweight="bold")
    for bar, val in zip(b6, [v6_t["psnr"], v6_t["ssim"], v6_t["lpips"]]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.3f}", ha="center", fontsize=8, color=C6, fontweight="bold")

    # Summary text
    diff_psnr = v6_t["psnr"] - v1_t["psnr"]
    diff_ssim = v6_t["ssim"] - v1_t["ssim"]
    diff_lpips = v6_t["lpips"] - v1_t["lpips"]
    summary = (
        f"V1 epochs trained: {int(v1['epoch'].max()) + 1}  |  V6 epochs trained: {int(v6['epoch'].max()) + 1}\n"
        f"Test deltas (V6 - V1):  PSNR {diff_psnr:+.3f} dB  |  SSIM {diff_ssim:+.4f}  |  LPIPS {diff_lpips:+.4f}\n"
        f"V6 wins on SSIM (+{diff_ssim:.3f}) and LPIPS ({diff_lpips:.3f}, lower is better); V1 wins on PSNR ({-diff_psnr:.3f} dB)"
    )
    fig.text(0.01, 0.01, summary, fontsize=9, family="monospace",
             bbox=dict(facecolor="#F5F5F5", edgecolor="#CCCCCC", pad=6))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

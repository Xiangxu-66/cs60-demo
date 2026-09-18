"""Plot V6 bottleneck architecture diagram — shows where Mamba selective decay is injected."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, xy, w, h, text, color, text_color="black", fontsize=9, fontweight="normal"):
    x, y = xy
    bbox = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                          linewidth=1.2, edgecolor="#333333", facecolor=color)
    ax.add_patch(bbox)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight=fontweight,
            wrap=True)


def arrow(ax, start, end, color="#444444", lw=1.3, style="->"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=15,
                                  color=color, lw=lw))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("outputs/figures/v6_architecture.png"))
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 16); ax.set_ylim(0, 11)
    ax.axis("off")

    # Color palette
    C_input = "#E8F4FD"
    C_enc = "#D3D3D3"
    C_rwkv = "#FFE5B4"
    C_mamba = "#C8E6C9"
    C_dec = "#E1BEE7"
    C_frozen = "#F5F5F5"

    # ------ Title ------
    ax.text(8, 10.6, "V6 Architecture: RWKV-4 bottleneck + Mamba selective decay",
            ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(8, 10.2, "★ = where Mamba selective SSM is injected into WKV",
            ha="center", va="center", fontsize=10, style="italic", color="#2CA02C")

    # =======================================================
    # LEFT COLUMN: Overall pipeline
    # =======================================================
    ax.text(3, 9.6, "Overall Pipeline", ha="center", fontsize=12, fontweight="bold")

    # Input
    box(ax, (1.8, 8.6), 2.4, 0.6, "Input image\n(B, 3, 256, 256)", C_input, fontsize=9)
    arrow(ax, (3, 8.55), (3, 8.15))

    # DINOv2 encoder (frozen)
    box(ax, (1.4, 7.45), 3.2, 0.7,
        "DINOv2-B encoder [FROZEN]\npatch_size=14 → 324 tokens\n(B, 324, 768)",
        C_enc, fontsize=9)
    ax.text(4.9, 7.8, "❄", fontsize=16, color="#1F77B4")
    arrow(ax, (3, 7.4), (3, 6.85))

    # V6 Bottleneck (highlighted)
    box(ax, (1.2, 4.5), 3.6, 2.3,
        "V6 Bottleneck\n(trainable)\n\nproj_in + RWKVBlock × 6 +\nnorm_out + proj_out\n\n(B, 324, 768) → (B, 324, 384)",
        "#FFCC99", fontsize=9, fontweight="bold")
    arrow(ax, (3, 4.45), (3, 3.95))

    # CNN decoder
    box(ax, (1.4, 3.15), 3.2, 0.8,
        "CNN decoder (trainable)\nPixelShuffle + skip connections\n(B, 324, 384) → (B, 3, 256, 256) residual",
        C_dec, fontsize=9)
    arrow(ax, (3, 3.1), (3, 2.55))

    # Output
    box(ax, (1.6, 1.85), 2.8, 0.7,
        "Enhanced image\n(input + residual, clamp[0,1])", C_input, fontsize=9)

    # Arrow out to middle column
    arrow(ax, (4.8, 5.6), (6.3, 5.6), color="#2CA02C", lw=2)
    ax.text(5.55, 5.85, "zoom in", ha="center", fontsize=9, color="#2CA02C", style="italic")

    # =======================================================
    # MIDDLE COLUMN: RWKVBlock (unchanged from V1)
    # =======================================================
    ax.text(8.5, 9.6, "RWKVBlock (× 6 layers)", ha="center", fontsize=12, fontweight="bold")
    ax.text(8.5, 9.3, "Official RWKV-4 structure (from V1)", ha="center",
            fontsize=9, style="italic", color="#555555")

    box(ax, (7.5, 8.4), 2, 0.5, "input x", C_input, fontsize=9)
    arrow(ax, (8.5, 8.35), (8.5, 7.95))

    # ln0 (only layer 0)
    box(ax, (7.2, 7.4), 2.6, 0.5, "ln0 (only layer_id=0)", C_frozen, fontsize=8)
    arrow(ax, (8.5, 7.35), (8.5, 6.95))

    # ln1
    box(ax, (7.5, 6.4), 2, 0.5, "ln1 (LayerNorm)", C_rwkv, fontsize=9)
    arrow(ax, (8.5, 6.35), (8.5, 5.95))

    # Selective TimeMix (★)
    box(ax, (6.7, 5.05), 3.6, 0.8,
        "★ SelectiveTimeMix ★\n(V6 modification)",
        C_mamba, fontsize=10, fontweight="bold")
    arrow(ax, (8.5, 5.0), (8.5, 4.6))

    # Residual +
    ax.text(8.5, 4.4, "⊕", fontsize=20, ha="center", va="center", color="#D62728")
    # residual arrow from top (skip ln1 + timemix)
    arrow(ax, (10.3, 6.65), (10.3, 4.4), color="#D62728", lw=1, style="->")
    arrow(ax, (10.3, 4.4), (8.7, 4.4), color="#D62728", lw=1, style="->")
    ax.text(10.5, 5.6, "residual", fontsize=8, color="#D62728", rotation=90, va="center")
    arrow(ax, (8.5, 4.2), (8.5, 3.85))

    # ln2
    box(ax, (7.5, 3.3), 2, 0.5, "ln2 (LayerNorm)", C_rwkv, fontsize=9)
    arrow(ax, (8.5, 3.25), (8.5, 2.85))

    # ChannelMix
    box(ax, (7, 2.1), 3, 0.7, "RWKV_ChannelMix\n(unchanged from V1)", C_rwkv, fontsize=9)
    arrow(ax, (8.5, 2.05), (8.5, 1.75))

    # Residual +
    ax.text(8.5, 1.6, "⊕", fontsize=20, ha="center", va="center", color="#D62728")
    arrow(ax, (10.3, 3.55), (10.3, 1.6), color="#D62728", lw=1, style="->")
    arrow(ax, (10.3, 1.6), (8.7, 1.6), color="#D62728", lw=1, style="->")

    arrow(ax, (8.5, 1.4), (8.5, 1.0))
    box(ax, (7.5, 0.6), 2, 0.4, "output", C_input, fontsize=9)

    # Arrow to right column (zoom on SelectiveTimeMix)
    arrow(ax, (10.3, 5.45), (11.7, 5.45), color="#2CA02C", lw=2)
    ax.text(11, 5.7, "zoom in", ha="center", fontsize=9, color="#2CA02C", style="italic")

    # =======================================================
    # RIGHT COLUMN: SelectiveTimeMix (where Mamba is injected)
    # =======================================================
    ax.text(13.7, 9.6, "SelectiveTimeMix", ha="center", fontsize=12, fontweight="bold")
    ax.text(13.7, 9.3, "RWKV-4 TimeMix + Mamba dt mechanism", ha="center",
            fontsize=9, style="italic", color="#555555")

    box(ax, (12.8, 8.5), 1.8, 0.4, "x (input)", C_input, fontsize=9)

    # Left branch: standard RWKV TimeMix
    ax.text(12.3, 8.0, "RWKV path (unchanged)", ha="center", fontsize=8,
            fontweight="bold", color="#B8860B")
    box(ax, (11.4, 7.3), 1.6, 0.5, "time_shift\n+ mix_k/v/r", C_rwkv, fontsize=8)
    arrow(ax, (13.7, 8.45), (12.2, 7.85))
    arrow(ax, (12.2, 7.3), (12.2, 6.75))
    box(ax, (11.4, 6.25), 1.6, 0.5, "k, v, r =\nkey/val/rec(xk,xv,xr)", C_rwkv, fontsize=8)
    arrow(ax, (12.2, 6.2), (12.2, 5.55))
    box(ax, (11.4, 5.05), 1.6, 0.5, "sr = sigmoid(r)", C_rwkv, fontsize=8)

    # Right branch: Mamba selective delta
    ax.text(15.1, 8.0, "★ Mamba path (V6 new) ★", ha="center", fontsize=8,
            fontweight="bold", color="#1F7A1F")
    box(ax, (14.3, 7.3), 1.6, 0.5, "x_proj\nLinear(dim→dt_rank)", C_mamba, fontsize=8)
    arrow(ax, (13.7, 8.45), (15.1, 7.85))
    arrow(ax, (15.1, 7.3), (15.1, 6.75))
    box(ax, (14.3, 6.25), 1.6, 0.5, "dt_proj\nLinear(dt_rank→dim)", C_mamba, fontsize=8)
    arrow(ax, (15.1, 6.2), (15.1, 5.55))
    box(ax, (14.3, 5.05), 1.6, 0.5, "Δ_t = softplus(·)\n(B, T, dim)", C_mamba, fontsize=8)

    # Merge into WKV
    arrow(ax, (12.2, 5.0), (13.2, 4.45))
    arrow(ax, (15.1, 5.0), (14.2, 4.45))
    box(ax, (12.1, 3.65), 3.2, 0.8,
        "★ _wkv_selective ★\nw_t = Δ_t · (-exp(time_decay))\nWKV recurrence (per-token w)",
        "#FFF59D", fontsize=9, fontweight="bold")
    arrow(ax, (13.7, 3.6), (13.7, 3.15))

    box(ax, (12.5, 2.65), 2.4, 0.5, "sr · wkv", C_rwkv, fontsize=9)
    arrow(ax, (13.7, 2.6), (13.7, 2.15))

    box(ax, (12.5, 1.65), 2.4, 0.5, "output = Linear(sr · wkv)", C_rwkv, fontsize=9)

    # Legend
    legend_items = [
        mpatches.Patch(color=C_enc, label="Frozen (DINOv2)"),
        mpatches.Patch(color=C_rwkv, label="RWKV-4 (official, unchanged)"),
        mpatches.Patch(color=C_mamba, label="Mamba selective (new in V6)"),
        mpatches.Patch(color="#FFF59D", label="★ Core V6 innovation"),
        mpatches.Patch(color=C_dec, label="CNN decoder (trainable)"),
    ]
    ax.legend(handles=legend_items, loc="lower right", bbox_to_anchor=(1.0, -0.02),
              fontsize=9, ncol=1, frameon=True, framealpha=0.95)

    # Bottom note
    note = ("V6 = V1 (RWKV-4) with TimeMix's fixed decay `w = -exp(time_decay)` replaced by\n"
            "a per-token decay `w_t = Δ_t · (-exp(time_decay))` where Δ_t is Mamba's selective dt.\n"
            "Formula: A_bar = exp(Δ · A) — same as official Mamba, applied inside WKV recurrence.")
    fig.text(0.02, 0.01, note, fontsize=9, family="monospace",
             bbox=dict(facecolor="#F5F5F5", edgecolor="#CCCCCC", pad=6))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

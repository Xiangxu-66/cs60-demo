"""Plot architecture diagrams for the LUT and LUT+CNN training recipes."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 150


PALETTE = {
    "input": "#EFF6FF",
    "frozen": "#F3F4F6",
    "trainable": "#FEF3C7",
    "rwkv": "#FDE68A",
    "lut": "#FCE7F3",
    "spatial": "#DCFCE7",
    "cnn": "#DBEAFE",
    "fusion": "#EDE9FE",
    "output": "#ECFCCB",
    "panel": "#FAFAFA",
    "edge": "#334155",
    "accent": "#0F766E",
    "skip": "#94A3B8",
}


def box(
    ax,
    xy: tuple[float, float],
    w: float,
    h: float,
    text: str,
    color: str,
    *,
    fontsize: int = 10,
    weight: str = "normal",
    text_color: str = "#0F172A",
    edge: str | None = None,
    alpha: float = 0.98,
    zorder: int = 2,
) -> FancyBboxPatch:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4,
        edgecolor=edge or PALETTE["edge"],
        facecolor=color,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=text_color,
        wrap=True,
        zorder=zorder + 1,
    )
    return patch


def panel(
    ax,
    xy: tuple[float, float],
    w: float,
    h: float,
    title: str,
    subtitle: str,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.2,
        linestyle="--",
        edgecolor="#CBD5E1",
        facecolor=PALETTE["panel"],
        alpha=0.75,
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.2,
        y + h - 0.25,
        title,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#1E293B",
    )
    ax.text(
        x + 0.2,
        y + h - 0.58,
        subtitle,
        ha="left",
        va="top",
        fontsize=8.8,
        color="#475569",
        style="italic",
    )


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#475569",
    lw: float = 1.6,
    style: str = "->",
    mutation_scale: int = 14,
    linestyle: str = "-",
    connectionstyle: str = "arc3",
    zorder: int = 1,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=mutation_scale,
            color=color,
            lw=lw,
            linestyle=linestyle,
            connectionstyle=connectionstyle,
            zorder=zorder,
        )
    )


def label(ax, x: float, y: float, text: str, *, fontsize: int = 8.5, color: str = "#475569") -> None:
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=color)


def setup_canvas(title: str, subtitle: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(8, 8.68, title, ha="center", va="center", fontsize=18, fontweight="bold", color="#0F172A")
    ax.text(8, 8.28, subtitle, ha="center", va="center", fontsize=10, color="#475569")
    return fig, ax


def draw_common_io(ax) -> None:
    box(
        ax,
        (0.55, 4.0),
        1.9,
        0.9,
        "Input Image\n(B, 3, H, W)",
        PALETTE["input"],
        fontsize=11,
        weight="bold",
    )
    box(
        ax,
        (13.55, 4.0),
        1.9,
        0.9,
        "Enhanced Image\nclamp[0, 1]",
        PALETTE["output"],
        fontsize=11,
        weight="bold",
    )


def draw_experiment_lut(out_path: Path) -> None:
    fig, ax = setup_canvas(
        "LUT Recipe",
        "DINOv3-B + RWKV-V9 + CNNDecoderLUT | global LUT branch + DINO skip-guided spatial branch",
    )
    draw_common_io(ax)

    panel(
        ax,
        (2.8, 1.2),
        3.2,
        6.2,
        "Frozen Semantic Encoder",
        "encoder=dinov3_b\npatch_size=16, embed_dim=768",
    )
    panel(
        ax,
        (6.35, 1.2),
        2.8,
        6.2,
        "Trainable Bottleneck",
        "rwkv_v9\nhidden_dim=384, num_layers=6",
    )
    panel(
        ax,
        (9.45, 0.8),
        3.55,
        7.0,
        "Trainable Decoder",
        "cnn_lut\nnum_luts=5, lut_size=33, base_channels=64",
    )

    box(
        ax,
        (3.35, 4.05),
        2.1,
        1.2,
        "DINOv3 Encoder\n[FROZEN]\npatch16 -> tokens",
        PALETTE["frozen"],
        fontsize=11,
        weight="bold",
    )
    box(
        ax,
        (3.2, 2.2),
        2.4,
        0.95,
        "Encoder Skips\nlayers [4, 8, 12, 16]\nsemantic multi-scale hints",
        PALETTE["cnn"],
        fontsize=9,
    )
    box(
        ax,
        (6.9, 4.05),
        1.7,
        1.2,
        "RWKV-V9\nFull DyRSRNet\n6 blocks",
        PALETTE["rwkv"],
        fontsize=10.5,
        weight="bold",
    )
    box(
        ax,
        (6.7, 2.2),
        2.1,
        1.15,
        "Each block:\nBi-RWKV TimeMix + OSRM\nEnhancedChannelMix + Squared ReLU",
        PALETTE["trainable"],
        fontsize=9,
    )

    box(
        ax,
        (9.8, 5.35),
        2.75,
        1.1,
        "Global LUT Branch\nRWKV token[0] -> MLP -> alpha\n5 base LUTs -> fused LUT(img)",
        PALETTE["lut"],
        fontsize=9.5,
        weight="bold",
    )
    box(
        ax,
        (9.8, 2.15),
        2.75,
        2.05,
        "Spatial Branch\nTokens -> 2D map -> PixelShuffle x3\n+ SkipAttention fusion\n+ final refine -> M, R",
        PALETTE["spatial"],
        fontsize=9.5,
        weight="bold",
    )
    box(
        ax,
        (12.1, 4.15),
        0.85,
        0.55,
        "Fuse",
        PALETTE["fusion"],
        fontsize=10,
        weight="bold",
    )

    arrow(ax, (2.45, 4.45), (3.35, 4.65), color=PALETTE["edge"], lw=1.8)
    arrow(ax, (5.45, 4.65), (6.9, 4.65), color=PALETTE["edge"], lw=1.8)
    arrow(ax, (8.6, 4.65), (9.8, 5.9), color=PALETTE["accent"], lw=1.8)
    arrow(ax, (8.6, 4.65), (9.8, 3.15), color=PALETTE["accent"], lw=1.8)
    arrow(ax, (2.45, 4.45), (9.8, 5.75), color="#7C3AED", lw=1.6, linestyle="--")
    label(ax, 6.2, 5.45, "original image for LUT sampling", color="#7C3AED")

    arrow(ax, (4.4, 4.05), (4.4, 3.15), color=PALETTE["skip"], lw=1.4, linestyle="--")
    arrow(ax, (5.6, 2.7), (9.8, 3.0), color=PALETTE["skip"], lw=1.6, linestyle="--")
    label(ax, 7.6, 1.95, "DINO skip features", color=PALETTE["skip"])

    arrow(ax, (12.55, 5.9), (12.55, 4.7), color=PALETTE["edge"])
    arrow(ax, (12.55, 4.2), (13.55, 4.45), color=PALETTE["edge"], lw=1.8)
    arrow(ax, (12.55, 3.15), (12.55, 4.15), color=PALETTE["edge"])

    label(ax, 11.35, 4.95, "(1-M) * LUT(img)", color="#BE185D")
    label(ax, 11.2, 3.8, "M * (img + tanh(R) * 0.5)", color="#166534")

    fig.text(
        0.03,
        0.03,
        "Decoder formula: enhanced = (1 - M) * LUT(img) + M * (img + R_scaled). "
        "In this experiment, local refinement is guided by DINO skip features rather than a second CNN encoder.",
        fontsize=9,
        color="#334155",
        bbox={"facecolor": "#F8FAFC", "edgecolor": "#CBD5E1", "pad": 6},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_experiment_lut_cnn(out_path: Path) -> None:
    fig, ax = setup_canvas(
        "LUT + CNN Recipe",
        "DINOv3-B + ShallowCNN + RWKV-V11 + CNNDecoderLUTCNN | LUT branch + gated CNN spatial fusion",
    )
    draw_common_io(ax)

    panel(
        ax,
        (2.55, 3.15),
        3.1,
        4.15,
        "Frozen Semantic Path",
        "encoder=dinov3_b\nprovides patch tokens only",
    )
    panel(
        ax,
        (2.55, 0.8),
        3.1,
        1.75,
        "Frozen Detail Path",
        "cnn_encoder=shallow_cnn\noutput_channels=64",
    )
    panel(
        ax,
        (6.0, 1.15),
        3.1,
        6.15,
        "Trainable Bottleneck",
        "rwkv_v11\nhidden_dim=384, num_layers=6",
    )
    panel(
        ax,
        (9.45, 0.7),
        3.65,
        7.1,
        "Trainable Decoder",
        "cnn_lut_cnn\nnum_luts=5, lut_size=33, gated skip fusion",
    )

    box(
        ax,
        (3.0, 5.05),
        2.2,
        1.2,
        "DINOv3 Encoder\n[FROZEN]\npatch16 -> tokens",
        PALETTE["frozen"],
        fontsize=11,
        weight="bold",
    )
    box(
        ax,
        (3.0, 1.25),
        2.2,
        0.95,
        "ShallowCNN\n[FROZEN]\n[f0, f1, f2, f3]",
        PALETTE["cnn"],
        fontsize=10,
        weight="bold",
    )
    box(
        ax,
        (6.55, 5.05),
        1.95,
        1.2,
        "RWKV-V11\nQuad-directional\n6 blocks",
        PALETTE["rwkv"],
        fontsize=10.5,
        weight="bold",
    )
    box(
        ax,
        (6.3, 2.2),
        2.45,
        1.75,
        "Per block:\n4-way RWKV scan\n+ VRSE module\n(Bi-RWKV TimeMix + OSRM)\n+ EnhancedChannelMix",
        PALETTE["trainable"],
        fontsize=9,
    )

    box(
        ax,
        (9.8, 5.45),
        2.85,
        1.0,
        "Global LUT Branch\nRWKV token[0] -> alpha\n5 base LUTs -> LUT(img)",
        PALETTE["lut"],
        fontsize=9.5,
        weight="bold",
    )
    box(
        ax,
        (9.8, 2.0),
        2.85,
        2.35,
        "Spatial Branch With CNN\nf0 init proj -> upsample blocks\nGatedSkipFusion with f1/f2/f3\nfinal refine -> M, R",
        PALETTE["spatial"],
        fontsize=9.3,
        weight="bold",
    )
    box(
        ax,
        (12.15, 4.15),
        0.8,
        0.55,
        "Fuse",
        PALETTE["fusion"],
        fontsize=10,
        weight="bold",
    )

    arrow(ax, (2.45, 4.45), (3.0, 5.65), color=PALETTE["edge"], lw=1.8)
    arrow(ax, (2.45, 4.45), (3.0, 1.72), color=PALETTE["edge"], lw=1.8)
    arrow(ax, (5.2, 5.65), (6.55, 5.65), color=PALETTE["edge"], lw=1.8)
    arrow(ax, (8.5, 5.65), (9.8, 5.95), color=PALETTE["accent"], lw=1.8)
    arrow(ax, (2.45, 4.45), (9.8, 5.82), color="#7C3AED", lw=1.6, linestyle="--")
    label(ax, 6.2, 5.25, "original image for LUT sampling", color="#7C3AED")

    arrow(ax, (5.2, 1.72), (9.8, 2.95), color=PALETTE["skip"], lw=1.7, linestyle="--")
    label(ax, 7.45, 1.82, "cnn_features = [64, 64, 64, 64]", color=PALETTE["skip"])

    arrow(ax, (12.65, 5.95), (12.65, 4.7), color=PALETTE["edge"])
    arrow(ax, (12.65, 2.95), (12.65, 4.15), color=PALETTE["edge"])
    arrow(ax, (12.95, 4.42), (13.55, 4.45), color=PALETTE["edge"], lw=1.8)

    label(ax, 11.45, 5.12, "(1-M) * LUT(img)", color="#BE185D")
    label(ax, 11.25, 3.72, "M * (img + tanh(R) * 0.5)", color="#166534")
    label(ax, 11.2, 1.25, "Gated fusion replaces DINO skip attention", color="#0F766E")

    fig.text(
        0.03,
        0.03,
        "Key difference vs the LUT-only recipe: the local branch no longer starts from RWKV tokens + DINO skips. "
        "It starts from a dedicated ShallowCNN pyramid and merges features with lightweight gated skip fusion.",
        fontsize=9,
        color="#334155",
        bbox={"facecolor": "#F8FAFC", "edgecolor": "#CBD5E1", "pad": 6},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LUT-recipe architecture diagrams")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()

    out_dir = args.out_dir
    draw_experiment_lut(out_dir / "lut_architecture.png")
    draw_experiment_lut_cnn(out_dir / "lut_cnn_architecture.png")

    print(f"Saved: {out_dir / 'lut_architecture.png'}")
    print(f"Saved: {out_dir / 'lut_cnn_architecture.png'}")


if __name__ == "__main__":
    main()

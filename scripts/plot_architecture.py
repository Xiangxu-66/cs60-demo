"""Draw architecture diagrams for Transformer and VMamba bottlenecks."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['figure.dpi'] = 150

# ═══════════════════════════════════════════════════════
# Helper drawing functions
# ═══════════════════════════════════════════════════════

def draw_box(ax, x, y, w, h, text, color='#E3F2FD', edge='#1565C0',
             fontsize=10, fontweight='normal', text_color='#212121',
             style='round,pad=0.1', alpha=0.95, subtext=None, zorder=2):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=style, facecolor=color, edgecolor=edge,
        linewidth=1.5, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)
    if subtext:
        ax.text(x, y + 0.15, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=text_color, zorder=zorder+1)
        ax.text(x, y - 0.2, subtext, ha='center', va='center',
                fontsize=fontsize - 2, color='#666666', fontstyle='italic', zorder=zorder+1)
    else:
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=text_color, zorder=zorder+1)
    return box


def draw_arrow(ax, x1, y1, x2, y2, color='#37474F', style='->', lw=1.5):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=1)


def draw_brace(ax, x, y1, y2, text, side='right'):
    """Draw a curly brace indicating repeated blocks."""
    mid_y = (y1 + y2) / 2
    offset = 0.3 if side == 'right' else -0.3
    ax.annotate('', xy=(x + offset, y1), xytext=(x + offset, y2),
                arrowprops=dict(arrowstyle='-', color='#888888', lw=1,
                                connectionstyle='arc3,rad=0'))
    ax.text(x + offset + (0.15 if side == 'right' else -0.15), mid_y,
            text, ha='left' if side == 'right' else 'right',
            va='center', fontsize=9, color='#D32F2F', fontweight='bold',
            fontstyle='italic')


# ═══════════════════════════════════════════════════════
# Colors
# ═══════════════════════════════════════════════════════
C_INPUT  = '#ECEFF1'  # gray
C_PROJ   = '#E8EAF6'  # indigo light
C_ATTN   = '#E3F2FD'  # blue light
C_FFN    = '#E8F5E9'  # green light
C_NORM   = '#FFF3E0'  # orange light
C_OUTPUT = '#FCE4EC'  # pink light
C_SSM    = '#E3F2FD'  # blue light
C_GATE   = '#F3E5F5'  # purple light
C_SCAN   = '#E0F7FA'  # cyan light
C_MERGE  = '#FFF9C4'  # yellow light
C_CONV   = '#FFECB3'  # amber light
C_BLOCK  = '#F5F5F5'  # block background

E_BLUE   = '#1565C0'
E_GREEN  = '#2E7D32'
E_ORANGE = '#E65100'
E_RED    = '#C62828'
E_PURPLE = '#6A1B9A'
E_CYAN   = '#00838F'
E_GRAY   = '#546E7A'


def draw_transformer(fig):
    """Draw Transformer bottleneck architecture."""
    ax = fig.add_axes([0.02, 0.02, 0.46, 0.96])
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 16.5)
    ax.axis('off')

    # Title
    ax.text(0, 16.2, 'Transformer Bottleneck', ha='center', va='center',
            fontsize=16, fontweight='bold', color=E_BLUE)
    ax.text(0, 15.8, 'O(N²) self-attention  |  6 layers  |  Pre-Norm',
            ha='center', va='center', fontsize=10, color='#666666')

    # ── Input ──
    draw_box(ax, 0, 15.0, 3.5, 0.55, 'Input Tokens', C_INPUT, E_GRAY,
             fontsize=11, subtext='(B, N, C_enc=768)')
    draw_arrow(ax, 0, 14.7, 0, 14.2)

    # ── proj_in ──
    draw_box(ax, 0, 13.9, 3.0, 0.5, 'Linear Projection (proj_in)', C_PROJ, E_PURPLE,
             fontsize=10, subtext='768 → 384')
    draw_arrow(ax, 0, 13.4, 0, 13.0)

    # ── + pos_embed ──
    draw_box(ax, 0, 12.7, 3.0, 0.5, '⊕ Learnable Positional Embedding', C_NORM, E_ORANGE,
             fontsize=10, subtext='(1, 4096, 384) truncated to N')
    draw_arrow(ax, 0, 12.2, 0, 11.7)

    # ── Transformer Block (detailed) ──
    # Background block
    block_bg = FancyBboxPatch(
        (-2.6, 5.5), 5.2, 6.0,
        boxstyle='round,pad=0.15', facecolor=C_BLOCK, edgecolor='#BDBDBD',
        linewidth=2, linestyle='--', alpha=0.6, zorder=0,
    )
    ax.add_patch(block_bg)
    ax.text(-2.3, 11.3, '×6 Transformer\nEncoder Layers',
            fontsize=10, fontweight='bold', color=E_RED,
            fontstyle='italic', va='top')

    # Layer Norm 1
    draw_box(ax, 0, 11.0, 2.6, 0.45, 'LayerNorm (Pre-Norm)', C_NORM, E_ORANGE, fontsize=10)
    draw_arrow(ax, 0, 10.75, 0, 10.35)

    # Multi-Head Self-Attention
    draw_box(ax, 0, 10.0, 2.6, 0.6, 'Multi-Head Self-Attention', C_ATTN, E_BLUE,
             fontsize=11, fontweight='bold', subtext='6 heads, d_k=64')

    # Residual arrow 1
    ax.annotate('', xy=(-1.6, 10.0), xytext=(-1.6, 11.5),
                arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2,
                                connectionstyle='arc3,rad=-0.4'))
    ax.text(-2.15, 10.7, 'residual', fontsize=8, color='#D32F2F', rotation=90,
            ha='center', va='center')

    # ⊕ Add
    draw_box(ax, 0, 9.2, 0.6, 0.4, '⊕', '#FFCDD2', E_RED, fontsize=12, fontweight='bold')
    draw_arrow(ax, 0, 9.65, 0, 9.42)
    draw_arrow(ax, 0, 8.98, 0, 8.6)

    # Layer Norm 2
    draw_box(ax, 0, 8.3, 2.6, 0.45, 'LayerNorm (Pre-Norm)', C_NORM, E_ORANGE, fontsize=10)
    draw_arrow(ax, 0, 8.05, 0, 7.65)

    # FFN
    draw_box(ax, 0, 7.3, 2.6, 0.6, 'Feed-Forward Network', C_FFN, E_GREEN,
             fontsize=11, fontweight='bold', subtext='384 → 1536 → 384, GELU')

    # Residual arrow 2
    ax.annotate('', xy=(-1.6, 7.3), xytext=(-1.6, 8.9),
                arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2,
                                connectionstyle='arc3,rad=-0.4'))
    ax.text(-2.15, 8.1, 'residual', fontsize=8, color='#D32F2F', rotation=90,
            ha='center', va='center')

    # ⊕ Add
    draw_box(ax, 0, 6.5, 0.6, 0.4, '⊕', '#FFCDD2', E_RED, fontsize=12, fontweight='bold')
    draw_arrow(ax, 0, 7.0, 0, 6.72)
    draw_arrow(ax, 0, 6.28, 0, 5.0)

    # Dropout
    ax.text(0.5, 10.5, 'dropout=0.1', fontsize=8, color='#999999', fontstyle='italic')
    ax.text(0.5, 7.8, 'dropout=0.1', fontsize=8, color='#999999', fontstyle='italic')

    # ── Final LayerNorm ──
    draw_box(ax, 0, 4.7, 2.6, 0.45, 'LayerNorm', C_NORM, E_ORANGE, fontsize=10)
    draw_arrow(ax, 0, 4.45, 0, 4.05)

    # ── proj_out ──
    draw_box(ax, 0, 3.7, 2.6, 0.5, 'Linear Projection (proj_out)', C_PROJ, E_PURPLE,
             fontsize=10, subtext='384 → 384')
    draw_arrow(ax, 0, 3.2, 0, 2.75)

    # ── Output ──
    draw_box(ax, 0, 2.45, 3.5, 0.55, 'Output Tokens', C_OUTPUT, E_RED,
             fontsize=11, subtext='(B, N, 384) → Decoder')

    # Complexity box
    draw_box(ax, 0, 1.2, 4.0, 0.8, '', '#FFF8E1', '#FFA000', fontsize=9)
    ax.text(0, 1.45, 'Complexity: O(N²) per layer', ha='center',
            fontsize=10, fontweight='bold', color='#E65100')
    ax.text(0, 1.0, 'N=256 tokens (16×16 patches)  |  6 heads  |  dim=384',
            ha='center', fontsize=8, color='#888888')


def draw_vmamba(fig):
    """Draw VMamba bottleneck architecture."""
    ax = fig.add_axes([0.52, 0.02, 0.46, 0.96])
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-0.5, 16.5)
    ax.axis('off')

    # Title
    ax.text(0, 16.2, 'VMamba Bottleneck (SS2D)', ha='center', va='center',
            fontsize=16, fontweight='bold', color=E_BLUE)
    ax.text(0, 15.8, 'O(N) state-space model  |  6 VSSBlocks  |  4-dir scan',
            ha='center', va='center', fontsize=10, color='#666666')

    # ── Input ──
    draw_box(ax, 0, 15.0, 3.8, 0.55, 'Input Tokens', C_INPUT, E_GRAY,
             fontsize=11, subtext='(B, N, C_enc=768)')
    draw_arrow(ax, 0, 14.7, 0, 14.2)

    # ── proj_in ──
    draw_box(ax, 0, 13.9, 3.2, 0.5, 'Linear Projection (proj_in)', C_PROJ, E_PURPLE,
             fontsize=10, subtext='768 → 384')
    draw_arrow(ax, 0, 13.4, 0, 13.05)

    # ── Reshape ──
    draw_box(ax, 0, 12.75, 3.2, 0.5, 'Reshape → 2D Spatial', C_CONV, E_ORANGE,
             fontsize=10, subtext='(B, N, C) → (B, H, W, C)')
    draw_arrow(ax, 0, 12.25, 0, 11.85)

    # ── VSSBlock (detailed) ──
    block_bg = FancyBboxPatch(
        (-3.2, 3.8), 6.4, 8.0,
        boxstyle='round,pad=0.15', facecolor=C_BLOCK, edgecolor='#BDBDBD',
        linewidth=2, linestyle='--', alpha=0.5, zorder=0,
    )
    ax.add_patch(block_bg)
    ax.text(-2.9, 11.6, '×6 VSSBlocks\n(DropPath ↑)',
            fontsize=10, fontweight='bold', color=E_RED,
            fontstyle='italic', va='top')

    # ── SS2D Branch ──
    # LayerNorm
    draw_box(ax, 0, 11.4, 2.8, 0.4, 'LayerNorm (Pre-Norm)', C_NORM, E_ORANGE, fontsize=9)
    draw_arrow(ax, 0, 11.18, 0, 10.85)

    # SS2D sub-block background
    ss2d_bg = FancyBboxPatch(
        (-2.8, 7.15), 5.6, 3.65,
        boxstyle='round,pad=0.1', facecolor='#E8F5E9', edgecolor='#66BB6A',
        linewidth=1.5, alpha=0.3, zorder=0,
    )
    ax.add_patch(ss2d_bg)
    ax.text(2.5, 10.65, 'SS2D', fontsize=10, fontweight='bold', color=E_GREEN,
            ha='center', fontstyle='italic')

    # in_proj → split
    draw_box(ax, 0, 10.6, 2.6, 0.4, 'in_proj → split(x, z)', C_PROJ, E_PURPLE, fontsize=9,
             subtext='C → 2×d_inner')

    # Branch: x path and z path
    draw_arrow(ax, -0.6, 10.15, -0.6, 9.85)  # x path
    draw_arrow(ax,  0.6, 10.15,  2.0, 9.85)   # z path

    ax.text(-1.1, 10.0, 'x', fontsize=9, fontweight='bold', color=E_BLUE)
    ax.text(1.3, 10.0, 'z', fontsize=9, fontweight='bold', color=E_PURPLE)

    # Gate: SiLU(z)
    draw_box(ax, 2.0, 9.6, 1.2, 0.4, 'SiLU(z)', C_GATE, E_PURPLE, fontsize=9)

    # DWConv + SiLU on x
    draw_box(ax, -0.6, 9.55, 2.0, 0.5, 'DWConv2d + SiLU', C_CONV, '#F57F17',
             fontsize=9, subtext='k=3, groups=d_inner')
    draw_arrow(ax, -0.6, 9.05, -0.6, 8.7)

    # 4-direction Cross Scan
    draw_box(ax, -0.6, 8.4, 2.4, 0.5, '4-Dir Cross Scan', C_SCAN, E_CYAN,
             fontsize=10, fontweight='bold', subtext='→  ↓  ←  ↑')
    draw_arrow(ax, -0.6, 7.9, -0.6, 7.6)

    # Selective Scan (S6)
    draw_box(ax, -0.6, 7.25, 2.4, 0.55, 'Selective Scan (S6)', C_SSM, E_BLUE,
             fontsize=10, fontweight='bold', subtext='A, B, C, Δ, D params')

    # x_proj annotation
    ax.text(-2.5, 8.0, 'x_proj:\nΔ, B, C', fontsize=8, color='#555555',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='#BDBDBD', alpha=0.8))
    ax.annotate('', xy=(-1.82, 7.8), xytext=(-2.2, 8.0),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=1))

    draw_arrow(ax, -0.6, 6.95, -0.6, 6.65)

    # Cross Merge
    draw_box(ax, -0.6, 6.35, 2.4, 0.5, 'Cross Merge (sum)', C_MERGE, '#F9A825',
             fontsize=10, fontweight='bold', subtext='4 dirs → 1 feature map')
    draw_arrow(ax, -0.6, 5.85, -0.6, 5.55)

    # LayerNorm + Gate multiply
    draw_box(ax, -0.6, 5.25, 1.8, 0.4, 'LayerNorm', C_NORM, E_ORANGE, fontsize=9)
    draw_arrow(ax, -0.6, 5.03, 0.4, 4.8)

    # Gate multiply ⊗
    draw_box(ax, 0.4, 4.55, 0.5, 0.4, '⊗', C_GATE, E_PURPLE, fontsize=12, fontweight='bold')
    # z path arrow down
    ax.annotate('', xy=(2.0, 4.55), xytext=(2.0, 9.38),
                arrowprops=dict(arrowstyle='->', color=E_PURPLE, lw=1.5,
                                connectionstyle='arc3,rad=0.0'))
    draw_arrow(ax, 1.75, 4.55, 0.67, 4.55)

    draw_arrow(ax, 0.4, 4.33, 0.4, 4.1)

    # out_proj
    draw_box(ax, 0, 3.85, 1.8, 0.4, 'out_proj', C_PROJ, E_PURPLE, fontsize=9)

    # SS2D residual
    ax.annotate('', xy=(-2.0, 3.85), xytext=(-2.0, 11.55),
                arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2,
                                connectionstyle='arc3,rad=-0.3'))
    ax.text(-2.9, 7.7, 'residual\n+ DropPath', fontsize=8, color='#D32F2F',
            ha='center', va='center', rotation=90)

    # ⊕
    draw_box(ax, 0, 3.3, 0.5, 0.35, '⊕', '#FFCDD2', E_RED, fontsize=11, fontweight='bold')
    draw_arrow(ax, 0, 3.65, 0, 3.49)
    draw_arrow(ax, 0, 3.11, 0, 2.85)

    # ── MLP Branch ──
    draw_box(ax, 0, 2.6, 2.8, 0.4, 'LayerNorm → MLP (GELU)', C_FFN, E_GREEN,
             fontsize=9, subtext='384 → 1536 → 384')

    # MLP residual
    ax.annotate('', xy=(-1.7, 2.6), xytext=(-1.7, 3.15),
                arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2,
                                connectionstyle='arc3,rad=-0.3'))
    ax.text(-2.1, 2.85, 'res', fontsize=7, color='#D32F2F')

    draw_box(ax, 0, 2.0, 0.5, 0.35, '⊕', '#FFCDD2', E_RED, fontsize=11, fontweight='bold')
    draw_arrow(ax, 0, 2.35, 0, 2.19)
    draw_arrow(ax, 0, 1.81, 0, 1.55)

    # ── Flatten + Norm + proj_out ──
    draw_box(ax, 0, 1.3, 3.2, 0.4, 'Flatten → LayerNorm → proj_out', C_PROJ, E_PURPLE,
             fontsize=9, subtext='(B, H, W, C) → (B, N, 384)')
    draw_arrow(ax, 0, 0.85, 0, 0.55)

    # ── Output ──
    draw_box(ax, 0, 0.25, 3.8, 0.55, 'Output Tokens', C_OUTPUT, E_RED,
             fontsize=11, subtext='(B, N, 384) → Decoder')

    # Complexity box
    draw_box(ax, 0, -0.55, 4.2, 0.7, '', '#FFF8E1', '#FFA000', fontsize=9)
    ax.text(0, -0.35, 'Complexity: O(N) per layer', ha='center',
            fontsize=10, fontweight='bold', color='#0D47A1')
    ax.text(0, -0.7, 'SS2D: 4-dir scan + S6 selective scan  |  d_state=16  |  ssm_ratio=2',
            ha='center', fontsize=8, color='#888888')


def main():
    import os
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'outputs', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(22, 16), facecolor='white')

    # Main title
    fig.suptitle(
        'Bottleneck Architecture Comparison: Transformer vs VMamba (SS2D)',
        fontsize=20, fontweight='bold', y=0.99, color='#212121'
    )

    draw_transformer(fig)
    draw_vmamba(fig)

    # Divider line
    line = plt.Line2D([0.49, 0.49], [0.03, 0.95], transform=fig.transFigure,
                      color='#BDBDBD', linewidth=2, linestyle='--')
    fig.add_artist(line)

    out_path = os.path.join(out_dir, 'architecture_transformer_vs_vmamba.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=150, facecolor='white')
    print(f'Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()

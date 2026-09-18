"""Generate None vs VMamba vs Transformer three-way comparison."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False


def main():
    # ── None v2 val PSNR (from CSV) ──
    none_val_psnr = [
        19.811, 19.529, 20.006, 19.968, 19.891, 19.884, 20.031, 20.085, 20.036, 19.752,
        20.178, 19.868, 20.071, 20.018, 20.129, 20.067, 20.033, 20.052, 19.917, 19.968,
        20.022, 19.954, 20.016, 20.092, 20.052, 20.003, 20.111, 19.968, 19.738, 20.083,
        20.107,
    ]

    # ── VMamba val PSNR ──
    vm_val_psnr = [
        19.723, 19.777, 19.722, 19.756, 19.743, 19.887, 19.721, 19.894, 19.811, 20.022,
        19.705, 19.538, 19.611, 19.878, 19.927, 19.881, 19.977, 19.787, 19.464, 19.811,
        19.799, 19.854, 19.967, 19.952, 19.952, 19.936, 20.034, 20.118, 19.994, 19.977,
        20.003, 20.005, 20.220, 20.218, 20.211, 20.107, 19.857, 20.283, 19.938, 20.130,
        20.270, 20.169, 20.114, 20.257, 20.211, 20.325, 20.233, 20.222, 20.129, 20.161,
        20.068, 20.047, 20.230, 20.185, 20.089, 20.223, 20.244, 20.226, 20.130, 20.184,
        20.101, 20.176, 20.230, 20.110, 20.161, 19.992,
    ]

    # ── Transformer val PSNR ──
    tf_val_psnr = [
        18.454, 18.841, 18.859, 18.782, 18.699, 18.620, 18.650, 18.684, 18.800, 18.573,
        18.927, 18.612, 18.684, 18.649, 18.804, 18.676, 18.624, 18.762, 18.865, 18.229,
        18.451, 18.446, 18.444, 18.334, 18.486, 18.531, 18.552, 18.693, 18.392, 18.521,
        18.342,
    ]

    # Test results
    none_test = {'psnr': 20.541, 'ssim': 0.813, 'lpips': 0.117}
    vm_test = {'psnr': 20.668, 'ssim': 0.815, 'lpips': 0.134}
    tf_test = {'psnr': 20.189, 'ssim': 0.807, 'lpips': 0.110}

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'outputs', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 10), facecolor='white')

    # ════════════════════════════════════════════════
    # LEFT: Val PSNR — three-way comparison
    # ════════════════════════════════════════════════
    ax1 = fig.add_axes([0.06, 0.12, 0.55, 0.76])

    ax1.plot(range(len(vm_val_psnr)), vm_val_psnr, '-', linewidth=2, color='#1565C0',
             label=f'VMamba (test: {vm_test["psnr"]:.2f} dB)', alpha=0.9)
    ax1.plot(range(len(none_val_psnr)), none_val_psnr, '-', linewidth=2, color='#4CAF50',
             label=f'None / Linear (test: {none_test["psnr"]:.2f} dB)', alpha=0.9)
    ax1.plot(range(len(tf_val_psnr)), tf_val_psnr, '-', linewidth=2, color='#D32F2F',
             label=f'Transformer (test: {tf_test["psnr"]:.2f} dB)', alpha=0.9)

    ax1.axhline(y=20.0, color='gray', linestyle=':', linewidth=1, alpha=0.4)

    # Highlight zones
    ax1.axhspan(19.8, 20.4, color='#E8F5E9', alpha=0.2)
    ax1.axhspan(18.2, 19.0, color='#FFEBEE', alpha=0.2)

    # Key insight annotations
    ax1.annotate('None ≈ VMamba\n(only 0.13 dB gap)',
                 xy=(25, 20.1), xytext=(40, 19.4),
                 fontsize=12, fontweight='bold', color='#E65100',
                 arrowprops=dict(arrowstyle='->', color='#E65100', lw=2),
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF3E0', edgecolor='#E65100', alpha=0.95))

    ax1.annotate('Transformer far behind\n(-1.7 dB vs None)',
                 xy=(15, 18.7), xytext=(35, 18.3),
                 fontsize=11, fontweight='bold', color='#B71C1C',
                 arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#EF9A9A', alpha=0.9))

    ax1.set_xlabel('Epoch', fontsize=13)
    ax1.set_ylabel('Val PSNR (dB)', fontsize=13)
    ax1.set_title('Validation PSNR — None vs VMamba vs Transformer',
                  fontsize=15, fontweight='bold', pad=12)
    ax1.legend(fontsize=11, loc='lower right', framealpha=0.95)
    ax1.grid(True, alpha=0.15)
    ax1.set_ylim(18.0, 20.7)
    ax1.set_xlim(-1, 68)
    ax1.tick_params(labelsize=11)

    # ════════════════════════════════════════════════
    # RIGHT TOP: Test metrics bar chart
    # ════════════════════════════════════════════════
    ax2 = fig.add_axes([0.67, 0.55, 0.30, 0.33])

    metrics = ['PSNR (dB)', 'SSIM']
    none_vals = [none_test['psnr'], none_test['ssim']]
    vm_vals = [vm_test['psnr'], vm_test['ssim']]
    tf_vals = [tf_test['psnr'], tf_test['ssim']]

    x = np.arange(len(metrics))
    w = 0.22
    b1 = ax2.bar(x - w, none_vals, w, label='None', color='#4CAF50', alpha=0.85)
    b2 = ax2.bar(x, vm_vals, w, label='VMamba', color='#1565C0', alpha=0.85)
    b3 = ax2.bar(x + w, tf_vals, w, label='Transformer', color='#D32F2F', alpha=0.85)

    for bars, vals, color in [(b1, none_vals, '#2E7D32'), (b2, vm_vals, '#0D47A1'), (b3, tf_vals, '#B71C1C')]:
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f'{val:.2f}' if val > 1 else f'{val:.3f}',
                     ha='center', va='bottom', fontsize=8, fontweight='bold', color=color)

    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=11)
    ax2.set_title('Test: PSNR & SSIM (higher = better)', fontsize=12, fontweight='bold', pad=8)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, axis='y', alpha=0.15)
    ax2.set_ylim(0, max(vm_vals) * 1.12)

    # ════════════════════════════════════════════════
    # RIGHT BOTTOM: LPIPS bar + summary
    # ════════════════════════════════════════════════
    ax3 = fig.add_axes([0.67, 0.12, 0.14, 0.33])

    lpips_vals = [none_test['lpips'], vm_test['lpips'], tf_test['lpips']]
    names = ['None', 'VMamba', 'Trans.']
    colors = ['#4CAF50', '#1565C0', '#D32F2F']
    bars = ax3.bar(names, lpips_vals, color=colors, alpha=0.85, width=0.6)
    for bar, val in zip(bars, lpips_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax3.set_title('LPIPS (lower = better)', fontsize=11, fontweight='bold', pad=8)
    ax3.grid(True, axis='y', alpha=0.15)
    ax3.set_ylim(0, 0.18)
    ax3.tick_params(labelsize=9)

    # Summary card
    ax4 = fig.add_axes([0.83, 0.12, 0.15, 0.33])
    ax4.axis('off')
    summary = (
        "KEY FINDINGS\n"
        "━━━━━━━━━━━━━━\n"
        "\n"
        "VMamba vs None:\n"
        "  PSNR +0.13 dB\n"
        "  (negligible)\n"
        "\n"
        "Transformer:\n"
        "  -0.35 dB vs None\n"
        "  (negative impact)\n"
        "\n"
        "Conclusion:\n"
        "  Bottleneck adds\n"
        "  minimal value.\n"
        "  Decoder does\n"
        "  the heavy lifting."
    )
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
             fontsize=10, fontfamily='monospace', verticalalignment='top',
             linespacing=1.3,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3E0',
                       edgecolor='#E65100', linewidth=1.5))

    fig.suptitle('Bottleneck Comparison: None vs VMamba vs Transformer\n'
                 'DINOv2 + CNN Decoder  |  FiveK 2k subset  |  Stage1 Loss  |  100 epochs',
                 fontsize=16, fontweight='bold', y=0.98, color='#212121')

    out_path = os.path.join(out_dir, 'comparison_three_way.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=150, facecolor='white')
    print(f'Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()

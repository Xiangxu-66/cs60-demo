"""Generate comprehensive comparison of ALL bottleneck architectures."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False


def main():
    # ══════════════════════════════════════════════════════
    # Validation PSNR data extracted from CSVs
    # ══════════════════════════════════════════════════════

    # CNN (20 epochs, stage1 loss, 125 steps/ep)
    cnn_val_psnr = [
        19.740, 19.857, 19.858, 19.928, 19.838, 19.901, 19.923, 19.925,
        19.971, 19.964, 19.916, 19.901, 19.841, 19.933, 19.890, 19.885,
        19.918, 19.914, 19.925, 19.923,
    ]

    # VMamba (66 epochs, stage1 loss, 125 steps/ep)
    vmamba_val_psnr = [
        19.723, 19.777, 19.722, 19.756, 19.743, 19.887, 19.721, 19.894,
        19.811, 20.022, 19.705, 19.538, 19.611, 19.878, 19.927, 19.881,
        19.977, 19.787, 19.464, 19.811, 19.799, 19.854, 19.967, 19.952,
        19.952, 19.936, 20.034, 20.118, 19.994, 19.977, 20.003, 20.005,
        20.220, 20.218, 20.211, 20.107, 19.857, 20.283, 19.938, 20.130,
        20.270, 20.169, 20.114, 20.257, 20.211, 20.325, 20.233, 20.222,
        20.129, 20.161, 20.068, 20.047, 20.230, 20.185, 20.089, 20.223,
        20.244, 20.226, 20.130, 20.184, 20.101, 20.176, 20.230, 20.110,
        20.161, 19.992,
    ]

    # Transformer (31 epochs, stage1 loss, 125 steps/ep)
    tf_val_psnr = [
        18.454, 18.841, 18.859, 18.783, 18.699, 18.620, 18.650, 18.684,
        18.800, 18.573, 18.927, 18.612, 18.684, 18.649, 18.804, 18.676,
        18.624, 18.762, 18.865, 18.229, 18.451, 18.446, 18.444, 18.334,
        18.486, 18.531, 18.552, 18.693, 18.392, 18.521, 18.342,
    ]

    # RWKV (8 epochs, stage1 loss, 200 steps/ep)
    rwkv_val_psnr = [
        20.219, 20.606, 20.997, 20.742, 20.911, 20.963, 20.273, 20.791,
    ]

    # ══════════════════════════════════════════════════════
    # Training loss data (epoch-level averages)
    # ══════════════════════════════════════════════════════
    cnn_train_loss = [
        0.1957, 0.1911, 0.1932, 0.1912, 0.1920, 0.1910, 0.1901, 0.1912,
        0.1908, 0.1903, 0.1908, 0.1893, 0.1904, 0.1903, 0.1918, 0.1894,
        0.1915, 0.1909, 0.1888, 0.1885,
    ]

    vmamba_train_loss = [
        0.1952, 0.1946, 0.1920, 0.1925, 0.1899, 0.1907, 0.1897, 0.1900,
        0.1920, 0.1887, 0.1904, 0.1897, 0.1883, 0.1882, 0.1875, 0.1873,
        0.1873, 0.1859, 0.1881, 0.1873, 0.1861, 0.1869, 0.1850, 0.1829,
        0.1844, 0.1840, 0.1834, 0.1827, 0.1826, 0.1818, 0.1827, 0.1846,
        0.1828, 0.1816, 0.1813, 0.1807, 0.1799, 0.1810, 0.1812, 0.1803,
        0.1800, 0.1801, 0.1788, 0.1781, 0.1787, 0.1772, 0.1783, 0.1785,
        0.1785, 0.1778, 0.1779, 0.1776, 0.1762, 0.1772, 0.1766, 0.1778,
        0.1746, 0.1766, 0.1766, 0.1755, 0.1748, 0.1764, 0.1758, 0.1764,
        0.1751, 0.1756,
    ]

    tf_train_loss = [
        0.1971, 0.1955, 0.1930, 0.1944, 0.1939, 0.1934, 0.1925, 0.1941,
        0.1938, 0.1913, 0.1922, 0.1920, 0.1919, 0.1921, 0.1918, 0.1898,
        0.1921, 0.1934, 0.1902, 0.1928, 0.1915, 0.1907, 0.1932, 0.1903,
        0.1914, 0.1921, 0.1908, 0.1914, 0.1906, 0.1910, 0.1907,
    ]

    rwkv_train_loss = [
        0.2130, 0.2080, 0.2078, 0.2093, 0.2082, 0.2071, 0.2066, 0.2060,
    ]

    # ══════════════════════════════════════════════════════
    # Test metrics (from JSON results / CSV final rows)
    # ══════════════════════════════════════════════════════
    test_results = {
        'None':        {'psnr': 20.323, 'ssim': 0.808, 'lpips': 0.115, 'color': '#9E9E9E'},
        'CNN':         {'psnr': 20.581, 'ssim': 0.814, 'lpips': 0.111, 'color': '#4CAF50'},
        'RWKV':        {'psnr': 20.186, 'ssim': 0.807, 'lpips': 0.118, 'color': '#FF9800'},
        'Transformer': {'psnr': 20.189, 'ssim': 0.807, 'lpips': 0.110, 'color': '#D32F2F'},
        'VMamba':      {'psnr': 20.668, 'ssim': 0.815, 'lpips': 0.134, 'color': '#1565C0'},
    }

    # Training info
    train_info = {
        'None':        {'epochs': '8 (L1-only)',    'early_stop': 'ep 7',  'time': '~10 min'},
        'CNN':         {'epochs': '20',              'early_stop': 'ep 8',  'time': '~30 min'},
        'RWKV':        {'epochs': '8',               'early_stop': 'ep 5',  'time': '~45 min'},
        'Transformer': {'epochs': '31 (early stop)', 'early_stop': 'ep 10', 'time': '~37 min'},
        'VMamba':      {'epochs': '66 (early stop)', 'early_stop': 'ep 45', 'time': '~128 min'},
    }

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'outputs', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════
    # Create figure: 2x2 layout
    # ══════════════════════════════════════════════════════
    fig = plt.figure(figsize=(22, 14), facecolor='white')

    # ────────────────────────────────────────────────────
    # (a) TOP LEFT: Validation PSNR curves
    # ────────────────────────────────────────────────────
    ax1 = fig.add_axes([0.06, 0.54, 0.55, 0.38])

    ax1.plot(range(len(vmamba_val_psnr)), vmamba_val_psnr, '-', linewidth=2,
             color='#1565C0', label='VMamba (SS2D)', alpha=0.9)
    ax1.plot(range(len(cnn_val_psnr)), cnn_val_psnr, '-', linewidth=2,
             color='#4CAF50', label='CNN', alpha=0.9)
    ax1.plot(range(len(rwkv_val_psnr)), rwkv_val_psnr, '-o', linewidth=2,
             color='#FF9800', label='RWKV', alpha=0.9, markersize=5)
    ax1.plot(range(len(tf_val_psnr)), tf_val_psnr, '-', linewidth=2,
             color='#D32F2F', label='Transformer', alpha=0.9)

    # Reference lines
    ax1.axhline(y=20.0, color='gray', linestyle=':', linewidth=1, alpha=0.4)
    ax1.axhline(y=19.0, color='gray', linestyle=':', linewidth=1, alpha=0.3)

    # Highlight zones
    ax1.axhspan(20.0, 21.1, color='#E3F2FD', alpha=0.15, label='_nolegend_')
    ax1.axhspan(18.2, 19.0, color='#FFEBEE', alpha=0.15, label='_nolegend_')

    # Annotations
    ax1.annotate('VMamba keeps\nimproving',
                 xy=(45, 20.33), xytext=(50, 20.6),
                 fontsize=10, fontweight='bold', color='#0D47A1',
                 arrowprops=dict(arrowstyle='->', color='#0D47A1', lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD',
                           edgecolor='#1565C0', alpha=0.9))

    ax1.annotate('CNN plateau\n~19.9 dB',
                 xy=(10, 19.97), xytext=(25, 19.5),
                 fontsize=10, fontweight='bold', color='#2E7D32',
                 arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9',
                           edgecolor='#4CAF50', alpha=0.9))

    ax1.annotate('RWKV high val PSNR\nbut only 8 epochs',
                 xy=(2, 21.0), xytext=(10, 21.0),
                 fontsize=9, fontweight='bold', color='#E65100',
                 arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0',
                           edgecolor='#FF9800', alpha=0.9))

    ax1.annotate('Transformer\nstagnates ~18.7',
                 xy=(15, 18.7), xytext=(35, 18.3),
                 fontsize=10, fontweight='bold', color='#B71C1C',
                 arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE',
                           edgecolor='#EF9A9A', alpha=0.9))

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation PSNR (dB)', fontsize=12)
    ax1.set_title('(a) Validation PSNR — All Bottlenecks', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(fontsize=10, loc='lower right', framealpha=0.95)
    ax1.grid(True, alpha=0.15)
    ax1.set_ylim(18.0, 21.2)
    ax1.set_xlim(-1, 68)
    ax1.tick_params(labelsize=10)

    # ────────────────────────────────────────────────────
    # (b) TOP RIGHT: Test PSNR bar chart
    # ────────────────────────────────────────────────────
    ax2 = fig.add_axes([0.67, 0.54, 0.30, 0.38])

    names = list(test_results.keys())
    psnr_vals = [test_results[n]['psnr'] for n in names]
    colors = [test_results[n]['color'] for n in names]

    bars = ax2.bar(names, psnr_vals, color=colors, alpha=0.85, width=0.6,
                   edgecolor='white', linewidth=1.5)

    # Value labels
    for bar, val, name in zip(bars, psnr_vals, names):
        fontw = 'bold' if name == 'VMamba' else 'normal'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11,
                 fontweight=fontw, color='#333333')

    # Highlight the best
    best_idx = psnr_vals.index(max(psnr_vals))
    bars[best_idx].set_edgecolor('#FFD600')
    bars[best_idx].set_linewidth(3)

    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title('(b) Test PSNR (higher = better)', fontsize=14, fontweight='bold', pad=10)
    ax2.set_ylim(19.8, 20.85)
    ax2.grid(True, axis='y', alpha=0.15)
    ax2.tick_params(labelsize=10)

    # Add rank annotation
    sorted_names = sorted(names, key=lambda n: test_results[n]['psnr'], reverse=True)
    rank_text = 'Rank: ' + ' > '.join(sorted_names)
    ax2.text(0.5, 0.02, rank_text, transform=ax2.transAxes, ha='center',
             fontsize=9, fontstyle='italic', color='#666666')

    # ────────────────────────────────────────────────────
    # (c) BOTTOM LEFT: Training loss curves
    # ────────────────────────────────────────────────────
    ax3 = fig.add_axes([0.06, 0.08, 0.35, 0.38])

    ax3.plot(range(len(vmamba_train_loss)), vmamba_train_loss, '-', linewidth=2,
             color='#1565C0', label='VMamba', alpha=0.9)
    ax3.plot(range(len(cnn_train_loss)), cnn_train_loss, '-', linewidth=2,
             color='#4CAF50', label='CNN', alpha=0.9)
    ax3.plot(range(len(rwkv_train_loss)), rwkv_train_loss, '-o', linewidth=2,
             color='#FF9800', label='RWKV', alpha=0.9, markersize=5)
    ax3.plot(range(len(tf_train_loss)), tf_train_loss, '-', linewidth=2,
             color='#D32F2F', label='Transformer', alpha=0.9)

    # Annotations
    ax3.annotate('VMamba: steady decline\n0.195 → 0.175',
                 xy=(30, 0.183), xytext=(40, 0.196),
                 fontsize=9, fontweight='bold', color='#0D47A1',
                 arrowprops=dict(arrowstyle='->', color='#0D47A1', lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD',
                           edgecolor='#1565C0', alpha=0.9))

    ax3.annotate('CNN & Transformer:\nflat ~0.19',
                 xy=(10, 0.192), xytext=(20, 0.200),
                 fontsize=9, fontweight='bold', color='#666666',
                 arrowprops=dict(arrowstyle='->', color='#888888', lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5',
                           edgecolor='#BDBDBD', alpha=0.9))

    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Training Loss', fontsize=12)
    ax3.set_title('(c) Training Loss Comparison', fontsize=14, fontweight='bold', pad=10)
    ax3.legend(fontsize=9, loc='upper right', framealpha=0.95)
    ax3.grid(True, alpha=0.15)
    ax3.set_xlim(-1, 68)
    ax3.tick_params(labelsize=10)

    # ────────────────────────────────────────────────────
    # (d) BOTTOM MIDDLE: SSIM & LPIPS bars
    # ────────────────────────────────────────────────────
    ax4 = fig.add_axes([0.46, 0.08, 0.22, 0.38])

    ssim_vals = [test_results[n]['ssim'] for n in names]
    lpips_vals = [test_results[n]['lpips'] for n in names]

    x = np.arange(len(names))
    w = 0.35
    bars_ssim = ax4.bar(x - w/2, ssim_vals, w, label='SSIM ↑', color=colors, alpha=0.7,
                        edgecolor='white', linewidth=1)
    bars_lpips = ax4.bar(x + w/2, lpips_vals, w, label='LPIPS ↓', color=colors, alpha=0.4,
                         edgecolor=colors, linewidth=1.5, hatch='///')

    for bar, val in zip(bars_ssim, ssim_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar, val in zip(bars_lpips, lpips_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax4.set_xticks(x)
    ax4.set_xticklabels(names, fontsize=9, rotation=20, ha='right')
    ax4.set_title('(d) SSIM & LPIPS', fontsize=14, fontweight='bold', pad=10)
    ax4.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax4.grid(True, axis='y', alpha=0.15)
    ax4.set_ylim(0, 0.95)
    ax4.tick_params(labelsize=9)

    # ────────────────────────────────────────────────────
    # (e) BOTTOM RIGHT: Summary table
    # ────────────────────────────────────────────────────
    ax5 = fig.add_axes([0.72, 0.08, 0.26, 0.38])
    ax5.axis('off')

    # Table data
    col_labels = ['PSNR↑', 'SSIM↑', 'LPIPS↓', 'Epochs', 'Best@']
    row_labels = list(test_results.keys())
    cell_text = []
    cell_colors = []
    for name in row_labels:
        t = test_results[name]
        info = train_info[name]
        row = [
            f'{t["psnr"]:.3f}',
            f'{t["ssim"]:.3f}',
            f'{t["lpips"]:.3f}',
            info['epochs'],
            info['early_stop'],
        ]
        cell_text.append(row)

    table = ax5.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        rowLoc='center',
        loc='center',
        bbox=[0, 0.25, 1.0, 0.7],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Color the row labels
    for i, name in enumerate(row_labels):
        table[i + 1, -1].set_facecolor(test_results[name]['color'])
        table[i + 1, -1].set_text_props(color='white', fontweight='bold')

    # Bold the best values
    best_psnr_idx = psnr_vals.index(max(psnr_vals))
    best_ssim_idx = [test_results[n]['ssim'] for n in names].index(
        max(test_results[n]['ssim'] for n in names))
    best_lpips_idx = [test_results[n]['lpips'] for n in names].index(
        min(test_results[n]['lpips'] for n in names))

    table[best_psnr_idx + 1, 0].set_text_props(fontweight='bold', color='#0D47A1')
    table[best_psnr_idx + 1, 0].set_facecolor('#E3F2FD')
    table[best_ssim_idx + 1, 1].set_text_props(fontweight='bold', color='#0D47A1')
    table[best_ssim_idx + 1, 1].set_facecolor('#E3F2FD')
    table[best_lpips_idx + 1, 2].set_text_props(fontweight='bold', color='#0D47A1')
    table[best_lpips_idx + 1, 2].set_facecolor('#E3F2FD')

    # Header styling
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#37474F')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Key findings text below table
    findings = (
        "KEY FINDINGS:\n"
        "• VMamba achieves best PSNR (20.668) & SSIM (0.815)\n"
        "• Transformer has best LPIPS (0.110) — better perceptual quality\n"
        "• CNN is a strong baseline — close to VMamba with fewer epochs\n"
        "• RWKV & Transformer similar PSNR (~20.19) but very different dynamics\n"
        "• VMamba learns steadily over 66 epochs; others plateau early"
    )
    ax5.text(0.5, 0.12, findings, transform=ax5.transAxes,
             fontsize=9, ha='center', va='top', fontfamily='monospace',
             linespacing=1.4,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8E1',
                       edgecolor='#FFA000', linewidth=1.5, alpha=0.95))

    ax5.set_title('(e) Comparison Summary', fontsize=14, fontweight='bold', pad=10)

    # ════════════════════════════════════════════════════
    # Global title
    # ════════════════════════════════════════════════════
    fig.suptitle(
        'All Bottleneck Architectures — Full Comparison\n'
        'DINOv2 Encoder + CNN Decoder  |  FiveK 2k subset  |  Stage-1 Loss',
        fontsize=17, fontweight='bold', y=0.98, color='#212121'
    )

    out_path = os.path.join(out_dir, 'comparison_all_bottlenecks.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=150, facecolor='white')
    print(f'Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()

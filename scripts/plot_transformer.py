"""Generate analysis plots for Transformer bottleneck experiment."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False


def main():
    val_epochs = list(range(0, 31))
    val_psnr = [
        18.454, 18.841, 18.859, 18.782, 18.699, 18.620, 18.650, 18.684, 18.800, 18.573,
        18.927, 18.612, 18.684, 18.649, 18.804, 18.676, 18.624, 18.762, 18.865, 18.229,
        18.451, 18.446, 18.444, 18.334, 18.486, 18.531, 18.552, 18.693, 18.392, 18.521,
        18.342,
    ]
    train_epochs = list(range(0, 31))
    train_loss = [
        0.1971, 0.1955, 0.1930, 0.1944, 0.1939, 0.1934, 0.1925, 0.1941, 0.1938, 0.1913,
        0.1922, 0.1920, 0.1919, 0.1921, 0.1918, 0.1898, 0.1921, 0.1934, 0.1902, 0.1928,
        0.1915, 0.1907, 0.1932, 0.1903, 0.1914, 0.1921, 0.1908, 0.1914, 0.1906, 0.1910,
        0.1907,
    ]
    val_loss = [
        0.1207, 0.1173, 0.1172, 0.1179, 0.1188, 0.1200, 0.1196, 0.1198, 0.1182, 0.1206,
        0.1170, 0.1205, 0.1198, 0.1201, 0.1188, 0.1196, 0.1208, 0.1187, 0.1176, 0.1247,
        0.1231, 0.1230, 0.1228, 0.1244, 0.1223, 0.1221, 0.1212, 0.1197, 0.1232, 0.1215,
        0.1234,
    ]

    test_psnr = 20.189
    test_ssim = 0.807
    test_lpips = 0.110
    best_ep = val_epochs[np.argmax(val_psnr)]
    best_val_psnr = max(val_psnr)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'outputs', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(18, 8), facecolor='white')

    # ════════════════════════════════════════════════
    # LEFT: Val PSNR — the main story
    # ════════════════════════════════════════════════
    ax1 = fig.add_axes([0.06, 0.12, 0.42, 0.72])

    # Stagnation zone
    ax1.axhspan(18.2, 19.0, color='#FFCDD2', alpha=0.3, label='Stagnation zone')
    # PSNR curve
    ax1.plot(val_epochs, val_psnr, '-o', markersize=5, linewidth=2.2,
             color='#D32F2F', markerfacecolor='white', markeredgecolor='#D32F2F', markeredgewidth=1.5,
             zorder=5)
    # Best point
    ax1.plot(best_ep, best_val_psnr, '*', markersize=18, color='#4CAF50', zorder=10,
             markeredgecolor='white', markeredgewidth=1)
    # Test line
    ax1.axhline(y=test_psnr, color='#1565C0', linestyle='--', linewidth=1.8, alpha=0.8)
    # Early stop line
    ax1.axvline(x=30, color='gray', linestyle=':', linewidth=1.2, alpha=0.6)

    # Annotations — well separated
    ax1.annotate(f'Best val: {best_val_psnr:.2f} dB\n(epoch {best_ep})',
                 xy=(best_ep, best_val_psnr), xytext=(16, 19.5),
                 fontsize=12, fontweight='bold', color='#2E7D32',
                 arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#2E7D32', alpha=0.9))

    ax1.annotate(f'Test: {test_psnr:.2f} dB',
                 xy=(2, test_psnr), xytext=(2, 20.45),
                 fontsize=12, fontweight='bold', color='#1565C0',
                 arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#1565C0', alpha=0.9))

    ax1.annotate('Early Stop\n(patience=20)', xy=(30, 18.34), xytext=(24, 17.95),
                 fontsize=10, color='#616161',
                 arrowprops=dict(arrowstyle='->', color='#616161', lw=1.2))

    # Big text: key insight
    ax1.text(15, 17.65, 'Val PSNR stagnates at ~18.7 dB\nNo learning progress after epoch 10',
             fontsize=11, color='#B71C1C', fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE', edgecolor='#EF9A9A', alpha=0.95))

    ax1.set_xlabel('Epoch', fontsize=13)
    ax1.set_ylabel('Validation PSNR (dB)', fontsize=13)
    ax1.set_title('Validation PSNR Over Training', fontsize=15, fontweight='bold', pad=12)
    ax1.set_ylim(17.4, 20.8)
    ax1.set_xlim(-1, 33)
    ax1.grid(True, alpha=0.15)
    ax1.tick_params(labelsize=11)

    # ════════════════════════════════════════════════
    # TOP RIGHT: Train vs Val Loss
    # ════════════════════════════════════════════════
    ax2 = fig.add_axes([0.56, 0.52, 0.40, 0.32])

    ax2.plot(train_epochs, train_loss, '-', linewidth=2.2, color='#D32F2F', label='Train Loss')
    ax2.plot(val_epochs, val_loss, '-', linewidth=2.2, color='#1565C0', label='Val Loss')
    ax2.fill_between(train_epochs, train_loss, val_loss, alpha=0.06, color='gray')

    # Key insight annotation
    ax2.annotate('Large gap, train barely decreases\n(0.197 → 0.191)',
                 xy=(15, 0.191), xytext=(18, 0.170),
                 fontsize=9, color='#B71C1C', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#EF9A9A', alpha=0.9))

    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title('Training vs Validation Loss', fontsize=13, fontweight='bold', pad=8)
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.15)
    ax2.set_xlim(-1, 33)
    ax2.tick_params(labelsize=10)

    # ════════════════════════════════════════════════
    # BOTTOM RIGHT: Test Results Summary
    # ════════════════════════════════════════════════
    ax3 = fig.add_axes([0.56, 0.06, 0.40, 0.36])
    ax3.axis('off')

    # Results card
    card_text = (
        f"{'━' * 40}\n"
        f"  TEST RESULTS\n"
        f"{'━' * 40}\n"
        f"  PSNR    {test_psnr:.2f} dB\n"
        f"  SSIM    {test_ssim:.3f}\n"
        f"  LPIPS   {test_lpips:.3f}  (best among all bottlenecks)\n"
        f"{'━' * 40}\n"
        f"  TRAINING SUMMARY\n"
        f"{'━' * 40}\n"
        f"  Epochs       31 / 100  (early stopped)\n"
        f"  Best val     {best_val_psnr:.2f} dB at epoch {best_ep}\n"
        f"  GPU          RTX 5090\n"
        f"  Total time   ~37 min\n"
        f"{'━' * 40}\n"
        f"  CONFIG\n"
        f"{'━' * 40}\n"
        f"  Encoder      DINOv2-B (frozen)\n"
        f"  Bottleneck   Transformer 6L/6H dim=384\n"
        f"  Decoder      CNN (PixelShuffle)\n"
        f"  Data         FiveK 2k subset, crop 252\n"
        f"  Complexity   O(N^2) self-attention\n"
    )
    ax3.text(0.05, 0.95, card_text, transform=ax3.transAxes,
             fontsize=10, fontfamily='monospace', verticalalignment='top',
             linespacing=1.4,
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#FAFAFA',
                       edgecolor='#BDBDBD', linewidth=1.5))

    # Main title
    fig.suptitle('Transformer Bottleneck — Training Analysis',
                 fontsize=18, fontweight='bold', y=0.97,
                 color='#212121')

    out_path = os.path.join(out_dir, 'transformer_analysis.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=150, facecolor='white')
    print(f'Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()

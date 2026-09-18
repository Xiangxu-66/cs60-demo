"""Generate analysis plots for VMamba (Official SS2D) bottleneck experiment."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False


def main():
    # Val PSNR per epoch (extracted from provided metrics.csv)
    val_psnr = [
        19.723, 19.777, 19.722, 19.756, 19.743, 19.887, 19.721, 19.894, 19.811, 20.022,
        19.705, 19.538, 19.611, 19.878, 19.927, 19.881, 19.977, 19.787, 19.464, 19.811,
        19.799, 19.854, 19.967, 19.952, 19.952, 19.936, 20.034, 20.118, 19.994, 19.977,
        20.003, 20.005, 20.220, 20.218, 20.211, 20.107, 19.857, 20.283, 19.938, 20.130,
        20.270, 20.169, 20.114, 20.257, 20.211, 20.325, 20.233, 20.222, 20.129, 20.161,
        20.068, 20.047, 20.230, 20.185, 20.089, 20.223, 20.244, 20.226, 20.130, 20.184,
        20.101, 20.176, 20.230, 20.110, 20.161, 19.992,
    ]
    val_epochs = list(range(len(val_psnr)))

    # Train loss per epoch
    train_loss = [
        0.1952, 0.1946, 0.1920, 0.1925, 0.1899, 0.1907, 0.1897, 0.1900, 0.1920, 0.1887,
        0.1904, 0.1897, 0.1883, 0.1882, 0.1875, 0.1873, 0.1873, 0.1859, 0.1881, 0.1873,
        0.1861, 0.1869, 0.1850, 0.1829, 0.1844, 0.1840, 0.1834, 0.1827, 0.1826, 0.1818,
        0.1827, 0.1846, 0.1828, 0.1816, 0.1813, 0.1807, 0.1799, 0.1810, 0.1812, 0.1803,
        0.1800, 0.1801, 0.1788, 0.1781, 0.1787, 0.1772, 0.1783, 0.1785, 0.1785, 0.1778,
        0.1779, 0.1776, 0.1762, 0.1772, 0.1766, 0.1778, 0.1746, 0.1766, 0.1766, 0.1755,
        0.1748, 0.1764, 0.1758, 0.1764, 0.1751, 0.1756,
    ]
    train_epochs = list(range(len(train_loss)))

    # Val loss per epoch
    val_loss = [
        0.1077, 0.1072, 0.1078, 0.1078, 0.1080, 0.1069, 0.1080, 0.1064, 0.1072, 0.1056,
        0.1085, 0.1111, 0.1100, 0.1078, 0.1080, 0.1078, 0.1072, 0.1092, 0.1120, 0.1085,
        0.1092, 0.1085, 0.1075, 0.1077, 0.1084, 0.1084, 0.1070, 0.1063, 0.1075, 0.1086,
        0.1083, 0.1081, 0.1061, 0.1061, 0.1059, 0.1071, 0.1095, 0.1051, 0.1091, 0.1067,
        0.1051, 0.1070, 0.1074, 0.1058, 0.1066, 0.1046, 0.1057, 0.1064, 0.1077, 0.1066,
        0.1081, 0.1091, 0.1067, 0.1066, 0.1084, 0.1063, 0.1061, 0.1066, 0.1076, 0.1072,
        0.1083, 0.1072, 0.1063, 0.1083, 0.1074, 0.1097,
    ]

    test_psnr = 20.668
    test_ssim = 0.815
    test_lpips = 0.134
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

    # Growth zone highlight
    ax1.axhspan(19.5, 20.0, color='#FFF9C4', alpha=0.4, label='Early learning phase')
    ax1.axhspan(20.0, 20.4, color='#C8E6C9', alpha=0.3, label='Mature phase (>20 dB)')

    # PSNR curve
    ax1.plot(val_epochs, val_psnr, '-o', markersize=4, linewidth=2,
             color='#1565C0', markerfacecolor='white', markeredgecolor='#1565C0', markeredgewidth=1.2,
             zorder=5)

    # Best point
    ax1.plot(best_ep, best_val_psnr, '*', markersize=18, color='#4CAF50', zorder=10,
             markeredgecolor='white', markeredgewidth=1)

    # Test line
    ax1.axhline(y=test_psnr, color='#D32F2F', linestyle='--', linewidth=1.8, alpha=0.8)

    # 20 dB reference
    ax1.axhline(y=20.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Early stop line
    ax1.axvline(x=65, color='gray', linestyle=':', linewidth=1.2, alpha=0.6)

    # Annotations
    ax1.annotate(f'Best val: {best_val_psnr:.2f} dB\n(epoch {best_ep})',
                 xy=(best_ep, best_val_psnr), xytext=(50, 20.55),
                 fontsize=12, fontweight='bold', color='#2E7D32',
                 arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#2E7D32', alpha=0.9))

    ax1.annotate(f'Test: {test_psnr:.2f} dB',
                 xy=(5, test_psnr), xytext=(5, 20.85),
                 fontsize=12, fontweight='bold', color='#D32F2F',
                 arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#D32F2F', alpha=0.9))

    ax1.annotate('Breaks 20 dB\n(epoch 9)', xy=(9, 20.022), xytext=(15, 19.45),
                 fontsize=10, fontweight='bold', color='#E65100',
                 arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', edgecolor='#E65100', alpha=0.9))

    ax1.annotate('Early Stop\n(patience=20)', xy=(65, val_psnr[-1]), xytext=(55, 19.25),
                 fontsize=9, color='#616161',
                 arrowprops=dict(arrowstyle='->', color='#616161', lw=1))

    # Key insight
    ax1.text(33, 19.1, 'Steady improvement from ep 0 to 45\nVal PSNR rises 19.7 → 20.3 dB (+0.6 dB)',
             fontsize=10, color='#0D47A1', fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor='#90CAF9', alpha=0.95))

    ax1.set_xlabel('Epoch', fontsize=13)
    ax1.set_ylabel('Validation PSNR (dB)', fontsize=13)
    ax1.set_title('Validation PSNR Over Training', fontsize=15, fontweight='bold', pad=12)
    ax1.set_ylim(19.0, 21.0)
    ax1.set_xlim(-1, 68)
    ax1.grid(True, alpha=0.15)
    ax1.tick_params(labelsize=11)

    # ════════════════════════════════════════════════
    # TOP RIGHT: Train vs Val Loss
    # ════════════════════════════════════════════════
    ax2 = fig.add_axes([0.56, 0.52, 0.40, 0.32])

    ax2.plot(train_epochs, train_loss, '-', linewidth=2.2, color='#1565C0', label='Train Loss')
    ax2.plot(val_epochs, val_loss, '-', linewidth=2.2, color='#43A047', label='Val Loss')
    ax2.fill_between(train_epochs, train_loss, val_loss, alpha=0.06, color='gray')

    ax2.annotate('Train loss steadily decreasing\n(0.195 → 0.175)',
                 xy=(50, 0.177), xytext=(35, 0.196),
                 fontsize=9, color='#0D47A1', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#0D47A1', lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#90CAF9', alpha=0.9))

    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title('Training vs Validation Loss', fontsize=13, fontweight='bold', pad=8)
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.15)
    ax2.set_xlim(-1, 68)
    ax2.tick_params(labelsize=10)

    # ════════════════════════════════════════════════
    # BOTTOM RIGHT: Summary card
    # ════════════════════════════════════════════════
    ax3 = fig.add_axes([0.56, 0.06, 0.40, 0.36])
    ax3.axis('off')

    card_text = (
        f"{'━' * 40}\n"
        f"  TEST RESULTS\n"
        f"{'━' * 40}\n"
        f"  PSNR    {test_psnr:.2f} dB\n"
        f"  SSIM    {test_ssim:.3f}\n"
        f"  LPIPS   {test_lpips:.3f}\n"
        f"{'━' * 40}\n"
        f"  TRAINING SUMMARY\n"
        f"{'━' * 40}\n"
        f"  Epochs       66 / 100  (early stopped)\n"
        f"  Best val     {best_val_psnr:.2f} dB at epoch {best_ep}\n"
        f"  20 dB at     epoch 9  (fast convergence)\n"
        f"  GPU          AutoDL CUDA\n"
        f"  Total time   ~128 min (~1.9 min/epoch)\n"
        f"{'━' * 40}\n"
        f"  CONFIG\n"
        f"{'━' * 40}\n"
        f"  Encoder      DINOv2-B (frozen)\n"
        f"  Bottleneck   VMamba Official SS2D 6L dim=384\n"
        f"  Decoder      CNN (PixelShuffle)\n"
        f"  Data         FiveK 2k subset, crop 252\n"
        f"  Complexity   O(N) selective scan\n"
    )
    ax3.text(0.05, 0.95, card_text, transform=ax3.transAxes,
             fontsize=10, fontfamily='monospace', verticalalignment='top',
             linespacing=1.4,
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#FAFAFA',
                       edgecolor='#BDBDBD', linewidth=1.5))

    fig.suptitle('VMamba (Official SS2D) Bottleneck — Training Analysis',
                 fontsize=18, fontweight='bold', y=0.97, color='#212121')

    out_path = os.path.join(out_dir, 'vmamba_analysis.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=150, facecolor='white')
    print(f'Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()

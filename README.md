# RWKV-Based Image Color Aesthetic Enhancement

A modular deep learning pipeline for image color aesthetic enhancement, comparing RWKV-based sequence modeling against CNN, Transformer, and other bottleneck architectures. Built with PyTorch Lightning and Hydra.

---

## Architecture Overview

The pipeline follows a frozen-encoder / trainable-bottleneck / trainable-decoder design:

```
Input Image (B, 3, H, W)
        ↓
[Frozen Encoder]      → patch tokens  (B, N, C_enc)
        ↓
[Trainable Bottleneck] → refined tokens (B, N, C_bot)
        ↓
[Trainable Decoder]   → enhanced image (B, 3, H, W)
```

Only the bottleneck and decoder are trained. The encoder is frozen throughout.

### Encoders

| Config Key | Model | Output dim |
|---|---|---|
| `dinov2_b` | DINOv2-B (default) | 768 |
| `vit_b16` | ViT-B/16 | 768 |
| `vrwkv_b` | Vision RWKV-B | 768 |

CLS tokens are discarded; only patch tokens are passed to the bottleneck.

### Bottlenecks

| Config Key | Description | Complexity |
|---|---|---|
| `none` | Linear projection only (lower-bound baseline) | O(N) |
| `cnn` | Residual conv blocks on 2D feature map | O(K²N) |
| `rwkv` | RWKV TimeMix + ChannelMix recurrence | O(N) |
| `transformer` | Multi-head self-attention (stub, not yet implemented) | O(N²) |
| `vmamba` | Vision Mamba state-space variant | — |

The RWKV bottleneck (`src/models/bottlenecks/rwkv_bottleneck.py`) implements the WKV recurrence with learned per-channel time decay and a pure-PyTorch reference kernel (no custom CUDA extension required). Six RWKV variants (`v1`–`v6`) are available under `configs/bottleneck_version/` for ablation.

### Decoder

A CNN decoder using PixelShuffle upsampling (avoids checkerboard artifacts). The output can be a full RGB image (sigmoid) or a residual delta (tanh, added to the input).

---

## Repository Structure

```
CS60-1/
├── configs/
│   ├── config.yaml              # Main entry: DINOv2 + CNN + CNN (default)
│   ├── base.yaml                # Training hyperparameters
│   ├── encoder/                 # dinov2_b, vit_b16, vrwkv_b
│   ├── bottleneck/              # none, cnn, rwkv, transformer, vmamba
│   ├── bottleneck_version/      # v1–v6 RWKV variants
│   ├── decoder/                 # cnn
│   ├── loss/                    # stage1 (L1+SSIM+Perceptual), stage2
│   ├── data/                    # fivek, ppr10k
│   └── ablation/                # 7 ablation configs
├── src/
│   ├── models/
│   │   ├── pipeline.py          # ImageEnhancementPipeline
│   │   ├── encoders/
│   │   ├── bottlenecks/
│   │   │   └── versions/        # v1–v6 RWKV implementations
│   │   └── decoders/
│   ├── losses/                  # l1, ssim, perceptual, color_histogram, nima
│   ├── data/                    # fivek_dataset, ppr10k_dataset, transforms
│   ├── evaluation/              # metrics, efficiency, visualization
│   ├── lit_module.py            # ColorEnhanceLitModule (Lightning)
│   └── utils/                  # device, checkpoint, seed, logging
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── benchmark_efficiency.py
│   ├── generate_comparison.py
│   ├── export_results.py
│   └── run_ablation.sh
├── tests/
│   ├── test_pipeline.py
│   ├── test_fivek_dataset.py
│   └── test_rwkv_bottleneck.py
├── conda/
│   ├── env-cuda.yml
│   └── env-mps.yml
├── docker/Dockerfile
├── Makefile
└── pyproject.toml
```

---

## Environment Setup

### Conda — CUDA (Linux/Windows)

```bash
conda env create -f conda/env-cuda.yml
conda activate rwkv-color
pip install -e .
```

Requires: Python 3.11, PyTorch 2.5.1, CUDA 12.4.

### Conda — MPS (macOS)

```bash
conda env create -f conda/env-mps.yml
conda activate rwkv-color
pip install -e .
```

### Docker

```bash
docker build -f docker/Dockerfile -t rwkv-color .
docker run --gpus all -e DATA_DIR=/data -v /your/data:/data rwkv-color
```

Base image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`.

---

## Data Preparation

### MIT-Adobe FiveK (default)

Expected directory layout (set via `DATA_DIR` env variable or `data.data_dir` override):

```
data/fivek/
├── input/ or raw/      # Raw input images
├── expertC/ or c/      # Expert C retouched targets (ground truth)
└── splits/
    ├── train.txt       # Optional explicit train split
    ├── val.txt         # Optional explicit validation split
    └── test.txt        # Optional explicit test split
```

The loader accepts either semantic directory names (`input/`, `expertC/`) or
the original FiveK-style names (`raw/`, `c/`).

If `splits/` is absent, the dataloader auto-splits by index using configurable
ratios. The default is:

- `train_ratio = 0.8`
- `val_ratio = 0.1`
- `test_ratio = 0.1`

Useful overrides:

```bash
# Use a different train / val / test ratio
python scripts/train.py data.train_ratio=0.7 data.val_ratio=0.15 data.test_ratio=0.15

# Keep 80 / 10 / 10, but cap each split for faster local experiments
python scripts/train.py data.train_subset_size=1000 data.val_subset_size=100 data.test_subset_size=100
```

### PPR10K

Configured via `configs/data/ppr10k.yaml`. Directory layout should mirror the FiveK structure. Refer to `src/data/ppr10k_dataset.py` for exact expectations.

---

## Training

Configuration is managed by [Hydra](https://hydra.cc/). All overrides use dot-notation on the command line.

```bash
# Default experiment: DINOv2-B + CNN bottleneck + CNN decoder, FiveK, Stage-1 loss
python scripts/train.py experiment_name=dino_cnn_cnn

# RWKV bottleneck
python scripts/train.py bottleneck=rwkv experiment_name=dino_rwkv_cnn

# No-bottleneck baseline
python scripts/train.py bottleneck=none experiment_name=dino_none_cnn

# Switch encoder
python scripts/train.py encoder=vit_b16 experiment_name=vit_cnn_cnn

# Switch dataset to PPR10K
python scripts/train.py data=ppr10k experiment_name=dino_cnn_cnn_ppr10k

# Stage-2 loss
python scripts/train.py loss=stage2 experiment_name=dino_cnn_cnn_stage2

# Quick smoke test (2 epochs, tiny train / val / test)
python scripts/train.py training.max_epochs=2 data.batch_size=2 \
  data.train_subset_size=8 data.val_subset_size=4 data.test_subset_size=4 \
  experiment_name=debug
```

Default training hyperparameters (from `configs/base.yaml`):

| Parameter | Value |
|---|---|
| Epochs | 200 |
| Batch size | 16 |
| Optimizer | AdamW (lr=1e-4, wd=1e-4) |
| Scheduler | Cosine annealing (T_max=200, eta_min=1e-6) |
| Grad clip | 1.0 |
| Early stopping | patience=20 on `val/psnr` |
| Checkpoint | Top-3 by `val/psnr` |

For smaller local experiments, it is often useful to cap validation and test
explicitly as well:

```bash
python scripts/train.py \
  data.train_subset_size=1000 \
  data.val_subset_size=100 \
  data.test_subset_size=100
```

### Makefile Shortcuts

```bash
make train               # Default training run
make train-debug         # 2-epoch smoke test
make train-100           # 100-sample subset run
make evaluate            # Standalone evaluation
make benchmark           # FLOPs + latency profiling
make ablation-bottleneck # none / cnn / rwkv comparison
make ablation-depth      # CNN depth sweep (2 / 4 / 6 layers)
make ablation-loss       # Loss component ablations
make test                # Run pytest
make lint                # Code linting
make format              # Auto-format
```

---

## Evaluation

```bash
# Evaluate a checkpoint
python scripts/evaluate.py checkpoint=outputs/checkpoints/<name>.ckpt

# Benchmark efficiency (FLOPs, latency, memory)
python scripts/benchmark_efficiency.py bottleneck=rwkv

# Generate visual comparison grids
python scripts/generate_comparison.py experiment_name=<name>

# Export metrics to JSON
python scripts/export_results.py experiment_name=<name>
```

Metrics computed: **PSNR** (dB), **SSIM** (0–1), **LPIPS** (AlexNet, lower is better).

---

## Loss Functions

Stage-1 loss (`configs/loss/stage1.yaml`):

```
total = L1 + 0.1 * (1 - SSIM) + 0.01 * Perceptual
```

Additional components available (used in ablations or stage 2): `ColorHistogramLoss`, `NIMALoss`. Weights are fully configurable via Hydra.

---

## RWKV Variant Ablations

Six RWKV bottleneck variants are defined under `configs/bottleneck_version/`:

| Version | Description |
|---|---|
| `v1_base` | Baseline VRWKV |
| `v2_hsv_scan` | HSV color-space sequential scan ordering |
| `v3_hierarchical` | Multi-scale hierarchical processing |
| `v4_cla` | Channel-wise and location-aware mixing |
| `v5_global_token` | Global token aggregation |
| `v6_selective_decay` | Learnable per-layer decay rates |

```bash
python scripts/train.py bottleneck=rwkv bottleneck_version=v3_hierarchical \
  experiment_name=rwkv_v3
```

This should be confirmed against the corresponding config and implementation files.

---

## Outputs

```
outputs/
├── checkpoints/
│   └── {experiment_name}_epoch{N}-{psnr:.2f}.ckpt
├── logs/{experiment_name}/
│   ├── config/config.yaml       # Full config snapshot
│   └── version_0/
│       ├── metrics.csv
│       └── hparams.yaml
└── figures/{experiment_name}/
    ├── epoch_0000.png           # Input | Prediction | Target
    └── ...
results/
└── {experiment_name}_results.json
```

---

## Tests

```bash
pytest tests/ -v
```

| Test file | Coverage |
|---|---|
| `test_pipeline.py` | Encoder → bottleneck → decoder integration |
| `test_fivek_dataset.py` | Dataset loading and transforms |
| `test_rwkv_bottleneck.py` | RWKV forward pass correctness |

---

## Dependencies

Core dependencies (see `pyproject.toml` for full list):

- `torch >= 2.5`, `torchvision >= 0.20`
- `pytorch-lightning >= 2.2`
- `hydra-core >= 1.3`, `omegaconf >= 2.3`
- `timm >= 0.9`
- `einops >= 0.7`
- `lpips >= 0.1.4`
- `scikit-image >= 0.22`
- `fvcore >= 0.1.5` (efficiency benchmarking)

---

## Notes

- The `transformer` bottleneck is listed in configs but its implementation is marked as TODO in `src/models/bottlenecks/transformer_bottleneck.py`.
- The RWKV WKV kernel uses a pure-PyTorch reference implementation by default. A CUDA-optimized path can be enabled via the `RWKV_FLOAT_MODE=bf16` environment variable (see `docker/Dockerfile`).
- Based on the current repository structure, the `vrwkv_b` encoder wraps a Vision RWKV backbone loaded via `timm`. This should be confirmed from `src/models/encoders/vrwkv_encoder.py`.

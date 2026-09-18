# RWKV-Based Image Color Aesthetic Enhancement

A modular deep learning pipeline for image color aesthetic enhancement, comparing RWKV-based sequence modeling against CNN, Transformer, and other bottleneck architectures. Built with PyTorch Lightning and Hydra.

---

## Architecture Overview

The default pipeline follows a frozen-encoder / trainable-bottleneck / trainable-decoder design:

```
Input Image (B, 3, H, W)
        ↓
[Frozen Encoder]      → patch tokens  (B, N, C_enc)
        ↓
[Trainable Bottleneck] → refined tokens (B, N, C_bot)
        ↓
[Trainable Decoder]   → enhanced image (B, 3, H, W)
```

Only the main encoder is frozen throughout. Some experiments also attach an
optional CNN encoder (for example ColorNAF) whose later stages can remain
trainable under a semi-frozen policy.

### Encoders

| Config Key | Model | Output dim |
| --- | --- | --- |
| `dinov2_b` | DINOv2-B (default) | 768 |
| `vit_b16` | ViT-B/16 | 768 |
| `vrwkv_b` | Vision RWKV-B | 768 |

CLS tokens are discarded; only patch tokens are passed to the bottleneck.

### Bottlenecks

| Config Key | Description | Complexity |
| --- | --- | --- |
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
│   ├── family/                  # Architecture baselines for experiment families
│   ├── variant/                 # Tuned multi-change recipes built on families
│   ├── encoder/                 # dinov2_b, vit_b16, vrwkv_b
│   ├── bottleneck/              # none, cnn, rwkv, transformer, vmamba
│   ├── bottleneck_version/      # v1–v6 RWKV variants
│   ├── decoder/                 # cnn
│   ├── loss/                    # stage1 (L1+SSIM+Perceptual), stage2
│   ├── data/                    # fivek, ppr10k
│   └── ablation/                # Legacy flat ablations + grouped family overrides
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

### Family + Ablation Workflow

Use `family=...` to select a canonical architecture baseline, then stack
single-variable overrides from grouped ablation configs. Use `variant=...`
for tuned multi-change recipes that should not be treated as ablations:

```bash
# Canonical ColorNAF + RWKV7 baseline
python scripts/train.py family=color_naf_rwkv7

# Remove ColorNAF skip fusion only
python scripts/train.py family=color_naf_rwkv7 \
  ablation/color_naf_rwkv7/naf_skip=none \
  experiment_name=color_naf_no_skip

# Enable FiLM only
python scripts/train.py family=color_naf_rwkv7 \
  ablation/color_naf_rwkv7/film=on \
  experiment_name=color_naf_film

```

The old one-file experiment wrappers under `configs/experiment*.yaml` have
been removed. Prefer direct Hydra composition with `family=...`, `variant=...`,
and explicit overrides.

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
└── expertC/ or c/      # Expert C retouched targets (ground truth)
```

The loader accepts either semantic directory names (`input/`, `expertC/`) or
the original FiveK-style names (`raw/`, `c/`).

Fixed split files live in `configs/splits/five/`, so the experiment protocol is
not tied to the dataset location. If those files do not match the current
dataset, the dataloader falls back to `DATA_DIR/splits/`, then auto-splits by
index using configurable ratios. The default ratio fallback is:

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

Recommended fixed-split workflow for FiveK:

- For paper-style comparison, keep a fixed 500-image `test.txt` that matches
  the protocol or public repo you want to compare against.
- For day-to-day development, sample `val.txt` from the remaining 4500 images
  with a fixed seed so you get a reproducible `4000 train / 500 val / 500 test`
  split.

```bash
# Seeded fallback: split the whole dataset once and write repo split files
python scripts/create_fixed_splits.py \
  --data-dir /path/to/fivek \
  --output-dir configs/splits/five \
  --seed 42

# Paper-aligned workflow: preserve a known 500-image test list, then sample
# 500 validation images from the remaining 4500.
python scripts/create_fixed_splits.py \
  --data-dir /path/to/fivek \
  --test-list /path/to/paper_test.txt \
  --output-dir configs/splits/five \
  --val-size 500 \
  --seed 42
```

When `--test-list` is provided, the script keeps those test samples untouched
and only samples validation from the remaining pool. If `--val-size` is omitted,
it defaults to the same size as the explicit test list. The script writes
`train.txt`, `val.txt`, `test.txt`, and `metadata.yaml` under
`configs/splits/five/`.

For final paper-style comparison runs, you can reuse the same fixed split files
but merge `train + val` into a 4500-image training pool and disable validation:

```bash
python scripts/train.py \
  family=color_naf_rwkv7 \
  ablation/color_naf_rwkv7/film=on \
  +run_mode=final_4500_no_val
```

This run mode:

- trains on the full 4500-image train+val pool
- keeps the fixed 500-image test split unchanged
- skips validation / early stopping
- saves and tests the `last` checkpoint
- appends `_final4500` to `experiment_name`

### PPR10K

Configured via `configs/data/ppr10k.yaml`. Directory layout should mirror the FiveK structure. Refer to `src/data/ppr10k_dataset.py` for exact expectations.

---

## Benchmark Protocol

This section fixes all evaluation choices so that results are comparable across experiments and with published papers (Image-Adaptive 3DLUT, CSRNet, AdaInt, CLUT-Net, ICELUT).

### Dataset

| Item | Choice |
| --- | --- |
| Dataset | MIT-Adobe FiveK |
| Preprocessing | Short-edge 512 PNG (standard preprocessed release) |
| Ground truth | Expert C |
| Task | Paired supervised image enhancement |

### Split

| Phase | Split | Purpose |
| --- | --- | --- |
| Development | 4000 train / 500 val / 500 test | Early stopping, hyperparameter search |
| Final paper comparison | 4500 train / 500 test | Merge train+val, no validation |

Both phases use the **same fixed `configs/splits/five/test.txt`** (500 images). The development val split is sampled from the remaining 4500 with seed 42. Always use the split files generated by `create_fixed_splits.py` — never rely on the automatic ratio fallback for reported results.

### Resolution

| Stage | Mode | Result |
| --- | --- | --- |
| Training | Short-edge resize to 480 px, aspect ratio preserved | Variable-size batches, padded at collate |
| Validation | Short-edge resize to 480 px, aspect ratio preserved | Metrics computed on valid pixels only (padding excluded) |
| Test | Short-edge resize to 480 px, aspect ratio preserved | Same as validation |

This matches the 480p evaluation space used by the reference papers.

### Metrics

| Metric | When | Notes |
| --- | --- | --- |
| PSNR (dB) | val + test | Higher is better |
| SSIM | val + test | Higher is better |
| LPIPS (AlexNet) | val + test | Lower is better; enabled when lpips loss is active, or set `enable_lpips: true` |
| ΔE (CIEDE2000) | test only | Lower is better; pure PyTorch, no extra dependency |
| NIMA | test only | Aesthetic quality; set `enable_nima: false` to skip |

All metrics are computed **per image** then averaged, not on a single stacked batch tensor.

### Running the final 4500/500 benchmark

```bash
# 1. Generate fixed split files (one-time setup)
python scripts/create_fixed_splits.py \
  --data-dir /path/to/fivek \
  --output-dir configs/splits/five \
  --seed 42

# 2. Develop with 4000/500/500 as usual, then run the final comparison:
python scripts/train.py \
  family=color_naf_rwkv7 \
  +run_mode=final_4500_no_val \
  experiment_name=my_model_final
```

The `final_4500_no_val` run mode merges train+val, disables early stopping, saves the last checkpoint, and appends `_final4500` to the experiment name.

---

## Experiment Report & Team Sharing

Every training run automatically produces a **self-contained experiment directory** under `experiments/` when `on_test_end` fires.  Team members download this directory and upload it to the shared Microsoft 365 folder — no Git or external tooling needed for result sharing.

### What each experiment directory contains

```text
experiments/{name}_{YYYYMMDD}_{commit}/
├── README.txt              Human-readable one-page summary (opens in M365 preview)
├── experiment_record.json  Full metadata: 38 fields covering arch, data, training, results
├── config.yaml             Fully-resolved Hydra config — sufficient to reproduce the run
├── training_curve.png      Loss + Val-PSNR vs Epoch with best-epoch marker
├── metrics.csv             Raw per-step/epoch data from Lightning CSVLogger
├── test_results.json       Test-phase metric averages generated from this run
├── test_per_image_metrics.csv   Per-test-image metrics using evaluation.test gates
├── test_per_image_metrics.json  Same per-image metrics in JSON form
├── test_best_psnr.png      5 best samples  (Input | Prediction | Expert C | Diff×5)
├── test_median_psnr.png    5 median samples
└── test_worst_psnr.png     5 worst samples — primary failure-analysis resource
```

**Checkpoint** is kept on the server; its path is recorded in `experiment_record.json` and `README.txt`.

### Global summary table

```text
experiments/results_summary.csv   ← one row per experiment, sorted by test PSNR
```

Rebuild this table at any time after adding new experiment directories:

```bash
python scripts/rebuild_summary.py                          # scans experiments/ locally
python scripts/rebuild_summary.py --experiments-dir /path  # custom location
```

The script prints a ranked comparison table to the terminal and writes the CSV. It is safe to run repeatedly — it rebuilds from scratch each time, so there are no merge conflicts.

### Team workflow (per experiment)

```bash
# 1. Train on server:
python scripts/train.py family=color_naf_rwkv7 experiment_name=my_exp
# → experiments/my_exp_20260508_a3f2c1/ is created automatically at end of training

# 2. Download the directory from the server (scp / rsync / file manager):
scp -r user@server:~/project/experiments/my_exp_20260508_a3f2c1 ./experiments/

# 3. Regenerate the summary table locally:
python scripts/rebuild_summary.py

4. Upload to M365 shared folder:
   - Upload experiments/my_exp_20260508_a3f2c1/ as a folder
   - Upload experiments/results_summary.csv (replaces the previous version)
```

### Regenerate report post-hoc

If you need to regenerate the visualisation from an existing checkpoint (e.g., for a rerun with a different test set):

```bash
python scripts/generate_test_report.py \
    --checkpoint /path/to/best.ckpt \
    --config     outputs/logs/my_exp/config/config.yaml \
    --output-dir experiments/my_exp_20260508_a3f2c1
```

---

## Training

Configuration is managed by [Hydra](https://hydra.cc/). All overrides use dot-notation on the command line.

```bash
# Default experiment: DINOv3-B + CNN bottleneck + CNN decoder, FiveK, Stage-1 loss
python scripts/train.py experiment_name=dinov3_cnn_cnn

# RWKV bottleneck
python scripts/train.py bottleneck=rwkv experiment_name=dino_rwkv_cnn

# No-bottleneck baseline
python scripts/train.py bottleneck=none experiment_name=dino_none_cnn

# Switch encoder
python scripts/train.py encoder=vit_b16 experiment_name=vit_cnn_cnn

# Switch dataset to PPR10K
python scripts/train.py data=ppr10k experiment_name=dinov3_cnn_cnn_ppr10k

# Stage-2 loss
python scripts/train.py loss=stage2 experiment_name=dinov3_cnn_cnn_stage2

# Quick smoke test (2 epochs, tiny train / val / test)
python scripts/train.py training.max_epochs=2 training.batch_size_per_device=2 \
  data.train_subset_size=8 data.val_subset_size=4 data.test_subset_size=4 \
  experiment_name=debug
```

Default training hyperparameters (from `configs/base.yaml`):

| Parameter | Value |
| --- | --- |
| Epochs | 20 |
| Per-device batch size | 8 |
| Gradient accumulation | 4 |
| Effective batch size | 32 |
| Optimizer | AdamW (lr=1e-4, wd=1e-4) |
| Scheduler | Cosine annealing (T_max=20, eta_min=1e-6) |
| Grad clip | 1.0 |
| Early stopping | patience=5 on `val/psnr` |
| Checkpoint | Top-3 by `val/psnr` |

For smaller local experiments, it is often useful to cap validation and test
explicitly as well:

```bash
python scripts/train.py \
  data.train_subset_size=1000 \
  data.val_subset_size=100 \
  data.test_subset_size=100
```

### Batch Size & Gradient Accumulation

The project supports **gradient accumulation** to enable large effective batch
sizes with limited GPU memory. The effective batch size is:

```text
effective_batch = batch_size_per_device × accumulate_grad_batches
```

**Why this matters:** This project uses LayerNorm (not BatchNorm), so changing
the batch size does not affect model behavior. Gradient accumulation lets you
train with the same gradient estimation as a large batch while using less memory.

| Configuration   | Per-device Batch | Accumulate | Effective Batch | GPU Memory |
|-----------------|------------------|------------|-----------------|------------|
| Default         | 8                | 4          | 32              | ~8GB       |
| Memory-saver    | 4                | 8          | 32              | ~5GB       |
| Ultra-low       | 2                | 16         | 32              | ~3GB       |

```bash
# Use smaller batch size with gradient accumulation (recommended for limited GPU)
python scripts/train.py training.batch_size_per_device=4 training.accumulate_grad_batches=8

# Even more aggressive: batch=2, accumulate=16 for the same effective batch
python scripts/train.py training.batch_size_per_device=2 training.accumulate_grad_batches=16

# LUT decoder with complex architecture (use smallest batch)
python scripts/train.py decoder=cnn_lut training.batch_size_per_device=2 training.accumulate_grad_batches=16
```

**Note:** Learning rate should remain based on the effective batch size, so no
adjustment is needed when using gradient accumulation.

### Makefile Shortcuts

```bash
make help                # Show grouped targets and the recommended ARGS workflow
make train               # Preferred training entry point; pass Hydra overrides in ARGS
make train-debug         # 2-epoch smoke test
make train-160           # 160-sample subset run
make evaluate ARGS='--checkpoint ... --config ...'
make benchmark           # FLOPs + latency profiling
make ablation-bottleneck # none / cnn comparison
make ablation-depth      # CNN depth sweep (2 / 4 / 6 layers)
make ablation-loss       # Loss component ablations
make test                # Run pytest
make lint                # Code linting
make format              # Auto-format
```

Preferred pattern for new runs:

```bash
make train ARGS='family=color_naf_rwkv7 experiment_name=my_run'
make train ARGS='family=color_naf_rwkv7 ablation/color_naf_rwkv7/film=on experiment_name=my_run'
```

---

## Evaluation

### Single Model Evaluation

```bash
# Evaluate a checkpoint
python scripts/evaluate.py --checkpoint outputs/checkpoints/<name>.ckpt \
    --config outputs/logs/<experiment>/config/config.yaml

# Benchmark efficiency (FLOPs, latency, memory)
python scripts/benchmark_efficiency.py bottleneck=rwkv

# Generate visual comparison grids
python scripts/generate_comparison.py experiment_name=<name>

# Export metrics to JSON
python scripts/export_results.py experiment_name=<name>
```

Metrics computed: **PSNR** (dB), **SSIM** (0–1), **LPIPS** (AlexNet, lower is better).

---

## Multi-Model Comparison

This project provides utilities for comparing multiple trained models side-by-side.

### Step 1: Generate Prediction Images (Per Model)

For each trained model, generate prediction images using `eval_single_model.py`:

```bash
python scripts/eval_single_model.py \
    --checkpoint outputs/checkpoints/model.ckpt \
    --config outputs/logs/experiment/config/config.yaml \
    --output-dir outputs/eval_single \
    --num-samples 30 \
    --pred-only \
    --device cpu
```

**Key Options:**
| Option | Description |
|--------|-------------|
| `--checkpoint` | Path to model checkpoint (.ckpt) |
| `--config` | Path to config snapshot (config.yaml) |
| `--output-dir` | Output directory for generated images |
| `--num-samples` | Number of samples to generate (default: 30) |
| `--strategy` | Sampling strategy: `equidistant` \| `difficulty` \| `scene` \| `custom` |
| `--indices` | Custom sample indices (for `custom` strategy) |
| `--pred-only` | Only save predictions (input/target are shared across models) |
| `--device` | Device to use: `auto` \| `cpu` \| `cuda` |

**Output Structure:**
```
outputs/eval_single/{model_name}/
├── pred_{model_name}_0000.png   # Prediction images
├── pred_{model_name}_0001.png
├── ...
└── metadata.json                 # Sample indices and file mappings
```

**Sampling Strategies** (configured in `configs/eval_samples.yaml`):
- `equidistant`: Uniform spacing across test set (default)
- `difficulty`: Based on PSNR quantiles (requires baseline)
- `scene`: Manual scene categories (portraits, landscape, etc.)
- `custom`: Explicit index list

### Step 2: Merge Multiple Models

After generating predictions for multiple models, merge them into side-by-side comparisons using the config file:

```bash
python scripts/merge_model_predictions.py --config configs/merge_img.yaml
```

**Configuration File** (`configs/merge_img.yaml`):

```yaml
# Dataset paths
data_dir: /path/to/dataset
input_subdir: Original        # or: raw, a
target_subdir: expertC       # or: c

# Model prediction zip files - specify directory OR list individual files
zip_dir: /path/to/predictions  # Directory containing .zip files (auto-scanned)

# Column display names (maps zip filename stem to display name)
model_columns:
  model1_zip_name: "Model 1 Display Name"
  model2_zip_name: "Model 2 Display Name"

# Output settings
output_dir: outputs/model_comparison
max_samples: null              # Limit number of comparison images (null = all)
```

**Key Options:**
| Option | Description |
|--------|-------------|
| `--config` | Path to YAML config file with all settings |
| `--zip-files` | Explicit zip file paths (overrides config) |
| `--data-dir` | Dataset root directory (overrides config) |
| `--input-subdir` | Input subdirectory name (overrides config) |
| `--target-subdir` | Target subdirectory name (overrides config) |
| `--max-samples` | Maximum number of comparison images |
| `--output-dir` | Output directory (overrides config) |

**Output Structure:**
```
outputs/model_comparison/
├── comparison_0000.png      # Single sample: [Input] [ModelA] [ModelB] [ModelC] [Target]
├── comparison_0016.png
├── ...
├── comparison_grid.png     # All samples in one grid image
├── comparison_summary.json  # Model names and sample indices
└── comparison_summary.md    # Human-readable markdown table
```

**Comparison Layout:**
- Columns: `[Input]` → `[Model1]` → `[Model2]` → ... → `[Target]`
- Column headers use display names from `model_columns` config
- Rows: Different test samples
- All images are resized to `480 × 480` for consistent comparison

**Workflow for Multi-Machine Teams:**

1. Each team member trains their model and saves the best checkpoint
2. Each member runs `eval_single_model.py` with `--pred-only` to generate a zip of predictions
3. Collect all zip files in one directory
4. Edit `configs/merge_img.yaml` to set `zip_dir` and add display names to `model_columns`
5. Run `merge_model_predictions.py --config configs/merge_img.yaml`
6. Share the `comparison_grid.png` and `comparison_summary.md`

---

## Loss Functions

Stage-1 loss (`configs/loss/stage1.yaml`):

```
total = L1 + 0.1 * (1 - SSIM) + 0.01 * LPIPSLoss
```

Additional components available (used in ablations or stage 2): `ColorHistogramLoss`, `NIMALoss`. Weights are fully configurable via Hydra.

---

## RWKV Variant Ablations

Six RWKV bottleneck variants are defined under `configs/bottleneck_version/`:

| Version | Description |
| --- | --- |
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
| --- | --- |
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

#!/usr/bin/env bash
# Cloud / AutoDL initialisation script.
# Run once after cloning the repo on a fresh GPU instance.
#
# Usage:
#   DATA_PATH=/root/autodl-tmp/fivek bash scripts/setup_cloud.sh
set -euo pipefail

echo "=== RWKV Color Enhancement — Cloud Setup ==="

# 1. Create conda environment
if conda env list | grep -q "rwkv-color"; then
    echo "[1/4] Updating existing conda env..."
    conda env update -f conda/env-cuda.yml --prune
else
    echo "[1/4] Creating conda env..."
    conda env create -f conda/env-cuda.yml
fi

# 2. Activate (for subshell — caller must activate manually)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rwkv-color

# 3. Link data directory
DATA_PATH="${DATA_PATH:-/root/autodl-tmp/fivek}"
if [ -d "${DATA_PATH}" ]; then
    ln -sfn "${DATA_PATH}" data/fivek
    echo "[2/4] Linked data: ${DATA_PATH} → data/fivek"
else
    echo "[2/4] WARNING: DATA_PATH='${DATA_PATH}' not found."
    echo "       Set DATA_PATH env var or create the symlink manually:"
    echo "       ln -s /your/path/to/fivek data/fivek"
fi

# 4. Create output directories
mkdir -p outputs/checkpoints outputs/logs outputs/figures outputs/eval
echo "[3/4] Created output directories."

# 5. Verify installation
echo "[4/4] Verifying installation..."
python -c "
import torch
print(f'  PyTorch    : {torch.__version__}')
print(f'  CUDA avail : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU        : {torch.cuda.get_device_name(0)}')
"
python -c "
from src.models.decoders.cnn_decoder import CNNDecoder
from src.models.pipeline import ImageEnhancementPipeline
print('  Imports    : OK')
"

echo ""
echo "=== Setup complete ==="
echo "Run a debug train with:"
echo "  make train-debug ARGS='bottleneck=none experiment_name=smoke_test'"

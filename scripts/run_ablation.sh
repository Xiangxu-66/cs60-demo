#!/usr/bin/env bash
# Batch ablation script for the supported Stage 5 baseline.
#
# Usage:
#   bash scripts/run_ablation.sh                    # full run
#   bash scripts/run_ablation.sh "training.max_epochs=50"  # shorter run
#
# All extra arguments are forwarded to every train.py call.
set -euo pipefail

EXTRA="${*}"
PYTHON_BIN="${PYTHON:-python}"
CONFIG_NAME="${CONFIG_NAME:-config}"
TRAIN_CMD=("${PYTHON_BIN}" scripts/train.py --config-name "${CONFIG_NAME}")

log() { echo -e "\n\033[1;36m>>> $*\033[0m"; }

# ── Line 1: Bottleneck lower-bound comparison ─────────────────────────
log "Line 1 — Bottleneck comparison (none vs cnn)"
for BN in none cnn; do
    log "  bottleneck=${BN}"
    "${TRAIN_CMD[@]}" \
        encoder=dinov2_b \
        decoder=cnn \
        bottleneck="${BN}" \
        experiment_name="A_bn_${BN}" \
        ${EXTRA}
done

# ── Line 2: CNN bottleneck depth ──────────────────────────────────────
log "Line 2 — CNN bottleneck depth"
for D in 2 4 6; do
    log "  depth=${D}"
    "${TRAIN_CMD[@]}" \
        encoder=dinov2_b \
        decoder=cnn \
        bottleneck=cnn \
        bottleneck.num_layers="${D}" \
        experiment_name="A_cnn_depth_${D}" \
        ${EXTRA}
done

# ── Line 3: Loss composition ───────────────────────────────────────────
log "Line 3 — Loss composition"
"${TRAIN_CMD[@]}" experiment_name=A_loss_stage1 ${EXTRA}
"${TRAIN_CMD[@]}" ablation=a_loss_l1_only experiment_name=A_loss_l1 ${EXTRA}
"${TRAIN_CMD[@]}" ablation=a_loss_l1_ssim experiment_name=A_loss_l1_ssim ${EXTRA}

log "All ablations complete."

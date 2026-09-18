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

log() { echo -e "\n\033[1;36m>>> $*\033[0m"; }

# ── Line 1: Bottleneck lower-bound comparison ─────────────────────────
log "Line 1 — Bottleneck comparison (none vs cnn)"
for BN in none cnn; do
    log "  bottleneck=${BN}"
    python scripts/train.py \
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
    python scripts/train.py \
        encoder=dinov2_b \
        decoder=cnn \
        bottleneck=cnn \
        bottleneck.num_layers="${D}" \
        experiment_name="A_cnn_depth_${D}" \
        ${EXTRA}
done

# ── Line 3: Loss composition ───────────────────────────────────────────
log "Line 3 — Loss composition"
python scripts/train.py experiment_name=A_loss_stage1 ${EXTRA}
python scripts/train.py ablation=a_loss_l1_only experiment_name=A_loss_l1 ${EXTRA}
python scripts/train.py ablation=a_loss_l1_ssim experiment_name=A_loss_l1_ssim ${EXTRA}

log "All ablations complete."

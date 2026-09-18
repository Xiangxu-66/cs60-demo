# Ablation Plan

This document is the index for the current RWKV color enhancement experiment plans.

Before running experiments:

```bash
conda activate rwkv-color
```

## Plan Documents

| Document | Scope |
| --- | --- |
| [Initial Baseline](plans/initial_baseline.md) | Minimal DINOv3 + RWKV7/RWKV7-vision + CNN-only reference runs |
| [Encoder Family Ablation](plans/encoder_family_ablation.md) | Three encoder-family groups with feature-skip on/off comparisons |
| [Loss Comparison](plans/loss_ablation.md) | Follow-up loss sweep on the selected carrier architecture |

## Shared Protocol

- Loss: use `loss=stage1` for the current architecture batches. Run loss
  ablations only as the separate follow-up batch in
  [Loss Comparison](plans/loss_ablation.md).
- Bottleneck: use `rwkv7_vision` by default. Use `ablation/common/bottleneck=rwkv7` only as the phase-1/fallback comparison.
- Image skip: disabled by default with `decoder.img_skip_enabled=false`; run img-skip ablation only after selecting a strong model.
- Primary comparison stage: `train-2000` first, then confirm selected runs with `train-5000`.
- Final paper-style stage: after selecting winners, run `train-final`
  (`run_mode=final_4500_no_val`) for 4500-image training with no validation and
  the fixed 500-image test split.
- Primary metrics: PSNR, SSIM, LPIPS, NIMA, CIEDE2000, runtime, and visual samples.
- Owner policy: use `TBD` until a team member is assigned.
- Use `make -n <target> ARGS='...'` to preview a command before running it.

## Promotion Summary

1. Run smoke jobs before training jobs.
2. Run all planned `train-2000` jobs.
3. Use the Initial-Baseline plan as the reference for how much the newer recipes improve over DINOv3 + RWKV7 + CNN-only decoding.
4. Use the Encoder Family Ablation plan to compare feature-skip on/off inside each family and compare the skip-on family bases.
5. After selecting a stable carrier architecture, run the Loss Comparison plan before spending final confirmation budget.
6. Run `train-5000` only for selected winners and necessary references.
7. Run `train-final` only for the final model(s) that will be reported as
   4500 train / 500 test results.

## Fixed Data Protocol

- Fixed split files live in `configs/splits/five/{train,val,test}.txt`.
  Training reads these files directly; it does not regenerate them.
- `train-2000`: first 2000 entries from `configs/splits/five/train.txt`,
  plus the fixed 500-image validation split and fixed 500-image test split.
- `train-5000`: all 4000 entries from `configs/splits/five/train.txt`,
  plus the fixed 500-image validation split and fixed 500-image test split.
- `train-final` / `run_mode=final_4500_no_val`: merge
  `configs/splits/five/train.txt` and `configs/splits/five/val.txt` into a
  4500-image training pool, disable validation and early stopping, and keep
  `configs/splits/five/test.txt` unchanged as the 500-image test set.
- The fixed test set is the standard FiveK final 500 images (`a4501`-`a5000`).

## Result Packaging

Follow the experiment directory convention in [README.md](../README.md). Each completed training run automatically creates a self-contained directory when `on_test_end` fires:

```text
experiments/{name}_{YYYYMMDD}_{commit}/
├── README.txt
├── experiment_record.json
├── config.yaml
├── training_curve.png
├── metrics.csv
├── test_results.json
├── test_per_image_metrics.csv
├── test_per_image_metrics.json
├── test_best_psnr.png
├── test_median_psnr.png
└── test_worst_psnr.png
```

The checkpoint stays on the server; its path is recorded in `experiment_record.json` and `README.txt`.

Keep the global summary table in sync:

```text
experiments/results_summary.csv
```

Rebuild it after adding new experiment directories:

```bash
python scripts/rebuild_summary.py
```

## Status Values

Use these values consistently:

```text
planned
config_ready
smoke_running
smoke_done
subset_running
subset_done
final_running
final_done
packaged
uploaded
summarized
blocked
```

# Loss Comparison Plan

This preliminary plan defines the loss-comparison batch for the RWKV color
enhancement experiments. It should run after the current architecture-focused
plans have selected a stable carrier model.

## Purpose

The current Initial-Baseline and Encoder Family Ablation plans intentionally
hold `loss=stage1` fixed so that architecture, feature-skip, and conditioning
effects are not mixed with optimization-objective effects.

This L-series plan answers:

- Which loss recipe is strongest on the selected carrier architecture?
- Does MS-SWC improve color/aesthetic metrics without giving up too much PSNR?
- Is the current `stage1` recipe still the best final-training objective under
  the newer RWKV7-vision + CNN skip decoder protocol?

Do not mix L-series results into the A/B-series architecture conclusions. The
L-series is a follow-up loss study using the architecture winner as its carrier.

## Dependency And Timing

Run this plan only after these gates are complete:

1. Complete the Initial-Baseline and Encoder Family A-series smoke jobs.
2. Complete all A-series `train-2000` jobs.
3. If `color_naf_rwkv7` wins Phase 1, complete the conditional B01/B02
   ColorNAF conditioning-path checks before choosing the final carrier.
4. Select one carrier architecture for loss screening. If the top two carriers
   are within seed noise, run only the highest-priority loss candidate on the
   second carrier as a transfer check.

Recommended timing:

```text
A/IB smoke -> A/IB train-2000 -> optional B train-2000 -> L smoke -> L train-2000 -> selected L train-5000 -> final run
```

Do not wait for every architecture candidate to finish `train-5000` before
starting the L-series. The loss screen is meant to decide which objective should
receive the expensive confirmation runs.

## Fixed Protocol

- Carrier: winner from `docs/plans/encoder_family_ablation.md`.
- Bottleneck: keep the winning carrier's bottleneck, normally `rwkv7_vision`.
- Decoder: keep the winning carrier's decoder, normally `cnn_skip`.
- Feature skip / conditioning: keep the winning carrier's settings.
- Image skip: keep disabled unless the carrier plan explicitly promotes it.
- Data protocol: use `configs/splits/five/{train,val,test}.txt`.
  `train-2000` is for screening, and `train-5000` is 4000 train + 500 val +
  500 test for development confirmation.
- Final paper-style reporting: after selecting the final loss objective, rerun
  it with `make train-final ARGS='...'` for 4500 train / 500 test with no
  validation.
- Seed: keep the project default seed unless doing a seed-noise check.
- Early stopping: keep `training.early_stopping.monitor=val/psnr` for the first
  clean loss comparison, even when LPIPS/NIMA/CIEDE2000 are used as secondary
  decision signals.
- Primary metrics: PSNR, SSIM, LPIPS, NIMA, CIEDE2000, runtime, and visual
  samples.

## Carrier Selection

Copy the winning carrier override from the architecture plan before running any
L-series job.

| Carrier source | Override |
| --- | --- |
| A01 ColorNAF skip on | `family=color_naf_rwkv7 ablation/common/feature_skip=on` |
| B01 ColorNAF global token off | `family=color_naf_rwkv7 ablation/common/feature_skip=on ablation/color_naf_rwkv7/global_token=off` |
| B02 ColorNAF FiLM on | `family=color_naf_rwkv7 ablation/common/feature_skip=on ablation/color_naf_rwkv7/film=on` |
| A05 V2 DINOv3 skip on | `family=v2_dinov3_rwkv7 ablation/common/feature_skip=on` |
| A03 DINOv3 CNN frequency skip on | `family=dinov3_cnn_freq_rwkv7 ablation/common/feature_skip=on` |

Only use A03 as a carrier if logs confirm the CNN-frequency branch actually
provides decoder-consumable skip features.

The concrete commands below use A01 as the provisional carrier. If a different
carrier wins, replace only the carrier override and the short name in
`experiment_name`.

## Config Inventory

| Purpose | Override | Notes |
| --- | --- | --- |
| Current reference | `loss=stage1` | L1 + SSIM + LPIPS in the current config |
| L1 only | `loss=l1` | Pixel-only lower/anchor objective |
| L1 + SSIM | `ablation=a_loss_l1_ssim` | Removes the perceptual term from stage1 |
| MS-SWC replacement | `loss=stage1_msswc` | L1 + SSIM + MS-SWC; high-priority candidate |
| LPIPS + MS-SWC stack | `loss=stage1_plus_msswc` | Optional; previous notes suggest possible conflict |
| PSNR-tuned objective | `loss=stage1_psnr` | Optional MSE-only PSNR check |
| MS-SWC only | `loss=msswc_only` | Diagnostic only; do not promote directly |

Note: `configs/ablation/common/loss/` contains smaller loss-ablation configs,
but the current top-level Hydra defaults do not include
`optional ablation/common/loss`. This plan therefore uses the currently
executable `loss=...` overrides and the existing legacy `ablation=a_loss_l1_ssim`
entry.

## Experiment Matrix

| ID | Experiment | Factor | Override | Owner | Stage | Status |
| --- | --- | --- | --- | --- | --- | --- |
| L00 | Carrier stage1 reference | none | `<CARRIER_OVERRIDES> loss=stage1` | TBD | 2000/5000-anchor | conditional |
| L01 | Carrier L1 only | loss | `<CARRIER_OVERRIDES> loss=l1` | TBD | smoke/2000 | planned |
| L02 | Carrier L1 + SSIM | loss | `<CARRIER_OVERRIDES> ablation=a_loss_l1_ssim` | TBD | smoke/2000 | planned |
| L03 | Carrier stage1 MS-SWC | loss | `<CARRIER_OVERRIDES> loss=stage1_msswc` | TBD | smoke/2000/5000-candidate | planned |
| L04 | Carrier stage1 plus MS-SWC | loss | `<CARRIER_OVERRIDES> loss=stage1_plus_msswc` | TBD | smoke/2000 optional | optional |
| L05 | Carrier PSNR objective | loss | `<CARRIER_OVERRIDES> loss=stage1_psnr` | TBD | smoke/2000 optional | optional |
| L06 | Carrier MS-SWC only diagnostic | loss diagnostic | `<CARRIER_OVERRIDES> loss=msswc_only` | TBD | smoke/diag160 only | optional |

Interpretation rule:

- L00 is the anchor. Reuse the existing carrier `train-2000` result if it has
  exactly the same carrier override and `loss=stage1`; otherwise run L00.
- L01 and L02 estimate how much the perceptual/color terms contribute.
- L03 tests whether MS-SWC can replace the current perceptual term.
- L04 tests whether LPIPS and MS-SWC are complementary. Treat this as optional
  because earlier MS-SWC notes suggest the two can conflict.
- L05 is only for a PSNR-first reporting angle.
- L06 is a sanity/diagnostic run. It can reveal MS-SWC scale and failure modes,
  but it is not a final objective candidate because it has no pixel anchor.

## Commands

Before running experiments:

```bash
conda activate rwkv-color
```

### L00 Carrier Stage1 Reference

Run only if the selected carrier does not already have an equivalent
`train-2000` result.

Smoke:

```bash
make train-debug ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on loss=stage1 experiment_name=L00_colornaf_skipon_stage1_smoke'
```

Train-2000:

```bash
make train-2000 ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on loss=stage1 experiment_name=L00_colornaf_skipon_stage1_2000'
```

Train-5000 anchor:

```bash
make train-5000 ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on loss=stage1 experiment_name=L00_colornaf_skipon_stage1_5000'
```

### L01 Carrier L1 Only

Smoke:

```bash
make train-debug ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on loss=l1 experiment_name=L01_colornaf_skipon_l1_smoke'
```

Train-2000:

```bash
make train-2000 ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on loss=l1 experiment_name=L01_colornaf_skipon_l1_2000'
```

### L02 Carrier L1 + SSIM

Smoke:

```bash
make train-debug ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on ablation=a_loss_l1_ssim experiment_name=L02_colornaf_skipon_l1ssim_smoke'
```

Train-2000:

```bash
make train-2000 ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on ablation=a_loss_l1_ssim experiment_name=L02_colornaf_skipon_l1ssim_2000'
```

### L03 Carrier Stage1 MS-SWC

Smoke:

```bash
make train-debug ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on loss=stage1_msswc experiment_name=L03_colornaf_skipon_stage1msswc_smoke'
```

Train-2000:

```bash
make train-2000 ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on loss=stage1_msswc experiment_name=L03_colornaf_skipon_stage1msswc_2000'
```

Train-5000 candidate:

```bash
make train-5000 ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on loss=stage1_msswc experiment_name=L03_colornaf_skipon_stage1msswc_5000'
```

### L04 Carrier Stage1 Plus MS-SWC

Smoke:

```bash
make train-debug ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on loss=stage1_plus_msswc experiment_name=L04_colornaf_skipon_stage1plusmsswc_smoke'
```

Train-2000:

```bash
make train-2000 ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on loss=stage1_plus_msswc experiment_name=L04_colornaf_skipon_stage1plusmsswc_2000'
```

Train-5000 candidate, only if L04 clearly beats both L00 and L03:

```bash
make train-5000 ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on loss=stage1_plus_msswc experiment_name=L04_colornaf_skipon_stage1plusmsswc_5000'
```

### L05 Carrier PSNR Objective

Optional PSNR-first check.

Smoke:

```bash
make train-debug ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on loss=stage1_psnr experiment_name=L05_colornaf_skipon_psnr_smoke'
```

Train-2000:

```bash
make train-2000 ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on loss=stage1_psnr experiment_name=L05_colornaf_skipon_psnr_2000'
```

### L06 Carrier MS-SWC Only Diagnostic

Use this only to confirm MS-SWC scale and behavior on the selected carrier. Do
not promote it to `train-5000`.

Smoke:

```bash
make train-debug ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on loss=msswc_only experiment_name=L06_colornaf_skipon_msswconly_smoke'
```

Diagnostic-160:

```bash
make train ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on loss=msswc_only data.train_subset_size=160 data.val_subset_size=20 data.test_subset_size=20 training.max_epochs=20 training.early_stopping.patience=5 experiment_name=L06_colornaf_skipon_msswconly_diag160'
```

## Promotion Rule

1. Run L01-L03 smoke jobs first. Run L04-L06 smoke jobs only if budget allows or
   if MS-SWC debugging is needed.
2. Run L00-L03 on `train-2000`. Reuse the carrier's existing stage1 result for
   L00 when valid.
3. Promote L03 to `train-5000` if it improves LPIPS, NIMA, CIEDE2000, or visual
   quality while keeping PSNR/SSIM within the seed-noise band of L00.
4. Promote L04 only if it beats both L00 and L03; otherwise treat it as evidence
   that LPIPS and MS-SWC are not complementary for this carrier.
5. Promote L05 only if the report needs a PSNR-first objective comparison.
6. Never promote L06; summarize it as a diagnostic run.
7. If two architecture carriers were close in the A/B-series, run L03 on the
   second carrier at `train-2000` before finalizing the loss conclusion.
8. Confirm the final objective with `train-5000` on L00 and the best L-series
   candidate. If PSNR and perceptual/aesthetic metrics disagree, promote both
   candidate objectives and report the tradeoff explicitly.
9. Run `train-final` for the final objective(s) selected for 4500/500
   paper-style reporting.

## Result Tracking

| ID | Experiment Name | Experiment Directory | Summary Row | Microsoft 365 Upload | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| L00 | `L00_colornaf_skipon_stage1_2000` | `experiments/L00_colornaf_skipon_stage1_2000_{YYYYMMDD}_{commit}/` | `experiments/results_summary.csv` | `Microsoft365/experiments/L00_colornaf_skipon_stage1_2000/` | conditional | Reuse A/B carrier result if exact match exists |
| L01 | `L01_colornaf_skipon_l1_2000` | `experiments/L01_colornaf_skipon_l1_2000_{YYYYMMDD}_{commit}/` | `experiments/results_summary.csv` | `Microsoft365/experiments/L01_colornaf_skipon_l1_2000/` | planned | Pixel-only anchor |
| L02 | `L02_colornaf_skipon_l1ssim_2000` | `experiments/L02_colornaf_skipon_l1ssim_2000_{YYYYMMDD}_{commit}/` | `experiments/results_summary.csv` | `Microsoft365/experiments/L02_colornaf_skipon_l1ssim_2000/` | planned | Removes perceptual/color term |
| L03 | `L03_colornaf_skipon_stage1msswc_2000` | `experiments/L03_colornaf_skipon_stage1msswc_2000_{YYYYMMDD}_{commit}/` | `experiments/results_summary.csv` | `Microsoft365/experiments/L03_colornaf_skipon_stage1msswc_2000/` | planned | High-priority MS-SWC replacement |
| L04 | `L04_colornaf_skipon_stage1plusmsswc_2000` | `experiments/L04_colornaf_skipon_stage1plusmsswc_2000_{YYYYMMDD}_{commit}/` | `experiments/results_summary.csv` | `Microsoft365/experiments/L04_colornaf_skipon_stage1plusmsswc_2000/` | optional | Tests LPIPS + MS-SWC compatibility |
| L05 | `L05_colornaf_skipon_psnr_2000` | `experiments/L05_colornaf_skipon_psnr_2000_{YYYYMMDD}_{commit}/` | `experiments/results_summary.csv` | `Microsoft365/experiments/L05_colornaf_skipon_psnr_2000/` | optional | PSNR-first objective |
| L06 | `L06_colornaf_skipon_msswconly_diag160` | `experiments/L06_colornaf_skipon_msswconly_diag160_{YYYYMMDD}_{commit}/` | `experiments/results_summary.csv` | `Microsoft365/experiments/L06_colornaf_skipon_msswconly_diag160/` | optional | Diagnostic only |

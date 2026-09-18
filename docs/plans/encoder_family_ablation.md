# Encoder Family Ablation Plan

This plan compares three encoder-family recipes under the shared `RWKV7-vision + CNN skip decoder` protocol.

## Purpose

The previous plan mixed several factors: FiLM, raw image skip, loss, encoder freeze, and RWKV depth. This plan splits the work into two phases so each phase varies one thing at a time.

Phase 1 — Family comparison (A01–A07): three encoder families × feature-skip on/off, plus a G1-only image-skip control. Answers:

- Which encoder family is the strongest base under the same RWKV7 + CNN skip decoder protocol?
- Does decoder-side feature skip matter consistently across encoder families?
- In ColorNAF, is decoder feature/cnn skip more useful than raw-image skip?

Phase 2 — ColorNAF conditioning-path depth dive (B01, B02): only triggers if Phase 1 selects `color_naf` as the winner family. Both knobs require a hist_token from the ColorNAF encoder, so they are not meaningful for the other two families. Answers:

- Is the prepended global histogram token worth its parameters and FLOPs (B01 vs A01)?
- Does FiLM conditioning on the hist_token add anything on top of the existing encoder→bottleneck injection (B02 vs A01)?

## Fixed Protocol

- Bottleneck: `rwkv7_vision` by default. If CUDA kernel or stability issues appear, add `ablation/common/bottleneck=rwkv7` as the phase-1 fallback and record it in Notes.
- Decoder: `cnn_skip`
- Image skip: disabled by default with `decoder.img_skip_enabled=false`
- Loss: `stage1`
- Primary comparison stage: `train-2000` first, then confirm selected runs with `train-5000`.
- Data split: use `configs/splits/five/{train,val,test}.txt`; `train-5000`
  means 4000 train + 500 val + 500 test for development confirmation.
- Final paper-style reporting: after selecting the architecture winner, rerun it
  with `make train-final ARGS='...'` to merge train+val into 4500 training
  images and keep the fixed 500-image test split.
- Primary metrics: PSNR, SSIM, LPIPS, NIMA, CIEDE2000, runtime, and visual samples.

## Baseline Families

| Group | Family | Encoder Setup | Bottleneck | Decoder | Status |
| --- | --- | --- | --- | --- | --- |
| G1 | `family=color_naf_rwkv7` | DINOv3-B + ColorNAF CNN encoder | `rwkv7_vision` | `cnn_skip` | config_ready |
| G2 | `family=dinov3_cnn_freq_rwkv7` | DINOv3-B + frequency-gated CNN encoder | `rwkv7_vision` | `cnn_skip` | config_ready |
| G3 | `family=v2_dinov3_rwkv7` | V2 DINOv3-B fused local/colour branches | `rwkv7_vision` | `cnn_skip` | config_ready |

Notes:

- G1 and G3 already expose decoder-consumable skip features.
- G2 uses `cnn_encoder=cnn_freq_gated`; in the current pipeline this branch is fused into token features, but its intermediate CNN feature list is not passed to `CNNSkipDecoder` as `skip_dict`. Treat the G2 skip on/off pair as a smoke-validated check first. If config snapshots show no effective decoder skip path, do not report G2 on/off as evidence about feature skip.

## Config Inventory

| Purpose | Override |
| --- | --- |
| ColorNAF family | `family=color_naf_rwkv7` |
| DINOv3 + frequency CNN family | `family=dinov3_cnn_freq_rwkv7` |
| V2 DINOv3 family | `family=v2_dinov3_rwkv7` |
| RWKV7 vision bottleneck | `ablation/common/bottleneck=rwkv7_vision` |
| RWKV7 phase-1 bottleneck | `ablation/common/bottleneck=rwkv7` |
| Feature skip on | `ablation/common/feature_skip=on` |
| Feature skip off | `ablation/common/feature_skip=off` |
| Image skip on | `ablation/common/img_skip=on` |
| Image skip off | `ablation/common/img_skip=off` |
| ColorNAF global token on | `ablation/color_naf_rwkv7/global_token=on` |
| ColorNAF global token off | `ablation/color_naf_rwkv7/global_token=off` |
| ColorNAF FiLM on | `ablation/color_naf_rwkv7/film=on` |
| ColorNAF FiLM off | `ablation/color_naf_rwkv7/film=off` |

## Experiment Matrix

### Phase 1 — Family × skip path

| ID | Group | Experiment | Factor | Override | Owner | Stage | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A01 | G1 | ColorNAF feature skip only | feature/cnn skip | `family=color_naf_rwkv7 ablation/common/feature_skip=on ablation/common/img_skip=off` | Jianwei,Xiangxu | smoke/2000/5000-candidate | config_ready |
| A02 | G1 | ColorNAF no decoder skip | skip control | `family=color_naf_rwkv7 ablation/common/feature_skip=off ablation/common/img_skip=off` | Jianwei,Xiangxu | smoke/2000/5000-candidate | config_ready |
| A03 | G1 | ColorNAF image skip only | img skip | `family=color_naf_rwkv7 ablation/common/feature_skip=off ablation/common/img_skip=on` | Jianwei,Xiangxu | smoke/2000/5000-candidate | config_ready |
| A04 | G2 | DINOv3 CNN frequency skip on | feature skip | `family=dinov3_cnn_freq_rwkv7 ablation/common/feature_skip=on` | Zihang | smoke/2000 | config_ready |
| A05 | G2 | DINOv3 CNN frequency skip off | feature skip | `family=dinov3_cnn_freq_rwkv7 ablation/common/feature_skip=off` | Zihang | smoke/2000 | config_ready |
| A06 | G3 | V2 DINOv3 skip on | feature skip | `family=v2_dinov3_rwkv7 ablation/common/feature_skip=on` | Yuxi | smoke/2000/5000-candidate | config_ready |
| A07 | G3 | V2 DINOv3 skip off | feature skip | `family=v2_dinov3_rwkv7 ablation/common/feature_skip=off` | Yuxi | smoke/2000/5000-candidate | config_ready |

### Phase 2 — ColorNAF conditioning path (conditional on G1 winning Phase 1)

Both rows are single-variable changes from A01. The fourth corner (`global_token=off, film=on`) is omitted because [cnn_skip_decoder.py:343](../../src/models/decoders/cnn_skip_decoder.py#L343) skips FiLM when `hist_token is None`, so it would degenerate to A01 with `global_token=off` (i.e. B01).

| ID | Group | Experiment | Factor | Override | Owner | Stage | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B01 | G1 | ColorNAF global token off | global token | `family=color_naf_rwkv7 ablation/common/feature_skip=on ablation/color_naf_rwkv7/global_token=off` | Jianwei,Xiangxu | 2000 (after Phase 1) | config_ready |
| B02 | G1 | ColorNAF FiLM on | FiLM | `family=color_naf_rwkv7 ablation/common/feature_skip=on ablation/color_naf_rwkv7/film=on` | Jianwei,Xiangxu | 2000 (after Phase 1) | config_ready |

Interpretation rule:

- Phase 1: within each group, compare feature skip on vs off to estimate feature-skip value. Within G1, also compare A01 vs A03 to separate feature/cnn skip from raw-image skip. Across G1–G3, compare the feature-skip-on runs first because they represent each family's intended base recipe. Use the Initial-Baseline plan only as an external reference; do not mix it into the feature-skip conclusion.
- Phase 2: A01 is the shared baseline. B01 − A01 isolates the global-token contribution; B02 − A01 isolates FiLM's added contribution on top of the encoder→bottleneck hist-token injection. Do not interpret B01 vs B02 directly — they vary different factors.

## Commands

### A01 ColorNAF Feature Skip Only

Smoke:

```bash
make train-debug ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on ablation/common/img_skip=off experiment_name=A01_colornaf_feature_skip_only_smoke'
```

Train-2000:

```bash
make train-2000 ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on ablation/common/img_skip=off experiment_name=A01_colornaf_feature_skip_only_2000'
```

Train-5000 candidate:

```bash
make train-5000 ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on ablation/common/img_skip=off experiment_name=A01_colornaf_feature_skip_only_5000'
```

### A02 ColorNAF No Decoder Skip

Smoke:

```bash
make train-debug ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=off ablation/common/img_skip=off experiment_name=A02_colornaf_no_decoder_skip_smoke'
```

Train-2000:

```bash
make train-2000 ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=off ablation/common/img_skip=off experiment_name=A02_colornaf_no_decoder_skip_2000'
```

Train-5000 candidate:

```bash
make train-5000 ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=off ablation/common/img_skip=off experiment_name=A02_colornaf_no_decoder_skip_5000'
```

### A03 ColorNAF Image Skip Only

Smoke:

```bash
make train-debug ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=off ablation/common/img_skip=on experiment_name=A03_colornaf_img_skip_only_smoke'
```

Train-2000:

```bash
make train-2000 ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=off ablation/common/img_skip=on experiment_name=A03_colornaf_img_skip_only_2000'
```

Train-5000 candidate:

```bash
make train-5000 ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=off ablation/common/img_skip=on experiment_name=A03_colornaf_img_skip_only_5000'
```

### A04 DINOv3 CNN Frequency Skip On

Smoke:

```bash
make train-debug ARGS='family=dinov3_cnn_freq_rwkv7 ablation/common/feature_skip=on experiment_name=A04_dinov3_cnnfreq_skip_on_smoke'
```

Train-2000:

```bash
make train-2000 ARGS='family=dinov3_cnn_freq_rwkv7 ablation/common/feature_skip=on experiment_name=A04_dinov3_cnnfreq_skip_on_2000'
```

Train-5000 candidate:

```bash
make train-5000 ARGS='family=dinov3_cnn_freq_rwkv7 ablation/common/feature_skip=on experiment_name=A04_dinov3_cnnfreq_skip_on_5000'
```

### A05 DINOv3 CNN Frequency Skip Off

Smoke:

```bash
make train-debug ARGS='family=dinov3_cnn_freq_rwkv7 ablation/common/feature_skip=off experiment_name=A05_dinov3_cnnfreq_skip_off_smoke'
```

Train-2000:

```bash
make train-2000 ARGS='family=dinov3_cnn_freq_rwkv7 ablation/common/feature_skip=off experiment_name=A05_dinov3_cnnfreq_skip_off_2000'
```

Train-5000 candidate:

```bash
make train-5000 ARGS='family=dinov3_cnn_freq_rwkv7 ablation/common/feature_skip=off experiment_name=A05_dinov3_cnnfreq_skip_off_5000'
```

### A06 V2 DINOv3 Skip On

Smoke:

```bash
make train-debug ARGS='family=v2_dinov3_rwkv7 ablation/common/feature_skip=on experiment_name=A06_v2dinov3_skip_on_smoke'
```

Train-2000:

```bash
make train-2000 ARGS='family=v2_dinov3_rwkv7 ablation/common/feature_skip=on experiment_name=A06_v2dinov3_skip_on_2000'
```

Train-5000 candidate:

```bash
make train-5000 ARGS='family=v2_dinov3_rwkv7 ablation/common/feature_skip=on experiment_name=A06_v2dinov3_skip_on_5000'
```

### A07 V2 DINOv3 Skip Off

Smoke:

```bash
make train-debug ARGS='family=v2_dinov3_rwkv7 ablation/common/feature_skip=off experiment_name=A07_v2dinov3_skip_off_smoke'
```

Train-2000:

```bash
make train-2000 ARGS='family=v2_dinov3_rwkv7 ablation/common/feature_skip=off experiment_name=A07_v2dinov3_skip_off_2000'
```

Train-5000 candidate:

```bash
make train-5000 ARGS='family=v2_dinov3_rwkv7 ablation/common/feature_skip=off experiment_name=A07_v2dinov3_skip_off_5000'
```

### B01 ColorNAF Global Token Off

Single-variable change from A01: drop the prepended histogram/global token. Run only after Phase 1 picks G1 as the winner family.

Smoke:

```bash
make train-debug ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on ablation/color_naf_rwkv7/global_token=off experiment_name=B01_colornaf_gt_off_smoke'
```

Train-2000:

```bash
make train-2000 ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on ablation/color_naf_rwkv7/global_token=off experiment_name=B01_colornaf_gt_off_2000'
```

### B02 ColorNAF FiLM On

Single-variable change from A01: enable FiLM conditioning at decoder stages 2 and 3, conditioned on the existing hist_token. Run only after Phase 1 picks G1 as the winner family.

Smoke:

```bash
make train-debug ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on ablation/color_naf_rwkv7/film=on experiment_name=B02_colornaf_film_on_smoke'
```

Train-2000:

```bash
make train-2000 ARGS='family=color_naf_rwkv7 ablation/common/feature_skip=on ablation/color_naf_rwkv7/film=on experiment_name=B02_colornaf_film_on_2000'
```

## Promotion Rule

### Phase 1 (A01–A07)

1. Run all seven A-series smoke jobs.
2. Run all seven A-series jobs on `train-2000`.
3. Verify config snapshots include the intended family, bottleneck, decoder, `decoder.skip_enabled`, and `decoder.img_skip_enabled` values.
4. Compare A01 vs A02 to estimate ColorNAF feature/cnn skip value, and A03 vs A02 to estimate raw-image skip value.
5. Compare A01 vs A03 directly for ColorNAF feature/cnn skip vs image skip.
6. Compare feature-skip on vs off inside G2 and G3, then compare A01/A04/A06 across groups.
7. Run `train-5000` only for the best family and its most relevant skip-control counterpart. Optionally run the second-best skip-on run if its metrics or visual samples are close.
8. After the final architecture is selected, run `train-final` for the model(s)
   that need 4500/500 paper-style results.

Do not promote G2's skip pair as a feature-skip conclusion unless the run logs confirm decoder-consumable skip features are actually used.

### Phase 2 (B01, B02)

1. Trigger condition: Phase 1 selected `color_naf` (G1) as the winner family. If a different family wins, skip Phase 2 entirely — these knobs do not exist on the other encoders.
2. Run B01 and B02 smoke jobs.
3. Run B01 and B02 on `train-2000`.
4. Verify config snapshots show `cnn_encoder.use_global_token=false` for B01 and `decoder.film_stages=[2, 3]` for B02; everything else should match A01.
5. Compute B01 − A01 and B02 − A01 on the primary metrics. Use seed-noise estimate from any Phase 1 reruns as the threshold for "real" effect.
6. Promote whichever of (A01, B01, B02) wins to `train-5000` only if the gain over A01 is larger than seed noise.

## Result Tracking

| ID | Experiment Name | Experiment Directory | Summary Row | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| A01 | `A01_colornaf_feature_skip_only_2000` | `experiments/A01_colornaf_feature_skip_only_2000_{YYYYMMDD}_{commit}/` | `experiments/results_summary.csv` | planned | ColorNAF with feature/cnn skip only |
| A02 | `A02_colornaf_no_decoder_skip_2000` | `experiments/A02_colornaf_no_decoder_skip_2000_{YYYYMMDD}_{commit}/` | `experiments/results_summary.csv` | planned | ColorNAF no-skip control |
| A03 | `A03_colornaf_img_skip_only_2000` | `experiments/A03_colornaf_img_skip_only_2000_{YYYYMMDD}_{commit}/` | `experiments/results_summary.csv` | planned | ColorNAF image-skip-only control |
| A04 | `A04_dinov3_cnnfreq_skip_on_2000` | `experiments/A04_dinov3_cnnfreq_skip_on_2000_{YYYYMMDD}_{commit}/` | `experiments/results_summary.csv` | planned | Confirm skip path before interpreting |
| A05 | `A05_dinov3_cnnfreq_skip_off_2000` | `experiments/A05_dinov3_cnnfreq_skip_off_2000_{YYYYMMDD}_{commit}/` | `experiments/results_summary.csv` | planned | Confirm skip path before interpreting |
| A06 | `A06_v2dinov3_skip_on_2000` | `experiments/A06_v2dinov3_skip_on_2000_{YYYYMMDD}_{commit}/` | `experiments/results_summary.csv` | planned | Intended V2 DINOv3 family base |
| A07 | `A07_v2dinov3_skip_off_2000` | `experiments/A07_v2dinov3_skip_off_2000_{YYYYMMDD}_{commit}/` | `experiments/results_summary.csv` | planned | Measures V2 decoder skip contribution |
| B01 | `B01_colornaf_gt_off_2000` | `experiments/B01_colornaf_gt_off_2000_{YYYYMMDD}_{commit}/` | `experiments/results_summary.csv` | conditional (Phase 1 winner = G1) | Measures global-token contribution vs A01 |
| B02 | `B02_colornaf_film_on_2000` | `experiments/B02_colornaf_film_on_2000_{YYYYMMDD}_{commit}/` | `experiments/results_summary.csv` | conditional (Phase 1 winner = G1) | Measures FiLM's added contribution vs A01 |

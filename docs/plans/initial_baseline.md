# Initial Baseline Plan

This plan defines the minimal reference runs for comparing later encoder/decoder recipes.

## Purpose

The Initial-Baseline group answers how strong the `DINOv3 + RWKV7 + CNN-only decoder` recipe is before adding encoder feature skip paths or auxiliary encoder families.

These runs are reference baselines only. They use `decoder=cnn_skip` with `decoder.skip_enabled=false` and `decoder.img_skip_enabled=false`, so the decoder does not consume encoder feature skips or image-skip guidance.

## Fixed Protocol

- Encoder: `dinov3_b`
- Decoder: `cnn_skip`
- Feature skip: disabled with `decoder.skip_enabled=false`
- Image skip: disabled with `decoder.img_skip_enabled=false`
- Loss: `stage1`
- Primary comparison stage: `train-2000` first, then confirm the stronger reference with `train-5000` if compute budget allows.
- Data split: use `configs/splits/five/{train,val,test}.txt`; `train-5000`
  means 4000 train + 500 val + 500 test for development confirmation, not the
  final 4500/500 paper run.
- Final paper-style reporting: after selecting the baseline to report, rerun it
  with `make train-final ARGS='...'` to merge train+val into 4500 training
  images and disable validation.
- Primary metrics: PSNR, SSIM, LPIPS, NIMA, CIEDE2000, runtime, and visual samples.

## Experiment Matrix

| ID | Experiment | Factor | Override | Owner | Stage | Status |
| --- | --- | --- | --- | --- | --- | --- |
| IB01 | Initial DINOv3 RWKV7 CNN | initial baseline | `family=initial_dinov3_rwkv7_cnn` | Jianwei,Xiangxu | smoke/2000/5000 | config_ready |
| IB02 | Initial DINOv3 RWKV7 vision CNN | bottleneck | `family=initial_dinov3_rwkv7_cnn ablation/common/bottleneck=rwkv7_vision` | Jianwei,Xiangxu | smoke/2000/5000 | config_ready |

## Commands

### IB01 Initial DINOv3 RWKV7 CNN

Smoke:

```bash
make train-debug ARGS='family=initial_dinov3_rwkv7_cnn experiment_name=IB01_initial_dinov3_rwkv7_cnn_smoke'
```

Train-2000:

```bash
make train-2000 ARGS='family=initial_dinov3_rwkv7_cnn experiment_name=IB01_initial_dinov3_rwkv7_cnn_2000'
```

Train-5000:

```bash
make train-5000 ARGS='family=initial_dinov3_rwkv7_cnn experiment_name=IB01_initial_dinov3_rwkv7_cnn_5000'
```

### IB02 Initial DINOv3 RWKV7 Vision CNN

Smoke:

```bash
make train-debug ARGS='family=initial_dinov3_rwkv7_cnn ablation/common/bottleneck=rwkv7_vision experiment_name=IB02_initial_bottleneck_rwkv7vision_smoke'
```

Train-2000:

```bash
make train-2000 ARGS='family=initial_dinov3_rwkv7_cnn ablation/common/bottleneck=rwkv7_vision experiment_name=IB02_initial_bottleneck_rwkv7vision_2000'
```

Train-5000:

```bash
make train-5000 ARGS='family=initial_dinov3_rwkv7_cnn ablation/common/bottleneck=rwkv7_vision experiment_name=IB02_initial_bottleneck_rwkv7vision_5000'
```

## Promotion Rule

1. Run both smoke jobs.
2. Run both `train-2000` jobs.
3. Compare IB01 vs IB02 to choose the stronger initial reference baseline.
4. Run `train-5000` for the stronger Initial-Baseline reference if compute budget allows.
5. Run `train-final` only if the initial baseline is needed as a final
   4500/500 paper-style reference.

## Result Tracking

| ID | Experiment Name | Experiment Directory | Summary Row | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| IB01 | `IB01_initial_dinov3_rwkv7_cnn_2000` | `experiments/IB01_initial_dinov3_rwkv7_cnn_2000_{YYYYMMDD}_{commit}/` | `experiments/results_summary.csv` | planned | Initial baseline with conservative RWKV7 |
| IB02 | `IB02_initial_bottleneck_rwkv7vision_2000` | `experiments/IB02_initial_bottleneck_rwkv7vision_2000_{YYYYMMDD}_{commit}/` | `experiments/results_summary.csv` | planned | Initial baseline with vision RWKV7 |

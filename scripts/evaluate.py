"""Standalone evaluation script for a trained checkpoint.

Usage::

    python scripts/evaluate.py \\
        --checkpoint outputs/checkpoints/dino_cnn_cnn_epoch050-24.30.ckpt \\
        --config     outputs/logs/dino_cnn_cnn/config/config.yaml \\
        --output-dir outputs/eval
"""
from __future__ import annotations

import argparse
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from src.data.transforms import to_image_range
from src.evaluation.metrics import MetricCollection
from src.evaluation.visualization import save_comparison_grid
from src.lit_module import ColorEnhanceLitModule
from src.utils.checkpoint import allow_trusted_checkpoint_loading
from src.utils.device import get_device
from src.utils.logging_utils import export_results_json, print_metrics_table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument("--config",     required=True, help="Path to config.yaml snapshot")
    p.add_argument("--data-dir",   default=None,  help="Override data directory")
    p.add_argument("--output-dir", default="outputs/eval")
    p.add_argument("--device",     default="auto")
    p.add_argument("--num-vis",    type=int, default=8,
                   help="Number of images in the comparison grid")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    allow_trusted_checkpoint_loading()

    # ── Config ────────────────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)
    if args.data_dir:
        cfg.data.data_dir = args.data_dir

    # ── Device ────────────────────────────────────────────────────────
    device = (
        get_device() if args.device == "auto" else torch.device(args.device)
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = ColorEnhanceLitModule.load_from_checkpoint(
        args.checkpoint, cfg=cfg, map_location=device
    )
    model.eval().to(device)

    # ── Data ──────────────────────────────────────────────────────────
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    loader = datamodule.test_dataloader()

    # ── Evaluation loop ───────────────────────────────────────────────
    metric_fn = MetricCollection(cfg)
    test_gates = model._resolve_metric_gates(cfg.evaluation.get("test"))
    metric_names = model._enabled_metric_names(test_gates)
    accum: dict[str, list[float]] = {name: [] for name in metric_names}
    out_dir = Path(args.output_dir) / cfg.experiment_name
    vis_saved = False
    inputs_are_normalized = bool(cfg.data.get("normalize_to_neg_one_one", True))

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 2:
                inputs, targets = batch
                sizes = None
            elif len(batch) == 3:
                inputs, targets, sizes = batch
                sizes = sizes.to(device)
            else:
                inputs, targets, sizes, _names = batch
                sizes = sizes.to(device)

            inputs = inputs.to(device)
            targets = targets.to(device)

            inputs_01 = to_image_range(
                inputs,
                normalized_to_neg_one_one=inputs_are_normalized,
            )
            targets_01 = to_image_range(
                targets,
                normalized_to_neg_one_one=inputs_are_normalized,
            )

            preds = model.pipeline(inputs_01)
            if sizes is None:
                m = metric_fn(
                    preds,
                    targets_01,
                    enable_psnr=True,
                    enable_ssim=True,
                    **test_gates,
                )
            else:
                metric_acc: dict[str, list[torch.Tensor]] = {}
                for idx, size in enumerate(sizes):
                    h = int(size[0].item())
                    w = int(size[1].item())
                    sample_metrics = metric_fn(
                        preds[idx:idx + 1, :, :h, :w],
                        targets_01[idx:idx + 1, :, :h, :w],
                        enable_psnr=True,
                        enable_ssim=True,
                        **test_gates,
                    )
                    for key, value in sample_metrics.items():
                        if key in metric_names:
                            metric_acc.setdefault(key, []).append(value)
                m = {
                    key: torch.stack(values).mean()
                    for key, values in metric_acc.items()
                }
            for k, v in m.items():
                if k in accum:
                    accum[k].append(v.item())

            # Save comparison grid for the first batch
            if not vis_saved:
                n = min(args.num_vis, inputs_01.shape[0])
                save_comparison_grid(
                    inputs_01[:n].cpu(),
                    preds[:n].cpu(),
                    targets_01[:n].cpu(),
                    save_path=out_dir / "comparison.png",
                )
                vis_saved = True

    # ── Aggregate & report ────────────────────────────────────────────
    results = {k: sum(v) / len(v) for k, v in accum.items() if v}
    print_metrics_table(results, title=f"Evaluation — {cfg.experiment_name}")

    export_results_json(
        {"experiment": cfg.experiment_name, "metrics": results},
        out_dir / "eval_results.json",
    )


if __name__ == "__main__":
    main()

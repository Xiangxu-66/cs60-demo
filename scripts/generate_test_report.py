"""Standalone post-hoc test report generator.

Loads a trained checkpoint, runs inference on the test set, and produces the
standardised 4-panel visualisation (Input | Pred | Expert C | Diff×5) for
the best / median / worst PSNR samples.

Use this to regenerate reports without re-training, or to compare multiple
checkpoints on the same fixed test images.

Usage::

    python scripts/generate_test_report.py \\
        --checkpoint outputs/checkpoints/my_exp.ckpt \\
        --config     outputs/logs/my_exp/config/config.yaml \\
        --output-dir outputs/logs/my_exp/visual_results

    # Override data directory
    python scripts/generate_test_report.py \\
        --checkpoint outputs/checkpoints/my_exp.ckpt \\
        --config     outputs/logs/my_exp/config/config.yaml \\
        --data-dir   /path/to/fivek \\
        --n-per-group 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from rich.console import Console

from src.data.transforms import to_image_range
from src.evaluation.report_builder import write_test_metric_exports
from src.evaluation.visualization import save_test_report, to_display_u8
from src.lit_module import ColorEnhanceLitModule
from src.utils.checkpoint import allow_trusted_checkpoint_loading
from src.utils.device import get_device

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate standardised test report from checkpoint")
    p.add_argument("--checkpoint",  required=True, help="Path to .ckpt file")
    p.add_argument("--config",      required=True, help="Path to config.yaml snapshot")
    p.add_argument("--data-dir",    default=None,  help="Override data directory")
    p.add_argument("--output-dir",  default=None,  help="Output directory (default: next to checkpoint)")
    p.add_argument("--device",      default="auto")
    p.add_argument("--n-per-group", type=int, default=5,
                   help="Samples per group (best / median / worst)")
    p.add_argument("--batch-size",  type=int, default=None,
                   help="Override batch size for inference")
    return p.parse_args()


def collect_samples(
    model: ColorEnhanceLitModule,
    loader,
    device: torch.device,
    metric_gates: dict[str, bool],
) -> list[dict]:
    """Run inference and return per-sample metrics plus display tensors."""
    samples: list[dict] = []
    inputs_normalized = model.inputs_are_normalized
    metric_names = model._enabled_metric_names(metric_gates)

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                inputs, targets = batch
                sizes = None
                names = None
            else:
                if len(batch) == 3:
                    inputs, targets, sizes = batch
                    names = None
                else:
                    inputs, targets, sizes, names = batch
                sizes = sizes.to(device)

            inputs = inputs.to(device)
            targets = targets.to(device)

            inputs_01 = to_image_range(inputs, normalized_to_neg_one_one=inputs_normalized)
            targets_01 = to_image_range(targets, normalized_to_neg_one_one=inputs_normalized)
            preds = model.pipeline(inputs_01)

            B = preds.shape[0]
            for i in range(B):
                if sizes is not None:
                    h = int(sizes[i, 0].item())
                    w = int(sizes[i, 1].item())
                else:
                    h, w = preds.shape[-2], preds.shape[-1]

                p = preds[i:i + 1, :, :h, :w]
                t = targets_01[i:i + 1, :, :h, :w]
                inp = inputs_01[i, :, :h, :w]

                sample_metrics = model.metrics(
                    p,
                    t,
                    enable_psnr=True,
                    enable_ssim=True,
                    **metric_gates,
                )

                row = {
                    "sample_index": len(samples),
                    "file_name": names[i] if names is not None else str(len(samples)),
                    "height": h,
                    "width": w,
                    "input_u8":  to_display_u8(inp.cpu()),
                    "pred_u8":   to_display_u8(p[0].cpu()),
                    "target_u8": to_display_u8(t[0].cpu()),
                }
                for metric_name in metric_names:
                    row[metric_name] = float(sample_metrics[metric_name].detach().cpu())
                samples.append(row)

    return samples


def main() -> None:
    args = parse_args()
    allow_trusted_checkpoint_loading()

    cfg = OmegaConf.load(args.config)
    if args.data_dir:
        cfg.data.data_dir = args.data_dir
    if args.batch_size:
        cfg.data.batch_size = args.batch_size

    device = get_device() if args.device == "auto" else torch.device(args.device)

    console.print(f"[cyan]Loading checkpoint:[/cyan] {args.checkpoint}")
    model = ColorEnhanceLitModule.load_from_checkpoint(
        args.checkpoint, cfg=cfg, map_location=device
    )
    model.eval().to(device)

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    loader = datamodule.test_dataloader()

    metric_gates = model._resolve_metric_gates(cfg.evaluation.get("test"))

    console.print(f"[cyan]Running inference on {len(datamodule._test)} test images …[/cyan]")
    samples = collect_samples(model, loader, device, metric_gates)

    if not samples:
        console.print("[red]No samples collected — check data path.[/red]")
        return

    test_psnrs = [s["psnr"] for s in samples]
    avg_psnr = sum(test_psnrs) / len(test_psnrs)
    console.print(
        f"[green]Collected {len(samples)} samples[/green]  "
        f"avg PSNR={avg_psnr:.2f} dB  "
        f"min={min(test_psnrs):.2f}  max={max(test_psnrs):.2f}"
    )

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ckpt_path = Path(args.checkpoint)
        out_dir = ckpt_path.parent / "visual_results" / ckpt_path.stem

    exp_name = cfg.get("experiment_name", Path(args.checkpoint).stem)
    saved = save_test_report(
        samples,
        out_dir,
        n_per_group=args.n_per_group,
        experiment_name=exp_name,
    )
    write_test_metric_exports({"experiment_name": exp_name}, samples, out_dir)

    for group, path in saved.items():
        console.print(f"  [green]→[/green] {group}: {path}")

    console.print(f"\n[bold green]Report saved to {out_dir}[/bold green]")


if __name__ == "__main__":
    main()

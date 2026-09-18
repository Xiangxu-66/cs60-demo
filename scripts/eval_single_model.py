#!/usr/bin/env python3
"""Single model evaluation with fixed sample indices for comparison.

Generates individual images (input, prediction, target) with standardized
naming for easy aggregation into comparison tables.

Usage::

    # Basic usage with default equidistant sampling
    python scripts/eval_single_model.py \\
        --checkpoint outputs/checkpoints/model.ckpt \\
        --config outputs/logs/experiment/config/config.yaml

    # Specify number of samples
    python scripts/eval_single_model.py \\
        --checkpoint outputs/checkpoints/model.ckpt \\
        --config outputs/logs/experiment/config/config.yaml \\
        --num-samples 20

    # Use scene-based sampling
    python scripts/eval_single_model.py \\
        --checkpoint outputs/checkpoints/model.ckpt \\
        --config outputs/logs/experiment/config/config.yaml \\
        --strategy scene --scene-category portraits

    # Custom indices
    python scripts/eval_single_model.py \\
        --checkpoint outputs/checkpoints/model.ckpt \\
        --config outputs/logs/experiment/config/config.yaml \\
        --strategy custom --indices 0 5 12 23 34 45
"""
from __future__ import annotations

import argparse
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from PIL import Image

from src.data.transforms import to_image_range
from src.evaluation.metadata import (
    EvalMetadata,
    extract_model_name,
    generate_filename,
)
from src.evaluation.samples import SampleIndexSelector, load_sample_indices
from src.lit_module import ColorEnhanceLitModule
from src.utils.checkpoint import allow_trusted_checkpoint_loading
from src.utils.device import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a single model with fixed sample indices"
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint (.ckpt file)",
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to config snapshot (config.yaml)",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="Override data directory from config",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/eval_single",
        help="Output directory for generated images",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Device to run on (auto/cpu/cuda)",
    )
    p.add_argument(
        "--strategy",
        choices=["equidistant", "difficulty", "scene", "custom"],
        default=None,
        help="Sampling strategy (overrides config)",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples for equidistant/difficulty strategies",
    )
    p.add_argument(
        "--scene-category",
        type=str,
        default=None,
        help="Scene category for scene-based sampling",
    )
    p.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Custom indices for custom strategy",
    )
    p.add_argument(
        "--sample-config",
        default="configs/eval_samples.yaml",
        help="Path to sample configuration file",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference (reduce if OOM)",
    )
    p.add_argument(
        "--model-name",
        default=None,
        help="Override model name (default: extracted from checkpoint path)",
    )
    p.add_argument(
        "--pred-only",
        action="store_true",
        help="Only save prediction images (input/target are shared across models)",
    )
    return p.parse_args()


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert [0, 1] tensor to PIL Image.

    Args:
        tensor: Image tensor (C, H, W) in [0, 1].

    Returns:
        PIL Image.
    """
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    """Save tensor as PNG image.

    Args:
        tensor: Image tensor (C, H, W) in [0, 1].
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_image(tensor).save(path)


def main() -> None:
    args = parse_args()
    allow_trusted_checkpoint_loading()

    # Load config
    cfg = OmegaConf.load(args.config)
    if args.data_dir:
        cfg.data.data_dir = args.data_dir

    # Setup device
    device = get_device() if args.device == "auto" else torch.device(args.device)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = ColorEnhanceLitModule.load_from_checkpoint(
        args.checkpoint, cfg=cfg, map_location=device
    )
    model.eval().to(device)

    # Extract model name
    model_name = args.model_name or extract_model_name(args.checkpoint)
    print(f"Model name: {model_name}")

    # Setup datamodule
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    test_dataset = datamodule._test

    # Get sample indices
    print(f"Loading sample configuration from: {args.sample_config}")

    if args.strategy == "custom" and args.indices is not None:
        # Direct custom indices
        sample_indices = [i for i in args.indices if i < len(test_dataset)]
        strategy_used = "custom"
    elif args.strategy:
        # Use strategy selector with overrides
        base_selector = SampleIndexSelector.from_config(args.sample_config)
        selector = SampleIndexSelector(
            strategy=args.strategy,
            num_samples=args.num_samples or base_selector.num_samples,
            start_idx=base_selector.start_idx,
            custom_indices=args.indices,
            scene_indices=base_selector.scene_indices,
        )
        sample_indices = selector.get_indices(
            len(test_dataset),
            scene_category=args.scene_category,
        )
        strategy_used = args.strategy
    else:
        # Use config file directly
        sample_indices = load_sample_indices(args.sample_config, len(test_dataset))
        strategy_used = OmegaConf.load(args.sample_config).get("active_strategy", "equidistant")

    print(f"Strategy: {strategy_used}")
    print(f"Selected {len(sample_indices)} indices: {sample_indices[:10]}...")

    # Setup output directory
    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata
    metadata = EvalMetadata(
        model_name=model_name,
        checkpoint_path=str(args.checkpoint),
        sampling_strategy=strategy_used,
        sample_indices=sample_indices,
        total_test_size=len(test_dataset),
    )

    # Process samples
    inputs_are_normalized = bool(cfg.data.get("normalize_to_neg_one_one", True))
    pred_only = args.pred_only

    print(f"\nGenerating images for {len(sample_indices)} samples...")
    if pred_only:
        print("  (Prediction only mode - skipping input/target)")

    with torch.no_grad():
        for global_idx, test_idx in enumerate(sample_indices):
            # Get single sample from dataset
            sample = test_dataset[test_idx]
            input_img, target_img = sample[0], sample[1]
            input_img = input_img.unsqueeze(0).to(device)
            target_img = target_img.unsqueeze(0).to(device)

            # Convert to [0, 1] if needed
            input_01 = to_image_range(
                input_img,
                normalized_to_neg_one_one=inputs_are_normalized,
            )
            target_01 = to_image_range(
                target_img,
                normalized_to_neg_one_one=inputs_are_normalized,
            )

            # Generate prediction
            pred_01 = model.pipeline(input_01)

            # Generate filenames
            pred_path = output_dir / generate_filename(model_name, "pred", test_idx)

            # Save prediction
            save_image(pred_01[0], pred_path)

            # Save input/target only if not pred-only mode
            if not pred_only:
                input_path = output_dir / generate_filename(model_name, "input", test_idx)
                target_path = output_dir / generate_filename(model_name, "target", test_idx)
                save_image(input_01[0], input_path)
                save_image(target_01[0], target_path)

                metadata.add_file_mapping(
                    test_idx,
                    str(input_path.relative_to(output_dir.parent)),
                    str(pred_path.relative_to(output_dir.parent)),
                    str(target_path.relative_to(output_dir.parent)),
                )
            else:
                # Pred-only mode: only store prediction path
                metadata.file_mappings[test_idx] = {
                    "pred": str(pred_path.relative_to(output_dir.parent))
                }

            if (global_idx + 1) % 5 == 0:
                print(f"  Processed {global_idx + 1}/{len(sample_indices)}")

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    metadata.save(metadata_path)
    print(f"\nMetadata saved to: {metadata_path}")

    print(f"\nDone! Output directory: {output_dir}")
    if pred_only:
        print(f"  - {len(sample_indices)} prediction images: pred_{model_name}_*.png")
    else:
        print(f"  - {len(sample_indices)} input images: input_{model_name}_*.png")
        print(f"  - {len(sample_indices)} prediction images: pred_{model_name}_*.png")
        print(f"  - {len(sample_indices)} target images: target_*.png")


if __name__ == "__main__":
    main()

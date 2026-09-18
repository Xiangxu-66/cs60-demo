#!/usr/bin/env python3
"""Merge prediction outputs from multiple models for side-by-side comparison.

Takes multiple model zip files (each containing pred_*.png and metadata.json),
extracts common samples, and generates side-by-side comparison images.

Usage::

    # Using config file (recommended)
    python scripts/merge_model_predictions.py \\
        --config configs/merge_img.yaml

    # Using command line arguments
    python scripts/merge_model_predictions.py \\
        --zip-files model_a.zip model_b.zip model_c.zip \\
        --data-dir /path/to/dataset \\
        --input-subdir a \\
        --target-subdir c

    # Override specific config values with CLI args
    python scripts/merge_model_predictions.py \\
        --config configs/merge_img.yaml \\
        --max-samples 10
"""
from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf
from PIL import Image
from PIL import ImageOps

from src.data.transforms import get_val_transforms, to_image_range
from src.evaluation.metadata import EvalMetadata, merge_metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge prediction outputs from multiple models"
    )
    p.add_argument(
        "--config",
        type=Path,
        help="Path to YAML config file with all settings (provides zip_files, data_dir, etc.)",
    )
    p.add_argument(
        "--zip-files",
        nargs="+",
        help="Path to model prediction zip files",
    )
    p.add_argument(
        "--data-dir",
        help="Path to dataset directory (for input/target images)",
    )
    p.add_argument(
        "--input-subdir",
        help="Subdirectory name for input images (default: a, alternatives: Original, raw)",
    )
    p.add_argument(
        "--target-subdir",
        help="Subdirectory name for target images (default: c, alternatives: expertC)",
    )
    p.add_argument(
        "--test-start-index",
        type=int,
        help="Global index where test set starts (default: 4500 for the standard FiveK final 500 test images)",
    )
    p.add_argument(
        "--crop-size",
        type=int,
        help="Size to resize/crop images to (default: 480, matching model output)",
    )
    p.add_argument(
        "--output-dir",
        help="Output directory for comparison images",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of comparison images to generate",
    )
    p.add_argument(
        "--tmp-dir",
        help="Temporary directory for extracting zips (default: auto)",
    )
    p.add_argument(
        "--keep-zip-content",
        action="store_true",
        help="Keep extracted directories after processing",
    )
    p.add_argument(
        "--create-output-zip",
        action="store_true",
        help="Create a zip file of the output directory",
    )
    return p.parse_args()


def load_column_config(config_path: Path) -> dict[str, str]:
    """Load column display name mapping from config file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Dictionary mapping zip filename (stem) to display name.
    """
    if config_path is None:
        return {}

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config_path)
        # Support both "model_columns" and direct dict format
        if "model_columns" in cfg:
            return dict(cfg.model_columns)
        return dict(cfg)
    except Exception as e:
        raise ValueError(f"Failed to load config from {config_path}: {e}")


def load_merge_config(config_path: Path) -> dict[str, Any]:
    """Load full configuration for merge_model_predictions from YAML file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Dictionary with all configuration values.
    """
    if config_path is None:
        return {}

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config_path)
        return dict(cfg)
    except Exception as e:
        raise ValueError(f"Failed to load config from {config_path}: {e}")


def extract_zip(zip_path: Path, output_dir: Path) -> Path:
    """Extract zip file to output directory.

    Args:
        zip_path: Path to zip file.
        output_dir: Directory to extract to.

    Returns:
        Path to extracted directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    # Find the extracted directory (usually one level deep)
    extracted_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    if len(extracted_dirs) == 1:
        return extracted_dirs[0]
    return output_dir


def load_metadata_from_dir(model_dir: Path) -> dict[str, Any] | None:
    """Load metadata.json from a model directory.

    Args:
        model_dir: Path to extracted model directory.

    Returns:
        Metadata dictionary, or None if metadata.json doesn't exist.
    """
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        return json.load(f)


def extract_model_name_from_dir(model_dir: Path) -> str:
    """Extract model name from directory name or prediction files.

    Args:
        model_dir: Path to extracted model directory.

    Returns:
        Model name string.
    """
    # First, try to extract from prediction files (more reliable)
    for file in model_dir.rglob("pred_*.png"):
        stem = file.stem
        # pred_model_name_0010 -> model_name
        if stem.startswith("pred_"):
            parts = stem.split("_")
            if len(parts) >= 2:
                # Remove "pred" prefix and index suffix (last 4 digits)
                # Handle names with extra numbers like model_056_22_98_0000
                # We want to keep the base name but drop common suffixes
                name_parts = parts[1:-1]  # Remove "pred" and index

                # Remove common numeric suffixes that look like epoch/metrics
                # e.g., "_048_22_78" or "_056_22_98"
                while name_parts and name_parts[-1].isdigit():
                    name_parts.pop()

                if name_parts:
                    return "_".join(name_parts)

    # Fallback: try directory name
    dir_name = model_dir.name
    # Remove common suffixes
    for suffix in ["_preds", "_predictions", "_output"]:
        if dir_name.endswith(suffix):
            dir_name = dir_name[:-len(suffix)]
            break

    if dir_name and dir_name != "pred" and not dir_name.startswith("."):
        return dir_name

    return "unknown_model"


def find_prediction_files(model_dir: Path, model_name: str = None) -> dict[int, Path]:
    """Find all prediction files in a model directory.

    Args:
        model_dir: Path to extracted model directory.
        model_name: Optional model name for filtering. If None, matches all pred_*.png.

    Returns:
        Mapping from test index to prediction file path.
    """
    pred_files = {}

    for file in model_dir.rglob("pred_*.png"):
        # Extract index from filename: pred_model_name_0010.png -> 10
        # Or: pred_model_name_xx_yy_zz_0010.png -> 10
        stem = file.stem

        # Check if file belongs to this model (if model_name specified)
        if model_name:
            # Check if model_name appears in the filename (not necessarily as prefix)
            # and is in the "pred_" section (before the final index)
            if f"pred_{model_name}" not in stem:
                continue

        # Extract index from the END (last 4 digits)
        # Format: ..._XXXX where XXXX is the index
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue

        idx_str = parts[-1]
        if not idx_str.isdigit() or len(idx_str) != 4:
            continue

        try:
            idx = int(idx_str)
            pred_files[idx] = file
        except ValueError:
            continue

    return pred_files


def get_common_indices(model_metadata: list[dict]) -> set[int]:
    """Get intersection of sample indices across all models.

    Args:
        model_metadata: List of metadata dictionaries.

    Returns:
        Set of common indices.
    """
    if not model_metadata:
        return set()

    common = set(model_metadata[0].get("sample_indices", []))
    for meta in model_metadata[1:]:
        common &= set(meta.get("sample_indices", []))

    return common


def create_comparison_image(
    images: list[Image.Image],
    labels: list[str],
    padding: int = 10,
) -> Image.Image:
    """Create a side-by-side comparison image.

    Args:
        images: List of PIL Images to compare.
        labels: Labels for each image.
        padding: Padding between images.

    Returns:
        Combined comparison image with labels.
    """
    from PIL import ImageDraw, ImageFont

    if not images:
        raise ValueError("No images provided")

    # Check that all images have the same size
    target_w, target_h = images[0].size
    mismatched = []
    for i, (img, label) in enumerate(zip(images, labels)):
        if img.size != (target_w, target_h):
            mismatched.append(f"{label}: {img.size} (expected {target_w}x{target_h})")

    if mismatched:
        raise ValueError(
            f"Image size mismatch detected:\n" + "\n".join(mismatched) +
            f"\n\nAll images must have the same size ({target_w}x{target_h})."
        )

    n = len(images)

    # Calculate total size
    total_w = n * target_w + (n - 1) * padding
    label_h = 30

    # Create combined image
    combined = Image.new("RGB", (total_w, target_h + label_h), (255, 255, 255))

    # Paste images
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, label_h))
        x_offset += target_w + padding

    # Add labels
    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        try:
            font = ImageFont.truetype("Arial", 14)
        except OSError:
            font = ImageFont.load_default()

    x_offset = 0
    for label in labels:
        # Get text size
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]

        # Center text above image
        text_x = x_offset + (target_w - text_w) // 2
        draw.text((text_x, 5), label, fill=(0, 0, 0), font=font)
        x_offset += target_w + padding

    return combined


def create_grid_comparison(
    rows: list[list[Image.Image]],
    row_labels: list[str],
    col_labels: list[str],
    padding: int = 10,
) -> Image.Image:
    """Create a grid comparison image (multiple samples, all models).

    Args:
        rows: List of rows, each row is a list of images for one sample.
        row_labels: Labels for each row (sample indices).
        col_labels: Labels for each column (model names).
        padding: Padding between images.

    Returns:
        Grid comparison image.
    """
    from PIL import ImageDraw, ImageFont

    if not rows or not rows[0]:
        raise ValueError("No images provided")

    n_rows = len(rows)
    n_cols = len(rows[0])

    # Check that all images have the same size
    target_w, target_h = rows[0][0].size
    mismatched = []

    for i, row in enumerate(rows):
        for j, img in enumerate(row):
            if img.size != (target_w, target_h):
                mismatched.append(
                    f"Row {i}, Col {j} ({col_labels[j] if j < len(col_labels) else '?'}): "
                    f"{img.size} (expected {target_w}x{target_h})"
                )

    if mismatched:
        raise ValueError(
            f"Image size mismatch detected:\n" + "\n".join(mismatched) +
            f"\n\nAll images must have the same size ({target_w}x{target_h})."
        )

    label_h = 30
    label_w = 60

    # Calculate total size
    total_w = label_w + n_cols * target_w + (n_cols - 1) * padding
    total_h = label_h + n_rows * target_h + (n_rows - 1) * padding

    # Create combined image
    combined = Image.new("RGB", (total_w, total_h), (255, 255, 255))

    # Add column labels
    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        try:
            font = ImageFont.truetype("Arial", 14)
            font_bold = ImageFont.truetype("Arial Bold", 14)
        except OSError:
            font = ImageFont.load_default()
            font_bold = font

    # Column labels (top)
    for j, label in enumerate(col_labels):
        x = label_w + j * (target_w + padding) + (target_w - len(label) * 6) // 2
        draw.text((x, 5), label, fill=(0, 0, 0), font=font_bold)

    # Row labels (left) and images
    for i, row in enumerate(rows):
        # Row label
        y = label_h + i * (target_h + padding) + target_h // 2
        draw.text((5, y), row_labels[i], fill=(0, 0, 0), font=font)

        # Images
        for j, img in enumerate(row):
            x = label_w + j * (target_w + padding)
            y = label_h + i * (target_h + padding)
            combined.paste(img, (x, y))

    return combined


def load_dataset_image_direct(
    data_dir: Path,
    input_subdir: str,
    target_subdir: str,
    test_idx: int,
    image_type: str,  # "input" or "target"
    file_names: list[str] = None,
    test_start_index: int = 0,
    crop_size: int = 480,
) -> Image.Image:
    """Load an image directly from dataset directory with resize/crop.

    Args:
        data_dir: Path to dataset root.
        input_subdir: Subdirectory name for input images.
        target_subdir: Subdirectory name for target images.
        test_idx: Index in test set (0-based).
        image_type: "input" or "target".
        file_names: Optional list of filenames (if None, scans directory).
        test_start_index: Global index where the standard FiveK final 500 test
            images start.
        crop_size: Size to resize images to (default: 480).

    Returns:
        PIL Image (RGB, resized to crop_size x crop_size).
    """
    if image_type == "input":
        img_dir = data_dir / input_subdir
    else:
        img_dir = data_dir / target_subdir

    if not img_dir.exists():
        raise FileNotFoundError(f"Directory not found: {img_dir}")

    # Get list of image files
    if file_names is None:
        file_names = sorted([
            f.stem for f in img_dir.glob("*.png")
            if f.stem.startswith("a")
        ])

    # Map test_idx to global index
    global_idx = test_idx + test_start_index

    if global_idx >= len(file_names):
        raise IndexError(f"global_idx {global_idx} (test_idx={test_idx} + start={test_start_index}) out of range ({len(file_names)} files)")

    # Find the image file
    filename = file_names[global_idx]
    img_path = img_dir / f"{filename}.png"

    if not img_path.exists():
        # Try .jpg
        img_path = img_dir / f"{filename}.jpg"

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Load and resize to match model output size
    img = Image.open(img_path).convert("RGB")

    # Resize to crop_size x crop_size using the same method as validation transforms
    # Using Lanczos for high quality
    img = img.resize((crop_size, crop_size), Image.Resampling.LANCZOS)

    return img


def main() -> None:
    args = parse_args()

    # Load config file if provided
    config = load_merge_config(args.config)

    # Merge config with command line args (CLI args take precedence)
    zip_dir = config.get("zip_dir")
    zip_files = args.zip_files or config.get("zip_files", [])

    # If zip_dir is specified, scan for zip files
    if zip_dir and not zip_files:
        zip_dir_path = Path(zip_dir)
        if not zip_dir_path.exists():
            raise FileNotFoundError(f"Zip directory not found: {zip_dir}")
        zip_files = sorted([
            str(f) for f in zip_dir_path.glob("*.zip")
        ])
        print(f"Found {len(zip_files)} zip files in {zip_dir}")

    data_dir = args.data_dir or config.get("data_dir")
    input_subdir = args.input_subdir or config.get("input_subdir", "a")
    target_subdir = args.target_subdir or config.get("target_subdir", "c")
    test_start_index = args.test_start_index if args.test_start_index is not None else config.get("test_start_index", 4500)
    crop_size = args.crop_size if args.crop_size is not None else config.get("crop_size", 480)
    output_dir = args.output_dir or config.get("output_dir", "outputs/model_comparison")
    max_samples = args.max_samples if args.max_samples is not None else config.get("max_samples")
    tmp_dir = args.tmp_dir or config.get("tmp_dir")
    keep_zip_content = args.keep_zip_content or config.get("keep_zip_content", False)
    create_output_zip = args.create_output_zip or config.get("create_output_zip", False)

    # Validate required settings
    if not zip_files:
        raise ValueError(
            "No zip files provided. Specify via --zip-files, zip_dir in config, or zip_files in config"
        )
    if not data_dir:
        raise ValueError(
            "No data directory provided. Specify via --data-dir or in config file under 'data_dir:'"
        )

    # Get column name mapping
    column_config = config.get("model_columns", {})

    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if column_config:
        print(f"Column config loaded: {len(column_config)} mappings")
    else:
        print("No column config provided, using auto-detected names")

    if tmp_dir:
        tmp_dir = Path(tmp_dir)
    else:
        tmp_dir = output_dir / ".tmp_extract"

    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Temporary directory: {tmp_dir}")

    # 1. Extract all zip files
    print("\n=== Extracting zip files ===")
    model_dirs = []
    model_metadata = []
    model_names = []
    zip_stem_to_model_name: dict[str, str] = {}  # For prediction file lookup

    for zip_path in zip_files:
        zip_path = Path(zip_path)
        zip_stem = zip_path.stem
        print(f"Extracting {zip_path.name}")

        extract_dir = tmp_dir / zip_stem
        model_dir = extract_zip(zip_path, extract_dir)
        model_dirs.append(model_dir)

        # Load metadata (may be None)
        meta = load_metadata_from_dir(model_dir)
        if meta is None:
            # Extract model name from directory/files for file lookup
            internal_model_name = extract_model_name_from_dir(model_dir)
            print(f"    Internal model name: {internal_model_name} (no metadata.json)")
            # Find prediction files to get indices
            pred_files = find_prediction_files(model_dir)
            indices = sorted(pred_files.keys())
            meta = {
                "model_name": internal_model_name,
                "checkpoint_path": str(zip_path),
                "sample_indices": indices,
                "total_test_size": len(indices),
            }
        else:
            internal_model_name = meta["model_name"]
            print(f"    Internal model name: {internal_model_name}")
            print(f"    Samples: {len(meta['sample_indices'])}")

        # Get display name from config or fall back to internal name
        display_name = column_config.get(zip_stem, internal_model_name)
        if column_config and zip_stem in column_config:
            print(f"    Display name: {display_name} (from config)")

        # Store mapping for prediction file lookup
        zip_stem_to_model_name[zip_stem] = internal_model_name

        # Store display_name in metadata for later use
        meta["display_name"] = display_name
        meta["zip_stem"] = zip_stem

        model_metadata.append(meta)
        model_names.append(display_name)

    # 2. Find common indices
    print("\n=== Finding common samples ===")
    common_indices = sorted(get_common_indices(model_metadata))
    print(f"Common samples across all models: {len(common_indices)}")

    if max_samples:
        common_indices = common_indices[:max_samples]
        print(f"Limited to {len(common_indices)} samples")

    # 3. Setup dataset for loading input/target
    print("\n=== Setting up dataset ===")
    data_dir = Path(data_dir)

    # Verify directories exist
    input_dir = data_dir / input_subdir
    target_dir = data_dir / target_subdir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")

    print(f"  Input directory: {input_dir}")
    print(f"  Target directory: {target_dir}")

    # Get list of image filenames (for indexing)
    input_files = sorted([
        f.stem for f in input_dir.glob("*.png")
        if f.stem.startswith("a")
    ])
    print(f"  Found {len(input_files)} input images")

    # 4. Build mapping: test_idx -> {display_name: pred_path}
    print("\n=== Indexing prediction files ===")
    pred_mapping: dict[int, dict[str, Path]] = defaultdict(dict)

    for meta, model_dir in zip(model_metadata, model_dirs):
        internal_name = meta["model_name"]
        display_name = meta["display_name"]
        pred_files = find_prediction_files(model_dir, internal_name)
        print(f"{display_name} (internal: {internal_name}): {len(pred_files)} prediction files")

        for idx, path in pred_files.items():
            if idx in common_indices:
                pred_mapping[idx][display_name] = path

    # Verify all samples have all models
    for idx in common_indices:
        if len(pred_mapping[idx]) != len(model_names):
            missing = set(model_names) - set(pred_mapping[idx].keys())
            print(f"Warning: Sample {idx} missing predictions from {missing}")

    # 5. Generate comparison images
    print("\n=== Generating comparison images ===")

    all_labels = ["Input"] + model_names + ["Target"]
    sample_images = []  # For grid layout
    row_labels = []

    for i, test_idx in enumerate(common_indices):
        print(f"[{i+1}/{len(common_indices)}] Sample {test_idx} (global: {test_idx + test_start_index})")

        # Load images
        images = []
        labels = []

        # Input
        input_img = load_dataset_image_direct(
            data_dir, input_subdir, target_subdir,
            test_idx, "input", input_files, test_start_index, crop_size
        )
        images.append(input_img)
        labels.append("Input")

        # Predictions from each model
        for model_name in model_names:
            pred_path = pred_mapping[test_idx].get(model_name)
            if pred_path:
                pred_img = Image.open(pred_path)
                images.append(pred_img)
                labels.append(model_name)
            else:
                # Placeholder if missing
                placeholder = Image.new("RGB", input_img.size, (128, 128, 128))
                images.append(placeholder)
                labels.append(f"{model_name} (missing)")

        # Target
        target_img = load_dataset_image_direct(
            data_dir, input_subdir, target_subdir,
            test_idx, "target", input_files, test_start_index, crop_size
        )
        images.append(target_img)
        labels.append("Target")

        # Save individual comparison
        comparison = create_comparison_image(images, labels)
        comparison_path = output_dir / f"comparison_{test_idx:04d}.png"
        comparison.save(comparison_path)

        # Store for grid
        sample_images.append(images)
        row_labels.append(f"#{test_idx}")

    # 6. Create grid comparison (all samples in one image)
    print("\n=== Creating grid comparison ===")
    grid = create_grid_comparison(sample_images, row_labels, all_labels)
    grid_path = output_dir / "comparison_grid.png"
    grid.save(grid_path)
    print(f"Grid saved: {grid_path}")

    # 7. Save summary metadata
    print("\n=== Saving summary ===")
    summary = {
        "models": [
            {
                "name": meta["model_name"],
                "display_name": meta.get("display_name", meta["model_name"]),
                "zip_stem": meta.get("zip_stem", ""),
                "checkpoint": meta["checkpoint_path"],
                "num_samples": len(meta["sample_indices"]),
            }
            for meta in model_metadata
        ],
        "common_indices": common_indices,
        "total_comparison_images": len(common_indices),
        "model_names": model_names,
        "output_directory": str(output_dir),
    }

    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")

    # Save summary as markdown table
    md_path = output_dir / "comparison_summary.md"
    with open(md_path, "w") as f:
        f.write("# Model Comparison Summary\n\n")
        f.write(f"**Output Directory:** `{output_dir}`\n\n")
        f.write(f"**Total Samples:** {len(common_indices)}\n\n")
        f.write(f"**Data Directory:** `{data_dir}`\n\n")
        f.write(f"**Input Subdir:** `{input_subdir}` | **Target Subdir:** `{target_subdir}`\n\n")

        f.write("## Models\n\n")
        f.write("| # | Display Name | Model Name | Checkpoint | Samples |\n")
        f.write("|---|--------------|------------|------------|--------|\n")
        for i, model in enumerate(summary["models"], 1):
            display_name = model["display_name"]
            model_name = model["name"]
            # Extract just the filename from checkpoint path
            checkpoint = model["checkpoint"]
            if "/" in checkpoint:
                checkpoint = checkpoint.split("/")[-1]
            num_samples = model["num_samples"]
            f.write(f"| {i} | {display_name} | `{model_name}` | `{checkpoint}` | {num_samples} |\n")

        f.write("\n## Sample Indices\n\n")
        f.write(f"Common test indices ({len(common_indices)} samples):\n\n")
        f.write("```\n")
        for i in range(0, len(common_indices), 10):
            f.write(", ".join(f"{idx:4d}" for idx in common_indices[i:i+10]) + "\n")
        f.write("```\n")
    print(f"Markdown summary saved: {md_path}")

    # Print table to console
    print("\n=== Model Comparison Summary ===")
    print(f"  Samples: {len(common_indices)}")
    print(f"  Data: {data_dir}")
    print(f"  Input: {input_subdir} | Target: {target_subdir}")
    print("\nModels:")
    print(f"  {'#':<2} {'Display Name':<25} {'Samples':<8}")
    print(f"  {'--':<2} {'-------------------------':<25} {'--------':<8}")
    for i, model in enumerate(summary["models"], 1):
        print(f"  {i:<2} {model['display_name']:<25} {model['num_samples']:<8}")

    # 8. Create output zip
    if create_output_zip:
        print("\n=== Creating output zip ===")
        import shutil
        zip_path = Path(str(output_dir) + ".zip")
        shutil.make_archive(str(output_dir), "zip", str(output_dir))
        print(f"Output zip: {zip_path}")

    # 9. Cleanup
    if not keep_zip_content:
        print(f"\n=== Cleaning up temporary directory ===")
        import shutil
        shutil.rmtree(tmp_dir)
        print(f"Removed: {tmp_dir}")
    else:
        print(f"\n=== Temporary files kept at: {tmp_dir} ===")

    print(f"\n=== Done! ===")
    print(f"Output: {output_dir}")
    print(f"  - {len(common_indices)} comparison images")
    print(f"  - 1 grid comparison image")
    print(f"  - 1 summary JSON")
    print(f"  - 1 summary Markdown")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
NIMA Image Quality Scoring Script

Usage:
    python scripts/nima_score.py <original_folder> <target_folder>

Example:
    python scripts/nima_score.py /path/to/Original /path/to/expertC
"""
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.models as models
import torchvision.transforms as T
from tqdm import tqdm
import json


class NIMA(torch.nn.Module):
    """NIMA (Neural Image Assessment) with MobileNetV2 backbone."""

    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v2(pretrained=True)
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = torch.nn.Linear(in_features, 10)
        self.backbone = backbone

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        logits = self.backbone(x)
        return F.softmax(logits, dim=1)


def load_image(image_path: Path) -> torch.Tensor:
    """Load image as tensor in [0,1] range."""
    img = Image.open(image_path).convert("RGB")
    transform = T.ToTensor()
    return transform(img).unsqueeze(0)


def compute_nima_score(model: NIMA, img: torch.Tensor) -> float:
    """Compute mean aesthetic score (1-10 scale)."""
    model.eval()
    with torch.no_grad():
        dist = model(img)  # (1, 10)
        weights = torch.arange(1, 11, device=img.device, dtype=dist.dtype)
        score = (dist * weights).sum()
    return score.item()


def find_image_pairs(original_folder: Path, target_folder: Path) -> list[tuple[Path, Path]]:
    """Find matching image pairs in two folders."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    original_files = {
        f.stem: f for f in original_folder.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    }
    target_files = {
        f.stem: f for f in target_folder.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    }

    common_names = set(original_files.keys()) & set(target_files.keys())

    pairs = [
        (original_files[name], target_files[name])
        for name in sorted(common_names)
    ]

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Compute NIMA aesthetic quality scores for image pairs"
    )
    parser.add_argument("original_folder", type=str, help="Path to Original images folder")
    parser.add_argument("target_folder", type=str, help="Path to Target (e.g., expertC) images folder")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda/mps/cpu). Default: auto-detect"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: nima_results_<timestamp>.json)"
    )
    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    # Generate output filename if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"nima_results_{timestamp}.json"
    output_path = Path(args.output)

    original_folder = Path(args.original_folder)
    target_folder = Path(args.target_folder)

    if not original_folder.is_dir():
        raise FileNotFoundError(f"Original folder not found: {original_folder}")
    if not target_folder.is_dir():
        raise FileNotFoundError(f"Target folder not found: {target_folder}")

    # Find image pairs
    pairs = find_image_pairs(original_folder, target_folder)
    if not pairs:
        print("No matching image pairs found!")
        return

    print(f"Found {len(pairs)} image pairs")

    # Load model
    print(f"Loading NIMA model on {args.device}...")
    model = NIMA().to(args.device)
    model.eval()

    # Compute scores
    results = []
    for original_path, target_path in tqdm(pairs, desc="Scoring"):
        original_img = load_image(original_path).to(args.device)
        target_img = load_image(target_path).to(args.device)

        original_score = compute_nima_score(model, original_img)
        target_score = compute_nima_score(model, target_img)

        results.append({
            "name": original_path.stem,
            "original": original_score,
            "target": target_score,
            "diff": target_score - original_score
        })

    # Output results
    print(f"\n{'Name':<40} {'Original':>10} {'Target':>10} {'Diff':>10}")
    print("-" * 75)

    orig_sum = 0
    targ_sum = 0
    for r in results:
        print(f"{r['name']:<40} {r['original']:>10.4f} {r['target']:>10.4f} {r['diff']:>10.4f}")
        orig_sum += r['original']
        targ_sum += r['target']

    print("-" * 75)
    n = len(results)
    avg_orig = orig_sum / n
    avg_targ = targ_sum / n
    print(f"{'Average':<40} {avg_orig:>10.4f} {avg_targ:>10.4f} {avg_targ-avg_orig:>10.4f}")

    # Save to JSON
    output_data = {
        "original_folder": str(original_folder),
        "target_folder": str(target_folder),
        "device": args.device,
        "total_pairs": n,
        "average_original": avg_orig,
        "average_target": avg_targ,
        "average_diff": avg_targ - avg_orig,
        "results": results
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

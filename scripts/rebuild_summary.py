"""Rebuild experiments/results_summary.csv from all experiment_record.json files.

Scans every subdirectory of experiments/ for an experiment_record.json,
then regenerates the global results_summary.csv in sorted order.

Run this any time after uploading new experiment directories, or to sync a
fresh download from the shared M365 folder:

    python scripts/rebuild_summary.py
    python scripts/rebuild_summary.py --experiments-dir /path/to/experiments

Output:
    experiments/results_summary.csv  — sorted by test_psnr descending
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.evaluation.experiment_record import SUMMARY_FIELDS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild results_summary.csv from experiment JSONs")
    p.add_argument(
        "--experiments-dir",
        default="experiments",
        help="Root directory containing per-experiment subdirectories (default: experiments/)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: {experiments_dir}/results_summary.csv)",
    )
    p.add_argument(
        "--sort-by",
        default="test_psnr",
        help="Field to sort results by, descending (default: test_psnr)",
    )
    return p.parse_args()


def load_records(experiments_dir: Path) -> list[dict]:
    """Scan for experiment_record.json files and return a list of record dicts."""
    records = []
    for json_path in sorted(experiments_dir.glob("*/experiment_record.json")):
        try:
            with open(json_path) as f:
                record = json.load(f)
            records.append(record)
        except Exception as e:
            print(f"  [skip] {json_path}: {e}")
    return records


def write_summary_csv(records: list[dict], out_path: Path, sort_by: str) -> None:
    """Write records to a CSV file, sorted descending by sort_by field."""
    def _sort_key(r: dict) -> float:
        v = r.get(sort_by, "")
        try:
            return -float(v)  # descending
        except (TypeError, ValueError):
            return float("inf")

    sorted_records = sorted(records, key=_sort_key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted_records)


def main() -> None:
    args = parse_args()
    experiments_dir = Path(args.experiments_dir)

    if not experiments_dir.exists():
        print(f"Directory not found: {experiments_dir}")
        return

    out_path = Path(args.output) if args.output else experiments_dir / "results_summary.csv"

    print(f"Scanning {experiments_dir} …")
    records = load_records(experiments_dir)

    if not records:
        print("No experiment_record.json files found.")
        return

    write_summary_csv(records, out_path, sort_by=args.sort_by)

    print(f"\nFound {len(records)} experiments:")
    for r in sorted(records, key=lambda x: -(float(x.get("test_psnr") or 0))):
        name = r.get("experiment_name", "?")
        psnr = r.get("test_psnr", "—")
        ssim = r.get("test_ssim", "—")
        lpips = r.get("test_lpips", "—")
        de = r.get("test_delta_e", "—")
        ts = r.get("timestamp", "")[:10]
        try:
            psnr_str = f"{float(psnr):.2f} dB"
        except (TypeError, ValueError):
            psnr_str = str(psnr)
        print(f"  {ts}  {name:<45}  PSNR={psnr_str}  SSIM={ssim}  LPIPS={lpips}  ΔE={de}")

    print(f"\nSummary written → {out_path}")


if __name__ == "__main__":
    main()

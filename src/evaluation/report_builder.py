"""Build a self-contained per-experiment report directory.

Called automatically from ColorEnhanceLitModule.on_test_end().
Can also be invoked via scripts/generate_test_report.py for post-hoc reports.

Output layout::

    experiments/{name}_{YYYYMMDD}_{commit}/
    ├── README.txt              human-readable one-page summary
    ├── experiment_record.json  full metadata (38 fields)
    ├── config.yaml             fully-resolved training config
    ├── training_curve.png      Loss + Val-PSNR vs Epoch (best-epoch marker)
    ├── metrics.csv             raw per-step/epoch data from Lightning
    ├── test_results.json       test-phase metrics
    ├── test_per_image_metrics.csv
    ├── test_per_image_metrics.json
    ├── test_best_psnr.png      5 best samples  (Input | Pred | GT | Diff×5)
    ├── test_median_psnr.png    5 median samples
    └── test_worst_psnr.png     5 worst samples — failure analysis
"""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from numbers import Number
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Directory naming
# ---------------------------------------------------------------------------

def make_experiment_dir_name(experiment_name: str, git_commit: str) -> str:
    """Return a sortable, unique directory name for this experiment run."""
    date = datetime.now().strftime("%Y%m%d")
    commit = (git_commit or "nocommit")[:7]
    return f"{experiment_name}_{date}_{commit}"


# ---------------------------------------------------------------------------
# Training curve
# ---------------------------------------------------------------------------

def _parse_metrics_csv(path: Path) -> dict[str, list]:
    """Extract per-epoch time series from a Lightning CSVLogger metrics.csv.

    Lightning writes separate rows for:
      - LR updates          (epoch blank, no metrics)
      - val epoch metrics   (epoch filled, val/* columns)
      - train step metrics  (epoch filled, train/*_step column)
      - train epoch totals  (epoch filled, train/*_epoch column)

    Returns a dict: metric_key -> list of (epoch, value) pairs, sorted by epoch.
    """
    by_epoch: dict[int, dict[str, float]] = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_epoch = row.get("epoch", "")
            if not raw_epoch:
                continue
            try:
                ep = int(float(raw_epoch))
            except (ValueError, TypeError):
                continue

            ep_dict = by_epoch.setdefault(ep, {})
            for k, v in row.items():
                if k in ("epoch", "step") or not v:
                    continue
                try:
                    ep_dict[k] = float(v)
                except (ValueError, TypeError):
                    pass

    if not by_epoch:
        return {}

    epochs = sorted(by_epoch)
    series: dict[str, list] = {"epochs": epochs}

    all_keys: set[str] = set()
    for d in by_epoch.values():
        all_keys.update(d.keys())

    for k in all_keys:
        # keep None for missing epochs so indices align
        series[k] = [by_epoch[ep].get(k) for ep in epochs]

    return series


def generate_training_curve(
    metrics_csv: Path,
    out_path: Path,
    best_epoch: int = -1,
) -> bool:
    """Plot Loss and Val-PSNR vs Epoch curves and save to out_path.

    Args:
        metrics_csv: Path to Lightning metrics.csv.
        out_path:    Destination PNG path.
        best_epoch:  If >= 0, draw a red dashed vertical line at this epoch.

    Returns:
        True if the plot was saved, False if no plottable data was found.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = _parse_metrics_csv(metrics_csv)
    if not series:
        return False

    epochs = series["epochs"]

    # Prefer epoch-aggregated train loss; fall back to step loss
    train_loss_key = next(
        (k for k in ["train/loss_epoch", "train/loss"] if k in series),
        None,
    )
    val_psnr_key  = "val/psnr"   if "val/psnr"   in series else None
    train_psnr_key = "train/psnr" if "train/psnr" in series else None

    has_psnr_panel = val_psnr_key is not None or train_psnr_key is not None
    if train_loss_key is None and not has_psnr_panel:
        return False

    n_panels = (train_loss_key is not None) + has_psnr_panel
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4), squeeze=False)
    axes = axes[0]

    panel = 0

    def _xy(key: str) -> tuple[list[int], list[float]]:
        xs, ys = [], []
        for ep, v in zip(epochs, series[key]):
            if v is not None:
                xs.append(ep)
                ys.append(v)
        return xs, ys

    def _vline(ax, colour: str = "#E53935") -> None:
        if best_epoch >= 0:
            ax.axvline(
                best_epoch, color=colour, linestyle="--",
                linewidth=1.2, alpha=0.8, label=f"best epoch {best_epoch}",
            )

    if train_loss_key is not None:
        ax = axes[panel]
        xs, ys = _xy(train_loss_key)
        ax.plot(xs, ys, color="#1565C0", linewidth=1.5, label="train loss")

        if "val/loss" in series:
            vxs, vys = _xy("val/loss")
            ax.plot(vxs, vys, color="#EF6C00", linewidth=1.2,
                    linestyle="--", label="val loss", alpha=0.8)

        _vline(ax)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Loss", fontsize=9)
        ax.set_title("Training Loss", fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        panel += 1

    if has_psnr_panel:
        ax = axes[panel]

        # Train PSNR — solid blue; if this is also low alongside val, it's underfitting
        if train_psnr_key is not None:
            txs, tys = _xy(train_psnr_key)
            ax.plot(txs, tys, color="#1565C0", linewidth=1.5,
                    label="train PSNR", alpha=0.85)

        # Val PSNR — solid green; gap between train and val indicates overfitting
        if val_psnr_key is not None:
            vxs, vys = _xy(val_psnr_key)
            ax.plot(vxs, vys, color="#2E7D32", linewidth=1.5, label="val PSNR")
            if vys:
                best_val = max(vys)
                ax.annotate(
                    f"val max {best_val:.2f} dB",
                    xy=(vxs[vys.index(best_val)], best_val),
                    xytext=(5, -12), textcoords="offset points",
                    fontsize=7, color="#2E7D32",
                )

        _vline(ax)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("PSNR (dB)", fontsize=9)
        ax.set_title("PSNR — train vs val  (gap = overfitting, both low = underfitting)",
                     fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Training Curve",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(pad=1.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Human-readable README.txt
# ---------------------------------------------------------------------------

def generate_readme_txt(record: dict[str, Any], out_path: Path) -> None:
    """Write a plain-text one-page experiment summary to out_path."""
    def _fmt(val: Any, precision: int = 4) -> str:
        if val == "" or val is None:
            return "—"
        if isinstance(val, float):
            return f"{val:.{precision}f}"
        return str(val)

    sep = "=" * 62
    lines = [
        sep,
        "  EXPERIMENT SUMMARY",
        sep,
        f"  Name       : {record.get('experiment_name', '—')}",
        f"  Timestamp  : {record.get('timestamp', '—')}",
        f"  Git commit : {record.get('git_commit', '—')}  "
        f"(branch: {record.get('git_branch', '—')})",
        "",
        "  ARCHITECTURE",
        f"    Encoder    : {record.get('encoder', '—')}",
        f"    Bottleneck : {record.get('bottleneck', '—')}",
        f"    Decoder    : {record.get('decoder', '—')}",
        f"    Fusion     : {record.get('fusion_mode', '—')}",
        f"    Skip conn  : {record.get('skip_connection', '—')}",
        f"    Loss       : {record.get('loss', '—')}",
        "",
        "  DATA",
        "    Dataset      : FiveK Expert C",
        f"    Image size   : {record.get('image_size', '—')} px (short edge)",
        f"    Crop mode    : {record.get('train_crop_mode', '—')}",
        f"    Eval resize  : {record.get('eval_resize_mode', '—')}",
        f"    Train / Val / Test : "
        f"{record.get('train_size', '—')} / "
        f"{record.get('val_size', '—')} / "
        f"{record.get('test_size', '—')}",
        "",
        "  TRAINING",
        f"    Batch        : {record.get('batch_size', '—')} × "
        f"{record.get('grad_accum', '—')} grad_accum = "
        f"{record.get('effective_batch', '—')} effective",
        f"    LR           : {record.get('lr', '—')}  "
        f"Scheduler: {record.get('scheduler', '—')}",
        f"    Max epochs   : {record.get('max_epochs', '—')}  "
        f"Best epoch: {record.get('best_epoch', '—')}",
        f"    Precision    : {record.get('precision', '—')}",
        f"    Training time: {_fmt(record.get('training_time_min'), 1)} min",
        "",
        "  EFFICIENCY",
        f"    Total params    : {record.get('total_params', '—')}",
        f"    Trainable params: {record.get('trainable_params', '—')}",
        f"    GFLOPs (480p)   : {_fmt(record.get('gflops'), 2)}",
        f"    Peak VRAM       : {_fmt(record.get('peak_vram_mb'), 1)} MB",
        "",
        "  RESULTS",
        f"    Val   PSNR : {_fmt(record.get('val_psnr'), 2)} dB  "
        f"SSIM : {_fmt(record.get('val_ssim'), 3)}  "
        f"LPIPS : {_fmt(record.get('val_lpips'), 3)}",
        f"    Test  PSNR : {_fmt(record.get('test_psnr'), 2)} dB  "
        f"SSIM : {_fmt(record.get('test_ssim'), 3)}  "
        f"LPIPS : {_fmt(record.get('test_lpips'), 3)}",
        f"    Test  ΔE   : {_fmt(record.get('test_delta_e'), 2)}  "
        f"NIMA : {_fmt(record.get('test_nima'), 2)}",
        "",
        "  CHECKPOINT (on server)",
        f"    {record.get('checkpoint_path', '—')}",
        "",
        "  NOTES",
        f"    {record.get('notes', '(none)')}",
        sep,
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Find Lightning metrics CSV
# ---------------------------------------------------------------------------

def find_metrics_csv(log_dir: Path) -> Path | None:
    """Return the latest version metrics.csv under log_dir/version_N/."""
    candidates = sorted(log_dir.glob("version_*/metrics.csv"))
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Test metric exports
# ---------------------------------------------------------------------------

_IMAGE_SAMPLE_KEYS = {"input_u8", "pred_u8", "target_u8"}
_PER_IMAGE_ID_FIELDS = ["sample_index", "file_name", "height", "width"]
_PREFERRED_METRIC_ORDER = [
    "psnr",
    "ssim",
    "lpips",
    "delta_e",
    "nima",
    "clip",
    "vgg_perceptual",
]


def _is_metric_value(value: Any) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool)


def _metric_keys_from_samples(test_samples: list[dict]) -> list[str]:
    keys: set[str] = set()
    excluded = _IMAGE_SAMPLE_KEYS | set(_PER_IMAGE_ID_FIELDS)
    for sample in test_samples:
        for key, value in sample.items():
            if key not in excluded and _is_metric_value(value):
                keys.add(key)

    ordered = [key for key in _PREFERRED_METRIC_ORDER if key in keys]
    ordered.extend(sorted(keys.difference(ordered)))
    return ordered


def _per_image_rows(test_samples: list[dict], metric_keys: list[str]) -> list[dict]:
    rows: list[dict] = []
    for idx, sample in enumerate(test_samples):
        row = {
            "sample_index": sample.get("sample_index", idx),
            "file_name": sample.get("file_name", str(idx)),
            "height": sample.get("height", ""),
            "width": sample.get("width", ""),
        }
        for key in metric_keys:
            value = sample.get(key, "")
            row[key] = float(value) if _is_metric_value(value) else value
        rows.append(row)
    return rows


def _mean_metrics(rows: list[dict], metric_keys: list[str]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in metric_keys:
        values = [row[key] for row in rows if _is_metric_value(row.get(key))]
        if values:
            summary[key] = sum(float(v) for v in values) / len(values)
    return summary


def write_test_metric_exports(
    record_data: dict[str, Any],
    test_samples: list[dict],
    exp_dir: Path,
) -> None:
    """Write test_results.json plus per-image CSV/JSON metric files."""
    exp_name = record_data.get("experiment_name", "unnamed")
    metric_keys = _metric_keys_from_samples(test_samples)
    rows = _per_image_rows(test_samples, metric_keys)
    metrics = _mean_metrics(rows, metric_keys)

    if rows:
        csv_path = exp_dir / "test_per_image_metrics.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=_PER_IMAGE_ID_FIELDS + metric_keys,
            )
            writer.writeheader()
            writer.writerows(rows)

        per_image_payload = {
            "experiment_name": exp_name,
            "num_samples": len(rows),
            "metric_names": metric_keys,
            "samples": rows,
        }
        (exp_dir / "test_per_image_metrics.json").write_text(
            json.dumps(per_image_payload, indent=2, default=str),
            encoding="utf-8",
        )
    else:
        for key in ["psnr", "ssim", "lpips", "delta_e", "nima"]:
            value = record_data.get(f"test_{key}", "")
            if value != "":
                metrics[key] = float(value)

    results_payload = {
        "experiment_name": exp_name,
        "num_samples": len(rows),
        "metrics": metrics,
    }
    if rows:
        results_payload["per_image_metrics_csv"] = "test_per_image_metrics.csv"
        results_payload["per_image_metrics_json"] = "test_per_image_metrics.json"

    (exp_dir / "test_results.json").write_text(
        json.dumps(results_payload, indent=2, default=str),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_experiment_dir(
    record_data: dict[str, Any],
    test_samples: list[dict],
    log_dir: Path,
    base_dir: Path = Path("experiments"),
) -> Path:
    """Consolidate all experiment artefacts into one self-contained directory.

    Args:
        record_data:  ExperimentRecord.data dict (already finalised).
        test_samples: Per-sample list from lit_module._test_samples.
        log_dir:      outputs/logs/{experiment_name}/ directory.
        base_dir:     Root for all experiment output dirs (default: experiments/).

    Returns:
        Path to the created experiment directory.
    """
    from src.evaluation.visualization import save_test_report

    exp_name = record_data.get("experiment_name", "unnamed")
    git_commit = record_data.get("git_commit", "unknown")
    dir_name = make_experiment_dir_name(exp_name, git_commit)
    exp_dir = base_dir / dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1. config.yaml
    config_src = log_dir / "config" / "config.yaml"
    if config_src.exists():
        shutil.copy2(config_src, exp_dir / "config.yaml")

    # 2. metrics.csv (Lightning training log)
    metrics_csv = find_metrics_csv(log_dir)
    if metrics_csv and metrics_csv.exists():
        shutil.copy2(metrics_csv, exp_dir / "metrics.csv")

    # 3. Test metrics generated directly from this run.
    write_test_metric_exports(record_data, test_samples, exp_dir)

    # 4. Training curve PNG
    if metrics_csv and metrics_csv.exists():
        best_epoch = int(record_data.get("best_epoch") or -1)
        generate_training_curve(metrics_csv, exp_dir / "training_curve.png", best_epoch)

    # 5. Test visualisation PNGs (best / median / worst)
    if test_samples:
        saved = save_test_report(
            test_samples,
            exp_dir,
            experiment_name=exp_name,
        )
        for group, path in saved.items():
            print(f"  [{group}] → {path.name}")

    # 6. experiment_record.json (with experiment_dir field filled in)
    record_data["experiment_dir"] = str(exp_dir)
    (exp_dir / "experiment_record.json").write_text(
        json.dumps(record_data, indent=2, default=str),
        encoding="utf-8",
    )

    # 7. Profiler detail JSON (only present when training was run with profile=true)
    profiler_src = Path("outputs/profiler_logs") / f"{exp_name}_profiler.json"
    if profiler_src.exists():
        shutil.copy2(profiler_src, exp_dir / "profiler_detail.json")

    # 8. Human-readable README.txt
    generate_readme_txt(record_data, exp_dir / "README.txt")

    return exp_dir

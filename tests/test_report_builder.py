from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from src.evaluation.report_builder import build_experiment_dir
from src.lit_module import ColorEnhanceLitModule


def test_build_experiment_dir_writes_run_local_test_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import src.evaluation.visualization as visualization

    monkeypatch.setattr(visualization, "save_test_report", lambda *args, **kwargs: {})

    log_dir = tmp_path / "logs" / "dinov3_2000img"
    (log_dir / "config").mkdir(parents=True)
    (log_dir / "config" / "config.yaml").write_text(
        "experiment_name: dinov3_2000img\n",
        encoding="utf-8",
    )

    samples = [
        {
            "sample_index": 0,
            "file_name": "a0001",
            "height": 8,
            "width": 10,
            "psnr": 20.0,
            "ssim": 0.80,
            "delta_e": 4.0,
        },
        {
            "sample_index": 1,
            "file_name": "a0002",
            "height": 9,
            "width": 11,
            "psnr": 22.0,
            "ssim": 0.82,
            "delta_e": 2.0,
        },
    ]
    record = {
        "experiment_name": "dinov3_2000img",
        "git_commit": "abcdef0",
        "git_branch": "test",
    }

    exp_dir = build_experiment_dir(
        record,
        samples,
        log_dir,
        base_dir=tmp_path / "experiments",
    )

    results = json.loads((exp_dir / "test_results.json").read_text())
    assert results["experiment_name"] == "dinov3_2000img"
    assert results["num_samples"] == 2
    assert results["metrics"] == {
        "psnr": 21.0,
        "ssim": 0.81,
        "delta_e": 3.0,
    }
    assert "lpips" not in results["metrics"]

    per_image_json = json.loads((exp_dir / "test_per_image_metrics.json").read_text())
    assert per_image_json["metric_names"] == ["psnr", "ssim", "delta_e"]
    assert per_image_json["samples"][0]["file_name"] == "a0001"

    with open(exp_dir / "test_per_image_metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["file_name"] == "a0001"
    assert rows[1]["psnr"] == "22.0"


def test_collect_test_samples_uses_enabled_test_metric_gates() -> None:
    class DummyMetrics:
        def __call__(self, *args, **kwargs):
            return {
                "psnr": torch.tensor(21.0),
                "ssim": torch.tensor(0.81),
                "lpips": torch.tensor(0.12),
                "delta_e": torch.tensor(3.0),
                "nima": torch.tensor(5.0),
                "clip": torch.tensor(0.0),
                "vgg_perceptual": torch.tensor(0.0),
            }

    module = ColorEnhanceLitModule.__new__(ColorEnhanceLitModule)
    module._test_samples = []
    module.metrics = DummyMetrics()

    image = torch.rand(1, 3, 8, 10)
    gates = {
        "enable_lpips": True,
        "enable_delta_e": True,
        "enable_nima": False,
        "enable_clip": False,
        "enable_vgg_perceptual": False,
    }

    ColorEnhanceLitModule._collect_test_samples(
        module,
        image,
        image,
        image,
        torch.tensor([[8, 10]]),
        ["a0001"],
        gates,
    )

    sample = module._test_samples[0]
    assert sample["file_name"] == "a0001"
    assert sample["psnr"] == 21.0
    assert sample["ssim"] == pytest.approx(0.81)
    assert sample["lpips"] == pytest.approx(0.12)
    assert sample["delta_e"] == 3.0
    assert "nima" not in sample

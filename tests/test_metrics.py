import importlib

import pytest
import torch

from src.evaluation.metrics import MetricCollection, compute_delta_e


def test_compute_delta_e_identical_images_is_zero():
    # Identical images must produce ΔE ≈ 0.
    img = torch.rand(2, 3, 16, 16).clamp(0.0, 1.0)
    de = compute_delta_e(img, img)
    assert de.item() < 1e-3, f"Expected ΔE ≈ 0 for identical images, got {de.item():.6f}"


def test_compute_delta_e_different_images_is_positive():
    # Black vs white image should give a large ΔE.
    black = torch.zeros(1, 3, 8, 8)
    white = torch.ones(1, 3, 8, 8)
    de = compute_delta_e(black, white)
    assert de.item() > 50.0, f"Expected ΔE > 50 for black vs white, got {de.item():.2f}"


def test_compute_delta_e_output_shape_is_scalar():
    pred = torch.rand(4, 3, 32, 32)
    target = torch.rand(4, 3, 32, 32)
    de = compute_delta_e(pred, target)
    assert de.shape == torch.Size([]), "compute_delta_e must return a scalar tensor"


def test_compute_delta_e_pure_red_vs_pure_green():
    # Pure red (1,0,0) and pure green (0,1,0) are maximally different hues.
    red = torch.zeros(1, 3, 4, 4)
    red[:, 0] = 1.0
    green = torch.zeros(1, 3, 4, 4)
    green[:, 1] = 1.0
    de = compute_delta_e(red, green)
    assert de.item() > 30.0, f"Expected ΔE > 30 for red vs green, got {de.item():.2f}"


@pytest.mark.parametrize("batch_size", [1, 4])
def test_compute_delta_e_batched(batch_size):
    pred = torch.rand(batch_size, 3, 16, 16)
    target = torch.rand(batch_size, 3, 16, 16)
    de = compute_delta_e(pred, target)
    assert de.shape == torch.Size([])
    assert 0.0 <= de.item() < 200.0


def test_metric_collection_returns_delta_e_key_when_enabled():
    cfg = {"evaluation": {"enable_nima": False, "enable_lpips": False, "enable_clip": False, "enable_delta_e": True}}
    metrics = MetricCollection(cfg)
    pred = torch.rand(1, 3, 8, 8)
    target = torch.rand(1, 3, 8, 8)
    results = metrics(pred, target, enable_ssim=False, enable_lpips=False, enable_clip=False, enable_delta_e=True)
    assert "delta_e" in results
    assert results["delta_e"].item() > 0.0


def test_metric_collection_skips_delta_e_when_disabled():
    cfg = {"evaluation": {"enable_nima": False, "enable_lpips": False, "enable_clip": False}}
    metrics = MetricCollection(cfg)
    pred = torch.rand(1, 3, 8, 8)
    target = torch.rand(1, 3, 8, 8)
    results = metrics(pred, target, enable_ssim=False, enable_lpips=False, enable_clip=False, enable_delta_e=False)
    assert results["delta_e"].item() == 0.0


def test_metric_collection_does_not_import_clip_when_metric_disabled(monkeypatch):
    called = False
    real_import_module = importlib.import_module

    def tracking_import_module(name, package=None):
        nonlocal called
        if name == "src.evaluation.clip_score":
            called = True
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", tracking_import_module)

    metrics = MetricCollection(
        {"evaluation": {"enable_clip": True, "enable_nima": False, "enable_lpips": False}}
    )
    pred = torch.zeros(1, 3, 8, 8)
    target = torch.zeros_like(pred)

    results = metrics(
        pred,
        target,
        enable_ssim=False,
        enable_lpips=False,
        enable_clip=False,
    )

    assert called is False
    assert results["clip"].item() == 0.0


def test_metric_collection_raises_clear_error_when_clip_requested_without_dependency(
    monkeypatch,
):
    real_import_module = importlib.import_module

    def failing_import_module(name, package=None):
        if name == "src.evaluation.clip_score":
            raise ImportError("No module named 'clip'")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", failing_import_module)

    metrics = MetricCollection(
        {"evaluation": {"enable_clip": True, "enable_nima": False, "enable_lpips": False}}
    )
    pred = torch.zeros(1, 3, 8, 8)
    target = torch.zeros_like(pred)

    try:
        metrics(
            pred,
            target,
            enable_ssim=False,
            enable_lpips=False,
            enable_clip=True,
        )
    except ImportError as exc:
        assert "CLIP metric requested" in str(exc)
    else:
        raise AssertionError("Expected CLIP metric initialization to fail without dependency")

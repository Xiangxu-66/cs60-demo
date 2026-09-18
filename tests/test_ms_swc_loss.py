from __future__ import annotations

import torch

from src.losses.ms_swc_loss import MSSWCLoss, rgb_to_lab


def _seed() -> None:
    torch.manual_seed(0)


def test_rgb_to_lab_reference_values() -> None:
    # Pure black / pure white reference points
    black = torch.zeros(1, 3, 1, 1)
    white = torch.ones(1, 3, 1, 1)
    lab_black = rgb_to_lab(black)
    lab_white = rgb_to_lab(white)
    assert torch.allclose(lab_black.squeeze(), torch.zeros(3), atol=1e-3)
    # White in D65 sRGB → L≈100, a≈0, b≈0
    assert abs(lab_white[0, 0, 0, 0].item() - 100.0) < 1e-2
    assert abs(lab_white[0, 1, 0, 0].item()) < 1e-2
    assert abs(lab_white[0, 2, 0, 0].item()) < 1e-2


def test_identical_inputs_give_zero_loss() -> None:
    _seed()
    loss_fn = MSSWCLoss(weight=1.0, num_scales=3, num_projections=64)
    x = torch.rand(2, 3, 64, 64)
    loss = loss_fn(x, x)
    assert loss.item() == 0.0


def test_forward_returns_finite_scalar() -> None:
    _seed()
    loss_fn = MSSWCLoss(weight=1.0, num_scales=3, num_projections=64)
    pred = torch.rand(2, 3, 64, 64)
    target = torch.rand(2, 3, 64, 64)
    loss = loss_fn(pred, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()
    assert loss.item() >= 0.0


def test_gradient_flows_to_pred() -> None:
    _seed()
    loss_fn = MSSWCLoss(weight=1.0, num_scales=3, num_projections=64)
    pred = torch.rand(1, 3, 32, 32, requires_grad=True)
    target = torch.rand(1, 3, 32, 32)
    loss = loss_fn(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all().item()
    assert pred.grad.abs().sum().item() > 0.0


def test_shifted_target_has_larger_loss_than_perturbation() -> None:
    # Hue-shifted target (far in color space) should give larger SWD
    # than a small additive perturbation in the same spatial layout.
    _seed()
    loss_fn = MSSWCLoss(weight=1.0, num_scales=3, num_projections=256)
    pred = torch.rand(1, 3, 64, 64)
    near = (pred + 0.02 * torch.randn_like(pred)).clamp(0, 1)
    far = pred.flip(dims=(1,))  # channel-flipped: completely different hue
    assert loss_fn(pred, far).item() > loss_fn(pred, near).item()


def test_weight_scales_output() -> None:
    _seed()
    base = MSSWCLoss(weight=1.0, num_scales=3, num_projections=64)
    scaled = MSSWCLoss(weight=2.5, num_scales=3, num_projections=64)
    pred = torch.rand(1, 3, 32, 32)
    target = torch.rand(1, 3, 32, 32)
    # Same RNG path inside each call, so ratio should be exact
    torch.manual_seed(42)
    a = base(pred, target).item()
    torch.manual_seed(42)
    b = scaled(pred, target).item()
    assert abs(b - 2.5 * a) < 1e-5

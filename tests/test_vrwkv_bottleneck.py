from __future__ import annotations

import torch

from src.models.bottlenecks.vrwkv_bottleneck import (
    ChannelMix,
    SpatialMix,
    VRWKVBottleneck,
    directional_scan_1d,
)


def test_vrwkv_bottleneck_forward_and_backward() -> None:
    model = VRWKVBottleneck(
        input_dim=768,
        hidden_dim=256,
        num_layers=2,
        drop_rate=0.0,
    )
    inputs = torch.randn(2, 324, 768, requires_grad=True)  # 18x18 tokens

    outputs = model(inputs)
    loss = outputs.mean()
    loss.backward()

    assert tuple(outputs.shape) == (2, 324, 256)
    assert model.proj_in.weight.grad is not None
    assert model.proj_out.weight.grad is not None


def test_directional_scan_stays_finite_cpu() -> None:
    x = torch.randn(8, 32, 18) * 10.0
    decay = torch.full((1, 32, 1), 0.8)
    y = directional_scan_1d(x, decay)
    assert tuple(y.shape) == tuple(x.shape)
    assert torch.isfinite(y).all()


def test_spatial_and_channel_mix_stay_finite_cpu() -> None:
    x = torch.randn(2, 64, 18, 18)

    sm = SpatialMix(64)
    cm = ChannelMix(64)

    y1 = sm(x)
    y2 = cm(x)

    assert tuple(y1.shape) == tuple(x.shape)
    assert tuple(y2.shape) == tuple(x.shape)
    assert torch.isfinite(y1).all()
    assert torch.isfinite(y2).all()


def test_vrwkv_train_step_finite_cpu() -> None:
    torch.manual_seed(0)
    model = VRWKVBottleneck(
        input_dim=768,
        hidden_dim=128,
        num_layers=2,
        drop_rate=0.0,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for _ in range(3):
        x = torch.randn(2, 18 * 18, 768)
        y = model(x)
        assert torch.isfinite(y).all()
        loss = (y**2).mean()
        assert torch.isfinite(loss)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
        optim.step()


def test_vrwkv_train_step_finite_mps_if_available() -> None:
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return

    device = torch.device("mps")
    torch.manual_seed(0)
    model = VRWKVBottleneck(
        input_dim=768,
        hidden_dim=64,
        num_layers=1,
        drop_rate=0.0,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)

    x = torch.randn(1, 18 * 18, 768, device=device)
    y = model(x)
    assert torch.isfinite(y).all()
    loss = (y**2).mean()
    assert torch.isfinite(loss)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
    optim.step()

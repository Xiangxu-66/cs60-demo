from __future__ import annotations

import torch

from src.models.bottlenecks.versions.v11_late_mhsa import RWKVBottleneckV11LateMHSA
from src.models.bottlenecks.versions.v11_quad_rwkv import RWKVBottleneckV11


def test_rwkv_v11_late_mhsa_forward_and_backward() -> None:
    model = RWKVBottleneckV11LateMHSA(
        input_dim=768,
        hidden_dim=384,
        num_layers=3,
        num_late_attention_blocks=1,
        mhsa_num_heads=6,
        drop_rate=0.0,
    )
    inputs = torch.randn(2, 35, 768, requires_grad=True)

    outputs = model(inputs)
    loss = outputs.mean()
    loss.backward()

    assert tuple(outputs.shape) == (2, 35, 384)
    assert model.proj_in.weight.grad is not None
    assert model.proj_out.weight.grad is not None


def test_rwkv_v11_late_mhsa_adds_capacity_over_v11() -> None:
    baseline = RWKVBottleneckV11(
        input_dim=768,
        hidden_dim=384,
        num_layers=6,
        drop_rate=0.1,
    )
    late_mhsa = RWKVBottleneckV11LateMHSA(
        input_dim=768,
        hidden_dim=384,
        num_layers=6,
        num_late_attention_blocks=2,
        mhsa_num_heads=6,
        drop_rate=0.1,
    )

    baseline_params = sum(p.numel() for p in baseline.parameters())
    late_mhsa_params = sum(p.numel() for p in late_mhsa.parameters())

    assert late_mhsa_params > baseline_params

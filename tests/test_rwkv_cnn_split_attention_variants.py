from __future__ import annotations

import torch

from src.models.bottlenecks.versions.v12_split import RWKVCNNSplitBottleneckV12
from src.models.bottlenecks.versions.v12_split_axial import RWKVCNNSplitAxialBottleneckV12
from src.models.bottlenecks.versions.v12_split_late_mhsa import (
    RWKVCNNSplitLateMHSABottleneckV12,
)


def test_rwkv_cnn_split_axial_forward_and_backward() -> None:
    model = RWKVCNNSplitAxialBottleneckV12(
        input_dim=768,
        hidden_dim=384,
        num_layers=2,
        axial_num_heads=4,
        drop_rate=0.0,
    )
    inputs = torch.randn(2, 36, 768, requires_grad=True)

    outputs = model(inputs)
    loss = outputs.mean()
    loss.backward()

    assert tuple(outputs.shape) == (2, 36, 384)
    assert model.proj_in.weight.grad is not None
    assert model.proj_out.weight.grad is not None


def test_rwkv_cnn_split_late_mhsa_forward_and_backward() -> None:
    model = RWKVCNNSplitLateMHSABottleneckV12(
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


def test_attention_variants_add_capacity_over_v12_lite() -> None:
    baseline = RWKVCNNSplitBottleneckV12(
        input_dim=768,
        hidden_dim=384,
        num_layers=6,
        drop_rate=0.1,
    )
    axial = RWKVCNNSplitAxialBottleneckV12(
        input_dim=768,
        hidden_dim=384,
        num_layers=6,
        axial_num_heads=4,
        drop_rate=0.1,
    )
    late_mhsa = RWKVCNNSplitLateMHSABottleneckV12(
        input_dim=768,
        hidden_dim=384,
        num_layers=6,
        num_late_attention_blocks=2,
        mhsa_num_heads=6,
        drop_rate=0.1,
    )

    baseline_params = sum(p.numel() for p in baseline.parameters())
    axial_params = sum(p.numel() for p in axial.parameters())
    late_mhsa_params = sum(p.numel() for p in late_mhsa.parameters())

    assert axial_params > baseline_params
    assert late_mhsa_params > baseline_params

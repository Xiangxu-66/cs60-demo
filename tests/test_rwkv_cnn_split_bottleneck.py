from __future__ import annotations

import torch

from src.models.bottlenecks.rwkv_bottleneck import RWKVBottleneckV1
from src.models.bottlenecks.versions.v12_split import RWKVCNNSplitBottleneckV12


def test_rwkv_cnn_split_forward_and_backward() -> None:
    model = RWKVCNNSplitBottleneckV12(
        input_dim=768,
        hidden_dim=384,
        num_layers=2,
        drop_rate=0.0,
    )
    inputs = torch.randn(2, 37, 768, requires_grad=True)

    outputs = model(inputs)
    loss = outputs.mean()
    loss.backward()

    assert tuple(outputs.shape) == (2, 37, 384)
    assert model.proj_in.weight.grad is not None
    assert model.proj_out.weight.grad is not None


def test_rwkv_cnn_split_has_lower_param_count_than_rwkv_v1() -> None:
    split_model = RWKVCNNSplitBottleneckV12(
        input_dim=768,
        hidden_dim=384,
        num_layers=6,
        drop_rate=0.1,
    )
    rwkv_model = RWKVBottleneckV1(
        input_dim=768,
        hidden_dim=384,
        num_layers=6,
        drop_rate=0.1,
    )

    split_params = sum(p.numel() for p in split_model.parameters())
    rwkv_params = sum(p.numel() for p in rwkv_model.parameters())

    assert split_params < rwkv_params

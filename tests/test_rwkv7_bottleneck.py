from __future__ import annotations

import torch

from src.models.bottlenecks.rwkv7_bottleneck import RWKV7Bottleneck


def test_rwkv7_bottleneck_forward_and_backward_simple_shift() -> None:
    model = RWKV7Bottleneck(
        input_dim=64,
        hidden_dim=128,
        num_layers=2,
        head_size=32,
        ffn_ratio=2.0,
        decay_lora_dim=16,
        aaa_lora_dim=16,
        mv_lora_dim=8,
        gate_lora_dim=16,
        use_omni_shift=False,
        bidirectional=False,
    )
    inputs = torch.randn(2, 37, 64, requires_grad=True)

    outputs = model(inputs)
    loss = outputs.mean()
    loss.backward()

    assert tuple(outputs.shape) == (2, 37, 128)
    assert model.proj_in.weight.grad is not None
    assert model.proj_out.weight.grad is not None
    assert model.blocks[0].att.receptance.weight.grad is not None


def test_rwkv7_bottleneck_supports_2d_shift_and_bidirectional_scan() -> None:
    model = RWKV7Bottleneck(
        input_dim=48,
        hidden_dim=96,
        num_layers=1,
        head_size=32,
        ffn_ratio=2.0,
        decay_lora_dim=12,
        aaa_lora_dim=12,
        mv_lora_dim=6,
        gate_lora_dim=12,
        use_omni_shift=True,
        bidirectional=True,
        use_cuda_kernel=False,
    )
    inputs = torch.randn(2, 35, 48, requires_grad=True)

    outputs = model(inputs)
    loss = outputs.square().mean()
    loss.backward()

    assert tuple(outputs.shape) == (2, 35, 96)
    assert model.blocks[0].att.output.weight.grad is not None


def test_rwkv7_bottleneck_accepts_explicit_non_square_resolution() -> None:
    model = RWKV7Bottleneck(
        input_dim=48,
        hidden_dim=96,
        num_layers=1,
        head_size=32,
        ffn_ratio=2.0,
        decay_lora_dim=12,
        aaa_lora_dim=12,
        mv_lora_dim=6,
        gate_lora_dim=12,
        use_omni_shift=True,
        bidirectional=False,
        use_cuda_kernel=False,
    )
    inputs = torch.randn(2, 24, 48, requires_grad=True)

    outputs = model(inputs, resolution=(3, 8))
    loss = outputs.abs().mean()
    loss.backward()

    assert tuple(outputs.shape) == (2, 24, 96)
    assert model.blocks[0].att.output.weight.grad is not None

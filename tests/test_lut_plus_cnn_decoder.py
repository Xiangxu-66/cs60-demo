"""Smoke tests for LUTPlusCNNDecoder (PR3.2 of TokenLUT-BG plan)."""
from __future__ import annotations

import torch

from src.models.decoders.lut_plus_cnn_decoder import LUTPlusCNNDecoder


def test_predict_residual_flag():
    decoder = LUTPlusCNNDecoder(input_dim=384)
    assert decoder.predict_residual is False
    # Inner CNN must stay in residual mode — outer wrapper does the addition.
    assert decoder.cnn.predict_residual is True


def test_output_shape():
    decoder = LUTPlusCNNDecoder(input_dim=384)
    tokens = torch.randn(2, 900, 384)
    img = torch.rand(2, 3, 480, 480)
    out = decoder(tokens, h=30, w=30, img=img)

    assert out.shape == (2, 3, 480, 480)
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0


def test_initial_output_is_input_image():
    """Identity LUT + zero-init CNN head + zero-init weight MLP last layer
    means initial output should be exactly the input (within FP noise).
    """
    torch.manual_seed(42)
    decoder = LUTPlusCNNDecoder(
        input_dim=384,
        random_init_scale=0.0,                  # exact-identity LUT bases
    )
    decoder.eval()

    tokens = torch.randn(2, 900, 384)
    img = torch.rand(2, 3, 256, 256)

    with torch.no_grad():
        out = decoder(tokens, h=16, w=16, img=img)

    assert torch.allclose(out, img, atol=1e-4), (
        f"max abs diff at init: {(out - img).abs().max().item():.6f}"
    )


def test_gradient_flows_to_both_branches():
    decoder = LUTPlusCNNDecoder(
        input_dim=384,
        num_basis=3,
        random_init_scale=0.05,
    )
    tokens = torch.randn(2, 100, 384, requires_grad=True)
    img = torch.rand(2, 3, 96, 96)

    out = decoder(tokens, h=10, w=10, img=img)
    out.sum().backward()

    # LUT branch
    assert decoder.basis_lut.grad is not None
    assert decoder.basis_lut.grad.abs().sum().item() > 0
    assert decoder.weight_mlp[-1].weight.grad.abs().sum().item() > 0

    # CNN branch — pick the first conv after the token projection
    cnn_first_conv = decoder.cnn.upsample_blocks[0].conv
    assert cnn_first_conv.weight.grad is not None
    assert cnn_first_conv.weight.grad.abs().sum().item() > 0


def test_with_photometric_token():
    decoder = LUTPlusCNNDecoder(
        input_dim=384,
        use_photometric_token=True,
        random_init_scale=0.15,
    )
    tokens = torch.randn(2, 900, 384)
    img = torch.rand(2, 3, 480, 480)
    out = decoder(tokens, h=30, w=30, img=img)
    assert out.shape == (2, 3, 480, 480)


def test_residual_scale_caps_delta():
    """Even with arbitrary tokens, the CNN delta must respect residual_scale.

    We can't observe delta directly through the wrapper, but we can
    verify that the inner CNN's residual_scale is set correctly.
    """
    decoder = LUTPlusCNNDecoder(input_dim=384, cnn_residual_scale=0.3)
    assert decoder.cnn.residual_scale == 0.3

    decoder = LUTPlusCNNDecoder(input_dim=384, cnn_residual_scale=0.5)
    assert decoder.cnn.residual_scale == 0.5

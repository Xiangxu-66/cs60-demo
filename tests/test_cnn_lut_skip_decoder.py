from __future__ import annotations

import torch

from src.models.decoders.cnn_lut_skip_decoder import CNNLUTSkipDecoder


def test_lut_skip_decoder_output_shape() -> None:
    decoder = CNNLUTSkipDecoder(
        input_dim=384,
        patch_size=16,
        num_upsample_blocks=4,
        base_channels=32,
        skip_enabled=False,
        img_skip_enabled=True,
    )
    tokens = torch.randn(1, 36, 384)
    img = torch.rand(1, 3, 96, 96)

    output = decoder(tokens, h=6, w=6, img=img)

    assert output.shape == img.shape
    assert torch.isfinite(output).all()


def test_lut_skip_blend_zero_uses_raw_image_skip() -> None:
    decoder = CNNLUTSkipDecoder(
        input_dim=384,
        patch_size=16,
        num_upsample_blocks=1,
        base_channels=16,
        skip_enabled=False,
        lut_skip_blend=0.0,
    )
    tokens = torch.randn(1, 4, 384)
    img = torch.rand(1, 3, 32, 32)

    skip_img = decoder._make_skip_image(tokens, img)

    assert torch.equal(skip_img, img)


def test_lut_skip_branch_receives_gradient() -> None:
    decoder = CNNLUTSkipDecoder(
        input_dim=384,
        patch_size=16,
        num_upsample_blocks=1,
        base_channels=16,
        skip_enabled=False,
        lut_skip_blend=1.0,
    )
    # The decoder head is zero-initialized for identity starts. Make it nonzero
    # here so this unit test can observe gradient through the LUT skip branch.
    torch.nn.init.normal_(decoder.head[-1].weight, mean=0.0, std=0.01)

    tokens = torch.randn(1, 4, 384, requires_grad=True)
    img = torch.rand(1, 3, 32, 32)

    decoder(tokens, h=2, w=2, img=img).sum().backward()

    assert decoder.global_lut_branch.base_luts.grad is not None
    assert decoder.global_lut_branch.base_luts.grad.abs().sum().item() > 0

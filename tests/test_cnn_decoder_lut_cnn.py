from __future__ import annotations

import pytest
import torch

from src.models.decoders.cnn_decoder_lut_cnn import (
    CNNDecoderLUTCNN,
    SpatialBranchWithCNN,
)
from src.models.decoders.cnn_decoder_lut_cnn_legacy import CNNDecoderLUTCNNLegacy
from src.models.encoders.shallow_cnn import ShallowCNNEncoder


def test_shallow_cnn_encoder_outputs_expected_pyramid() -> None:
    encoder = ShallowCNNEncoder(output_channels=64)
    x = torch.rand(1, 3, 96, 96)

    features = encoder(x)
    sizes = [feat.shape[-2:] for feat in features]

    assert sizes == [(48, 48), (24, 24), (12, 12), (12, 12)]


def test_spatial_branch_expected_sizes_for_480_input() -> None:
    assert SpatialBranchWithCNN._expected_feature_sizes((480, 480)) == [
        (240, 240),
        (120, 120),
        (60, 60),
        (60, 60),
    ]


def test_spatial_branch_restores_resolution_coarse_to_fine() -> None:
    branch = SpatialBranchWithCNN(
        cnn_feature_channels=[64, 64, 64, 64],
        base_channels=64,
        num_upsample_blocks=4,
    )
    cnn_features = [
        torch.randn(1, 64, 48, 48),
        torch.randn(1, 64, 24, 24),
        torch.randn(1, 64, 12, 12),
        torch.randn(1, 64, 12, 12),
    ]

    seen_sizes: list[tuple[int, int]] = []
    detail_sizes: list[tuple[int, int]] = []

    def _record_size(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        seen_sizes.append(tuple(output.shape[-2:]))

    def _record_detail_size(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        detail_sizes.append(tuple(output.shape[-2:]))

    hooks = [block.register_forward_hook(_record_size) for block in branch.upsample_blocks]
    hooks.extend([
        branch.halfres_detail_fusion.register_forward_hook(_record_detail_size),
        branch.output_detail_fusion.register_forward_hook(_record_detail_size),
    ])
    try:
        weight_map, residual = branch(cnn_features, target_size=(96, 96))
    finally:
        for hook in hooks:
            hook.remove()

    assert seen_sizes == [(24, 24), (48, 48), (96, 96)]
    assert detail_sizes == [(48, 48), (96, 96)]
    assert weight_map.shape == (1, 1, 96, 96)
    assert residual.shape == (1, 3, 96, 96)


def test_spatial_branch_rejects_misaligned_feature_shapes() -> None:
    branch = SpatialBranchWithCNN(
        cnn_feature_channels=[64, 64, 64, 64],
        base_channels=64,
        num_upsample_blocks=4,
    )
    bad_features = [
        torch.randn(1, 64, 12, 12),
        torch.randn(1, 64, 24, 24),
        torch.randn(1, 64, 48, 48),
        torch.randn(1, 64, 12, 12),
    ]

    with pytest.raises(ValueError, match="size mismatch"):
        branch(bad_features, target_size=(96, 96))


def test_cnn_decoder_lut_cnn_output_shape_with_real_cnn_features() -> None:
    encoder = ShallowCNNEncoder(output_channels=64)
    decoder = CNNDecoderLUTCNN(
        input_dim=384,
        cnn_feature_channels=[64, 64, 64, 64],
        patch_size=16,
        num_upsample_blocks=4,
    )
    img = torch.rand(1, 3, 96, 96)
    tokens = torch.randn(1, 36, 384)

    cnn_features = encoder(img)
    output = decoder(tokens, h=6, w=6, img=img, cnn_features=cnn_features)

    assert output.shape == img.shape
    assert torch.isfinite(output).all()


def test_cnn_decoder_lut_cnn_legacy_output_shape_with_real_cnn_features() -> None:
    encoder = ShallowCNNEncoder(output_channels=64)
    decoder = CNNDecoderLUTCNNLegacy(
        input_dim=384,
        cnn_feature_channels=[64, 64, 64, 64],
        patch_size=16,
        num_upsample_blocks=1,
    )
    img = torch.rand(1, 3, 96, 96)
    tokens = torch.randn(1, 36, 384)

    cnn_features = encoder(img)
    output = decoder(tokens, h=6, w=6, img=img, cnn_features=cnn_features)

    assert output.shape == img.shape
    assert torch.isfinite(output).all()

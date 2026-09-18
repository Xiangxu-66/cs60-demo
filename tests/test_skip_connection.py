"""Test skip connection functionality for encoder and decoder."""
from __future__ import annotations

import importlib.util

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from src.models.bottlenecks.base import BaseBottleneck
from src.models.decoders.cnn_skip_decoder import CNNSkipDecoder, SimpleUpsampleBlock
from src.models.encoders.v2_dinov3 import DINOv3LoRAEncoder, ColourContextBranch, GhostDWBlock
from src.models.pipeline import ImageEnhancementPipeline

ENCODER_DEPS_AVAILABLE = all(
    importlib.util.find_spec(pkg) is not None
    for pkg in ("transformers", "peft")
)
requires_encoder_deps = pytest.mark.skipif(
    not ENCODER_DEPS_AVAILABLE,
    reason="encoder tests require `transformers` and `peft`",
)


class DummyBottleneck(BaseBottleneck):
    """Simple bottleneck for testing."""
    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self._output_dim = dim

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.proj(x)

    @property
    def output_dim(self) -> int:
        return self._output_dim


def test_ghost_dw_block() -> None:
    """Test GhostDWBlock output shape."""
    block = GhostDWBlock(in_channels=3, out_channels=64)
    x = torch.randn(2, 3, 32, 32)
    out = block(x)
    assert out.shape == (2, 64, 32, 32)
    print("✓ GhostDWBlock test passed")


def test_colour_context_branch() -> None:
    """Test ColourContextBranch with intermediate outputs."""
    branch = ColourContextBranch(embed_dim=768, return_intermediate=True)
    x = torch.randn(2, 6, 16, 16)
    out, intermediates = branch(x)

    assert out.shape == (2, 768, 16, 16)
    assert "layer0" in intermediates
    assert "layer1" in intermediates
    assert intermediates["layer0"].shape == (2, 32, 16, 16)
    assert intermediates["layer1"].shape == (2, 64, 16, 16)
    print("✓ ColourContextBranch test passed")


def test_colour_context_branch_no_intermediate() -> None:
    """Test ColourContextBranch without intermediate outputs."""
    branch = ColourContextBranch(embed_dim=768, return_intermediate=False)
    x = torch.randn(2, 6, 16, 16)
    out = branch(x)

    assert out.shape == (2, 768, 16, 16)
    print("✓ ColourContextBranch (no intermediate) test passed")


@requires_encoder_deps
def test_encoder_branch_combinations() -> None:
    """Test different encoder branch configurations."""
    img = torch.rand(1, 3, 224, 224)

    # Test: DINO only
    encoder = DINOv3LoRAEncoder(
        enable_dino=True,
        enable_local=False,
        enable_colour=False,
        skip_enabled=False,
    )
    tokens = encoder(img)
    assert tokens.shape == (1, 196, 768)  # 14x14 = 196
    print("✓ Encoder (DINO only) test passed")

    # Test: Local only (no DINO backbone)
    encoder = DINOv3LoRAEncoder(
        enable_dino=False,
        enable_local=True,
        enable_colour=False,
        skip_enabled=False,
    )
    tokens = encoder(img)
    assert tokens.shape == (1, 196, 768)
    print("✓ Encoder (Local only) test passed")

    # Test: Colour only
    encoder = DINOv3LoRAEncoder(
        enable_dino=False,
        enable_local=False,
        enable_colour=True,
        skip_enabled=False,
    )
    tokens = encoder(img)
    assert tokens.shape == (1, 196, 768)
    print("✓ Encoder (Colour only) test passed")

    # Test: All branches
    encoder = DINOv3LoRAEncoder(
        enable_dino=True,
        enable_local=True,
        enable_colour=True,
        skip_enabled=False,
    )
    tokens = encoder(img)
    assert tokens.shape == (1, 196, 768)
    print("✓ Encoder (All branches) test passed")


@requires_encoder_deps
def test_encoder_skip_features() -> None:
    """Test encoder skip feature output."""
    img = torch.rand(1, 3, 224, 224)

    encoder = DINOv3LoRAEncoder(
        enable_dino=True,
        enable_local=True,
        enable_colour=True,
        skip_enabled=True,
        return_skip=True,
    )

    tokens, skip_dict = encoder(img)

    assert tokens.shape == (1, 196, 768)
    assert "local" in skip_dict
    assert "colour" in skip_dict
    assert "raw_input" in skip_dict

    # Check local skip features
    assert "layer0" in skip_dict["local"]
    assert "layer1" in skip_dict["local"]
    assert skip_dict["local"]["layer0"].shape == (1, 64, 14, 14)
    assert skip_dict["local"]["layer1"].shape == (1, 128, 14, 14)

    # Check colour skip features
    assert "layer0" in skip_dict["colour"]
    assert "layer1" in skip_dict["colour"]
    assert skip_dict["colour"]["layer0"].shape == (1, 32, 14, 14)
    assert skip_dict["colour"]["layer1"].shape == (1, 64, 14, 14)

    # Check raw input skip features
    assert "local" in skip_dict["raw_input"]
    assert "colour" in skip_dict["raw_input"]
    assert skip_dict["raw_input"]["local"].shape == (1, 3, 14, 14)
    assert skip_dict["raw_input"]["colour"].shape == (1, 6, 14, 14)

    print("✓ Encoder skip features test passed")


@requires_encoder_deps
def test_encoder_skip_disabled() -> None:
    """Test encoder with skip disabled."""
    img = torch.rand(1, 3, 224, 224)

    encoder = DINOv3LoRAEncoder(
        skip_enabled=False,
        return_skip=False,
    )

    output = encoder(img)

    # Should return only tokens, not tuple
    assert isinstance(output, Tensor)
    assert output.shape == (1, 196, 768)
    print("✓ Encoder (skip disabled) test passed")


def test_simple_upsample_block() -> None:
    """Test SimpleUpsampleBlock without image skip."""
    block = SimpleUpsampleBlock(in_channels=64, out_channels=32)
    x = torch.randn(1, 64, 16, 16)
    out = block(x)

    # 2x upsampling
    assert out.shape == (1, 32, 32, 32)
    print("✓ SimpleUpsampleBlock test passed")


def test_cnn_skip_decoder_shapes() -> None:
    """Test CNNSkipDecoder output shapes."""
    decoder = CNNSkipDecoder(
        input_dim=768,
        patch_size=16,
        num_upsample_blocks=4,
        base_channels=64,
        skip_enabled=True,
        img_skip_enabled=True,
        skip_fusion_mode="gate",
        skip_stages=[1, 2, 3],
    )

    tokens = torch.randn(1, 196, 768)  # 14x14
    img = torch.rand(1, 3, 224, 224)

    # Create mock skip features
    skip_dict = {
        "local": {
            "layer0": torch.randn(1, 64, 14, 14),
            "layer1": torch.randn(1, 128, 14, 14),
        },
        "colour": {
            "layer0": torch.randn(1, 32, 14, 14),
            "layer1": torch.randn(1, 64, 14, 14),
        },
        "raw_input": {
            "local": torch.randn(1, 3, 14, 14),
            "colour": torch.randn(1, 6, 14, 14),
        }
    }

    output = decoder(tokens, h=14, w=14, img=img, skip_dict=skip_dict)

    assert output.shape == (1, 3, 224, 224)
    print("✓ CNNSkipDecoder (full) test passed")


def test_cnn_skip_decoder_no_img_skip() -> None:
    """Test CNNSkipDecoder with img_skip disabled."""
    decoder = CNNSkipDecoder(
        input_dim=768,
        patch_size=16,
        skip_enabled=True,
        img_skip_enabled=False,
    )

    tokens = torch.randn(1, 196, 768)
    img = torch.rand(1, 3, 224, 224)
    skip_dict = {
        "local": {"layer0": torch.randn(1, 64, 14, 14)},
        "colour": {"layer0": torch.randn(1, 32, 14, 14)},
        "raw_input": {
            "local": torch.randn(1, 3, 14, 14),
            "colour": torch.randn(1, 6, 14, 14),
        }
    }

    output = decoder(tokens, h=14, w=14, img=img, skip_dict=skip_dict)
    assert output.shape == (1, 3, 224, 224)
    print("✓ CNNSkipDecoder (no img skip) test passed")


def test_cnn_skip_decoder_no_skip() -> None:
    """Test CNNSkipDecoder with all skip disabled."""
    decoder = CNNSkipDecoder(
        input_dim=768,
        patch_size=16,
        skip_enabled=False,
        img_skip_enabled=True,
    )

    tokens = torch.randn(1, 196, 768)
    img = torch.rand(1, 3, 224, 224)

    output = decoder(tokens, h=14, w=14, img=img, skip_dict=None)
    assert output.shape == (1, 3, 224, 224)
    print("✓ CNNSkipDecoder (no skip) test passed")


def test_cnn_skip_decoder_skip_stages() -> None:
    """Test CNNSkipDecoder with selective skip stages."""
    decoder = CNNSkipDecoder(
        input_dim=768,
        patch_size=16,
        skip_enabled=True,
        skip_stages=[2, 3],  # Only stages 2 and 3
        img_skip_enabled=True,
    )

    tokens = torch.randn(1, 196, 768)
    img = torch.rand(1, 3, 224, 224)
    skip_dict = {
        "local": {
            "layer0": torch.randn(1, 64, 14, 14),
            "layer1": torch.randn(1, 128, 14, 14),
        },
        "colour": {
            "layer0": torch.randn(1, 32, 14, 14),
            "layer1": torch.randn(1, 64, 14, 14),
        },
        "raw_input": {
            "local": torch.randn(1, 3, 14, 14),
            "colour": torch.randn(1, 6, 14, 14),
        }
    }

    output = decoder(tokens, h=14, w=14, img=img, skip_dict=skip_dict)
    assert output.shape == (1, 3, 224, 224)
    print("✓ CNNSkipDecoder (selective stages) test passed")


@requires_encoder_deps
def test_pipeline_with_skip() -> None:
    """Test full pipeline with skip connections."""
    encoder = DINOv3LoRAEncoder(
        enable_dino=True,
        enable_local=True,
        enable_colour=True,
        skip_enabled=True,
    )
    bottleneck = DummyBottleneck(dim=768)
    decoder = CNNSkipDecoder(
        input_dim=768,
        skip_enabled=True,
        img_skip_enabled=True,
    )

    pipeline = ImageEnhancementPipeline(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
    )

    img = torch.rand(1, 3, 224, 224)
    output = pipeline(img)

    assert output.shape == img.shape
    assert output.min() >= 0.0 and output.max() <= 1.0
    print("✓ Pipeline (with skip) test passed")


@requires_encoder_deps
def test_pipeline_no_skip() -> None:
    """Test full pipeline without skip connections."""
    encoder = DINOv3LoRAEncoder(
        enable_dino=True,
        enable_local=False,
        enable_colour=False,
        skip_enabled=False,
    )
    bottleneck = DummyBottleneck(dim=768)
    decoder = CNNSkipDecoder(
        input_dim=768,
        skip_enabled=False,
        img_skip_enabled=True,
    )

    pipeline = ImageEnhancementPipeline(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
    )

    img = torch.rand(1, 3, 224, 224)
    output = pipeline(img)

    assert output.shape == img.shape
    assert output.min() >= 0.0 and output.max() <= 1.0
    print("✓ Pipeline (no skip) test passed")


def test_all_fusion_modes() -> None:
    """Test all decoder skip fusion modes."""
    tokens = torch.randn(1, 196, 768)
    img = torch.rand(1, 3, 224, 224)
    skip_dict = {
        "local": {
            "layer0": torch.randn(1, 64, 14, 14),
            "layer1": torch.randn(1, 128, 14, 14),
        },
        "colour": {
            "layer0": torch.randn(1, 32, 14, 14),
            "layer1": torch.randn(1, 64, 14, 14),
        },
        "raw_input": {
            "local": torch.randn(1, 3, 14, 14),
            "colour": torch.randn(1, 6, 14, 14),
        }
    }

    for mode in ["concat", "add", "gate", "gated_residual", "attention_gate"]:
        decoder = CNNSkipDecoder(
            input_dim=768,
            skip_enabled=True,
            skip_fusion_mode=mode,
            img_skip_enabled=False,
        )
        output = decoder(tokens, h=14, w=14, img=img, skip_dict=skip_dict)
        assert output.shape == (1, 3, 224, 224)
        print(f"✓ CNNSkipDecoder (mode={mode}) test passed")


@requires_encoder_deps
def test_encoder_fusion_modes() -> None:
    """Test encoder fusion modes."""
    img = torch.rand(1, 3, 224, 224)

    for mode in ["concat", "add", "gate"]:
        encoder = DINOv3LoRAEncoder(
            enable_dino=True,
            enable_local=True,
            enable_colour=True,
            fusion_mode=mode,
            skip_enabled=False,
        )
        tokens = encoder(img)
        assert tokens.shape == (1, 196, 768)
        print(f"✓ Encoder (fusion_mode={mode}) test passed")

    # Test new fusion modes
    for mode in ["gated_residual", "attention_gate"]:
        encoder = DINOv3LoRAEncoder(
            enable_dino=True,
            enable_local=True,
            enable_colour=True,
            fusion_mode=mode,
            skip_enabled=False,
        )
        tokens = encoder(img)
        assert tokens.shape == (1, 196, 768)
        print(f"✓ Encoder (fusion_mode={mode}) test passed")


@requires_encoder_deps
def test_encoder_new_fusion_with_skip() -> None:
    """Test encoder with new fusion modes and skip enabled."""
    img = torch.rand(1, 3, 224, 224)

    for mode in ["gated_residual", "attention_gate"]:
        encoder = DINOv3LoRAEncoder(
            enable_dino=True,
            enable_local=True,
            enable_colour=True,
            fusion_mode=mode,
            skip_enabled=True,
        )
        tokens, skip_dict = encoder(img)
        assert tokens.shape == (1, 196, 768)
        assert "local" in skip_dict
        assert "colour" in skip_dict
        print(f"✓ Encoder with skip (fusion_mode={mode}) test passed")


@requires_encoder_deps
def test_pipeline_with_new_fusion_modes() -> None:
    """Test full pipeline with new fusion modes."""
    img = torch.rand(1, 3, 224, 224)

    for mode in ["gated_residual", "attention_gate"]:
        encoder = DINOv3LoRAEncoder(
            enable_dino=True,
            enable_local=True,
            enable_colour=True,
            fusion_mode=mode,
            skip_enabled=True,
        )
        bottleneck = DummyBottleneck(dim=768)
        decoder = CNNSkipDecoder(
            input_dim=768,
            skip_enabled=True,
            skip_fusion_mode=mode,
            img_skip_enabled=True,
        )

        pipeline = ImageEnhancementPipeline(
            encoder=encoder,
            bottleneck=bottleneck,
            decoder=decoder,
        )

        output = pipeline(img)
        assert output.shape == img.shape
        assert output.min() >= 0.0 and output.max() <= 1.0
        print(f"✓ Pipeline (fusion_mode={mode}) test passed")


if __name__ == "__main__":
    print("Running skip connection tests...\n")

    test_ghost_dw_block()
    test_colour_context_branch()
    test_colour_context_branch_no_intermediate()
    test_encoder_branch_combinations()
    test_encoder_skip_features()
    test_encoder_skip_disabled()
    test_simple_upsample_block()
    test_cnn_skip_decoder_shapes()
    test_cnn_skip_decoder_no_img_skip()
    test_cnn_skip_decoder_no_skip()
    test_cnn_skip_decoder_skip_stages()
    test_pipeline_with_skip()
    test_pipeline_no_skip()
    test_all_fusion_modes()
    test_encoder_fusion_modes()
    test_encoder_new_fusion_with_skip()
    test_pipeline_with_new_fusion_modes()

    print("\n✓ All tests passed!")

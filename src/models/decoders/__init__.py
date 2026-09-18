__all__ = [
    "BaseDecoder",
    "CNNDecoder",
    "CNNDecoderV1",
    "CNNDecoderWithViTSkip",
    "CNNDecoderWithGlobal",
    "CNNDecoderWithPyramid",
    "CNNDecoderWithOSRM",
    "CNNDecoderWithSkip",
    "CNNDecoderWithDualBranch",
    "GatedResidualDecoder",
    "CNNDecoderLUT",
    "CNNDecoderLUTCNN",
    "CNNDecoderLUTCNNLegacy",
    "LUTPlusCNNDecoder",
]


def __getattr__(name: str):
    if name == "BaseDecoder":
        from src.models.decoders.base import BaseDecoder

        return BaseDecoder
    if name == "CNNDecoder":
        from src.models.decoders.cnn_decoder import CNNDecoder

        return CNNDecoder
    if name == "CNNDecoderV1":
        from src.models.decoders.cnn_decoder_v1 import CNNDecoderV1

        return CNNDecoderV1
    if name == "CNNDecoderWithViTSkip":
        from src.models.decoders.cnn_decoder_with_vit_skip import CNNDecoderWithViTSkip

        return CNNDecoderWithViTSkip
    if name == "CNNDecoderWithGlobal":
        from src.models.decoders.cnn_decoder_with_global import CNNDecoderWithGlobal

        return CNNDecoderWithGlobal
    if name == "CNNDecoderWithPyramid":
        from src.models.decoders.cnn_decoder_with_pyramid import CNNDecoderWithPyramid

        return CNNDecoderWithPyramid
    if name == "CNNDecoderWithOSRM":
        from src.models.decoders.cnn_decoder_with_osrm import CNNDecoderWithOSRM

        return CNNDecoderWithOSRM
    if name == "CNNDecoderWithSkip":
        from src.models.decoders.cnn_decoder_with_skip import CNNDecoderWithSkip

        return CNNDecoderWithSkip
    if name == "GatedResidualDecoder":
        from src.models.decoders.cnn_decoder_gated import GatedResidualDecoder

        return GatedResidualDecoder
    if name == "CNNDecoderWithDualBranch":
        from src.models.decoders.cnn_decoder_dual_branch import CNNDecoderWithDualBranch

        return CNNDecoderWithDualBranch
    if name == "CNNDecoderLUT":
        from src.models.decoders.cnn_decoder_lut import CNNDecoderLUT

        return CNNDecoderLUT
    if name == "CNNDecoderLUTCNN":
        from src.models.decoders.cnn_decoder_lut_cnn import CNNDecoderLUTCNN

        return CNNDecoderLUTCNN
    if name == "CNNDecoderLUTCNNLegacy":
        from src.models.decoders.cnn_decoder_lut_cnn_legacy import CNNDecoderLUTCNNLegacy

        return CNNDecoderLUTCNNLegacy
    if name == "LUTPlusCNNDecoder":
        from src.models.decoders.lut_plus_cnn_decoder import LUTPlusCNNDecoder

        return LUTPlusCNNDecoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

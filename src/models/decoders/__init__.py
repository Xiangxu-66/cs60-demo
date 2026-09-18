__all__ = ["BaseDecoder", "CNNDecoder"]


def __getattr__(name: str):
    if name == "BaseDecoder":
        from src.models.decoders.base import BaseDecoder

        return BaseDecoder
    if name == "CNNDecoder":
        from src.models.decoders.cnn_decoder import CNNDecoder

        return CNNDecoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

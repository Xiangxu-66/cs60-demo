__all__ = ["BaseEncoder", "DINOv2Encoder"]


def __getattr__(name: str):
    if name == "BaseEncoder":
        from src.models.encoders.base import BaseEncoder

        return BaseEncoder
    if name == "DINOv2Encoder":
        from src.models.encoders.dinov2 import DINOv2Encoder

        return DINOv2Encoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["BaseEncoder", "DINOv2Encoder", "DINOv3Encoder", "VRWKVEncoder", "DINOv3LoRAEncoder", "ShallowCNNEncoder", "ColorNAFEncoder"]


def __getattr__(name: str):
    if name == "BaseEncoder":
        from src.models.encoders.base import BaseEncoder
        return BaseEncoder
    if name == "DINOv2Encoder":
        from src.models.encoders.dinov2 import DINOv2Encoder
        return DINOv2Encoder
    if name == "DINOv3Encoder":
        from src.models.encoders.dinov3 import DINOv3Encoder
        return DINOv3Encoder
    if name == "DINOv3LoRAEncoder":
        from src.models.encoders.v2_dinov3 import DINOv3LoRAEncoder
        return DINOv3LoRAEncoder
    if name == "VRWKVEncoder":
        from src.models.encoders.vrwkv_encoder import VRWKVEncoder
        return VRWKVEncoder
    if name == "ShallowCNNEncoder":
        from src.models.encoders.shallow_cnn import ShallowCNNEncoder
        return ShallowCNNEncoder
    if name == "ColorNAFEncoder":
        from src.models.encoders.color_naf import ColorNAFEncoder
        return ColorNAFEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
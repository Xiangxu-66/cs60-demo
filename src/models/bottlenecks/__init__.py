__all__ = [
    "BaseBottleneck",
    "NoneBottleneck",
    "CNNBottleneck",
    "RWKVBottleneckV1",
    "RWKV7Bottleneck",
    "TransformerBottleneck",
    "VMambaBottleneck",
    "DSSMBottleneck",
]


def __getattr__(name: str):
    if name == "BaseBottleneck":
        from src.models.bottlenecks.base import BaseBottleneck

        return BaseBottleneck
    if name == "NoneBottleneck":
        from src.models.bottlenecks.none_bottleneck import NoneBottleneck

        return NoneBottleneck
    if name == "CNNBottleneck":
        from src.models.bottlenecks.cnn_bottleneck import CNNBottleneck

        return CNNBottleneck
    if name == "RWKVBottleneckV1":
        from src.models.bottlenecks.rwkv_bottleneck import RWKVBottleneckV1

        return RWKVBottleneckV1
    if name == "RWKV7Bottleneck":
        from src.models.bottlenecks.rwkv7_bottleneck import RWKV7Bottleneck

        return RWKV7Bottleneck
    if name == "TransformerBottleneck":
        from src.models.bottlenecks.transformer_bottleneck import TransformerBottleneck

        return TransformerBottleneck
    if name == "VMambaBottleneck":
        from src.models.bottlenecks.vmamba_bottleneck import VMambaBottleneck

        return VMambaBottleneck
    if name == "DSSMBottleneck":
        from src.models.bottlenecks.dssm_bottleneck import DSSMBottleneck

        return DSSMBottleneck
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

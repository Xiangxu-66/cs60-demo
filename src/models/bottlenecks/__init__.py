__all__ = ["BaseBottleneck", "NoneBottleneck", "CNNBottleneck", "RWKVBottleneckV1"]


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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

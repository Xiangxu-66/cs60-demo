__all__ = [
    "BaseBottleneck",
    "NoneBottleneck",
    "CNNBottleneck",
    "RWKVBottleneckV1",
    "RWKVBottleneckV11",
    "RWKVBottleneckV11LateMHSA",
    "RWKVCNNSplitBottleneckV12",
    "RWKVCNNSplitAxialBottleneckV12",
    "RWKVCNNSplitLateMHSABottleneckV12",
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
    if name == "RWKVBottleneckV11":
        from src.models.bottlenecks.versions.v11_quad_rwkv import RWKVBottleneckV11

        return RWKVBottleneckV11
    if name == "RWKVBottleneckV11LateMHSA":
        from src.models.bottlenecks.versions.v11_late_mhsa import RWKVBottleneckV11LateMHSA

        return RWKVBottleneckV11LateMHSA
    if name == "RWKVCNNSplitBottleneckV12":
        from src.models.bottlenecks.versions.v12_split import RWKVCNNSplitBottleneckV12

        return RWKVCNNSplitBottleneckV12
    if name == "RWKVCNNSplitAxialBottleneckV12":
        from src.models.bottlenecks.versions.v12_split_axial import (
            RWKVCNNSplitAxialBottleneckV12,
        )

        return RWKVCNNSplitAxialBottleneckV12
    if name == "RWKVCNNSplitLateMHSABottleneckV12":
        from src.models.bottlenecks.versions.v12_split_late_mhsa import (
            RWKVCNNSplitLateMHSABottleneckV12,
        )

        return RWKVCNNSplitLateMHSABottleneckV12
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

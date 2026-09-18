"""V3: RWKV bottleneck + hierarchical dual-branch (fine + coarse)."""
from __future__ import annotations

# TODO: Implement RWKVBottleneckV3(RWKVBottleneckV2)
#
# Problem solved: Single scale cannot model multi-scale color perception
#   (global tone vs local color detail).
# Solution: Parallel fine-grained and coarse-grained branches fused via
#   content-adaptive gate.
# Source: HiRWKV (IEEE TIP)
#
# Architecture:
#   Fine branch  : V2 blocks on full-resolution token grid (h×w)
#   Coarse branch: Separate VRWKVBlocks on spatially pooled tokens
#                  adaptive_avg_pool2d(x_2d, (h//coarse_scale, w//coarse_scale))
#                  then bilinear upsample back to h×w
#   Fusion       : HierarchicalFusion(dim)
#                  gate = sigmoid(Linear(2C → C))
#                  out  = gate * fine + (1-gate) * coarse
#
# Note: both branches use separate block lists to allow independent learning.
#   Do NOT share weights between fine and coarse blocks.

from src.models.bottlenecks.versions.v2_hsv_scan import RWKVBottleneckV2
from torch import Tensor


class RWKVBottleneckV3(RWKVBottleneckV2):
    """V3: + Hierarchical dual-branch (fine + coarse).

    TODO: Full implementation pending (depends on V2).
    """

    def __init__(self, *, fine_scale: int = 1, coarse_scale: int = 4, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fine_scale = fine_scale
        self.coarse_scale = coarse_scale
        # TODO: self.coarse_blocks = nn.ModuleList([...])
        # TODO: self.coarse_norm  = nn.LayerNorm(dim)
        # TODO: self.fusion       = HierarchicalFusion(dim)

    def forward(self, x: Tensor) -> Tensor:
        # TODO: implement dual-branch forward + fusion
        raise NotImplementedError("RWKVBottleneckV3 not yet implemented.")

"""V2: RWKV bottleneck + color-aware HSV scan ordering."""
from __future__ import annotations

# TODO: Implement RWKVBottleneckV2(RWKVBottleneckV1)
#
# Problem solved: Raster scan breaks 2D color locality (adjacent pixels in
#   color space are not adjacent in raster order).
# Solution: Sort tokens by HSV similarity before RWKV recurrence.
# Source: DRWKV (WACV 2026)
#
# New module — HSVScanModule:
#   - Input: (B, N, C) tokens + optional (B, N, 3) HSV color hint
#   - Sort tokens by chosen HSV channel (hue/saturation/value)
#   - Return (sorted_tokens, inverse_indices) so order can be restored
#   - If no color_hint: use feature L2 norm as proxy sort criterion
#     TODO: Replace proxy with a learned linear projection → HSV space
#
# Changes to forward():
#   1. proj_in(x)
#   2. sorted_x, inv_idx = hsv_scan(x)
#   3. Apply VRWKVBlocks on sorted_x
#   4. norm + restore order: gather(out, inv_idx)
#   5. proj_out
#
# Sort adds O(N log N) overhead but recurrence remains O(N)

from src.models.bottlenecks.rwkv_bottleneck import RWKVBottleneckV1
from torch import Tensor


class RWKVBottleneckV2(RWKVBottleneckV1):
    """V2: + HSV color-aware token scan order.

    TODO: Full implementation pending (depends on V1 being implemented first).
    """

    def __init__(self, *, hsv_sort_key: str = "hue", **kwargs) -> None:
        super().__init__(**kwargs)
        self.hsv_sort_key = hsv_sort_key
        # TODO: self.hsv_scan = HSVScanModule(sort_key=hsv_sort_key)

    def forward(self, x: Tensor) -> Tensor:
        # TODO: implement HSV sort → V1 blocks → restore order
        raise NotImplementedError("RWKVBottleneckV2 not yet implemented.")

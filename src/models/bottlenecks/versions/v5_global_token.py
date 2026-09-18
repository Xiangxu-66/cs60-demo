"""V5: RWKV bottleneck + global color tokens."""
from __future__ import annotations

# TODO: Implement RWKVBottleneckV5(RWKVBottleneckV4)
#
# Problem solved: RWKV recurrence is purely local (each token only sees
#   past context). No mechanism for global color tone consistency.
# Solution: Prepend k learnable global tokens to the sequence before RWKV.
#   These tokens see and influence all spatial tokens via the recurrence.
#   After processing, global tokens are discarded.
# Source: ViT CLS Token concept, adapted for linear recurrence
#
# Key constraint: complexity must remain O(N).
#   Total tokens = N + k, so O(N + k) = O(N) since k is a small constant (e.g. 4).
#   Do NOT use full Cross-Attention between global and spatial tokens.
#
# Implementation:
#   self.global_tokens : nn.Parameter(torch.zeros(1, num_global_tokens, dim))
#                        init with trunc_normal_(std=0.02)
#   In forward():
#     global_exp = global_tokens.expand(B, -1, -1)       # (B, k, C)
#     seq = cat([global_exp, spatial_tokens], dim=1)      # (B, k+N, C)
#     seq = run VRWKVBlocks(seq)                          # (B, k+N, C)
#     spatial_out = seq[:, num_global_tokens:, :]         # (B, N, C)  discard globals
#   Continue with norm + proj_out on spatial_out.
#
# Note: QShift grid size should still be set to (h, w) for spatial tokens.
#   Global tokens at positions 0..k-1 will receive zero shift (padding effect) — acceptable.

from src.models.bottlenecks.versions.v4_cla import RWKVBottleneckV4
from torch import Tensor


class RWKVBottleneckV5(RWKVBottleneckV4):
    """V5: + Global color tokens (O(N) preserved).

    TODO: Full implementation pending (depends on V4).
    """

    def __init__(self, *, num_global_tokens: int = 4, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_global_tokens = num_global_tokens
        # TODO: self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, self._hidden_dim))
        #       nn.init.trunc_normal_(self.global_tokens, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        # TODO: prepend global tokens → V4 blocks → discard globals → proj_out
        raise NotImplementedError("RWKVBottleneckV5 not yet implemented.")

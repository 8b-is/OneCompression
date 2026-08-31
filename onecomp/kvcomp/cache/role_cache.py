"""RoleKV cache: live role-graded KV precision (no eviction).

The role-aware counterpart of :class:`~onecomp.kvcomp.cache.taper_cache.
TaperKVCache`: instead of grading precision by token *age*, each token's
bit-width is decided by the semantic *role* of the span it belongs to
(system / user / assistant / reasoning / tool), so verbose tool outputs can
be stored at 2 bits while the system prompt stays at 8.

All cache mechanics (HF ``DynamicCache`` integration, re-quantization
cadence, ``skip_layers`` handling, honest whole-cache bit accounting) are
inherited from :class:`~onecomp.kvcomp.cache._graded_base.GradedQuantCache`;
this class only supplies the bit schedule.  Length-preserving and RoPE-safe:
no token is ever dropped.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from logging import getLogger

from ..recipe.rolekv import role_token_bits, validate_spans
from ._graded_base import GradedQuantCache

logger = getLogger(__name__)


class RoleKVCache(GradedQuantCache):
    """Graded cache whose per-token bits follow semantic role spans.

    Args:
        policy: a :class:`~onecomp.kvcomp.recipe.rolekv.RolePrecisionPolicy`.
        spans: :class:`~onecomp.kvcomp.recipe.rolekv.RoleSpan` list tiling the
            annotated prompt prefix (e.g. from ``annotate_spans_from_chat``).
            Tokens generated past the last span are graded as ``assistant``.
        config: model config (preserves sliding-window layer typing).
        requant_stride: re-apply the schedule to the full cache every this
            many decode steps (1 = every step; larger = cheaper, staler).
    """

    def __init__(self, policy, spans, config=None, requant_stride: int = 16, **kwargs):
        super().__init__(
            config=config,
            skip_layers=policy.skip_layers,
            requant_stride=requant_stride,
            **kwargs,
        )
        self.policy = policy
        # Validate tiling against the annotated prefix length (the last
        # span's end); generation may extend the sequence beyond it.
        ordered = sorted(spans, key=lambda s: s.start)
        self.spans = validate_spans(ordered, ordered[-1].end if ordered else 0)

    def token_bits(self, total, device, layer_idx):
        bits = role_token_bits(self.spans, total, self.policy, device)
        # Roles apply the same bit-width to keys and values; the K/V
        # asymmetry lives in the quantization axis, not the bits.
        return bits, bits

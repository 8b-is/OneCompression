"""Static (fixed) compression policy.

The default policy: always apply the codec/selector bound in the
:class:`~onecomp.kvcomp.ComposedPress`.  Adaptive policies (ThinKV,
service-aware) override :meth:`plan`.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass

from .._base import KVPolicy
from .._registry import register_kv_policy
from ..config import Budget, CompressionPlan


@register_kv_policy("static")
@dataclass
class StaticPolicy(KVPolicy):
    """Return a fixed plan; the cache uses the composed codec/selector."""

    budget: Budget = None

    def plan(self, request_state=None, system_state=None):
        return CompressionPlan(budget=self.budget or Budget())

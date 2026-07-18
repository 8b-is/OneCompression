"""Formal-method identity carried by HayateQuant benchmark rows.

The codec recipe and the scientific role of a result are deliberately
separate.  :mod:`onecomp.kvcomp.codec.hayate.recipes` owns the former; this
module supplies the small, JSON-native provenance record that every completed
primary-quality runner must attach to its result rows.  The public family has
exactly four identities: HQ, HQ-Frontier, AdaptiveHQ, and RotationHQ.
"""

from __future__ import annotations

from typing import Any

from .recipes import (
    HQ_ADAPTIVE_NAME,
    HQ_FRONTIER_NAME,
    HQ_NAME,
    HQ_PUBLIC_METHODS,
    HQ_ROTATION_NAME,
)

# The public family and formal matrix are identical. Every recipe is
# data-free: RotationHQ is the fixed HQ payload with a deterministic dense
# rotation, while HQ-Frontier requires its own content-addressed codebook
# bundle across all three rates.
PAPER_HAYATE_METHODS = HQ_PUBLIC_METHODS
FORMAL_HAYATE_METHODS = HQ_PUBLIC_METHODS
LEGACY_HAYATE_RESULT_LABELS = ("HayateQuant-frontier",)


def hayate_formal_provenance(method: str) -> dict[str, Any]:
    """Return the exact proposal-role record for a new formal result."""

    if method == HQ_NAME:
        variant = "fixed_global"
        role = "paper_proposal_fixed"
        allocation = "globally_fixed_kv_code_bits"
    elif method == HQ_FRONTIER_NAME:
        variant = "integer_frontier"
        role = "paper_proposal_frontier"
        allocation = "integer_key_value_code_bits"
    elif method == HQ_ADAPTIVE_NAME:
        variant = "causal_adaptive"
        role = "paper_proposal_adaptive"
        allocation = "cache_local_layer_local_causal_online_stats"
    elif method == HQ_ROTATION_NAME:
        variant = "rotation_dense"
        role = "paper_proposal_rotation"
        allocation = "globally_fixed_kv_code_bits_dense_rotation"
    else:
        raise ValueError(
            f"primary formal HayateQuant method must be one of {FORMAL_HAYATE_METHODS}, "
            f"got {method!r}"
        )
    return {
        "method_role": role,
        "external_quality_baseline": False,
        "proposal_method": True,
        "proposal_family": "HayateQuant",
        "proposal_variant": variant,
        "paper_comparable": True,
        "speed_comparison": True,
        "formal_rate_slots": [2, 3, 4],
        "allocation_scope": allocation,
    }


def reject_legacy_hayate_result_label(name: str) -> None:
    """Reject historical proposal labels instead of silently normalising."""

    if name in LEGACY_HAYATE_RESULT_LABELS:
        raise ValueError(
            f"legacy result label {name!r} is forbidden for new runs; "
            f"use one of {HQ_PUBLIC_METHODS!r}"
        )


__all__ = [
    "FORMAL_HAYATE_METHODS",
    "PAPER_HAYATE_METHODS",
    "HQ_PUBLIC_METHODS",
    "LEGACY_HAYATE_RESULT_LABELS",
    "hayate_formal_provenance",
    "reject_legacy_hayate_result_label",
]

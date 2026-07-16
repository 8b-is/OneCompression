"""Formal-method identity carried by HayateQuant benchmark rows.

The codec recipe and the scientific role of a result are deliberately
separate.  :mod:`onecomp.kvcomp.codec.hayate.recipes` owns the former; this
module supplies the small, JSON-native provenance record that every formal
quality runner must attach to the two proposal rows.
"""

from __future__ import annotations

from typing import Any

from .recipes import HQ_ADAPTIVE_NAME, HQ_NAME

FORMAL_HAYATE_METHODS = (HQ_NAME, HQ_ADAPTIVE_NAME)
LEGACY_HAYATE_RESULT_LABELS = ("HayateQuant-frontier",)


def hayate_formal_provenance(method: str) -> dict[str, Any]:
    """Return the exact proposal-role record for a new formal result."""

    if method == HQ_NAME:
        variant = "fixed_global"
        role = "paper_proposal_fixed"
        allocation = "globally_fixed_kv_code_bits"
    elif method == HQ_ADAPTIVE_NAME:
        variant = "causal_adaptive"
        role = "paper_proposal_adaptive"
        allocation = "cache_local_layer_local_causal_online_stats"
    else:
        raise ValueError(
            f"formal HayateQuant method must be one of {FORMAL_HAYATE_METHODS}, " f"got {method!r}"
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
            f"use {HQ_NAME!r} or {HQ_ADAPTIVE_NAME!r}"
        )


__all__ = [
    "FORMAL_HAYATE_METHODS",
    "LEGACY_HAYATE_RESULT_LABELS",
    "hayate_formal_provenance",
    "reject_legacy_hayate_result_label",
]

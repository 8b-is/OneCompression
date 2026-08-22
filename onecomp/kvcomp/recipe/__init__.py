"""Composite KV compression recipes (codec + selector + policy).

Copyright 2025-2026 Fujitsu Ltd.
"""

from .thinkv import (
    ThinKVPolicy,
    ThinKVEngine,
    ThoughtClassifier,
    attention_sparsity,
    kmeans_keep_indices,
    kde_estimate,
    kde_thresholds,
    auto_configure_thinkv,
    detect_outlier_layers,
)
from .taperkv import TaperKVPolicy, BlockTaperKVPolicy, BudgetTaperKVPolicy
from .pmkvq import PMKVQPolicy
from .rolekv import (
    RolePrecisionPolicy,
    RoleSpan,
    annotate_spans_from_chat,
    role_token_bits,
    validate_spans,
)
from .pmkvq_calib import (
    calibrate_pmkvq,
    estimate_block_sensitivity,
    solve_block_allocation,
    calibrate_channel_reparam,
)

__all__ = [
    "ThinKVPolicy",
    "ThinKVEngine",
    "ThoughtClassifier",
    "attention_sparsity",
    "kmeans_keep_indices",
    "kde_estimate",
    "kde_thresholds",
    "auto_configure_thinkv",
    "detect_outlier_layers",
    "TaperKVPolicy",
    "BlockTaperKVPolicy",
    "BudgetTaperKVPolicy",
    "PMKVQPolicy",
    "RolePrecisionPolicy",
    "RoleSpan",
    "annotate_spans_from_chat",
    "role_token_bits",
    "validate_spans",
    "calibrate_pmkvq",
    "estimate_block_sensitivity",
    "solve_block_allocation",
    "calibrate_channel_reparam",
]

# Copyright 2026 Fujitsu Ltd.
"""Pure configuration helpers for the experimental Hayate Triton backend.

This module deliberately has no vLLM, Triton, CUDA, or torch dependency.  The
plugin's mapping/configuration tests run in the ordinary OneComp environment,
while the kernels themselves are imported only in the vLLM environment.
"""

from __future__ import annotations

import os


def _direct_key_scale(n_groups: int, precompute_key_scale: bool) -> bool:
    """Whether one store program can write the complete key scale itself."""
    if not precompute_key_scale or n_groups != 1:
        return False
    mode = os.environ.get("KVCOMP_VLLM_HAYATE_DIRECT_KEY_SCALE", "auto")
    mode = mode.strip().lower().replace("-", "_")
    if mode in {"auto", "1", "true", "yes", "on", "always"}:
        return True
    if mode in {"0", "false", "no", "off", "never"}:
        return False
    raise ValueError(
        "KVCOMP_VLLM_HAYATE_DIRECT_KEY_SCALE must be auto/always/never " "(or a boolean spelling)"
    )


def _store_group_codes(code_bits: int, n_key_blocks: int) -> int:
    """Resolve the number of adjacent key blocks handled by one store program."""
    mode = os.environ.get("KVCOMP_VLLM_HAYATE_STORE_GROUP_CODES", "auto")
    mode = mode.strip().lower().replace("-", "_")
    if mode == "auto":
        if code_bits == 8:
            return 64 if n_key_blocks <= 64 else 128
        if n_key_blocks >= 128:
            return 128
        if n_key_blocks >= 64:
            return 64
        if n_key_blocks >= 32:
            return 32
        if n_key_blocks >= 16:
            return 16
        return 8 if n_key_blocks >= 8 else 4
    group_codes = int(mode)
    if group_codes not in {1, 2, 4, 8, 16, 32, 64, 128}:
        raise ValueError(
            "KVCOMP_VLLM_HAYATE_STORE_GROUP_CODES must be one of " "1,2,4,8,16,32,64,128"
        )
    if code_bits != 8 and group_codes not in {4, 8, 16, 32, 64, 128}:
        raise ValueError(
            "6-bit Hayate key packing requires STORE_GROUP_CODES=4, 8, " "16, 32, 64, or 128"
        )
    if code_bits != 8 and group_codes > n_key_blocks:
        raise ValueError(
            "6-bit Hayate key packing requires STORE_GROUP_CODES <= " "the number of key VQ blocks"
        )
    return group_codes

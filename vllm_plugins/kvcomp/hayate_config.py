"""Pure-Python configuration helpers for the native HayateQuant backend.

Keep environment parsing that does not require vLLM or Triton in this module so
launch planning and unit tests remain usable when the optional vLLM dependency
is not installed.
"""

from __future__ import annotations

import os


def resolve_store_group_codes(code_bits: int, n_key_blocks: int) -> int:
    """Resolve the number of VQ codes handled by one store-kernel program."""
    if not isinstance(code_bits, int) or isinstance(code_bits, bool) or code_bits not in {6, 8}:
        raise ValueError("code_bits must be 6 or 8 for native HayateQuant packing")
    if not isinstance(n_key_blocks, int) or isinstance(n_key_blocks, bool) or n_key_blocks <= 0:
        raise ValueError("n_key_blocks must be a positive integer")
    mode = os.environ.get("KVCOMP_VLLM_HAYATE_STORE_GROUP_CODES", "auto")
    mode = mode.strip().lower().replace("-", "_")
    if mode == "auto":
        if code_bits == 8:
            return 64 if n_key_blocks <= 64 else 128
        if n_key_blocks < 4:
            raise ValueError("6-bit Hayate key packing requires at least 4 key VQ blocks")
        if n_key_blocks >= 128:
            return 128
        if n_key_blocks >= 64:
            return 64
        if n_key_blocks >= 32:
            return 32
        if n_key_blocks >= 16:
            return 16
        return 8 if n_key_blocks >= 8 else 4

    try:
        group_codes = int(mode)
    except ValueError as exc:
        raise ValueError(
            "KVCOMP_VLLM_HAYATE_STORE_GROUP_CODES must be 'auto' or an integer"
        ) from exc
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


__all__ = ["resolve_store_group_codes"]

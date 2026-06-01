"""Low-bit floating / ternary KV codecs used by ThinKV's TBQ stage.

* :class:`NVFP4Codec` -- 4-bit E2M1 float with a shared per-group scale.
* :class:`TernaryCodec` -- 2-bit ``{-1, 0, +1}`` with a per-group scale.

Both use group quantization along the head dimension (group size 16 by
default, matching the ThinKV paper) and are data-free.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .._base import KVCodec, CompressedKV
from .._registry import register_kv_codec

# E2M1 (FP4) representable magnitudes.
_E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_MAX = 6.0


def _group_reshape(x: torch.Tensor, group_size: int):
    """Reshape ``(..., head_dim)`` to ``(..., n_groups, group_size)``."""
    head_dim = x.shape[-1]
    if head_dim % group_size != 0:
        group_size = head_dim  # single group fallback
    return x.reshape(*x.shape[:-1], head_dim // group_size, group_size), group_size


def _nearest_level(x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Round each element of ``x`` to the nearest value in ``levels`` (signed)."""
    sign = torch.sign(x)
    mag = x.abs()
    lv = levels.to(x.device)
    # nearest magnitude
    idx = torch.bucketize(mag, (lv[1:] + lv[:-1]) / 2.0)
    idx = idx.clamp(0, lv.numel() - 1)
    return sign * lv[idx]


def _fake_quant_nvfp4(x: torch.Tensor, group_size: int) -> torch.Tensor:
    xg, gs = _group_reshape(x, group_size)
    amax = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = amax / _E2M1_MAX  # shared per-group scale (stored as FP8 in production)
    q = _nearest_level(xg / scale, _E2M1_LEVELS) * scale
    return q.reshape(x.shape).to(x.dtype)


def _fake_quant_ternary(x: torch.Tensor, group_size: int) -> torch.Tensor:
    xg, gs = _group_reshape(x, group_size)
    amax = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    z = xg / amax
    t = torch.where(z > 0.5, 1.0, torch.where(z < -0.5, -1.0, 0.0))
    q = t * amax
    return q.reshape(x.shape).to(x.dtype)


@register_kv_codec("nvfp4")
@dataclass
class NVFP4Codec(KVCodec):
    """4-bit E2M1 float KV quantization with per-group scale."""

    bits: int = 4
    group_size: int = 16
    key_axis: str = "token"
    value_axis: str = "token"

    def encode(self, key, value, ctx=None) -> CompressedKV:
        kq = _fake_quant_nvfp4(key, self.group_size)
        vq = _fake_quant_nvfp4(value, self.group_size)
        stored = (key.numel() + value.numel()) * 4 + (
            (key.numel() + value.numel()) // self.group_size * 8
        )
        return CompressedKV(
            key_q=kq, value_q=vq,
            meta={"dtype": key.dtype, "stored_bits": int(stored), "bits": 4, "passthrough": True},
        )

    def decode(self, compressed: CompressedKV, ctx=None):
        dtype = compressed.meta.get("dtype", torch.float16)
        return compressed.key_q.to(dtype), compressed.value_q.to(dtype)


@register_kv_codec("ternary")
@dataclass
class TernaryCodec(KVCodec):
    """2-bit ternary ``{-1, 0, +1}`` KV quantization with per-group scale."""

    bits: int = 2
    group_size: int = 16
    key_axis: str = "token"
    value_axis: str = "token"

    def encode(self, key, value, ctx=None) -> CompressedKV:
        kq = _fake_quant_ternary(key, self.group_size)
        vq = _fake_quant_ternary(value, self.group_size)
        stored = (key.numel() + value.numel()) * 2 + (
            (key.numel() + value.numel()) // self.group_size * 8
        )
        return CompressedKV(
            key_q=kq, value_q=vq,
            meta={"dtype": key.dtype, "stored_bits": int(stored), "bits": 2, "passthrough": True},
        )

    def decode(self, compressed: CompressedKV, ctx=None):
        dtype = compressed.meta.get("dtype", torch.float16)
        return compressed.key_q.to(dtype), compressed.value_q.to(dtype)

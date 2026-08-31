"""Low-bit floating / ternary KV codecs used by ThinKV's TBQ stage.

* :class:`NVFP4Codec` -- 4-bit E2M1 float with a shared per-group scale.
* :class:`TernaryCodec` -- 2-bit ``{-1, 0, +1}`` with a per-group scale.

Keys use per-channel groups along the token axis and values use per-token groups
along head dimension (group size 16 by default), matching ThinKV's TBQ layout.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .._base import CompressedKV, KVCodec
from .._registry import register_kv_codec

# E2M1 (FP4) representable magnitudes.
_E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_MAX = 6.0


def _group_reshape(x: torch.Tensor, group_size: int, axis: str):
    """Group ``(b,h,seq,dim)`` along dim (token) or seq (channel)."""
    if axis not in ("token", "channel"):
        raise ValueError(f"axis must be 'token' or 'channel', got {axis!r}")
    moved = x if axis == "token" else x.transpose(-1, -2)
    length = moved.shape[-1]
    group_size = length if group_size is None or group_size <= 0 else group_size
    pad = (-length) % group_size
    if pad:
        moved = torch.nn.functional.pad(moved, (0, pad))
    grouped = moved.reshape(*moved.shape[:-1], moved.shape[-1] // group_size, group_size)

    def restore(grouped_tensor):
        restored = grouped_tensor.reshape(*moved.shape)[..., :length]
        return restored if axis == "token" else restored.transpose(-1, -2)

    return grouped, restore


def _nearest_level(x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Round each element of ``x`` to the nearest value in ``levels`` (signed)."""
    sign = torch.sign(x)
    mag = x.abs()
    lv = levels.to(x.device)
    # nearest magnitude
    idx = torch.bucketize(mag.contiguous(), ((lv[1:] + lv[:-1]) / 2.0).contiguous())
    idx = idx.clamp(0, lv.numel() - 1)
    return sign * lv[idx]


def _fake_quant_nvfp4(x: torch.Tensor, group_size: int, axis: str = "token") -> torch.Tensor:
    xg, restore = _group_reshape(x, group_size, axis)
    amax = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = amax / _E2M1_MAX  # shared per-group scale (stored as FP8 in production)
    q = _nearest_level(xg / scale, _E2M1_LEVELS) * scale
    return restore(q).to(x.dtype)


def _fake_quant_ternary(x: torch.Tensor, group_size: int, axis: str = "token") -> torch.Tensor:
    xg, restore = _group_reshape(x, group_size, axis)
    amax = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    z = xg / amax
    t = torch.where(z > 0.5, 1.0, torch.where(z < -0.5, -1.0, 0.0))
    q = t * amax
    return restore(q).to(x.dtype)


def _num_groups(x: torch.Tensor, group_size: int, axis: str) -> int:
    length = x.shape[-1] if axis == "token" else x.shape[-2]
    gs = length if group_size is None or group_size <= 0 else group_size
    vectors = x.numel() // length
    return vectors * ((length + gs - 1) // gs)


@register_kv_codec("nvfp4")
@dataclass
class NVFP4Codec(KVCodec):
    """4-bit E2M1 float KV quantization with per-group scale."""

    bits: int = 4
    group_size: int = 16
    key_axis: str = "channel"
    value_axis: str = "token"

    def encode(self, key, value, ctx=None) -> CompressedKV:
        kq = _fake_quant_nvfp4(key, self.group_size, self.key_axis)
        vq = _fake_quant_nvfp4(value, self.group_size, self.value_axis)
        stored = (key.numel() + value.numel()) * 4 + 8 * (
            _num_groups(key, self.group_size, self.key_axis)
            + _num_groups(value, self.group_size, self.value_axis)
        )
        return CompressedKV(
            key_q=kq,
            value_q=vq,
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
    key_axis: str = "channel"
    value_axis: str = "token"

    def encode(self, key, value, ctx=None) -> CompressedKV:
        kq = _fake_quant_ternary(key, self.group_size, self.key_axis)
        vq = _fake_quant_ternary(value, self.group_size, self.value_axis)
        stored = (key.numel() + value.numel()) * 2 + 8 * (
            _num_groups(key, self.group_size, self.key_axis)
            + _num_groups(value, self.group_size, self.value_axis)
        )
        return CompressedKV(
            key_q=kq,
            value_q=vq,
            meta={"dtype": key.dtype, "stored_bits": int(stored), "bits": 2, "passthrough": True},
        )

    def decode(self, compressed: CompressedKV, ctx=None):
        dtype = compressed.meta.get("dtype", torch.float16)
        return compressed.key_q.to(dtype), compressed.value_q.to(dtype)

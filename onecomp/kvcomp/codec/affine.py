"""Affine integer KV codecs (INT8/INT4/INT2) and KIVI.

These are data-free (``flag_calibration=False``): scales/zeros are computed
online from each K/V block, exactly like KIVI's tuning-free quantization.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .._base import KVCodec, CompressedKV
from .._registry import register_kv_codec
from ._quant_primitives import quantize_affine, dequantize_affine, effective_bits


@dataclass
class AffineKVCodec(KVCodec):
    """Per-axis affine integer quantization of the KV cache.

    Keys and values may use different quantization axes (``key_axis`` /
    ``value_axis``) and a different number of bits is *not* supported here --
    see :class:`KIVICodec` for the asymmetric K/V configuration.  Use
    :class:`Int4Codec` / :class:`Int2Codec` / :class:`Int8Codec` for the common
    presets.
    """

    bits: int = 4

    def validate_params(self):
        if not (isinstance(self.bits, int) and 1 <= self.bits <= 8):
            raise ValueError(f"AffineKVCodec.bits must be int in 1..8, got {self.bits!r}")
        for attr in ("key_axis", "value_axis"):
            if getattr(self, attr) not in ("token", "channel"):
                raise ValueError(f"AffineKVCodec.{attr} must be 'token' or 'channel'")

    def encode(self, key, value, ctx=None) -> CompressedKV:
        kq, ks, kz = quantize_affine(
            key, self.bits, axis=self.key_axis, symmetric=self.symmetric, group_size=self.group_size
        )
        vq, vs, vz = quantize_affine(
            value, self.bits, axis=self.value_axis, symmetric=self.symmetric,
            group_size=self.group_size,
        )
        head_dim = key.shape[-1]
        stored = (kq.numel() + vq.numel()) * effective_bits(self.bits, self.group_size, head_dim)
        return CompressedKV(
            key_q=kq,
            value_q=vq,
            key_meta={"scale": ks, "zero": kz, "axis": self.key_axis},
            value_meta={"scale": vs, "zero": vz, "axis": self.value_axis},
            meta={"dtype": key.dtype, "stored_bits": int(stored), "bits": self.bits},
        )

    def decode(self, compressed: CompressedKV, ctx=None):
        dtype = compressed.meta.get("dtype", torch.float16)
        key = dequantize_affine(
            compressed.key_q, compressed.key_meta["scale"], compressed.key_meta["zero"]
        ).to(dtype)
        value = dequantize_affine(
            compressed.value_q, compressed.value_meta["scale"], compressed.value_meta["zero"]
        ).to(dtype)
        return key, value


@register_kv_codec("int8")
@dataclass
class Int8Codec(AffineKVCodec):
    """8-bit affine KV quantization baseline."""

    bits: int = 8


@register_kv_codec("int4")
@dataclass
class Int4Codec(AffineKVCodec):
    """4-bit affine KV quantization baseline."""

    bits: int = 4


@register_kv_codec("int2")
@dataclass
class Int2Codec(AffineKVCodec):
    """2-bit affine KV quantization (aggressive)."""

    bits: int = 2


@register_kv_codec("kivi")
@dataclass
class KIVICodec(KVCodec):
    """KIVI-style asymmetric KV quantization.

    Keys are quantized per-channel and values per-token (the KIVI recipe),
    which empirically preserves attention quality far better than a uniform
    axis at the same bit-width.  Tuning-free / data-free.

    Reference: Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization
    for KV Cache" (arXiv:2402.02750).
    """

    bits: int = 2
    key_axis: str = "channel"
    value_axis: str = "token"
    group_size: int = 32
    residual_window: int = 128

    def validate_params(self):
        if not (isinstance(self.bits, int) and 1 <= self.bits <= 8):
            raise ValueError(f"KIVICodec.bits must be int in 1..8, got {self.bits!r}")

    def encode(self, key, value, ctx=None) -> CompressedKV:
        kq, ks, kz = quantize_affine(
            key, self.bits, axis=self.key_axis, symmetric=self.symmetric, group_size=self.group_size
        )
        vq, vs, vz = quantize_affine(
            value, self.bits, axis=self.value_axis, symmetric=self.symmetric,
            group_size=self.group_size,
        )
        head_dim = key.shape[-1]
        stored = (kq.numel() + vq.numel()) * effective_bits(self.bits, self.group_size, head_dim)
        return CompressedKV(
            key_q=kq,
            value_q=vq,
            key_meta={"scale": ks, "zero": kz, "axis": self.key_axis},
            value_meta={"scale": vs, "zero": vz, "axis": self.value_axis},
            meta={"dtype": key.dtype, "stored_bits": int(stored), "bits": self.bits},
        )

    def decode(self, compressed: CompressedKV, ctx=None):
        dtype = compressed.meta.get("dtype", torch.float16)
        key = dequantize_affine(
            compressed.key_q, compressed.key_meta["scale"], compressed.key_meta["zero"]
        ).to(dtype)
        value = dequantize_affine(
            compressed.value_q, compressed.value_meta["scale"], compressed.value_meta["zero"]
        ).to(dtype)
        return key, value

"""Bit-packed affine KV codec (real storage, serializable).

Unlike the simulated codecs in this package (which keep the dequantized
float tensor), :class:`PackedAffineCodec` stores actual ``uint8``-packed
codes plus compact per-group scales/zeros, so its :class:`CompressedKV`
payload is byte-serializable.  It is the live-cache counterpart of the
``.okv`` session checkpoint (:mod:`onecomp.kvcomp.checkpoint`), which uses
the same primitives -- a session compressed live with this codec and one
restored from a checkpoint see bit-identical quantization.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass

from .._base import KVCodec, CompressedKV
from .._registry import register_kv_codec
from ._packing import (
    SUPPORTED_PACKED_BITS,
    dequantize_packed,
    quantize_packed,
)


@register_kv_codec("packed_affine")
@dataclass
class PackedAffineCodec(KVCodec):
    """Per-group asymmetric affine quantization with real bit-packing.

    Keys default to per-channel and values to per-token grouping (the KIVI
    recipe).  Data-free / tuning-free; ``decode(encode(...))`` matches the
    ``.okv`` checkpoint reconstruction exactly.
    """

    bits: int = 4
    key_axis: str = "channel"
    value_axis: str = "token"
    group_size: int = 64

    def validate_params(self):
        if self.bits not in SUPPORTED_PACKED_BITS:
            raise ValueError(
                f"PackedAffineCodec.bits must be one of {SUPPORTED_PACKED_BITS}, "
                f"got {self.bits!r}"
            )
        for attr in ("key_axis", "value_axis"):
            if getattr(self, attr) not in ("token", "channel"):
                raise ValueError(f"PackedAffineCodec.{attr} must be 'token' or 'channel'")
        if not (isinstance(self.group_size, int) and self.group_size > 0):
            raise ValueError(
                f"PackedAffineCodec.group_size must be a positive int, got {self.group_size!r}"
            )

    def encode(self, key, value, ctx=None) -> CompressedKV:
        kq = quantize_packed(key, self.bits, group_size=self.group_size, axis=self.key_axis)
        vq = quantize_packed(value, self.bits, group_size=self.group_size, axis=self.value_axis)
        return CompressedKV(
            key_q=kq,
            value_q=vq,
            meta={
                "dtype": key.dtype,
                "bits": self.bits,
                # Real storage: the packed payload size is exact, not modelled.
                "stored_bits": (kq.num_bytes() + vq.num_bytes()) * 8,
            },
        )

    def decode(self, compressed: CompressedKV, ctx=None):
        return dequantize_packed(compressed.key_q), dequantize_packed(compressed.value_q)

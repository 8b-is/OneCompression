"""KV memory / effective-bits accounting for compression reports.

These helpers turn a codec's nominal bit-width (and per-vector metadata cost)
into a realised KV cache footprint, so every quality number can be reported
next to the memory it bought -- never compression ratio alone.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class KVMemoryReport:
    """Realised KV cache memory for one (model, sequence, codec) point."""

    seq_len: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    fp16_bytes: int
    stored_bytes: int

    def __post_init__(self) -> None:
        for name in ("seq_len", "num_layers", "num_kv_heads", "head_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("fp16_bytes", "stored_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def compression_ratio(self) -> float:
        if self.stored_bytes == 0:
            return float("inf") if self.fp16_bytes > 0 else 1.0
        return self.fp16_bytes / self.stored_bytes

    @property
    def kv_fraction(self) -> float:
        """Stored / FullKV(fp16) memory fraction (lower is more compressed)."""
        if self.fp16_bytes == 0:
            return 1.0
        return self.stored_bytes / self.fp16_bytes

    def as_dict(self) -> dict:
        return {
            "seq_len": self.seq_len,
            "fp16_MB": self.fp16_bytes / 2**20,
            "stored_MB": self.stored_bytes / 2**20,
            "compression_ratio": round(self.compression_ratio, 3),
            "kv_fraction_pct": round(100.0 * self.kv_fraction, 3),
        }


def effective_bits(stored_bits: int, num_elems: int) -> float:
    """Average logical fixed-width data-plane bits per quantized K/V element.

    ``stored_bits`` may include enabled scale/mask/QJL/outlier data-plane
    fields, but not Python control tags or container/allocator bytes unless a
    codec explicitly documents a different contract.
    """
    if isinstance(stored_bits, bool) or not isinstance(stored_bits, int) or stored_bits < 0:
        raise ValueError("stored_bits must be a non-negative integer")
    if isinstance(num_elems, bool) or not isinstance(num_elems, int) or num_elems < 0:
        raise ValueError("num_elems must be a non-negative integer")
    if num_elems == 0:
        if stored_bits:
            raise ValueError("stored_bits must be zero when num_elems is zero")
        return 0.0
    return stored_bits / num_elems


def kv_cache_bytes(
    seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    bits: float = 16.0,
    meta_bits_per_vector: float = 0.0,
) -> int:
    """Bytes for a K+V cache of ``seq_len`` tokens at ``bits`` per element.

    ``meta_bits_per_vector`` accounts for per-token scales/norms (e.g. INT/
    TurboQuant keep an fp16 scale per (layer, head, token) vector).
    """
    for name, value in (
        ("seq_len", seq_len),
        ("num_layers", num_layers),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    for name, value in (("bits", bits), ("meta_bits_per_vector", meta_bits_per_vector)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be a finite non-negative number")
    elems = 2 * num_layers * num_kv_heads * seq_len * head_dim  # K and V
    vectors = 2 * num_layers * num_kv_heads * seq_len
    total_bits = elems * bits + vectors * meta_bits_per_vector
    # A packed byte stream still needs storage for its final partial byte.
    # Truncation silently reported zero bytes for 1--7 bits and under-counted
    # non-byte-aligned identity ledgers (for example BlockQuant channel IDs).
    return math.ceil(total_bits / 8)


def summarize_compression(
    seq_len: int,
    model_config,
    stored_bits: Optional[int] = None,
    avg_bits: Optional[float] = None,
    meta_bits_per_vector: float = 16.0,
) -> KVMemoryReport:
    """Build a :class:`KVMemoryReport` from a HF model config.

    Provide either ``stored_bits`` (codec-accounted) or ``avg_bits``
    (nominal).  When ``stored_bits`` comes from :class:`CompressedKV`, the
    resulting ``stored_bytes`` is the corresponding logical packed data-plane
    size, not the allocator-resident size of the current Python/PyTorch
    payload and not a measurement of persistent codec state.
    """
    cfg = (
        model_config.get_text_config()
        if hasattr(model_config, "get_text_config")
        else model_config
    )
    num_layers = cfg.num_hidden_layers
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    fp16_bytes = kv_cache_bytes(seq_len, num_layers, num_kv_heads, head_dim, bits=16.0)
    if stored_bits is not None:
        if isinstance(stored_bits, bool) or not isinstance(stored_bits, int) or stored_bits < 0:
            raise ValueError("stored_bits must be a non-negative integer")
        stored_bytes = math.ceil(stored_bits / 8)
    else:
        stored_bytes = kv_cache_bytes(
            seq_len,
            num_layers,
            num_kv_heads,
            head_dim,
            bits=avg_bits if avg_bits is not None else 16.0,
            meta_bits_per_vector=meta_bits_per_vector,
        )
    return KVMemoryReport(
        seq_len=seq_len,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        fp16_bytes=fp16_bytes,
        stored_bytes=stored_bytes,
    )

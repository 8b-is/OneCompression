"""Exact per-generation KV-cache accounting for evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

PER_EXAMPLE_CACHE_LIFETIME = "released_after_scalar_accounting_before_next_cache_factory"
PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION = (
    "none; retained_records_are_host_scalar_text_and_mappings_only"
)


def _positive_ratio(value: Any, *, label: str) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{label} must be a positive real number") from exc
    if math.isnan(ratio) or ratio <= 0.0:
        raise RuntimeError(f"{label} must be positive and not NaN, got {ratio!r}")
    return ratio


def cache_mem_fraction(cache: object) -> float | None:
    """Return a validated stored/FP16 fraction for display summaries."""

    mem_fraction = getattr(cache, "mem_fraction", None)
    if mem_fraction is not None:
        if not callable(mem_fraction):
            raise RuntimeError("cache.mem_fraction must be callable")
        try:
            fraction = float(mem_fraction())
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                "cache.mem_fraction() must return a finite non-negative number"
            ) from exc
        if not math.isfinite(fraction) or fraction < 0.0:
            raise RuntimeError(
                "cache.mem_fraction() must return a finite non-negative number, "
                f"got {fraction!r}"
            )
        return fraction

    compression_ratio = getattr(cache, "compression_ratio", None)
    if compression_ratio is not None:
        if not callable(compression_ratio):
            raise RuntimeError("cache.compression_ratio must be callable")
        ratio = _positive_ratio(
            compression_ratio(),
            label="cache.compression_ratio()",
        )
        return 0.0 if math.isinf(ratio) else 1.0 / ratio
    return None


def aggregate_mem_fraction(
    compression_statistics: Mapping[str, Any],
    legacy_observations: Sequence[float] = (),
) -> tuple[float, str]:
    """Derive stored/FP16 from exact totals, with an explicit legacy fallback."""

    mode = compression_statistics.get("compression_aggregation")
    if mode == "sum_exact_per_generation_cache_bit_counters":
        fp16_bits = compression_statistics.get("compression_fp16_bits")
        stored_bits = compression_statistics.get("compression_stored_bits")
        if (
            isinstance(fp16_bits, bool)
            or not isinstance(fp16_bits, int)
            or fp16_bits <= 0
            or isinstance(stored_bits, bool)
            or not isinstance(stored_bits, int)
            or stored_bits < 0
        ):
            raise RuntimeError("invalid exact counters for aggregate memory fraction")
        return stored_bits / fp16_bits, "sum_stored_bits_over_sum_fp16_bits"
    if mode == "fullkv_no_cache":
        return 1.0, "fullkv_no_cache"
    if mode != "legacy_arithmetic_mean_per_generation_ratio_no_exact_counters":
        raise RuntimeError(f"unknown compression aggregation for memory fraction: {mode!r}")
    if not legacy_observations:
        raise RuntimeError("legacy memory-fraction aggregation has no observations")
    values = [float(value) for value in legacy_observations]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise RuntimeError("legacy memory-fraction observations must be finite and non-negative")
    return (
        math.fsum(values) / len(values),
        "legacy_arithmetic_mean_per_generation_mem_fraction_no_exact_counters",
    )


def cache_compression_sufficient_statistics(cache: object) -> dict[str, Any]:
    """Snapshot exact resident cache-bit counters when the cache exposes them."""

    has_fp16_bits = hasattr(cache, "fp16_bits")
    has_stored_bits = hasattr(cache, "stored_bits")
    if has_fp16_bits != has_stored_bits:
        raise RuntimeError("compressed cache exposes only one of fp16_bits/stored_bits")
    if not has_fp16_bits:
        fraction = cache_mem_fraction(cache)
        if fraction is not None:
            ratio = float("inf") if fraction == 0.0 else 1.0 / fraction
            return {
                "compression_fp16_bits": None,
                "compression_stored_bits": None,
                "compression_aggregation": "legacy_cache_ratio_only",
                "compression_ratio": ratio,
            }
        return {
            "compression_fp16_bits": None,
            "compression_stored_bits": None,
            "compression_aggregation": "fullkv_no_cache",
        }

    fp16_bits = getattr(cache, "fp16_bits")
    stored_bits = getattr(cache, "stored_bits")
    if (
        isinstance(fp16_bits, bool)
        or not isinstance(fp16_bits, int)
        or fp16_bits <= 0
        or isinstance(stored_bits, bool)
        or not isinstance(stored_bits, int)
        or stored_bits < 0
    ):
        raise RuntimeError(
            "compressed cache must expose positive fp16_bits and non-negative "
            "integer stored_bits"
        )
    return {
        "compression_fp16_bits": fp16_bits,
        "compression_stored_bits": stored_bits,
        "compression_aggregation": "exact_cache_bit_counters",
    }


def aggregate_compression_sufficient_statistics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate exact per-generation cache counters."""

    if not records:
        raise ValueError("compression accounting needs at least one record")
    modes = {row.get("compression_aggregation") for row in records}
    if modes == {"fullkv_no_cache"}:
        if any(
            row.get("compression_fp16_bits") is not None
            or row.get("compression_stored_bits") is not None
            for row in records
        ):
            raise RuntimeError("FullKV records contain compressed-cache counters")
        return {
            "compression_fp16_bits": None,
            "compression_stored_bits": None,
            "compression_aggregation": "fullkv_no_cache",
            "compression_observations": len(records),
        }

    if modes == {"legacy_cache_ratio_only"}:
        ratios = [
            _positive_ratio(
                row.get("compression_ratio"),
                label="legacy cache compression_ratio",
            )
            for row in records
        ]
        finite = [ratio for ratio in ratios if math.isfinite(ratio)]
        mean = float("inf") if len(finite) != len(ratios) else math.fsum(finite) / len(finite)
        return {
            "compression_fp16_bits": None,
            "compression_stored_bits": None,
            "compression_aggregation": (
                "legacy_arithmetic_mean_per_generation_ratio_no_exact_counters"
            ),
            "compression_observations": len(records),
            "compression_ratio_arithmetic_mean": mean,
        }

    if modes != {"exact_cache_bit_counters"}:
        raise RuntimeError("mixed or missing compression accounting modes")
    fp16_bits = 0
    stored_bits = 0
    for row in records:
        fp16 = row.get("compression_fp16_bits")
        stored = row.get("compression_stored_bits")
        if (
            isinstance(fp16, bool)
            or not isinstance(fp16, int)
            or fp16 <= 0
            or isinstance(stored, bool)
            or not isinstance(stored, int)
            or stored < 0
        ):
            raise RuntimeError("compressed record has invalid exact bit counters")
        fp16_bits += fp16
        stored_bits += stored
    return {
        "compression_fp16_bits": fp16_bits,
        "compression_stored_bits": stored_bits,
        "compression_aggregation": ("sum_exact_per_generation_cache_bit_counters"),
        "compression_observations": len(records),
    }

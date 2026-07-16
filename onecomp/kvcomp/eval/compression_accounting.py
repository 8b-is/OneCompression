"""Exact per-generation KV-cache accounting for formal evaluation axes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..codec.hayate.codec import (
    ADAPTIVE_HQ_POLICY_VERSION,
    ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION,
)

PER_EXAMPLE_CACHE_LIFETIME = "released_after_scalar_accounting_before_next_cache_factory"
PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION = (
    "none; retained_records_are_host_scalar_text_and_mappings_only"
)


def _adaptive_integer(value: Any, *, label: str, minimum: int = 1) -> int:
    """Return an AdaptiveHQ provenance counter without accepting ``bool``."""

    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return value


def _validated_adaptive_hq_runtime_summary(value: Any) -> dict[str, Any]:
    """Validate and detach the compact v5 one-selector-bit/update ledger."""

    if not isinstance(value, Mapping):
        raise RuntimeError("AdaptiveHQ cache runtime provenance is incomplete")
    if (
        value.get("schema_version") != 1
        or value.get("policy_version") != ADAPTIVE_HQ_POLICY_VERSION
        or value.get("selector_carrier_version") != ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION
        or value.get("selector_incremental_stored_bits_total") != 0
    ):
        raise RuntimeError("AdaptiveHQ cache policy/carrier contract changed")

    encoded_updates = _adaptive_integer(
        value.get("encoded_updates"), label="AdaptiveHQ encoded_updates"
    )
    encoded_tokens = _adaptive_integer(
        value.get("encoded_tokens"), label="AdaptiveHQ encoded_tokens"
    )
    selector_bits = _adaptive_integer(
        value.get("selector_bits_total"), label="AdaptiveHQ selector_bits_total"
    )
    if selector_bits != encoded_updates:
        raise RuntimeError("AdaptiveHQ v5 must carry exactly one selector bit per update")

    raw_histogram = value.get("split_histogram")
    if not isinstance(raw_histogram, Mapping) or not raw_histogram:
        raise RuntimeError("AdaptiveHQ selector histogram is missing")
    histogram: dict[str, dict[str, Any]] = {}
    for key, raw in sorted(raw_histogram.items()):
        if (
            not isinstance(key, str)
            or not isinstance(raw, Mapping)
            or not isinstance(raw.get("pair"), list)
            or len(raw["pair"]) != 2
            or any(
                isinstance(bit, bool) or not isinstance(bit, int) or bit <= 0
                for bit in raw["pair"]
            )
            or key != f"{raw['pair'][0]}:{raw['pair'][1]}"
        ):
            raise RuntimeError("AdaptiveHQ selector histogram is malformed")
        histogram[key] = {
            "pair": [int(raw["pair"][0]), int(raw["pair"][1])],
            "updates": _adaptive_integer(
                raw.get("updates"), label=f"AdaptiveHQ histogram {key}/updates"
            ),
            "tokens": _adaptive_integer(
                raw.get("tokens"), label=f"AdaptiveHQ histogram {key}/tokens"
            ),
            "vectors": _adaptive_integer(
                raw.get("vectors"), label=f"AdaptiveHQ histogram {key}/vectors"
            ),
        }
    if (
        sum(bucket["updates"] for bucket in histogram.values()) != encoded_updates
        or sum(bucket["tokens"] for bucket in histogram.values()) != encoded_tokens
        or sum(bucket["vectors"] for bucket in histogram.values()) < 1
        or any(
            bucket["updates"] > bucket["tokens"] or bucket["tokens"] > bucket["vectors"]
            for bucket in histogram.values()
        )
    ):
        raise RuntimeError(
            "AdaptiveHQ selector histogram totals/order do not match "
            "encoded updates/tokens/vectors"
        )
    return {
        "schema_version": 1,
        "policy_version": ADAPTIVE_HQ_POLICY_VERSION,
        "encoded_updates": encoded_updates,
        "encoded_tokens": encoded_tokens,
        "selector_bits_total": selector_bits,
        "selector_incremental_stored_bits_total": 0,
        "selector_carrier_version": ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION,
        "split_histogram": histogram,
    }


def _positive_ratio(value: Any, *, label: str) -> float:
    """Return a validated cache compression ratio (``+inf`` is meaningful)."""
    try:
        ratio = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{label} must be a positive real number") from exc
    if math.isnan(ratio) or ratio <= 0.0:
        raise RuntimeError(f"{label} must be positive and not NaN, got {ratio!r}")
    return ratio


def cache_mem_fraction(cache: object) -> float | None:
    """Return a validated stored/FP16 fraction for display-only summaries.

    Exact formal aggregation must use integer bit counters below.  This helper
    merely prevents NaN, negative, or non-callable legacy ratio accessors from
    contaminating diagnostic averages in long-context evaluators.
    """

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
        ratio = _positive_ratio(compression_ratio(), label="cache.compression_ratio()")
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
    """Snapshot exact resident cache-bit counters and compact AdaptiveHQ state.

    A plain Hugging Face cache has no compressed payload, so its counters are
    explicitly ``None``.  A compressed cache must expose integer counters;
    silently substituting a displayed ratio is forbidden.
    """

    has_fp16_bits = hasattr(cache, "fp16_bits")
    has_stored_bits = hasattr(cache, "stored_bits")
    if has_fp16_bits != has_stored_bits:
        raise RuntimeError(
            "compressed cache exposes only one of fp16_bits/stored_bits; "
            "exact accounting requires both"
        )
    if not has_fp16_bits:
        fraction = cache_mem_fraction(cache)
        if fraction is not None:
            ratio = float("inf") if fraction == 0.0 else 1.0 / fraction
            return {
                "compression_fp16_bits": None,
                "compression_stored_bits": None,
                "compression_aggregation": "legacy_cache_ratio_only",
                "compression_ratio": ratio,
                "adaptive_hq_runtime": None,
            }
        return {
            "compression_fp16_bits": None,
            "compression_stored_bits": None,
            "compression_aggregation": "fullkv_no_cache",
            "adaptive_hq_runtime": None,
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
            "compressed cache did not expose positive fp16_bits and non-negative "
            "exact integer stored_bits"
        )

    adaptive = None
    callback = getattr(cache, "get_adaptive_hq_provenance", None)
    if callable(callback):
        provenance = callback()
        runtime = provenance.get("runtime") if isinstance(provenance, Mapping) else None
        if runtime:
            if not isinstance(runtime, Mapping):
                raise RuntimeError("AdaptiveHQ cache runtime provenance is incomplete")
            aggregate = runtime.get("aggregate")
            if runtime.get("kind") != "adaptive_hq" or not isinstance(aggregate, Mapping):
                raise RuntimeError("AdaptiveHQ cache runtime provenance is incomplete")
            adaptive = _validated_adaptive_hq_runtime_summary(
                {
                    "schema_version": runtime.get("schema_version"),
                    "policy_version": runtime.get("policy_version"),
                    **dict(aggregate),
                }
            )
    return {
        "compression_fp16_bits": fp16_bits,
        "compression_stored_bits": stored_bits,
        "compression_aggregation": "exact_cache_bit_counters",
        "adaptive_hq_runtime": adaptive,
    }


def aggregate_compression_sufficient_statistics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Sum exact per-generation counters and AdaptiveHQ provenance."""

    if not records:
        raise ValueError("compression accounting needs at least one generation record")
    compressed = [row.get("compression_aggregation") for row in records]
    if set(compressed) == {"fullkv_no_cache"}:
        if any(
            row.get("compression_fp16_bits") is not None
            or row.get("compression_stored_bits") is not None
            or row.get("adaptive_hq_runtime") is not None
            for row in records
        ):
            raise RuntimeError("FullKV records unexpectedly contain compressed-cache counters")
        return {
            "compression_fp16_bits": None,
            "compression_stored_bits": None,
            "compression_aggregation": "fullkv_no_cache",
            "compression_observations": len(records),
            "adaptive_hq_runtime": None,
        }
    if set(compressed) == {"legacy_cache_ratio_only"}:
        ratios = [
            _positive_ratio(row.get("compression_ratio"), label="legacy cache compression_ratio")
            for row in records
        ]
        finite = [ratio for ratio in ratios if math.isfinite(ratio)]
        arithmetic_mean = (
            float("inf") if len(finite) != len(ratios) else math.fsum(finite) / len(finite)
        )
        if any(row.get("adaptive_hq_runtime") is not None for row in records):
            raise RuntimeError("legacy cache ratios cannot carry AdaptiveHQ exact provenance")
        return {
            "compression_fp16_bits": None,
            "compression_stored_bits": None,
            "compression_aggregation": (
                "legacy_arithmetic_mean_per_generation_ratio_no_exact_counters"
            ),
            "compression_observations": len(records),
            "compression_ratio_arithmetic_mean": arithmetic_mean,
            "adaptive_hq_runtime": None,
        }
    if set(compressed) != {"exact_cache_bit_counters"}:
        raise RuntimeError("mixed or missing compression accounting modes")
    fp16_bits = 0
    stored_bits = 0
    adaptive_rows = []
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
            raise RuntimeError("compressed generation record has invalid exact bit counters")
        fp16_bits += fp16
        stored_bits += stored
        if row.get("adaptive_hq_runtime") is not None:
            adaptive_rows.append(
                _validated_adaptive_hq_runtime_summary(row["adaptive_hq_runtime"])
            )
    adaptive = None
    if adaptive_rows:
        if len(adaptive_rows) != len(records):
            raise RuntimeError("mixed AdaptiveHQ and non-adaptive runtime provenance")
        policies = {row.get("policy_version") for row in adaptive_rows}
        carriers = {row.get("selector_carrier_version") for row in adaptive_rows}
        if len(policies) != 1 or None in policies or len(carriers) != 1 or None in carriers:
            raise RuntimeError("AdaptiveHQ policy/carrier changed across generation records")
        histogram: dict[str, dict[str, Any]] = {}
        for row in adaptive_rows:
            if row.get("selector_incremental_stored_bits_total") != 0:
                raise RuntimeError("AdaptiveHQ selector unexpectedly added stored bits")
            for key, bucket in row["split_histogram"].items():
                merged = histogram.setdefault(
                    key,
                    {"pair": list(bucket["pair"]), "updates": 0, "tokens": 0, "vectors": 0},
                )
                if merged["pair"] != bucket["pair"]:
                    raise RuntimeError("AdaptiveHQ histogram pair identity changed")
                for field in ("updates", "tokens", "vectors"):
                    merged[field] += int(bucket[field])
        adaptive = _validated_adaptive_hq_runtime_summary(
            {
                "schema_version": 1,
                "policy_version": next(iter(policies)),
                "encoded_updates": sum(int(row["encoded_updates"]) for row in adaptive_rows),
                "encoded_tokens": sum(int(row["encoded_tokens"]) for row in adaptive_rows),
                "selector_bits_total": sum(
                    int(row.get("selector_bits_total", 0)) for row in adaptive_rows
                ),
                "selector_incremental_stored_bits_total": 0,
                "selector_carrier_version": next(iter(carriers)),
                "split_histogram": histogram,
            }
        )
    return {
        "compression_fp16_bits": fp16_bits,
        "compression_stored_bits": stored_bits,
        "compression_aggregation": "sum_exact_per_generation_cache_bit_counters",
        "compression_observations": len(records),
        "adaptive_hq_runtime": adaptive,
    }

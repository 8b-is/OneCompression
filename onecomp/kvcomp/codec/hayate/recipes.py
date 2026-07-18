"""Named HayateQuant operating points.

The codec implementation deliberately stays parameterized: ``bits``,
``value_bits`` and scale modes are primitive knobs.  Paper/benchmark recipes
should live here instead of being re-spelled in every runner.  That keeps the
definition of HQ auditable and makes future single-codec candidates compare
against the same baseline.

The paper benchmark exposes four named rows: fixed HQ, the stronger
frontier operating point, AdaptiveHQ, and RotationHQ.  These are distinct
recipes; old ``HayateQuant-*`` spellings are not emitted by new runs.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

from .codec import HayateQuantCodec

HQ_NAME = "HQ"
HQ_FRONTIER_NAME = "HQ-Frontier"
HQ_ADAPTIVE_NAME = "AdaptiveHQ"
HQ_ROTATION_NAME = "RotationHQ"
# These are the only user-facing HayateQuant identities.
HQ_PUBLIC_METHODS = (HQ_NAME, HQ_FRONTIER_NAME, HQ_ADAPTIVE_NAME, HQ_ROTATION_NAME)
HQ_PAPER_AVERAGE_RATES = (2, 3, 4)
HQ_BLOCK_SIZE = 2
HQ_KEY_SCALE_MODE = "nc"
HQ_VALUE_SCALE_MODE = "ls_iter"
HQ_CODEBOOK_BUNDLE = "fixed_hq_b2_v1"
HQ_FRONTIER_CODEBOOK_BUNDLE = "frontier_hq_b2_v1"
HQ_ADAPTIVE_CODEBOOK_BUNDLE = "adaptive_hq_b2_safe_v1"

# Scale constants for the fixed HQ recipe.  The frontier row has its own
# integer K/V split and UB key scale below.
HQ_FRONTIER_KEY_SCALE_MODE = HQ_KEY_SCALE_MODE
HQ_FRONTIER_VALUE_SCALE_MODE = HQ_VALUE_SCALE_MODE
HQ_FRONTIER_SCALE_KWARGS = MappingProxyType(
    {
        "key_scale_mode": HQ_FRONTIER_KEY_SCALE_MODE,
        "value_scale_mode": HQ_FRONTIER_VALUE_SCALE_MODE,
    }
)
HQ_FIB_INIT_PROXY_NAME = "HQ-Fib-init-proxy"
HQ_BLOCK_SCALE_PROXY_NAME = "HQ-Block-scale-proxy"
# Read-only compatibility for result bundles produced before the proxy rows
# were named honestly.  These aliases must never be emitted by a new run.
HQ_PROXY_LEGACY_ALIASES = MappingProxyType(
    {
        HQ_FIB_INIT_PROXY_NAME: ("FibQuant", "HayateQuant-fibinit"),
        HQ_BLOCK_SCALE_PROXY_NAME: ("BlockQuant", "HayateQuant-BQscale"),
    }
)
HQ_FRACTIONAL_2BIT_NAME = "HayateQuant-frac2"
HQ_ADAPTIVE_2BIT_NAME = "HayateQuant-adaptive2"


def _paper_rate(avg_bits: int | float) -> int | None:
    if isinstance(avg_bits, bool) or not isinstance(avg_bits, (int, float)):
        return None
    value = float(avg_bits)
    if not value.is_integer() or int(value) not in HQ_PAPER_AVERAGE_RATES:
        return None
    return int(value)


def hayate_hq_role_rates(avg_bits: int | float) -> tuple[float, float] | None:
    """Return HQ's globally fixed K/V payload rates in bits per coordinate."""

    rate = _paper_rate(avg_bits)
    if rate is None:
        return None
    return rate + 0.5, rate - 0.5


def hayate_hq_code_bits(avg_bits: int | float) -> tuple[int, int] | None:
    """Return exact fixed-length K/V code bits for one two-coordinate block."""

    rate = _paper_rate(avg_bits)
    if rate is None:
        return None
    return 2 * rate + 1, 2 * rate - 1


def hayate_hq_kwargs(avg_bits: int | float, **overrides: Any) -> dict[str, Any]:
    """Keyword arguments for the paper's canonical fixed HQ recipe.

    Development used one complete seven-checkpoint by three-rate panel, never
    per-model or per-task selection. The frozen short-block recipe is a
    ``b=2`` Gaussian-source block VQ with K/V code bits
    ``(5,3)/(7,5)/(9,7)``, key norm correction, value one-step LS refit,
    rate-aware folded rotation, and one fp16 scale per vector. Every codebook
    remains at or below the default 512-entry cap.
    """

    rate = _paper_rate(avg_bits)
    code_bits = hayate_hq_code_bits(avg_bits)
    if rate is None or code_bits is None:
        raise ValueError(
            "HQ is defined only for the predeclared average rates 2, 3, and 4; "
            f"got {avg_bits!r}"
        )
    key_code_bits, value_code_bits = code_bits
    kwargs: dict[str, Any] = {
        "bits": rate,
        "value_bits": rate,
        "avg_bits": float(rate),
        "block_size": HQ_BLOCK_SIZE,
        "key_code_bits_per_block": key_code_bits,
        "value_code_bits_per_block": value_code_bits,
        "max_codebook_bits": 9,
        "dist": "gaussian",
        "required_codebook_bundle": HQ_CODEBOOK_BUNDLE,
        "rotation_mode": "auto",
        # Keep the formal recipe shape-independent and quality-safe while the
        # fixed T-candidate table is under its pinned WikiText gate.  A table
        # candidate may replace this exact contract only after that gate passes.
        "encoder": "exact",
        "key_scale_mode": HQ_KEY_SCALE_MODE,
        "value_scale_mode": HQ_VALUE_SCALE_MODE,
    }
    kwargs.update(overrides)
    return kwargs


def hayate_frontier_bits(avg_bits: int | float) -> tuple[float, float] | None:
    """Compatibility alias for :func:`hayate_hq_role_rates`."""

    return hayate_hq_role_rates(avg_bits)


def hayate_frontier_kwargs(avg_bits: int | float, **overrides: Any) -> dict[str, Any]:
    """Return the legacy callable fixed-HQ recipe.

    The function name is retained for internal callers; it is not a result
    label.  New benchmark rows use :data:`HQ_NAME` explicitly.
    """

    return hayate_hq_kwargs(avg_bits, **overrides)


def canonical_hayate_method_name(name: str) -> str:
    """Normalize legacy proxy labels for a new benchmark invocation.

    The returned name is always the honest HQ-proxy label.  This preserves old
    CLI/config spellings without creating any new result named as if it were an
    independent FibQuant or BlockQuant implementation.
    """

    for canonical, legacy_names in HQ_PROXY_LEGACY_ALIASES.items():
        if name == canonical or name in legacy_names:
            return canonical
    return name


def hayate_result_method_candidates(name: str) -> tuple[str, ...]:
    """Result keys to try, canonical first and legacy read aliases second."""

    canonical = canonical_hayate_method_name(name)
    return (canonical, *HQ_PROXY_LEGACY_ALIASES.get(canonical, ()))


def build_hayate_hq(avg_bits: int | float, **overrides: Any) -> HayateQuantCodec:
    """Construct the canonical fixed HQ codec."""

    return HayateQuantCodec(**hayate_hq_kwargs(avg_bits, **overrides))


def build_hayate_frontier(avg_bits: int | float, **overrides: Any) -> HayateQuantCodec:
    """Construct fixed HQ for internal callers that still use this helper."""

    return build_hayate_hq(avg_bits, **overrides)


def hayate_frontier_variant(
    avg_bits: int, label: str | None = None, **overrides: Any
) -> dict[str, Any]:
    """Hydra/YAML variant dictionary for long-context benchmark runners."""

    variant = {"label": label or HQ_NAME}
    variant.update(hayate_hq_kwargs(avg_bits, **overrides))
    return variant


def hayate_hq_frontier_kwargs(avg_bits: int | float, **overrides: Any) -> dict[str, Any]:
    """Keyword arguments for the second fixed HQ frontier operating point.

    This recipe must not pass through the generic ``avg_bits`` UI without an
    explicit block budget: that UI defaults to a four-coordinate block and
    would turn the 4-bit point into a 20-bit (``2**20``) codebook.  Besides
    being impractical, that made a production run fail only after model/PPL
    work had completed.  Keep the frontier's stronger key allocation on the
    same audited two-coordinate block contract as HQ instead.
    """

    rate = _paper_rate(avg_bits)
    if rate is None:
        raise ValueError(f"HQ-Frontier is defined only for rates 2, 3, and 4; got {avg_bits!r}")
    # K/V payload rates are (r+1, r-1) bpc, stored as fixed whole-block
    # widths.  The largest point is K10/V6 (K=1024), which is deliberately
    # bounded and feasible for the exact research encoder.
    key_code_bits, value_code_bits = 2 * int(rate) + 2, 2 * int(rate) - 2
    kwargs: dict[str, Any] = {
        "bits": int(rate),
        "value_bits": int(rate),
        "avg_bits": float(rate),
        "block_size": HQ_BLOCK_SIZE,
        "key_code_bits_per_block": key_code_bits,
        "value_code_bits_per_block": value_code_bits,
        "max_codebook_bits": 10,
        "dist": "gaussian",
        "required_codebook_bundle": HQ_FRONTIER_CODEBOOK_BUNDLE,
        "key_scale_mode": "ub",
        "value_scale_mode": HQ_VALUE_SCALE_MODE,
        "rotation_mode": "auto",
        "encoder": "exact",
    }
    kwargs.update(overrides)
    return kwargs


def build_hayate_hq_frontier(avg_bits: int | float, **overrides: Any) -> HayateQuantCodec:
    """Construct the distinct HQ-Frontier operating point."""

    return HayateQuantCodec(**hayate_hq_frontier_kwargs(avg_bits, **overrides))


def hayate_hq_frontier_variant(
    avg_bits: int, label: str | None = None, **overrides: Any
) -> dict[str, Any]:
    variant = {"label": label or HQ_FRONTIER_NAME}
    variant.update(hayate_hq_frontier_kwargs(avg_bits, **overrides))
    return variant


def hayate_fractional2_kwargs(**overrides: Any) -> dict[str, Any]:
    """Keyword arguments for the experimental 2-bit fractional block-rate HQ.

    This keeps the same average payload rate as 2-bit HQ, but uses
    block-size-four fixed-length VQ codes with K10/V6 bits per block, i.e.
    2.5 key bits/dim and 1.5 value bits/dim before the shared fp16 scale.
    Short probes on Llama-3.1-8B, Qwen3-4B, and R1-Qwen-1.5B showed lower
    WikiText-2 PPL than the historical integer split at the same stored bits. It remains an
    opt-in research operating point rather than the canonical HQ definition.
    """

    kwargs = hayate_fractional_rate_kwargs(
        avg_bits=2,
        block_size=4,
        key_bit_fraction=0.625,
        max_codebook_bits=10,
    )
    kwargs.update(overrides)
    return kwargs


def hayate_fractional_rate_kwargs(
    avg_bits: float,
    *,
    block_size: int = 4,
    key_bit_fraction: float = 0.625,
    adaptive: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    """Human-facing fixed-budget fractional block-rate HQ kwargs.

    ``avg_bits`` is the average K/V payload rate per coordinate.  The helper
    converts it to the integer number of fixed-length bits stored for one
    key/value block pair, avoiding manual K10/V6-style bookkeeping.  With the
    defaults, ``avg_bits=2`` and ``block_size=4`` give a 16-bit K/V pair
    budget; ``key_bit_fraction=0.625`` resolves to K10/V6.  Set
    ``adaptive=True`` to let the codec choose the K/V split from cheap online
    statistics under the same total budget, then quantize once.
    """

    nominal = max(1, min(8, int(round(float(avg_bits)))))
    kwargs: dict[str, Any] = {
        "bits": nominal,
        "value_bits": nominal,
        "avg_bits": float(avg_bits),
        "block_size": int(block_size),
        "key_bit_fraction": float(key_bit_fraction),
        "max_codebook_bits": 9,
        "dist": "gaussian",
        "rotation_mode": "auto",
        "key_scale_mode": HQ_KEY_SCALE_MODE,
        "value_scale_mode": HQ_VALUE_SCALE_MODE,
    }
    if adaptive:
        kwargs["adaptive_bit_split"] = "online_stats"
        # Adaptive HQ changes only the causal K/V bit selector and inherits the
        # same quality-safe assignment contract as fixed HQ.
        kwargs["encoder"] = "exact"
    kwargs.update(overrides)
    return kwargs


def build_hayate_fractional_rate(avg_bits: float, **overrides: Any) -> HayateQuantCodec:
    """Construct a fractional block-rate HQ codec from average-bit UI knobs."""

    return HayateQuantCodec(**hayate_fractional_rate_kwargs(avg_bits, **overrides))


def hayate_fractional_rate_variant(
    avg_bits: float,
    label: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Hydra/YAML variant dictionary for fractional block-rate HQ."""

    variant = {"label": label or f"HayateQuant-frac-{float(avg_bits):g}bit"}
    variant.update(hayate_fractional_rate_kwargs(avg_bits, **overrides))
    return variant


def build_hayate_fractional2(**overrides: Any) -> HayateQuantCodec:
    """Construct the experimental 2-bit fractional block-rate HQ codec."""

    return HayateQuantCodec(**hayate_fractional2_kwargs(**overrides))


def hayate_fractional2_variant(label: str | None = None, **overrides: Any) -> dict[str, Any]:
    """Hydra/YAML variant dictionary for the fractional 2-bit HQ candidate."""

    variant = {"label": label or f"{HQ_FRACTIONAL_2BIT_NAME}-2bit"}
    variant.update(hayate_fractional2_kwargs(**overrides))
    return variant


def hayate_adaptive2_kwargs(**overrides: Any) -> dict[str, Any]:
    """Keyword arguments for opt-in stats-adaptive 2-bit K/V bit splitting.

    ``2-bit`` denotes the fixed average *code* budget.  The candidate-pair
    selector is embedded in low fraction bits of the first existing fp16
    value scale, reducing that scale's precision without adding data-plane
    bits.
    """

    fixed_key_bits, fixed_value_bits = hayate_hq_code_bits(2)
    kwargs = hayate_fractional_rate_kwargs(
        avg_bits=2,
        block_size=HQ_BLOCK_SIZE,
        key_bit_fraction=fixed_key_bits / (fixed_key_bits + fixed_value_bits),
        adaptive=True,
    )
    kwargs.update(
        {
            "adaptive_bit_margin": 4,
            "key_error_weight": 4.0,
            "value_error_weight": 1.0,
            "required_codebook_bundle": HQ_ADAPTIVE_CODEBOOK_BUNDLE,
        }
    )
    kwargs.update(overrides)
    return kwargs


def build_hayate_adaptive2(**overrides: Any) -> HayateQuantCodec:
    """Construct the opt-in stats-adaptive 2-bit HQ codec."""

    return HayateQuantCodec(**hayate_adaptive2_kwargs(**overrides))


def hayate_adaptive2_variant(label: str | None = None, **overrides: Any) -> dict[str, Any]:
    """Hydra/YAML variant dictionary for opt-in stats-adaptive 2-bit HQ."""

    variant = {"label": label or f"{HQ_ADAPTIVE_2BIT_NAME}-2bit"}
    variant.update(hayate_adaptive2_kwargs(**overrides))
    return variant


def hayate_adaptive_rate_kwargs(
    avg_bits: float,
    *,
    block_size: int | None = None,
    key_bit_fraction: float | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Keyword arguments for the paper's opt-in Adaptive HQ row.

    Adaptive HQ keeps a fixed total K/V code-bit budget for each matched
    block pair and selects the key/value split from cache-local, layer-local
    causal raw-power statistics and actual data-free codebook distortion
    constants before quantizing once. Its data-dependent candidate-pair
    selector occupies low fraction bits of the first existing fp16 value
    scale; the scale is reconstructed after clearing those bits, so the total
    logical rate remains exactly matched. The immature/default incumbent is
    the matching fixed-HQ split (K5/V3, K7/V5, or K9/V7); the symmetric pair is
    entered only through the codec's frozen conservative margin/hysteresis.
    The block size is the same short ``b=2`` used by fixed HQ at every paper
    rate; only the causal split selector is added.
    """

    rate = _paper_rate(avg_bits)
    if rate is None:
        raise ValueError(
            "Adaptive HQ is defined only for the predeclared average rates "
            f"2, 3, and 4; got {avg_bits!r}"
        )
    avg = float(rate)
    if block_size is None:
        block_size = HQ_BLOCK_SIZE
    if key_bit_fraction is None:
        fixed_key_bits, fixed_value_bits = hayate_hq_code_bits(rate)
        key_bit_fraction = fixed_key_bits / (fixed_key_bits + fixed_value_bits)
    kwargs = hayate_fractional_rate_kwargs(
        avg_bits=avg,
        block_size=int(block_size),
        key_bit_fraction=key_bit_fraction,
        adaptive=True,
    )
    kwargs.update(
        {
            "adaptive_bit_margin": 4,
            "key_error_weight": 4.0,
            "value_error_weight": 1.0,
            "required_codebook_bundle": HQ_ADAPTIVE_CODEBOOK_BUNDLE,
        }
    )
    kwargs.update(overrides)
    return kwargs


def build_hayate_adaptive_rate(avg_bits: float, **overrides: Any) -> HayateQuantCodec:
    """Construct the paper's opt-in Adaptive HQ codec."""

    return HayateQuantCodec(**hayate_adaptive_rate_kwargs(avg_bits, **overrides))


def hayate_adaptive_rate_variant(
    avg_bits: float,
    label: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Hydra/YAML variant dictionary for the paper's Adaptive HQ row."""

    variant = {"label": label or HQ_ADAPTIVE_NAME}
    variant.update(hayate_adaptive_rate_kwargs(avg_bits, **overrides))
    return variant


__all__ = [
    "HQ_NAME",
    "HQ_FRONTIER_NAME",
    "HQ_PUBLIC_METHODS",
    "HQ_PAPER_AVERAGE_RATES",
    "HQ_BLOCK_SIZE",
    "HQ_KEY_SCALE_MODE",
    "HQ_VALUE_SCALE_MODE",
    "HQ_CODEBOOK_BUNDLE",
    "HQ_FRONTIER_CODEBOOK_BUNDLE",
    "HQ_ADAPTIVE_CODEBOOK_BUNDLE",
    "HQ_FRONTIER_KEY_SCALE_MODE",
    "HQ_FRONTIER_VALUE_SCALE_MODE",
    "HQ_FRONTIER_SCALE_KWARGS",
    "HQ_FIB_INIT_PROXY_NAME",
    "HQ_BLOCK_SCALE_PROXY_NAME",
    "HQ_PROXY_LEGACY_ALIASES",
    "HQ_FRACTIONAL_2BIT_NAME",
    "HQ_ADAPTIVE_2BIT_NAME",
    "HQ_ADAPTIVE_NAME",
    "HQ_ROTATION_NAME",
    "hayate_hq_role_rates",
    "hayate_hq_code_bits",
    "hayate_hq_kwargs",
    "hayate_frontier_bits",
    "hayate_frontier_kwargs",
    "canonical_hayate_method_name",
    "hayate_result_method_candidates",
    "build_hayate_frontier",
    "build_hayate_hq",
    "hayate_frontier_variant",
    "hayate_hq_frontier_kwargs",
    "build_hayate_hq_frontier",
    "hayate_hq_frontier_variant",
    "hayate_fractional_rate_kwargs",
    "build_hayate_fractional_rate",
    "hayate_fractional_rate_variant",
    "hayate_fractional2_kwargs",
    "build_hayate_fractional2",
    "hayate_fractional2_variant",
    "hayate_adaptive2_kwargs",
    "build_hayate_adaptive2",
    "hayate_adaptive2_variant",
    "hayate_adaptive_rate_kwargs",
    "build_hayate_adaptive_rate",
    "hayate_adaptive_rate_variant",
]

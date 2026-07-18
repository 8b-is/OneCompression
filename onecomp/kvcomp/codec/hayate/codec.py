"""HayateQuant: block-vector-quantized, hybrid-rotation KV codec.

HayateQuant is a **data-free block-VQ generalization of TurboQuant**
(arXiv:2504.19874) that targets part of TurboQuant's stated distortion slack
and combines HALO's storage-neutral research levers in one codec. TurboQuant
and HALO remain independent baselines; HayateQuant does not replace them by
definition.

Recap: TurboQuant rotates a KV vector by a random orthogonal transform so its
coordinates are identically Beta-distributed for finite ``d`` and become
approximately independent Gaussian coordinates as ``d -> inf`` (finite-sphere
coordinates are dependent through the fixed norm). It then fits a **scalar**
Lloyd-Max quantizer to that marginal law, *entirely
data-free* (the codebook only depends on the bit-width, not on any real KV
data). The paper proves this is within a constant factor
(``<= pi*sqrt(3)/2 ~= 2.72x``) of the information-theoretic
distortion-rate optimum. Classical
high-rate quantization theory (Gersho & Gray, *Vector Quantization and Signal
Compression*, 1992; Zamir, *Lattice Coding for Signals and Networks*, 2014)
identifies one relevant source of rate-distortion loss: **independent scalar
quantization of a memoryless Gaussian source loses the "space-filling" and
"shape" gains available to any joint (vector/lattice) quantizer of dimension
> 1** -- a classical ``1.53`` dB (``~0.255`` bit/dim) gap at high rate that
only closes as the joint quantization dimension grows (``E8``: ``~0.65`` dB
gap remaining; this repo's own HALO analysis flags exactly this as unexplored
"Lever 5", see ``benchmark/kvcomp-longcontext/paper/HALO_ENHANCED_REPORT.md``).

HayateQuant targets this gap with two data-free, storage-neutral additions on
top of TurboQuant's own skeleton:

1. **Block vector quantization (the core contribution).** Instead of an
   independent scalar Lloyd-Max codebook, HayateQuant fits a
   ``block_size``-dimensional **vector** codebook (``2**(bits*block_size)``
   codewords) via Lloyd's algorithm (batched K-means, :mod:`._vq`) to
   *synthetic samples drawn exactly from TurboQuant's own source law* --
   i.i.d. ``N(0,1)^{block_size}`` (the ``d -> inf`` limit) or the exact
   finite-``d`` ``block_size``-dim marginal of a random point on ``S^{d-1}``.
   This is data-free in exactly the sense TurboQuant's own scalar codebook is
   data-free (fit offline to the theoretical law, not to real KV) -- just
   generalized from dimension 1 to ``block_size``. Nearest-codeword search is
   a single GEMM (``z @ cb.T``) + ``argmax`` (:func:`._vq.quantize_to_vq_codebook`),
   so encode/decode remain O(1) matmuls per block; no explicit lattice
   geometry (e.g. ``E8``) or per-lattice decoding algorithm is required, and
   the codebook is *fit* on the target source law. Lloyd fitting is a local,
   finite-sample optimization, so quality against a fixed lattice is an
   empirical comparison rather than a universal guarantee. The total number of
   stored bits is unchanged (``bits`` per coordinate, same as TurboQuant --
   see ``_stored_bits``); only *how* those bits are allocated changes (jointly
   per block instead of independently per coordinate), which is exactly the
   controlled comparison needed to isolate the vector-quantization gain.
   ``block_size=1`` reduces the block codebook to the ordinary scalar
   Lloyd-Max quantizer.  For the exact Beta law, TurboQuant stores an L2
   scale while HayateQuant uses the mathematically equivalent RMS scale plus
   a ``sqrt(d)``-rescaled codebook; after signed-fp16 scale rounding their
   codes and rate match exactly but their reconstructed tensors need not be
   bit-identical.  The scalar-control test bounds the resulting vectorwise
   relative difference at the output dtype's machine epsilon.

2. **Hybrid rotation (resolves the Hadamard-vs-Dense trade-off).**
   TurboQuant/HALO's own benchmarks (``benchmark/kvcomp-codec/README.md``)
   found the *sequential* Hadamard transform gives marginally better
   Gaussianisation on real, anisotropic KV than a single Haar-random dense
   rotation (better incoherence => better codebook fit), while dense/folded-
   Hadamard-GEMM is 3-4x faster (one BLAS GEMM vs. a memory-bound butterfly).
   Block-VQ raises the stakes on this trade-off: the codebook is fit to an
   *isotropic* ``block_size``-dim law, so each rotated block must itself be
   close to isotropic for the codebook to fit well -- a fixed Hadamard
   transform alone (a deterministic matrix, randomized only by a global sign
   flip) is not Haar-random at the block level, so directly block-quantizing
   its output would mismatch the codebook. HayateQuant's ``rotation_mode=
   "hybrid"`` resolves this with a **two-level rotation**:

       R = HS_D  (global signed-Hadamard-GEMM, O(D log D) or one O(D^2) GEMM)
           followed by
           block-diagonal(Q_1, ..., Q_{D/b})  (independent dense Haar-random
           rotations, one per block, O(D * b), b << D)

   Composing two orthogonal transforms is orthogonal (norm-preserving,
   invertible via the transposes in reverse order, :mod:`.rotation`
   ``hybrid_rotate``/``hybrid_unrotate``). The explicit two-stage reference
   transform costs ``O(D log D + D*b)``. The shipped runtime instead folds the
   two matrices offline into one BLAS-friendly ``D x D`` GEMM, so its online
   asymptotic cost is ``O(D^2)``--the same as dense QR rotation--while avoiding
   two launches. It therefore makes no sub-quadratic runtime claim. The block
   size controls the local randomization and VQ dimension, but is not by itself
   a measured guarantee of better end-to-end accuracy.

3. **HALO's validated, storage-neutral levers (inherited, not reinvented).**
   Role-asymmetric reconstruction scale (:mod:`._scale`): keys use the
   norm-correcting (``"nc"``) scale (preserves the attention logit, shrunk by
   ``cos(theta)`` instead of ``cos^2(theta)``), values use the least-squares
   (``"ls"``/``"ls_iter"``) scale (the correct MSE objective for values, which
   are consumed linearly). Outlier mixed-precision (``n_outlier``/
   ``outlier_bits``) for the nominal 2.5/3.5 operating points at head_dim 128;
   a data-dependent implementation must additionally store its channel bitmap,
   so those labels are not exact realised rates.
   Optional QJL-style 1-bit residual (``use_qjl``, off by default; its
   high-variance attention effect requires separate downstream validation).
   These levers are dimension-agnostic (:func:`._scale.fit_scale` reduces
   over the concatenated per-block codewords exactly as it would over
   concatenated scalar centroids), so they compose for free with block-VQ.

Throughout this module, ``stored_bits`` means only fixed-width encoded-KV
data-plane fields: codes, signed-fp16 scales, and enabled outlier bitmap/QJL
fields.  Self-describing Python tags (rotation, distribution, codebook recipe,
seed, shapes) and container bytes are uncharged out-of-band/cache-level control
state, as are global codebooks, LUTs, rotations, and calibration objects.  The
materialized index tensors are not bit packed, so ``stored_bits`` is not their
physical resident memory or a packed-serving measurement.

**HayateQuant contains the scalar TurboQuant path and HALO's default
non-preconditioned scale path**: setting
``block_size=1`` and ``rotation_mode`` to ``"hadamard"``/``"hadamard_gemm"``/
``"dense"`` with ``key_scale_mode=value_scale_mode="rms"`` reproduces
TurboQuant's scalar math (and is bit-identical for the Gaussian/RMS
parameterization).  The finite-Beta L2/RMS parameterizations can differ only
at the stored-fp16 scale rounding described above.  Adding the same
role-asymmetric scale modes reproduces HALO's quantization equations when
HALO's optional centering/whitening is disabled. The
principal additional lever is
block-VQ (default on, via ``block_size=None`` auto-resolution), which is
exactly the "Lever 5 / future work" gap this repository's own HALO analysis
identified and left unimplemented.

See ``docs/design/kvcomp-architecture.md`` (HayateQuant section) and
``papers/hayatequant_paper/README.md`` for the full derivation, and
``benchmark/kvcomp-longcontext/paper/tests/verify_hayatequant_vq_gain.py`` /
``verify_hayatequant_isotropy.py`` for CPU-only, data-free checks on sampled
source draws: (a) block-VQ distortion at equal bit-rate as ``block_size`` grows
from 1, and (b) the hybrid rotation's measured block isotropy relative to the
Hadamard-only rotation. These checks are regression evidence, not universal
end-to-end quality proofs.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from functools import lru_cache

import torch

from ..._base import KVCodec, CompressedKV
from ..._registry import register_kv_codec
from ..rotation import (
    is_power_of_two,
    random_sign,
    random_orthogonal,
    rotate,
    unrotate,
    signed_hadamard_matrix,
    lloyd_max_gaussian,
    lloyd_max_beta,
    hybrid_rotation_matrix,
)
from .vq import (
    resolve_block_size,
    lloyd_vq_gaussian_block,
    lloyd_vq_beta_block,
    lloyd_vq_gaussian_codebook,
    lloyd_vq_beta_codebook,
    quantize_and_lookup,
    quantize_exact_pre,
    lookup_codewords,
    lut_spec_for,
    build_vq_lut,
    quantize_via_lut,
    gaussian_codebook_artifact_provenance,
    require_shipped_gaussian_codebook_bundle,
)
from .._scale import SCALE_MODES, resolve_scale_mode, fit_scale
from ..turboquant import _qjl_projection_on

ADAPTIVE_HQ_POLICY_VERSION = "adaptive_hq_cache_causal_fixed_incumbent_hysteresis_v6"
ADAPTIVE_HQ_DISTORTION_TABLE_VERSION = (
    "source_qmc128_exact_codebook_fp16scale_safe_two_pair_update_carrier_v4"
)
ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION = "value_scale_fp16_fraction_lsb_v1"
ADAPTIVE_HQ_OBJECTIVE = "causal_raw_power_times_qmc_base_plus_one_carrier_per_update"
_ADAPTIVE_QMC_VECTORS = 128
# The adaptive table is a deterministic, data-free 128-vector QMC estimate.
# Use one nominal 1/sqrt(N) sampling scale as a *predeclared* robustness margin
# before departing from fixed HQ.  This value follows only from the frozen QMC
# design above; it was not fitted to a model, task, or post-hoc benchmark row.
ADAPTIVE_HQ_SWITCH_MARGIN_FRACTION = 1.0 / math.sqrt(_ADAPTIVE_QMC_VECTORS)
ADAPTIVE_HQ_SWITCH_MARGIN_BASIS = "one_over_sqrt_fixed_qmc_vectors_128"
ADAPTIVE_HQ_HYSTERESIS = (
    "enter_challenger_only_beyond_margin_retain_while_strictly_better_"
    "fallback_to_fixed_incumbent_on_tie_or_loss"
)
# The exact/LUT implementation is audited through 2**15 entries.  Larger
# codebooks are not a benign quality knob: they can turn a typo in a rate UI
# into an infeasible first-use Lloyd fit or an OOM after a long model run.
MAX_AUDITED_CODEBOOK_BITS = 15


@lru_cache(maxsize=16)
def _scalar_codebook(bits: int, dist: str, m: int) -> torch.Tensor:
    if dist == "beta":
        # ``lloyd_max_beta`` lives on the *unit-sphere* coordinate scale
        # ([-1, 1], TurboQuant's L2-normalisation convention); HayateQuant
        # normalises by the RMS (= L2 / sqrt(m)) everywhere, so rescale the
        # codebook by sqrt(m) to match.  Decision boundaries scale with the
        # codewords, so the assigned codes and real-arithmetic reconstruction
        # match TurboQuant's L2-normalised beta path.  The stored fp16 scale is
        # parameterized differently (RMS here versus L2 in TurboQuant), so its
        # final rounding does not guarantee bit-identical reconstruction. This
        # convention is consistent with ``lloyd_vq_beta_block``.
        return lloyd_max_beta(bits, m) * math.sqrt(m)
    if dist == "gaussian":
        return lloyd_max_gaussian(bits)
    raise ValueError(f"unknown HayateQuant source law {dist!r}")


@lru_cache(maxsize=64)
def _block_codebook(
    bits: int,
    block_size: int,
    dist: str,
    m: int,
    codebook_init: str = "auto",
) -> torch.Tensor:
    """``(2**(bits*block_size), block_size)`` data-free block-VQ codebook.

    ``block_size=1`` returns the ordinary scalar Lloyd-Max codebook reshaped
    to ``(2**bits, 1)``.  It reduces to the scalar quantizer exactly; for the
    Beta convention, see :func:`_scalar_codebook` for the harmless fp16
    scale-parameterization rounding distinction from TurboQuant.
    """
    if block_size == 1:
        return _scalar_codebook(bits, dist, m).unsqueeze(-1)
    if dist == "beta":
        return lloyd_vq_beta_block(bits, block_size, m, init=codebook_init)
    if dist == "gaussian":
        return lloyd_vq_gaussian_block(bits, block_size, init=codebook_init, ambient_dim=m)
    raise ValueError(f"unknown HayateQuant source law {dist!r}")


@lru_cache(maxsize=64)
def _block_codebook_by_code_bits(
    code_bits: int,
    block_size: int,
    dist: str,
    m: int,
    codebook_init: str = "auto",
) -> torch.Tensor:
    """Data-free block codebook with explicit stored bits per block.

    This is the fractional-rate counterpart to :func:`_block_codebook`.
    ``code_bits == bits * block_size`` reproduces the ordinary HayateQuant
    codebook, while other integer ``code_bits`` values define fixed-length
    rates ``code_bits / block_size`` without changing the random-access cache
    contract.
    """
    code_bits = int(code_bits)
    if not 1 <= code_bits <= MAX_AUDITED_CODEBOOK_BITS:
        raise ValueError(
            f"code_bits must be in 1..{MAX_AUDITED_CODEBOOK_BITS}, got {code_bits}"
        )
    if block_size == 1:
        return _scalar_codebook(code_bits, dist, m).unsqueeze(-1)
    if dist == "beta":
        return lloyd_vq_beta_codebook(code_bits, block_size, m, init=codebook_init)
    if dist == "gaussian":
        return lloyd_vq_gaussian_codebook(code_bits, block_size, init=codebook_init, ambient_dim=m)
    raise ValueError(f"unknown HayateQuant source law {dist!r}")


@lru_cache(maxsize=256)
def _adaptive_datafree_relative_distortion(
    code_bits: int,
    block_size: int,
    dist: str,
    m: int,
    codebook_init: str,
    scale_mode: str,
    scale_fraction_lsb_clear_bits: int = 0,
) -> float:
    """Deterministic source-law distortion of one actual HQ codebook.

    Adaptive HQ must not rank candidate rates with the old high-rate
    ``2**(-2R)`` surrogate: that expression ignores finite-rate codebook
    geometry and the role-specific reconstruction scale.  This cached table
    entry instead evaluates the *actual* data-free codebook, exact CPU
    nearest-codeword assignment, requested scale law, one LS refinement when
    applicable, and the signed-fp16 scale rounding charged by ``stored_bits``.
    With ``scale_fraction_lsb_clear_bits > 0`` it clears those bits in every
    scale in the QMC panel.  Subtracting the carrier-free result therefore
    estimates the signed distortion delta for one carrier-bearing vector.
    Adaptive HQ divides that delta by the current update's actual vector count,
    because the payload carries exactly one selector per update.  This keeps
    the source-law proxy deterministic while matching the carrier's update-size
    amortization instead of implicitly assuming a 128-vector update.

    The source expectation is approximated by a fixed scrambled-Sobol panel
    of :data:`_ADAPTIVE_QMC_VECTORS` full vectors.  It uses no model activation
    or request data, is deterministic across cache instances, and is immutable
    global codec state (unlike the causal raw-power accumulator, which must be
    owned by each cache/layer).  ``ADAPTIVE_HQ_DISTORTION_TABLE_VERSION`` makes
    this finite QMC contract explicit in result provenance.
    """
    code_bits = int(code_bits)
    block_size = int(block_size)
    m = int(m)
    carrier_bits = int(scale_fraction_lsb_clear_bits)
    if carrier_bits < 0 or carrier_bits > 10:
        raise ValueError(
            "adaptive distortion scale carrier must clear 0..10 fp16 fraction "
            f"bits, got {scale_fraction_lsb_clear_bits!r}"
        )
    if m <= 0 or m % block_size:
        raise ValueError(f"adaptive distortion needs block_size={block_size} to divide m={m}")
    # A fixed low-discrepancy panel is substantially more stable than a small
    # pseudo-random Monte-Carlo sample at the same one-time initialization
    # cost.  Clamp before the inverse normal CDF to keep every source finite.
    sobol = torch.quasirandom.SobolEngine(dimension=m, scramble=True, seed=20260716)
    uniform = sobol.draw(_ADAPTIVE_QMC_VECTORS, dtype=torch.float64).clamp_(1e-12, 1.0 - 1e-12)
    source = (math.sqrt(2.0) * torch.erfinv(2.0 * uniform - 1.0)).to(torch.float32)
    if dist == "beta":
        source = source / source.norm(dim=-1, keepdim=True).clamp_min(1e-12) * math.sqrt(m)
    elif dist != "gaussian":
        raise ValueError(f"unknown HayateQuant source law {dist!r}")

    cb = _block_codebook_by_code_bits(code_bits, block_size, dist, m, codebook_init).to(
        dtype=torch.float32, device="cpu"
    )
    n_blocks = m // block_size
    rms = source.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)

    def _lookup(vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        blocks = vector.reshape(vector.shape[0], n_blocks, block_size)
        codes, chat_blocks = quantize_and_lookup(blocks, cb)
        return codes, chat_blocks.reshape_as(vector)

    def _requant(vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        codes, chat = _lookup(vector)
        return chat, codes

    codes, chat = _lookup(source / rms)
    scale, refined_codes = fit_scale(
        source,
        chat,
        scale_mode,
        rms,
        requantize_fn=_requant,
    )
    if refined_codes is not None:
        chat = lookup_codewords(cb, refined_codes).reshape_as(source)
    stored_scale = scale.to(torch.float16).clone().contiguous()
    if carrier_bits:
        words = stored_scale.view(torch.int16).reshape(-1)
        unsigned = words.to(torch.int32) & 0xFFFF
        words.copy_((unsigned & (0xFFFF ^ ((1 << carrier_bits) - 1))).to(torch.int16))
    reconstruction = chat * stored_scale.to(torch.float32)
    residual64 = (source - reconstruction).to(torch.float64)
    source64 = source.to(torch.float64)
    relative = residual64.square().sum() / source64.square().sum().clamp_min(1e-24)
    value = float(relative)
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError(
            "non-finite adaptive data-free distortion for "
            f"code_bits={code_bits}, block_size={block_size}, dist={dist}, "
            f"scale_mode={scale_mode}"
        )
    return max(value, 1e-12)


def _adaptive_value_distortion_for_update(constants: dict, update_vectors: int) -> float:
    """Amortize exactly one value-scale carrier over an encode update.

    ``delta`` is deliberately signed: clearing an fp16 fraction bit can very
    occasionally reduce squared error for a finite QMC panel.  Clamping it
    would bias the frozen rate--distortion proxy.  The final effective
    distortion, however, must remain finite and positive.
    """

    if (
        isinstance(update_vectors, bool)
        or not isinstance(update_vectors, int)
        or update_vectors <= 0
    ):
        raise ValueError("Adaptive HQ update vector count must be a positive integer")
    try:
        base = float(constants["value_relative_distortion_base"])
        delta = float(constants["value_selector_carrier_relative_distortion_delta_per_vector"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Adaptive HQ value distortion constants are malformed") from exc
    effective = base + delta / float(update_vectors)
    if not math.isfinite(base) or base <= 0.0 or not math.isfinite(delta):
        raise ValueError("Adaptive HQ value distortion constants must be finite")
    if not math.isfinite(effective) or effective <= 0.0:
        raise ValueError("Adaptive HQ effective value distortion must be finite and positive")
    return effective


def _adaptive_hysteretic_choice(
    scored: list[dict],
    incumbent_pair: tuple[int, int],
    previous_pair: tuple[int, int] | None,
    *,
    used_stats: bool,
) -> tuple[dict, dict]:
    """Select a two-pair AdaptiveHQ candidate with a fixed-HQ safety band.

    The fixed HQ pair is always used before the causal statistic matures.  A
    cache that currently uses fixed HQ enters the symmetric challenger only
    when its predicted objective improves by at least the frozen QMC margin.
    Once entered, the challenger is retained while it is *strictly* better;
    a tie or loss immediately falls back to fixed HQ.  The unequal entry/exit
    thresholds form the hysteresis band and avoid boundary chatter without
    accepting a proxy-predicted degradation relative to fixed HQ.
    """

    by_pair = {tuple(row["pair"]): row for row in scored}
    if len(by_pair) != len(scored) or incumbent_pair not in by_pair:
        raise ValueError("AdaptiveHQ scored candidates must uniquely contain the incumbent")
    if len(scored) not in (1, 2):
        raise ValueError("the frozen AdaptiveHQ safety policy supports one or two candidates")
    incumbent = by_pair[incumbent_pair]
    incumbent_objective = float(incumbent["weighted_objective"])
    if not math.isfinite(incumbent_objective) or incumbent_objective < 0.0:
        raise ValueError("AdaptiveHQ incumbent objective must be finite and non-negative")

    diagnostics = {
        "incumbent_pair": [int(incumbent_pair[0]), int(incumbent_pair[1])],
        "previous_pair": (
            None if previous_pair is None else [int(previous_pair[0]), int(previous_pair[1])]
        ),
        "switch_margin_fraction": float(ADAPTIVE_HQ_SWITCH_MARGIN_FRACTION),
        "switch_margin_basis": ADAPTIVE_HQ_SWITCH_MARGIN_BASIS,
        "hysteresis": ADAPTIVE_HQ_HYSTERESIS,
        "incumbent_objective": incumbent_objective,
        "challenger_objective": None,
        "challenger_relative_improvement": None,
    }
    if not used_stats:
        diagnostics["decision_reason"] = "immature_fixed_hq_incumbent"
        return incumbent, diagnostics
    if len(scored) == 1:
        diagnostics["decision_reason"] = "singleton_fixed_hq_incumbent"
        return incumbent, diagnostics

    challenger_pair = next(pair for pair in by_pair if pair != incumbent_pair)
    challenger = by_pair[challenger_pair]
    challenger_objective = float(challenger["weighted_objective"])
    if not math.isfinite(challenger_objective) or challenger_objective < 0.0:
        raise ValueError("AdaptiveHQ challenger objective must be finite and non-negative")
    relative_improvement = (
        0.0
        if incumbent_objective == 0.0
        else (incumbent_objective - challenger_objective) / incumbent_objective
    )
    diagnostics["challenger_objective"] = challenger_objective
    diagnostics["challenger_relative_improvement"] = float(relative_improvement)

    if previous_pair == challenger_pair and challenger_objective < incumbent_objective:
        diagnostics["decision_reason"] = "retain_strictly_better_challenger"
        return challenger, diagnostics
    entry_limit = incumbent_objective * (1.0 - ADAPTIVE_HQ_SWITCH_MARGIN_FRACTION)
    if (
        previous_pair != challenger_pair
        and challenger_objective < incumbent_objective
        and challenger_objective <= entry_limit
    ):
        diagnostics["decision_reason"] = "enter_challenger_beyond_frozen_margin"
        return challenger, diagnostics
    diagnostics["decision_reason"] = (
        "fallback_to_fixed_hq_incumbent"
        if previous_pair == challenger_pair
        else "fixed_hq_incumbent_within_safety_margin"
    )
    return incumbent, diagnostics


@lru_cache(maxsize=16)
def _dense_rotation(dim: int, seed: int) -> torch.Tensor:
    return random_orthogonal(dim, seed)


@lru_cache(maxsize=64)
def _dense_rotation_on(dim: int, seed: int, device_str: str, dtype) -> torch.Tensor:
    """Device-resident dense orthogonal matrix, cached per (dim, seed, device, dtype)."""
    return _dense_rotation(dim, seed).to(device=torch.device(device_str), dtype=dtype)


@lru_cache(maxsize=64)
def _block_codebook_on(
    bits: int,
    block_size: int,
    dist: str,
    m: int,
    codebook_init: str,
    device_str: str,
) -> torch.Tensor:
    """Device-resident block codebook, cached so the host->device copy happens once."""
    return _block_codebook(bits, block_size, dist, m, codebook_init).to(torch.device(device_str))


@lru_cache(maxsize=64)
def _block_codebook_code_bits_on(
    code_bits: int,
    block_size: int,
    dist: str,
    m: int,
    codebook_init: str,
    device_str: str,
) -> torch.Tensor:
    """Device-resident explicit-code-bits block codebook."""
    return _block_codebook_by_code_bits(code_bits, block_size, dist, m, codebook_init).to(
        torch.device(device_str)
    )


@lru_cache(maxsize=64)
def _block_lut_on(
    bits: int,
    block_size: int,
    dist: str,
    m: int,
    codebook_init: str,
    bins: int,
    top_t: int,
    device_str: str,
) -> torch.Tensor:
    """Device-resident candidate-list LUT for the block codebook.

    The build pays the exact O(cells * K) scoring a single time per
    (codebook, resolution, device); every subsequent encode is an O(T) exact
    argmin over the cell's candidate list (see :func:`._vq.build_vq_lut`).
    """
    cb = _block_codebook_on(bits, block_size, dist, m, codebook_init, device_str)
    return build_vq_lut(cb, bins, top_t)


@lru_cache(maxsize=64)
def _block_lut_code_bits_on(
    code_bits: int,
    block_size: int,
    dist: str,
    m: int,
    codebook_init: str,
    bins: int,
    top_t: int,
    device_str: str,
) -> torch.Tensor:
    """Device-resident candidate-list LUT for an explicit-code-bit codebook."""
    cb = _block_codebook_code_bits_on(code_bits, block_size, dist, m, codebook_init, device_str)
    return build_vq_lut(cb, bins, top_t)


@lru_cache(maxsize=64)
def _codebook_aux_on(
    bits: int,
    block_size: int,
    dist: str,
    m: int,
    codebook_init: str,
    device_str: str,
):
    """Precomputed per-codebook constants for the fused round-trip path.

    Returns ``(cb32, cb_score, csq, bounds)``:

    * ``cb32``: the fp32 device codebook used for reconstruction gathers
      (identical dtype/precision to the standard encode/decode path).
    * ``cb_score``/``csq``: the scoring codebook (fp16 on CUDA, mirroring the
      tensor-core GEMM inside :func:`.vq.quantize_to_vq_codebook`) and its
      squared norms ``cb_score.pow(2).sum(1)``.
    * ``bounds``: the sorted scalar decision boundaries (``block_size == 1``
      only, else ``None``) -- Lloyd-Max centroids are sorted, so a plain
      ``searchsorted`` reproduces the scalar path exactly.

    The per-token encode is launch-bound (GPU-busy time is a fraction of
    wall-clock time), so hoisting these small per-call kernels (dtype casts,
    the ``csq`` reduction, boundary construction, host->device codebook
    moves) out of the hot loop is a measurable win on its own and a
    precondition for capturing the remainder in a single CUDA graph.
    """
    cb32 = _block_codebook_on(bits, block_size, dist, m, codebook_init, device_str)
    if block_size == 1:
        c1 = cb32.squeeze(-1)
        bounds = (c1[1:] + c1[:-1]) / 2.0
        return cb32, cb32, None, bounds.contiguous()
    if torch.device(device_str).type == "cuda":
        cb_score = cb32.half()
    else:
        cb_score = cb32
    csq = cb_score.pow(2).sum(dim=1)
    return cb32, cb_score, csq, None


@lru_cache(maxsize=64)
def _codebook_aux_code_bits_on(
    code_bits: int,
    block_size: int,
    dist: str,
    m: int,
    codebook_init: str,
    device_str: str,
):
    """Precomputed constants for explicit-code-bit codebooks."""
    cb32 = _block_codebook_code_bits_on(code_bits, block_size, dist, m, codebook_init, device_str)
    if block_size == 1:
        c1 = cb32.squeeze(-1)
        bounds = (c1[1:] + c1[:-1]) / 2.0
        return cb32, cb32, None, bounds.contiguous()
    cb_score = cb32.half() if torch.device(device_str).type == "cuda" else cb32
    csq = cb_score.pow(2).sum(dim=1)
    return cb32, cb_score, csq, None


def _codebook_aux_from_tensor(cb32: torch.Tensor, device) -> tuple:
    """Per-codebook constants for an already materialised learned codebook."""
    cb32 = cb32.to(device=device, dtype=torch.float32).contiguous()
    block_size = int(cb32.shape[1])
    if block_size == 1:
        # Calibrated scalar artifacts are not required to store their rows in
        # centroid order.  The public encoder defensively sorts such a
        # codebook before ``searchsorted`` and maps the selected row back to
        # the original table.  The fused round trip has no externally visible
        # code indices, so sorting the reconstruction table itself gives the
        # identical centroid values while keeping its cached-boundary path
        # valid.  Building boundaries from an unsorted artifact silently sent
        # most queries to the wrong centroid.
        cb32 = cb32[cb32.squeeze(-1).argsort(stable=True)]
        c1 = cb32.squeeze(-1)
        bounds = (c1[1:] + c1[:-1]) / 2.0
        return cb32, cb32, None, bounds.contiguous()
    cb_score = cb32.half() if torch.device(device).type == "cuda" else cb32
    csq = cb_score.pow(2).sum(dim=1)
    return cb32, cb_score, csq, None


def _roundtrip_tensors(x, w, cb32, cb_score, csq, bounds, lut, bins, block_size, modes):
    """Fused encode+decode over ``R`` stacked role tensors (pure tensor ops).

    Args:
        x: ``(R, batch, heads, seq, m)`` fp32 input -- key and value stacked
            along dim 0 when their bit-widths (hence codebook and rotation)
            agree, else a single role at a time.
        w: ``(m, m)`` fp32 orthogonal rotation (the folded hybrid / signed-
            Hadamard / dense GEMM matrix).
        cb32 / cb_score / csq / bounds: From :func:`_codebook_aux_on`.
        lut / bins: Candidate-list LUT (or ``None``) from :func:`_block_lut_on`.
        block_size: Resolved VQ block size.
        modes: Per-role scale modes, e.g. ``("nc", "ls_iter")`` -- a static
            Python tuple, so ``torch.compile`` specialises on it.

    The body reproduces ``_quant_rotor`` followed by ``_decode_rotor``
    *without* materialising a payload: one batched rotation GEMM for all
    roles, one RMS, the nearest-codeword assignment, the role-asymmetric
    scale fit, and one batched inverse-rotation GEMM -- and the reconstruction
    reuses the codeword gather already performed at encode time instead of
    repeating it at decode time.  Every branch below depends only on config
    and shapes (never on tensor *values*), so there is no host-device
    synchronisation and the whole function is capturable by
    ``torch.compile(mode="reduce-overhead")`` into a single CUDA graph --
    which is what eliminates the launch-bound overhead that dominates the
    per-token cost at decode time.
    """
    m = x.shape[-1]
    n_blocks = m // block_size
    xr = x @ w
    rms = xr.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)

    def _assign(vec):
        """Nearest-codeword reconstruction of ``vec`` (``(..., m)`` -> fp32)."""
        vb = vec.reshape(*vec.shape[:-1], n_blocks, block_size)
        if block_size == 1:
            codes = torch.searchsorted(bounds, vb.squeeze(-1).contiguous())
        elif lut is not None:
            codes = quantize_via_lut(
                vb,
                lut,
                bins,
                cb32,
                scoring_cb=cb_score,
                scoring_csq=csq,
            )
        else:
            zb = vb
            if zb.is_cuda and zb.dtype == torch.float32 and cb_score.dtype == torch.float16:
                zb = zb.half()
            codes = quantize_exact_pre(zb, cb_score, csq)
        return lookup_codewords(cb32, codes).reshape(vec.shape)

    outs = []
    for i, mode in enumerate(modes):
        if mode == "bsm":
            mode = "ls"
        elif mode == "ub":
            mode = "ip"
        xi, ri = xr[i], rms[i]
        chat = _assign(xi / ri)
        if mode == "rms":
            scale = ri
        elif mode == "nc":
            cn = chat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            scale = xi.norm(dim=-1, keepdim=True) / cn
        else:
            num = (xi * chat).sum(dim=-1, keepdim=True)
            den = (chat * chat).sum(dim=-1, keepdim=True).clamp_min(1e-8)
            if mode == "ip":
                num_abs = num.abs().clamp_min(1e-8)
                num_safe = torch.where(num < 0, -num_abs, num_abs)
                scale = (xi * xi).sum(dim=-1, keepdim=True) / num_safe
            else:
                scale = num / den
                if mode == "ls_iter":
                    scale_abs = scale.abs().clamp_min(1e-8)
                    scale_safe = torch.where(scale < 0, -scale_abs, scale_abs)
                    chat = _assign(xi / scale_safe)
                    num2 = (xi * chat).sum(dim=-1, keepdim=True)
                    den2 = (chat * chat).sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    scale = num2 / den2
        # Match the payload/native cache contract: a single signed fp16 scale
        # is stored per vector.  The fused fake path skips the payload, so it
        # must explicitly round through fp16 before reconstructing or it would
        # silently evaluate a higher-precision codec than ``stored_bits``.
        stored_scale = scale.to(dtype=torch.float16)
        outs.append(chat * stored_scale.to(dtype=chat.dtype))
    return torch.stack(outs, dim=0) @ w.t()


_RT_COMPILED = None


def _rt_dispatch(compile_allowed: bool = True):
    """Return ``(roundtrip_fn, compiled)`` honouring ``ONECOMP_COMPILE_CODEC``.

    ``ONECOMP_COMPILE_CODEC=1`` wraps :func:`_roundtrip_tensors` in
    ``torch.compile(mode="reduce-overhead", dynamic=False)``: with static
    shapes (a serving decode step is always ``(R, b, h, 1, d)``) the whole
    fused round-trip replays as ONE CUDA graph, collapsing the several dozen
    small kernel launches into a single launch.  Off by default: compilation
    adds tens of seconds of warmup per (shape, config) and CUDA-graph memory
    pools, which is the right trade for serving-style benchmarks but not for
    unit tests or one-shot evaluation.  Callers that need bit-identical
    prefill quality can set ``ONECOMP_COMPILE_DECODE_ONLY=1``; ``compile_allowed``
    then disables graphing for ``seq_len != 1`` while keeping the decode-step
    speedup.
    """
    if not compile_allowed or os.getenv("ONECOMP_COMPILE_CODEC", "0") != "1":
        return _roundtrip_tensors, False
    global _RT_COMPILED  # pylint: disable=global-statement
    if _RT_COMPILED is None:
        import torch._dynamo  # noqa: F401  pylint: disable=import-outside-toplevel

        # One specialisation per (config, shape) pair; the benchmark matrix
        # sweeps several codec variants x bit-widths x regimes, which easily
        # exceeds dynamo's default recompile budget (after which it silently
        # falls back to eager -- exactly what this flag exists to avoid).
        torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
        torch._dynamo.config.accumulated_cache_size_limit = max(
            256, torch._dynamo.config.accumulated_cache_size_limit
        )
        _RT_COMPILED = torch.compile(_roundtrip_tensors, mode="reduce-overhead", dynamic=False)
    return _RT_COMPILED, True


@register_kv_codec("hayatequant")
@dataclass
class HayateQuantCodec(KVCodec):
    """Block-VQ + hybrid-rotation + role-asymmetric-scale KV quantization.

    Attributes:
        bits: Bits per coordinate for keys (and values unless ``value_bits``).
        value_bits: Optional separate bit-width for values (K>V asymmetry).
        dist: ``"gaussian"`` (RMS-normalised, high-d limit, fast default) or
            ``"beta"`` (the exact finite-``d`` law).
        block_size: Vector-quantization block size in ``{None, 1, 2, 4, 8}``.
            ``None`` (default) auto-resolves to the largest of ``(8,4,2,1)``
            whose codebook (``2**(bits*block_size)`` entries) fits under
            ``max_codebook_bits`` and evenly divides the subset dimension.
            ``block_size=1`` is the scalar TurboQuant/HALO quantizer exactly.
        max_codebook_bits: Log2 cap on the block-VQ codebook size (``K =
            2**max_codebook_bits`` worst case). This bounds three costs at
            once: the Lloyd-fit budget, the nearest-codeword search's
            ``O(n_blocks * K)`` GEMM (measured to dominate encode latency at
            ``K>~1000`` on real ``batch*heads*seq`` block counts -- see
            ``codec_speed.py`` / ``e2e_speed.py``), and Lloyd-fit sample
            efficiency (``K`` codewords need enough samples to fit well).
            Default ``9`` (``K<=512``) keeps every paper bit-width's encode
            latency within a small constant factor of TurboQuant/HALO while
            still using ``block_size>1`` (block-VQ) at 2/3/4-bit -- raise it
            (e.g. 12+) to trade encode latency for a larger, more accurate
            block codebook at higher bit-widths.
        codebook_init: Lloyd warm-start for block codebooks. ``"auto"``
            preserves shipped high-budget Gaussian artifacts when available
            and otherwise uses the FibQuant radial-angular initialization
            before Lloyd refinement; ``"forgy"`` is the historical random
            sample K-means initialization; ``"fibquant"`` forces the
            FibQuant warm-start even when a Gaussian artifact exists.
        rotation_mode: ``"auto"`` (default; rate-aware: picks ``"dense"``
            when ``block_size>1`` and the subset's rate is coarse
            (``bits<=2``, where the fixed +-1 Hadamard structure measurably
            hurts PPL -- see ``_resolved_mode``), ``"hybrid"`` when
            ``block_size>1`` at ``bits>=3`` on a power-of-two dim,
            ``"hadamard_gemm"``/``"hadamard"`` when ``block_size==1``, else
            ``"dense"``), ``"hybrid"`` (global Hadamard-GEMM + per-block dense,
            requires power-of-two dim; folded offline into ONE orthogonal GEMM
            via :func:`rotation.hybrid_rotation_matrix`, so it costs exactly
            as much as ``"dense"`` at runtime), ``"hadamard_gemm"`` (single
            folded Hadamard GEMM), ``"hadamard"`` (sequential WHT butterfly),
            or ``"dense"`` (full QR random orthogonal, any dim).
        encoder: Nearest-codeword assignment algorithm for ``block_size>1``.
            ``"auto"`` (default) picks *per call* between the exact O(K)
            chunked-GEMM search (a tensor-core matmul -- fastest while the
            ``n_blocks x K`` score matrix stays small) and the O(T)-per-block
            candidate-list LUT encoder (:func:`._vq.quantize_via_lut`: grid
            cell -> top-``T`` candidate codewords -> exact argmin over the
            ``T``; ~10x faster once ``n_blocks * K`` crosses the measured
            crossover, e.g. long-prefill encodes against a large codebook).
            ``"lut"`` forces the LUT (error if infeasible for the block
            size); ``"exact"`` forces the exact search. The LUT restricts
            only the *candidate set* (distances over the candidates, and
            reconstruction, are exact); its excess distortion at the default
            specs is measured to be negligible (see
            ``test_hayatequant_lut_encoder_matches_exact_distortion``) while
            making encode latency independent of the codebook size ``K``.
        lut_bins / lut_candidates: Optional explicit fixed grid resolution and
            candidate-list length for ``encoder="lut"``.  They must be
            supplied together.  Canonical recipe selection never tunes them
            per model or workload; these knobs exist so a single proposed
            table contract can pass quality/parity gates before adoption.
        hadamard_gemm_max_dim: Above this head dim, ``"auto"`` falls back from
            the materialised Hadamard GEMM to the sequential WHT butterfly.
        ls_scale: Master switch for the role-asymmetric scale (``True``:
            smart per-role defaults below; ``False``: per-vector RMS on both
            roles, the TurboQuant-faithful baseline).
        key_scale_mode / value_scale_mode: Explicit override of the per-role
            scale (``"rms"``/``"nc"``/``"ls"``/``"ip"``/``"ls_iter"`` plus
            BlockQuant aliases ``"bsm"`` = ``"ls"`` and ``"ub"`` = ``"ip"``);
            ``None`` defers to the ``ls_scale`` smart default
            (``"nc"`` for keys, ``"ls_iter"`` for values).
        n_outlier / outlier_bits: Outlier mixed-precision (ported from
            TurboQuant/HALO): split the ``n_outlier`` highest-energy channels
            off and quantize them at ``outlier_bits`` (default ``bits+1``)
            for true fractional-bit operating points (e.g. 2.5/3.5-bit).
            ``key_n_outlier``/``value_n_outlier`` and
            ``key_outlier_bits``/``value_outlier_bits`` optionally override
            that split per role, enabling BlockQuant-style key-only outlier
            protection with lower-bit values.
        use_qjl: Optional 1-bit Gaussian QJL residual refinement (off by
            default).  It stores ``sign(S r)`` and ``||r||_2`` for the
            quantization residual, not the coordinate-wise sign of ``r``.
        seed: Seed for the deterministic rotations.
        calibration: Optional SpinQuant-style per-model artifact
            (:class:`.hayatequant_calib.HayateQuantCalibration` instance, or a
            path to one saved with ``.save()``). When set, the per-role
            *learned* orthogonal rotation replaces the data-free one (same
            single GEMM, zero runtime cost) and, if present in the artifact,
            the real-data-refit codebook replaces the shipped Gaussian one
            (same ``K``). Payload rot tags then carry the artifact's content
            digest, and decode verifies the same artifact is loaded (fail
            fast instead of silently decoding under a different rotation).
            Default ``None`` keeps the codec fully data-free.
        key_code_bits_per_block / value_code_bits_per_block: Experimental
            fixed-length fractional-rate block-VQ knobs for the HF/research
            path.  When set, the role stores exactly this many bits per VQ
            block, so the payload rate is ``code_bits_per_block / block_size``
            bits per coordinate.  vLLM native kernels remain tied to their
            integer-bit cache layouts and do not consume these knobs.
        avg_bits / key_bit_fraction: User-facing fractional-rate interface.
            For example ``avg_bits=2, block_size=4, key_bit_fraction=0.625``
            resolves to K10/V6 code bits per block, avoiding manual
            ``10``/``6`` bookkeeping. ``block_size`` defaults to 4 when this
            interface is used.
        adaptive_bit_split: Optional online K/V bit-allocation policy for the
            explicit-rate HF/research path. ``"online_stats"`` chooses a
            K/V split from cache-owned, layer-local cumulative raw K/V power,
            including the current update, then quantizes exactly once. Each
            candidate is scored with the actual source-law codebook's
            deterministic data-free relative-distortion constant under the
            resolved role scale (including fp16 scale rounding), rather than
            the former high-rate ``2**(-2R)`` proxy. The default key/value
            weights are the globally frozen ``4:1`` development choice.
            Before the cumulative-token gate matures, an explicit
            ``key_bit_fraction`` supplies the incumbent; otherwise the neutral
            uniform split is used by the generic codec.  The named AdaptiveHQ
            paper recipe explicitly supplies the matching fixed-HQ incumbent.
            Departing from that incumbent requires a predeclared ``1/sqrt(128)``
            relative-objective margin derived from the frozen QMC table size;
            an asymmetric retain/fallback threshold provides hysteresis.
            Because this choice is data-dependent, a packed decoder needs one
            fixed-width candidate-pair selector per jointly encoded K/V
            update: ``ceil(log2(n_pairs))`` bits (zero for a singleton set).
            Adaptive HQ embeds those bits in the low fraction bits of the
            first existing fp16 value-scale word and clears them before scale
            reconstruction.  It therefore sacrifices that word's scale
            precision rather than increasing the logical data-plane rate.
            This is opt-in and never changes the canonical HQ/frontier path.
        vllm_mode: Optional explicit serving backend selection persisted in
            ``kv_compression_config``. The HF codec ignores this field. The
            vLLM mapper validates it and fails closed on incompatible recipes.
    """

    flag_calibration: bool = False
    bits: int = 4
    value_bits: int = None
    dist: str = "gaussian"
    block_size: int = None
    max_codebook_bits: int = 9
    codebook_init: str = "auto"
    required_codebook_bundle: str = None
    rotation_mode: str = "auto"
    encoder: str = "auto"
    lut_bins: int = None
    lut_candidates: int = None
    hadamard_gemm_max_dim: int = 4096
    ls_scale: bool = True
    key_scale_mode: str = None
    value_scale_mode: str = None
    n_outlier: int = 0
    outlier_bits: int = None
    key_n_outlier: int = None
    value_n_outlier: int = None
    key_outlier_bits: int = None
    value_outlier_bits: int = None
    use_qjl: bool = False
    seed: int = 0
    calibration: object = None
    avg_bits: float = None
    code_bits_per_pair_block: int = None
    key_bit_fraction: float = None
    adaptive_bit_split: str = None
    adaptive_bit_margin: int = 4
    adaptive_stats_min_tokens: int = 16
    key_error_weight: float = 4.0
    value_error_weight: float = 1.0
    key_code_bits_per_block: int = None
    value_code_bits_per_block: int = None
    # Optional serving selection persisted with the codec. The HF codec does
    # not interpret it; the vLLM mapper validates the requested backend and
    # fails closed when the serialized recipe cannot be represented.
    vllm_mode: str = None
    key_axis: str = "token"
    value_axis: str = "token"

    _SCALE_MODES = SCALE_MODES
    _BLOCK_SIZES = (None, 1, 2, 4, 8)

    def _calib(self):
        """The loaded :class:`HayateQuantCalibration`, or None (data-free)."""
        if self.calibration is None:
            return None
        if isinstance(self.calibration, str):
            from .calibration import HayateQuantCalibration

            # Load once; replace the path with the object (idempotent).
            self.calibration = HayateQuantCalibration.load(self.calibration)
        return self.calibration

    def _fractional_rate_requested(self) -> bool:
        return self.avg_bits is not None or self.code_bits_per_pair_block is not None

    def _adaptive_bit_split_enabled(self) -> bool:
        return self.adaptive_bit_split not in (None, "", "off", "none", False)

    def teacher_forced_temporal_dependency(self):
        dependency = super().teacher_forced_temporal_dependency()
        if dependency is not None:
            return dependency
        if self._n_outlier_for("k") > 0 or self._n_outlier_for("v") > 0:
            return "dynamic_outlier_channels_over_update_tokens"
        if self._adaptive_bit_split_enabled():
            return "cache_local_causal_bit_allocation_statistics"
        return None

    def _pair_code_bits_total(self, block_size: int) -> int | None:
        if self.code_bits_per_pair_block is not None:
            return int(self.code_bits_per_pair_block)
        if self.avg_bits is None:
            return None
        return max(2, int(round(float(self.avg_bits) * 2.0 * int(block_size))))

    def _configure_fractional_rate_ui(self):
        """Resolve the human-facing fractional-rate interface into code bits."""
        if not self._fractional_rate_requested():
            return
        if self.block_size is None:
            self.block_size = 4
        if not (isinstance(self.block_size, int) and self.block_size >= 1):
            return
        block_size = int(self.block_size)
        total = self._pair_code_bits_total(block_size)
        if total is None:
            return
        if total < 2:
            raise ValueError("fractional-rate UI needs at least 2 total K/V bits per block")
        if self.avg_bits is not None:
            nominal = max(1, min(8, int(round(float(self.avg_bits)))))
            self.bits = nominal
            if self.value_bits is None:
                self.value_bits = nominal
        if self._adaptive_bit_split_enabled():
            if (
                self.key_code_bits_per_block is not None
                or self.value_code_bits_per_block is not None
            ):
                raise ValueError(
                    "adaptive_bit_split cannot be combined with explicit "
                    "key/value_code_bits_per_block"
                )
            frac = 0.5 if self.key_bit_fraction is None else float(self.key_bit_fraction)
            if not 0.0 < frac < 1.0:
                raise ValueError(f"key_bit_fraction must be in (0, 1), got {frac!r}")
            return
        key_explicit = self.key_code_bits_per_block
        value_explicit = self.value_code_bits_per_block
        if key_explicit is not None or value_explicit is not None:
            # ``avg_bits`` / ``code_bits_per_pair_block`` is a fixed TOTAL
            # K/V budget.  A former early return accepted one manual role and
            # silently left the other at its nominal integer rate; e.g. K10 +
            # implicit V8 advertised average-2 at block4 while actually
            # storing 2.25 payload bits/coordinate.  Infer the missing role,
            # or reject a pair that contradicts the declared total.
            if key_explicit is None:
                key_explicit = total - int(value_explicit)
                if key_explicit < 1:
                    raise ValueError(
                        "value_code_bits_per_block must leave at least one bit "
                        "for keys under the declared K/V pair budget"
                    )
                self.key_code_bits_per_block = key_explicit
            elif value_explicit is None:
                value_explicit = total - int(key_explicit)
                if value_explicit < 1:
                    raise ValueError(
                        "key_code_bits_per_block must leave at least one bit "
                        "for values under the declared K/V pair budget"
                    )
                self.value_code_bits_per_block = value_explicit
            elif int(key_explicit) + int(value_explicit) != total:
                raise ValueError(
                    "explicit key/value_code_bits_per_block must sum to the "
                    f"declared pair budget {total}, got "
                    f"{int(key_explicit) + int(value_explicit)}"
                )
            return
        frac = 0.625 if self.key_bit_fraction is None else float(self.key_bit_fraction)
        if not 0.0 < frac < 1.0:
            raise ValueError(f"key_bit_fraction must be in (0, 1), got {frac!r}")
        key_bits = min(total - 1, max(1, int(round(total * frac))))
        value_bits = total - key_bits
        self.key_code_bits_per_block = key_bits
        self.value_code_bits_per_block = value_bits

    def validate_params(self):
        if self.vllm_mode is not None and (
            not isinstance(self.vllm_mode, str) or not self.vllm_mode.strip()
        ):
            raise ValueError("vllm_mode must be a non-empty string or None")
        if self.block_size is not None and (
            not isinstance(self.block_size, int) or isinstance(self.block_size, bool)
        ):
            raise ValueError(f"block_size must be an int or None, got {self.block_size!r}")
        if not (
            isinstance(self.max_codebook_bits, int)
            and not isinstance(self.max_codebook_bits, bool)
            and self.max_codebook_bits >= 1
        ):
            raise ValueError(
                f"max_codebook_bits must be a positive int, got {self.max_codebook_bits!r}"
            )
        if self.max_codebook_bits > MAX_AUDITED_CODEBOOK_BITS:
            raise ValueError(
                f"max_codebook_bits={self.max_codebook_bits} exceeds the audited limit "
                f"{MAX_AUDITED_CODEBOOK_BITS}"
            )
        if not (
            isinstance(self.hadamard_gemm_max_dim, int)
            and not isinstance(self.hadamard_gemm_max_dim, bool)
            and self.hadamard_gemm_max_dim >= 1
        ):
            raise ValueError(
                "hadamard_gemm_max_dim must be a positive int, got "
                f"{self.hadamard_gemm_max_dim!r}"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an int, got {self.seed!r}")
        if self.avg_bits is not None and (
            not math.isfinite(float(self.avg_bits)) or float(self.avg_bits) <= 0.0
        ):
            raise ValueError(f"avg_bits must be finite and positive, got {self.avg_bits!r}")
        if self.code_bits_per_pair_block is not None and (
            not isinstance(self.code_bits_per_pair_block, int)
            or isinstance(self.code_bits_per_pair_block, bool)
            or self.code_bits_per_pair_block < 2
        ):
            raise ValueError(
                "code_bits_per_pair_block must be None or an int >= 2, "
                f"got {self.code_bits_per_pair_block!r}"
            )
        self._configure_fractional_rate_ui()
        if self.rotation_mode not in ("auto", "hybrid", "hadamard_gemm", "hadamard", "dense"):
            raise ValueError(
                "rotation_mode must be 'auto', 'hybrid', 'hadamard_gemm', 'hadamard' or 'dense'"
            )
        if self.encoder not in ("auto", "lut", "exact"):
            raise ValueError(f"encoder must be 'auto', 'lut' or 'exact', got {self.encoder!r}")
        explicit_lut = (self.lut_bins is not None, self.lut_candidates is not None)
        if any(explicit_lut) and not all(explicit_lut):
            raise ValueError("lut_bins and lut_candidates must be supplied together")
        if all(explicit_lut):
            if self.encoder != "lut":
                raise ValueError("explicit LUT geometry requires encoder='lut'")
            for name in ("lut_bins", "lut_candidates"):
                value = getattr(self, name)
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    raise ValueError(f"{name} must be a positive int, got {value!r}")
            if self.lut_bins > 256:
                raise ValueError("lut_bins above 256 are outside the audited grid contract")
            if self.lut_candidates > 16:
                raise ValueError("lut_candidates above 16 are outside the audited contract")
        if self.dist not in ("gaussian", "beta"):
            raise ValueError("dist must be 'gaussian' or 'beta'")
        if self.codebook_init not in ("auto", "forgy", "fibquant"):
            raise ValueError(
                "codebook_init must be 'auto', 'forgy' or 'fibquant', "
                f"got {self.codebook_init!r}"
            )
        if not (
            isinstance(self.bits, int) and not isinstance(self.bits, bool) and 1 <= self.bits <= 8
        ):
            raise ValueError(f"bits must be int in 1..8, got {self.bits!r}")
        if self.adaptive_bit_split not in (
            None,
            "",
            "off",
            "none",
            False,
            "online",
            "online_stats",
        ):
            raise ValueError(
                "adaptive_bit_split must be None/'off' or 'online_stats', "
                f"got {self.adaptive_bit_split!r}"
            )
        if self._adaptive_bit_split_enabled() and not self._fractional_rate_requested():
            raise ValueError(
                "adaptive_bit_split needs avg_bits or code_bits_per_pair_block "
                "to define a fixed total K/V bit budget"
            )
        if not (
            isinstance(self.adaptive_bit_margin, int)
            and not isinstance(self.adaptive_bit_margin, bool)
            and self.adaptive_bit_margin >= 0
        ):
            raise ValueError(
                f"adaptive_bit_margin must be a non-negative int, got {self.adaptive_bit_margin!r}"
            )
        if not (
            isinstance(self.adaptive_stats_min_tokens, int)
            and not isinstance(self.adaptive_stats_min_tokens, bool)
            and self.adaptive_stats_min_tokens >= 1
        ):
            raise ValueError(
                "adaptive_stats_min_tokens must be a positive int, "
                f"got {self.adaptive_stats_min_tokens!r}"
            )
        if (
            not math.isfinite(float(self.key_error_weight))
            or not math.isfinite(float(self.value_error_weight))
            or float(self.key_error_weight) <= 0.0
            or float(self.value_error_weight) <= 0.0
        ):
            raise ValueError("key_error_weight and value_error_weight must be finite and positive")
        for label, b in (
            ("value_bits", self.value_bits),
            ("outlier_bits", self.outlier_bits),
            ("key_outlier_bits", self.key_outlier_bits),
            ("value_outlier_bits", self.value_outlier_bits),
        ):
            if b is not None and not (
                isinstance(b, int) and not isinstance(b, bool) and 1 <= b <= 8
            ):
                raise ValueError(f"{label} must be None or int in 1..8, got {b!r}")
        for label, b in (
            ("key_code_bits_per_block", self.key_code_bits_per_block),
            ("value_code_bits_per_block", self.value_code_bits_per_block),
        ):
            if b is not None and not (isinstance(b, int) and not isinstance(b, bool) and b >= 1):
                raise ValueError(f"{label} must be None or a positive int, got {b!r}")
            if b is not None and b > self.max_codebook_bits:
                raise ValueError(
                    f"{label}={b} exceeds max_codebook_bits={self.max_codebook_bits}; "
                    "raise max_codebook_bits explicitly if this large codebook is intended"
                )
        if self.block_size not in self._BLOCK_SIZES:
            raise ValueError(
                f"block_size must be one of {self._BLOCK_SIZES}, got {self.block_size!r}"
            )
        for mode in (self.key_scale_mode, self.value_scale_mode):
            if mode is not None and mode not in self._SCALE_MODES:
                raise ValueError(
                    f"scale_mode must be one of {self._SCALE_MODES} or None, got {mode!r}"
                )
        if not (
            isinstance(self.n_outlier, int)
            and not isinstance(self.n_outlier, bool)
            and self.n_outlier >= 0
        ):
            raise ValueError(f"n_outlier must be a non-negative int, got {self.n_outlier!r}")
        for label, n in (
            ("key_n_outlier", self.key_n_outlier),
            ("value_n_outlier", self.value_n_outlier),
        ):
            if n is not None and not (isinstance(n, int) and not isinstance(n, bool) and n >= 0):
                raise ValueError(f"{label} must be None or a non-negative int, got {n!r}")
        for role, role_bits in (("k", self.bits), ("v", self._value_bits())):
            if self._n_outlier_for(role) > 0 and self._outlier_bits_for(role, role_bits) > 8:
                raise ValueError(f"8-bit HayateQuant {role} cannot default outlier bits to bits+1")
        if self.calibration is not None:
            from .calibration import HayateQuantCalibration

            if not isinstance(self.calibration, (str, HayateQuantCalibration)):
                raise ValueError(
                    "calibration must be None, a path, or a HayateQuantCalibration, "
                    f"got {type(self.calibration).__name__}"
                )
            if self._n_outlier_for("k") or self._n_outlier_for("v"):
                raise ValueError(
                    "calibration is not supported with n_outlier>0 "
                    "(subset dims differ from the calibrated head_dim)"
                )
        if self._has_explicit_code_bits() and (
            self._n_outlier_for("k") or self._n_outlier_for("v")
        ):
            raise ValueError(
                "explicit code_bits_per_block is currently supported only "
                "without outlier channel splits"
            )
        # A rotation-only calibration is compatible with the fixed HQ
        # K/V-asymmetric block-code contract: it replaces only the orthogonal
        # matrix and leaves the exact data-free, explicit-code-bit codebooks
        # and every per-token code/scale field unchanged.  A learned codebook
        # is checked fail-closed at encode time below, after a path-backed
        # artifact has been loaded.
        if self._adaptive_bit_split_enabled() and (
            self._n_outlier_for("k") or self._n_outlier_for("v") or self.calibration is not None
        ):
            raise ValueError(
                "adaptive_bit_split is currently supported only without "
                "outlier splits or calibration"
            )
        if self.required_codebook_bundle is not None:
            if (
                not isinstance(self.required_codebook_bundle, str)
                or not self.required_codebook_bundle.strip()
            ):
                raise ValueError("required_codebook_bundle must be None or a non-empty string")
            if self.dist != "gaussian" or self.codebook_init != "auto":
                raise ValueError(
                    "a required shipped codebook bundle needs dist='gaussian' and "
                    "codebook_init='auto'"
                )
            if self.block_size != 2:
                raise ValueError("the current formal HQ codebook bundles require block_size=2")
            if self._n_outlier_for("k") or self._n_outlier_for("v"):
                raise ValueError(
                    "a required shipped codebook bundle cannot be combined with "
                    "outlier splits"
                )
            # Canonical HQ/AdaptiveHQ must never fall through to an online fit
            # when a committed artifact is absent, corrupt, or overridden.
            require_shipped_gaussian_codebook_bundle(self.required_codebook_bundle)
        # OneComp convention: the flag (not the folder) marks calibrated codecs.
        self.flag_calibration = self.calibration is not None

    def _value_bits(self) -> int:
        return self.value_bits if self.value_bits is not None else self.bits

    def _code_bits_override_for(self, role: str) -> int | None:
        return self.key_code_bits_per_block if role == "k" else self.value_code_bits_per_block

    def _has_explicit_code_bits(self) -> bool:
        return (
            self.key_code_bits_per_block is not None or self.value_code_bits_per_block is not None
        )

    def _code_bits_for(self, role: str, bits: int, block_size: int) -> int:
        override = self._code_bits_override_for(role)
        return int(override) if override is not None else int(bits) * int(block_size)

    def _resolve_block_size_for(
        self,
        role: str,
        bits: int,
        m: int,
        *,
        code_bits_override: int | None = None,
    ) -> int:
        """Resolve a role's real VQ block dimension.

        Ordinary integer-rate HQ has ``bits * block_size`` code bits, so the
        legacy resolver correctly uses that product against the codebook cap.
        Explicit/fractional HQ instead stores ``code_bits_override`` for the
        WHOLE block.  Applying the nominal rule there silently shrank an
        explicitly requested block (for example b=4, K=2**5 became b=2 merely
        because the nominal ``bits`` field was 4), doubling its realised
        payload rate.  For fixed block-code bits the codebook-size constraint
        is independent of the block dimension: honour the requested block and
        fail closed if the head dimension cannot be partitioned exactly.
        """
        selected_code_bits = (
            self._code_bits_override_for(role)
            if code_bits_override is None
            else int(code_bits_override)
        )
        if selected_code_bits is None:
            return resolve_block_size(bits, self.block_size, self.max_codebook_bits, m)
        selected_code_bits = int(selected_code_bits)
        if not 1 <= selected_code_bits <= int(self.max_codebook_bits):
            raise ValueError(
                f"{role} code_bits_per_block={selected_code_bits} must be in "
                f"1..max_codebook_bits={self.max_codebook_bits}"
            )
        if self.block_size is None:
            block_size = next((b for b in (8, 4, 2, 1) if m % b == 0), 1)
        else:
            block_size = int(self.block_size)
            if m % block_size:
                raise ValueError(
                    f"explicit code bits require block_size={block_size} to divide "
                    f"the quantized dimension m={m}; silent block-size fallback "
                    "would change the requested stored rate"
                )
        if block_size == 1 and selected_code_bits > 8:
            raise ValueError(
                "scalar HayateQuant codebooks support at most 8 code bits; "
                f"got {selected_code_bits}"
            )
        return block_size

    def _outlier_bits(self) -> int:
        return self.outlier_bits if self.outlier_bits is not None else self.bits + 1

    def _n_outlier_for(self, role: str) -> int:
        explicit = self.key_n_outlier if role == "k" else self.value_n_outlier
        return self.n_outlier if explicit is None else explicit

    def _outlier_bits_for(self, role: str, bits: int) -> int:
        explicit = self.key_outlier_bits if role == "k" else self.value_outlier_bits
        if explicit is not None:
            return explicit
        if self.outlier_bits is not None:
            return self.outlier_bits
        return bits + 1

    def _scale_mode_for(self, role: str) -> str:
        explicit = self.key_scale_mode if role == "k" else self.value_scale_mode
        return resolve_scale_mode(explicit, self.ls_scale, role)

    def _resolved_mode(self, d: int, block_size: int, bits: float) -> str:
        """Resolve ``"auto"`` to a concrete rotation given block size and rate.

        Block-VQ (``block_size>1``) needs each rotated block to be close to
        isotropic (the codebook is fit to an isotropic law); both the hybrid
        (Hadamard + per-block Haar) and the plain Haar-dense rotation provide
        that at identical runtime cost (one folded GEMM). Which one wins is
        rate-dependent, measured per-lever on real KV (Llama-3.1-8B PPL,
        ``paper_hayatequant_ablation`` + the v7 rotation-x-scale attribution
        run, scale held fixed at NC/ls_iter):

        * ``bits >= 3``: hybrid wins (3-bit 7.013 vs dense 7.034; 4-bit
          6.759 vs 6.770) -- the Hadamard factor's stronger global
          incoherence still pays at moderate rates.
        * ``bits <= 2``: dense wins decisively (2-bit 8.915 vs 9.123,
          -0.21 PPL; confirmed at both scale settings, and neutral on
          R1-Qwen-1.5B) -- at coarse rates the fixed +-1 Hadamard structure
          correlates the quantization error across tokens, which a fully
          Haar-random rotation decorrelates. The 2-bit distortion is
          slightly *higher* under dense, so this is an attention-level
          effect, not an MSE one -- exactly why the policy is set by
          measured PPL, not by distortion.
        """
        if self.rotation_mode != "auto":
            # Structured transforms require a power-of-two subset.  Encode
            # has always fallen back to dense QR for ragged/outlier subsets;
            # resolve that fallback here too so provenance and fused-path
            # dispatch describe the transform that is actually applied.
            if self.rotation_mode in (
                "hybrid",
                "hadamard_gemm",
                "hadamard",
            ) and not is_power_of_two(d):
                return "dense"
            return self.rotation_mode
        if block_size > 1:
            if bits <= 2:
                return "dense"
            return "hybrid" if is_power_of_two(d) else "dense"
        if is_power_of_two(d):
            return "hadamard_gemm" if d <= self.hadamard_gemm_max_dim else "hadamard"
        return "dense"

    def _rotation_rate_for(
        self,
        role: str,
        nominal_bits: int,
        block_size: int,
        code_bits_override: int | None = None,
    ) -> float:
        """Rate that controls auto rotation for fixed/fractional block codes."""
        code_bits = (
            self._code_bits_for(role, nominal_bits, block_size)
            if code_bits_override is None
            else int(code_bits_override)
        )
        return float(code_bits) / float(block_size)

    # --- rotation ----------------------------------------------------------
    def _rotate(
        self,
        x: torch.Tensor,
        m: int,
        block_size: int,
        bits: float,
        role: str = "k",
        layer_idx=None,
    ):
        """Rotate ``x`` (``..., m``); returns ``(x_rotated, rot_tag)``.

        Any rotation that structurally requires a power-of-two ``m``
        (``"hybrid"``, ``"hadamard"``, ``"hadamard_gemm"``) transparently
        falls back to the dense QR rotation when ``m`` is not a power of two
        (e.g. an outlier-split subset) -- correctness is never compromised,
        only the specific transform requested.

        The rot tag ``(kind, block_size, seed)`` is stored in the payload so
        decode reconstructs the *exact* rotation used at encode time, even if
        the codec instance is reconfigured (e.g. a different ``seed``)
        between encode and decode -- the same self-describing-payload
        convention as the stored ``dist``/``bits``/``block_size``.
        """
        cal = self._calib()
        if cal is not None:
            w = cal.rotation_on(role, m, x.device, layer_idx=layer_idx)
            if w is not None:
                # Learned (SpinQuant-style) rotation: same single GEMM as the
                # data-free path; the tag carries the artifact digest so
                # decode can verify the matching artifact is loaded.
                return x.float() @ w, ("c", role, cal.digest, layer_idx)
        mode = self._resolved_mode(m, block_size, bits)
        x = x.float()
        if mode == "hybrid" and is_power_of_two(m):
            # Folded hybrid rotation: the global signed-Hadamard and the
            # per-block dense rotations are pre-composed into ONE orthogonal
            # GEMM (see rotation.hybrid_rotation_matrix), so the hybrid
            # rotation costs exactly as much as a plain dense rotation at
            # runtime -- the Hadamard/Dense interpolation is free.
            w = hybrid_rotation_matrix(m, block_size, self.seed, str(x.device), torch.float32)
            return x @ w, ("y", block_size, self.seed)
        if mode == "hadamard_gemm" and is_power_of_two(m):
            hs = signed_hadamard_matrix(m, self.seed, str(x.device), torch.float32)
            return x @ hs, ("hg", None, self.seed)
        if mode == "hadamard" and is_power_of_two(m):
            sign = random_sign(m, self.seed, device=x.device, dtype=torch.float32)
            return rotate(x, sign), ("h", None, self.seed)
        q = _dense_rotation_on(m, self.seed, str(x.device), torch.float32)
        return x @ q, ("d", None, self.seed)

    def _unrotate(self, xr: torch.Tensor, m: int, rot) -> torch.Tensor:
        kind, bs = rot[0], rot[1]
        if kind == "c":  # calibrated: rot == ("c", role, digest[, layer_idx])
            role, digest = rot[1], rot[2]
            layer_idx = rot[3] if len(rot) > 3 else None
            cal = self._calib()
            if cal is None or cal.digest != digest:
                raise ValueError(
                    "payload was encoded with a calibrated rotation "
                    f"(digest {digest}) but the codec has "
                    f"{'no' if cal is None else 'a different'} calibration loaded"
                )
            return xr @ cal.rotation_on(role, m, xr.device, layer_idx=layer_idx).t()
        # Pre-fix payloads stored 2-tuples without the seed; fall back to the
        # instance seed for those (the historical behaviour).
        seed = rot[2] if len(rot) > 2 else self.seed
        if kind == "y":
            w = hybrid_rotation_matrix(m, bs, seed, str(xr.device), torch.float32)
            return xr @ w.t()
        if kind == "hg":
            hs = signed_hadamard_matrix(m, seed, str(xr.device), torch.float32)
            return xr @ hs.t()
        if kind == "h":
            sign = random_sign(m, seed, device=xr.device, dtype=torch.float32)
            return unrotate(xr, sign)
        if kind == "d":
            q = _dense_rotation_on(m, seed, str(xr.device), torch.float32)
            return xr @ q.t()
        raise ValueError(f"unknown HayateQuant rotation payload tag {kind!r}")

    # Below this many (blocks * codewords) score products the exact GEMM is a
    # single trivially small kernel launch (decode-step encodes: one new
    # token), so the LUT's multi-kernel gather pipeline only adds overhead.
    _LUT_MIN_WORK = 2_000_000

    def _use_lut(
        self, block_size: int, bits: int, n_total_blocks: int, is_cuda: bool
    ) -> "tuple[int, int] | None":
        """``(bins, T)`` LUT spec to use for this call, or None for exact search.

        Measured policy (A100, real ``batch*heads*seq`` block counts,
        ``paper/tests/bench_encoder_paths.py``, fp16 candidate distances):

        * Tiny calls (``n_blocks * K`` below :data:`_LUT_MIN_WORK`, i.e. the
          per-decode-step encode of one new token): the exact GEMM is one
          small kernel; the LUT's several-kernel gather pipeline only adds
          launch overhead -- use exact.
        * ``b=2`` (``T=2``) prefill: the LUT wins outright -- 0.26ms vs
          1.5-3.1ms for the exact chunked GEMM at K=64-256 (5.8-11.7x); this
          is what puts 3/4-bit prefill on par with TurboQuant's scalar encode.
        * ``b=4`` prefill: with the fp16 candidate-distance step the LUT now
          wins at *every* codebook size -- 1.21ms vs 1.56ms at the default
          2-bit K=256 (1.3x; it used to lose at 5.4ms with fp32 gathers,
          which is why older revisions excluded ``b>=4, K<=512``), and
          1.80ms vs 19.4ms at the ``max_codebook_bits=12`` 3-bit K=4096
          point (10.8x) -- so the small-K exclusion is gone.
        * CPU: the crossover above was measured on GPU only, and the one-time
          table build (O(bins**b * K) scoring, up to ~seconds on CPU at the
          large-``K`` specs) is much harder to amortise there -- ``"auto"``
          therefore stays on the exact chunked GEMM off-CUDA. Forcing
          ``encoder="lut"`` still works on any device (used by the CPU unit
          tests, which exercise the LUT path on small codebooks).
        """
        if block_size <= 1 or self.encoder == "exact":
            return None
        k = 1 << (bits * block_size)
        spec = lut_spec_for(block_size, k)
        if spec is None:
            if self.encoder == "lut":
                raise ValueError(f"encoder='lut' infeasible for block_size={block_size}")
            return None
        if self.lut_bins is not None:
            spec = (int(self.lut_bins), min(int(self.lut_candidates), k))
        if self.encoder == "auto" and (not is_cuda or n_total_blocks * k < self._LUT_MIN_WORK):
            return None
        return spec

    def _use_lut_code_bits(
        self, block_size: int, code_bits: int, n_total_blocks: int, is_cuda: bool
    ) -> "tuple[int, int] | None":
        """LUT policy for explicit block-code-bit codebooks."""
        if block_size <= 1 or self.encoder == "exact":
            return None
        k = 1 << int(code_bits)
        spec = lut_spec_for(block_size, k)
        if spec is None:
            if self.encoder == "lut":
                raise ValueError(f"encoder='lut' infeasible for block_size={block_size}")
            return None
        if self.lut_bins is not None:
            spec = (int(self.lut_bins), min(int(self.lut_candidates), k))
        if self.encoder == "auto" and (not is_cuda or n_total_blocks * k < self._LUT_MIN_WORK):
            return None
        return spec

    # --- rotate + block-VQ + role-asymmetric scale for one channel subset --
    def _quant_rotor(
        self,
        xsub: torch.Tensor,
        bits: int,
        role: str,
        layer_idx=None,
        code_bits_override: int | None = None,
    ) -> dict:
        m = xsub.shape[-1]
        selected_override = (
            int(code_bits_override)
            if code_bits_override is not None
            else self._code_bits_override_for(role)
        )
        block_size = self._resolve_block_size_for(
            role, bits, m, code_bits_override=selected_override
        )
        code_bits = (
            int(selected_override)
            if selected_override is not None
            else int(bits) * int(block_size)
        )
        explicit_code_bits = code_bits != int(bits) * int(block_size)
        rotation_rate = self._rotation_rate_for(
            role,
            bits,
            block_size,
            code_bits_override=(code_bits if explicit_code_bits else None),
        )
        xr, rot = self._rotate(xsub, m, block_size, rotation_rate, role, layer_idx=layer_idx)
        device_str = str(xsub.device)
        cal = self._calib()
        if cal is not None and explicit_code_bits and cal.codebook:
            raise ValueError(
                "explicit code_bits_per_block supports rotation-only calibration; "
                "the loaded artifact also contains learned codebooks"
            )
        if cal is not None and self.required_codebook_bundle is not None and cal.codebook:
            raise ValueError(
                "a required fixed codebook bundle supports rotation-only calibration; "
                "the loaded artifact also contains learned codebooks"
            )
        cb_cal = (
            cal.codebook_on(role, bits, block_size, xsub.device)
            if cal is not None and m == cal.head_dim and not explicit_code_bits
            else None
        )
        # The scalar public encoder used to call ``quantize_to_vq_codebook``
        # for every payload.  That generic helper defensively sorts the
        # codebook and then calls ``torch.equal`` to discover that all of our
        # data-free Lloyd codebooks were already sorted.  On CUDA the equality
        # test is a host synchronisation in the encode hot path.  Reuse the
        # decision boundaries prepared at the same cache boundary as the
        # device-resident codebook instead.  Learned/calibrated codebooks keep
        # the defensive generic path because their ordering is not part of the
        # artifact contract.
        scalar_bounds = None
        exact_cb_score = None
        exact_csq = None
        if cb_cal is not None:
            cb = cb_cal
        elif explicit_code_bits and block_size == 1:
            cb, _, _, scalar_bounds = _codebook_aux_code_bits_on(
                code_bits, block_size, self.dist, m, self.codebook_init, device_str
            )
        elif explicit_code_bits:
            cb, exact_cb_score, exact_csq, _ = _codebook_aux_code_bits_on(
                code_bits, block_size, self.dist, m, self.codebook_init, device_str
            )
        elif block_size == 1:
            cb, _, _, scalar_bounds = _codebook_aux_on(
                bits, block_size, self.dist, m, self.codebook_init, device_str
            )
        elif block_size > 1:
            cb, exact_cb_score, exact_csq, _ = _codebook_aux_on(
                bits, block_size, self.dist, m, self.codebook_init, device_str
            )
        n_blocks = m // block_size
        n_total_blocks = max(1, xr.numel() // block_size)
        lut_spec = (
            self._use_lut_code_bits(block_size, code_bits, n_total_blocks, xr.is_cuda)
            if explicit_code_bits
            else self._use_lut(block_size, bits, n_total_blocks, xr.is_cuda)
        )
        if lut_spec is None:
            lut = None
        elif cb_cal is not None:
            lut = cal.lut_on(role, bits, block_size, lut_spec[0], lut_spec[1], xsub.device)
        elif explicit_code_bits:
            lut = _block_lut_code_bits_on(
                code_bits,
                block_size,
                self.dist,
                m,
                self.codebook_init,
                lut_spec[0],
                lut_spec[1],
                device_str,
            )
        else:
            lut = _block_lut_on(
                bits,
                block_size,
                self.dist,
                m,
                self.codebook_init,
                lut_spec[0],
                lut_spec[1],
                device_str,
            )

        def _payload(codes_: torch.Tensor, stored_scale_: torch.Tensor) -> dict:
            result = {
                "codes": codes_,
                "scale": stored_scale_,
                "bits": bits,
                "m": m,
                "block_size": block_size,
                "rot": rot,
                "dist": self.dist,
                "codebook_init": self.codebook_init,
            }
            if explicit_code_bits:
                result["code_bits"] = code_bits
            if cb_cal is not None:
                # Decode must gather from the SAME learned codebook; record the
                # role + artifact digest (same convention as the "c" rot tag).
                result["cb_cal"] = (role, cal.digest)
            return result

        # Do not dispatch the monolithic Triton round-trip kernel as a payload
        # encoder.  Its reduction order changed roughly 0.5--0.8% of codes on
        # a canonical rate-3 B200 fixture.  The narrow assignment kernels used
        # below are different: they materialise the audited eager fp16 rounding
        # points and are enabled only for the b=2/4,T=2 LUT and b=8 half-GEMM
        # layouts covered by bit-exact large-fixture/tie tests.  RMS and scale
        # fitting remain on the established tensor implementation.

        # One RMS scale per whole subset vector (not per block) -- matches
        # TurboQuant/HALO's storage convention exactly (a single fp16 scale
        # per vector regardless of block_size).
        rms = xr.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)

        triton_exact_b8 = None
        if (
            block_size == 8
            and exact_cb_score is not None
            and xr.is_cuda
            and not torch.compiler.is_compiling()
        ):
            from .triton_step import (
                quantize_exact_b8_triton,
                triton_exact_b8_supported,
            )

            if triton_exact_b8_supported(exact_cb_score, exact_csq, xr.device):
                triton_exact_b8 = quantize_exact_b8_triton

        def _lookup(vec_full):
            vb = vec_full.reshape(*vec_full.shape[:-1], n_blocks, block_size)
            if scalar_bounds is not None:
                codes_ = torch.searchsorted(scalar_bounds, vb.squeeze(-1).contiguous())
                chat_b = lookup_codewords(cb, codes_)
            elif lut is not None:
                codes_ = quantize_via_lut(
                    vb,
                    lut,
                    lut_spec[0],
                    cb,
                    scoring_cb=exact_cb_score,
                    scoring_csq=exact_csq,
                )
                chat_b = lookup_codewords(cb, codes_)
            elif triton_exact_b8 is not None:
                codes_ = triton_exact_b8(vb, exact_cb_score, exact_csq)
                chat_b = lookup_codewords(cb, codes_)
            elif exact_cb_score is not None:
                score_vb = (
                    vb.half()
                    if vb.is_cuda and vb.dtype == torch.float32
                    else vb.to(dtype=exact_cb_score.dtype)
                )
                codes_ = quantize_exact_pre(score_vb, exact_cb_score, exact_csq)
                chat_b = lookup_codewords(cb, codes_)
            else:
                codes_, chat_b = quantize_and_lookup(vb, cb)
            return codes_, chat_b.reshape(*vec_full.shape[:-1], m)

        def _requant(z_full):
            codes2, chat2 = _lookup(z_full)
            return chat2, codes2

        codes, chat = _lookup(xr / rms)
        scale, codes2 = fit_scale(
            xr, chat, self._scale_mode_for(role), rms, requantize_fn=_requant
        )
        if codes2 is not None:
            codes = codes2
        # ``dist`` is stored (as in TurboQuant) so decode picks the same
        # codebook even if the codec instance is reconfigured between
        # encode and decode.
        # The logical accounting and native backend both store one signed
        # fp16 scale per vector.  Materialise that precision in the research
        # payload as well so quality and logical data-plane rate describe the
        # same codec.
        stored_scale = scale.to(dtype=torch.float16)
        payload = _payload(codes, stored_scale)
        if self.use_qjl:
            recon = lookup_codewords(cb, codes).reshape(*codes.shape[:-1], m) * stored_scale.to(
                dtype=cb.dtype
            )
            resid = xr - recon
            # TurboQuant-prod QJL: S is a deterministic Gaussian projection.
            # Storing sign(Sr), rather than sign(r), is essential to the
            # unbiased inner-product estimator used at decode.
            qjl_seed = int(self.seed) + 104729
            proj = _qjl_projection_on(m, qjl_seed, str(xsub.device))
            projected = resid.float().reshape(-1, m) @ proj.t()
            payload["qjl_sign"] = (projected >= 0).reshape_as(resid)
            payload["qjl_norm"] = resid.float().norm(dim=-1, keepdim=True).to(dtype=torch.float16)
            payload["qjl_seed"] = qjl_seed
        return payload

    def _decode_rotor(self, p: dict, device, dtype) -> torch.Tensor:
        dist = p.get("dist", self.dist)  # pre-fix payloads lack "dist"
        if "cb_cal" in p:
            role, digest = p["cb_cal"]
            cal = self._calib()
            if cal is None or cal.digest != digest:
                raise ValueError(
                    "payload was encoded with a calibrated codebook "
                    f"(digest {digest}) but the codec has "
                    f"{'no' if cal is None else 'a different'} calibration loaded"
                )
            cb = cal.codebook_on(role, p["bits"], p["block_size"], device)
        else:
            codebook_init = p.get("codebook_init", self.codebook_init)
            if "code_bits" in p:
                cb = _block_codebook_code_bits_on(
                    p["code_bits"], p["block_size"], dist, p["m"], codebook_init, str(device)
                )
            else:
                cb = _block_codebook_on(
                    p["bits"], p["block_size"], dist, p["m"], codebook_init, str(device)
                )
        if p["codes"].is_cuda and "qjl_sign" not in p:
            from .triton_step import (
                decode_payload_triton,
                triton_payload_decode_supported,
            )

            if triton_payload_decode_supported(p["m"], p["block_size"], cb.shape[0]):
                xr = decode_payload_triton(p["codes"], p["scale"], cb, p["m"], p["block_size"])
            else:
                chat = lookup_codewords(cb, p["codes"]).reshape(*p["codes"].shape[:-1], p["m"])
                scale = p["scale"].to(device=device, dtype=chat.dtype)
                xr = chat * scale
        else:
            chat = lookup_codewords(cb, p["codes"]).reshape(*p["codes"].shape[:-1], p["m"])
            # New payloads contain fp16 metadata; legacy fp32 payloads remain
            # readable.  Restore explicitly to the codebook arithmetic dtype.
            scale = p["scale"].to(device=device, dtype=chat.dtype)
            xr = chat * scale
        if "qjl_sign" in p:
            sign = torch.where(p["qjl_sign"], 1.0, -1.0).to(torch.float32)
            if "qjl_norm" in p:
                proj = _qjl_projection_on(p["m"], p["qjl_seed"], str(device))
                qjl = (sign.reshape(-1, p["m"]) @ proj).reshape_as(xr)
                qjl = qjl * (math.sqrt(math.pi / 2.0) / float(p["m"]))
                qjl_norm = p["qjl_norm"].to(device=device, dtype=torch.float32)
                xr = xr.float() + qjl * qjl_norm
            else:
                # Decode payloads produced by the old coordinate-sign
                # prototype, while all new payloads use the Gaussian sketch.
                xr = xr + sign.to(xr.dtype) * p["qjl_rms"]
        return self._unrotate(xr, p["m"], p["rot"])

    # --- full-tensor encode / decode (handles the outlier split) -----------
    def _encode_tensor(self, x: torch.Tensor, bits: int, role: str, layer_idx=None) -> dict:
        head_dim = x.shape[-1]
        n_outlier = self._n_outlier_for(role)
        if n_outlier >= head_dim and n_outlier > 0:
            raise ValueError(
                f"{role} n_outlier={n_outlier} must be smaller than head_dim={head_dim}"
            )
        if n_outlier <= 0:
            return {
                "mode": "single",
                "main": self._quant_rotor(x, bits, role, layer_idx=layer_idx),
                "head_dim": head_dim,
            }
        # Outlier mixed-precision: top-n_outlier channels by energy over tokens.
        energy = x.float().abs().amax(dim=-2)
        energy = energy.reshape(-1, head_dim).mean(dim=0)
        out_idx = torch.argsort(energy, descending=True)[:n_outlier]
        out_idx, _ = torch.sort(out_idx)
        mask = torch.zeros(head_dim, dtype=torch.bool, device=x.device)
        mask[out_idx] = True
        return {
            "mode": "outlier",
            "head_dim": head_dim,
            "mask": mask,
            "out": self._quant_rotor(
                x[..., mask], self._outlier_bits_for(role, bits), role, layer_idx=layer_idx
            ),
            "in": self._quant_rotor(x[..., ~mask], bits, role, layer_idx=layer_idx),
        }

    def _decode_tensor(self, payload: dict, dtype) -> torch.Tensor:
        if payload["mode"] == "single":
            device = payload["main"]["codes"].device
            return self._decode_rotor(payload["main"], device, dtype).to(dtype)
        device = payload["out"]["codes"].device
        head_dim, mask = payload["head_dim"], payload["mask"]
        x_out = self._decode_rotor(payload["out"], device, dtype)
        x_in = self._decode_rotor(payload["in"], device, dtype)
        shape = list(x_out.shape[:-1]) + [head_dim]
        x = torch.empty(shape, device=device, dtype=x_out.dtype)
        x[..., mask] = x_out
        x[..., ~mask] = x_in
        return x.to(dtype)

    def _stored_bits_from_main_payload(self, x: torch.Tensor, payload: dict) -> int:
        """Count code and fp16-scale data-plane fields for one main payload.

        The self-describing recipe tags in ``payload`` are cache-level control
        state and their Python/container representation is intentionally not
        part of this logical fixed-width rate.
        """
        b, h, s, d = x.shape
        vecs = b * h * s
        block_size = int(payload["block_size"])
        code_bits = int(payload.get("code_bits", int(payload["bits"]) * block_size))
        return vecs * (d // block_size) * code_bits + vecs * 16

    @staticmethod
    def _adaptive_selector_mask(selector_bits: int) -> int:
        """Return the fp16 fraction-bit mask used by Adaptive HQ's carrier.

        A binary16 word has ten low-order fraction bits.  Adaptive HQ stores
        its per-update split selector in the low-order fraction bits of the
        first value-scale word and reconstructs that scale after clearing the
        carrier bits.  The selector therefore consumes scale precision, not
        extra data-plane bits.  Keeping this limit explicit prevents a future
        candidate-set expansion from silently overwriting exponent/sign bits.
        """

        bits = int(selector_bits)
        if bits < 0 or bits > 10:
            raise ValueError(
                "Adaptive HQ selector needs between 0 and 10 fp16 fraction "
                f"bits, got {selector_bits!r}"
            )
        return (1 << bits) - 1

    @classmethod
    def _embed_adaptive_selector(
        cls,
        scale: torch.Tensor,
        selected_pair_index: int,
        selector_bits: int,
    ) -> torch.Tensor:
        """Embed a split index in one existing fp16 value-scale word.

        The returned tensor remains a 16-bit field.  Its first word contains
        the original sign/exponent and upper fraction bits plus the selector
        in the low fraction bits.  Decode clears those low bits before using
        the scale, so the rate match is exact and the quality cost is explicit
        reduced precision for one value scale per encode update.
        """

        if scale.dtype != torch.float16:
            raise TypeError("Adaptive HQ selector carrier requires an fp16 scale tensor")
        if scale.numel() < 1:
            raise ValueError("Adaptive HQ selector carrier needs at least one value scale")
        mask = cls._adaptive_selector_mask(selector_bits)
        selected = int(selected_pair_index)
        if selected < 0 or selected > mask:
            raise ValueError(
                f"Adaptive HQ selector index {selected} does not fit in " f"{selector_bits} bits"
            )
        carried = scale.clone().contiguous()
        if mask == 0:
            if selected != 0:
                raise ValueError("a zero-bit Adaptive HQ selector can encode only index zero")
            return carried
        words = carried.view(torch.int16).reshape(-1)
        word = words[0].to(torch.int32) & 0xFFFF
        clear_mask = 0xFFFF ^ mask
        words[0] = ((word & clear_mask) | selected).to(torch.int16)
        return carried

    @classmethod
    def _extract_adaptive_selector(
        cls,
        scale: torch.Tensor,
        selector_bits: int,
    ) -> int:
        """Read the split index stored in the first fp16 value-scale word."""

        if scale.dtype != torch.float16:
            raise TypeError("Adaptive HQ selector carrier requires an fp16 scale tensor")
        if scale.numel() < 1:
            raise ValueError("Adaptive HQ selector carrier needs at least one value scale")
        mask = cls._adaptive_selector_mask(selector_bits)
        if mask == 0:
            return 0
        word = scale.contiguous().view(torch.int16).reshape(-1)[0].to(torch.int32)
        return int((word & mask).item())

    @classmethod
    def _clear_adaptive_selector(
        cls,
        scale: torch.Tensor,
        selector_bits: int,
    ) -> torch.Tensor:
        """Recover the reduced-precision fp16 scale carried by the payload."""

        if scale.dtype != torch.float16:
            raise TypeError("Adaptive HQ selector carrier requires an fp16 scale tensor")
        if scale.numel() < 1:
            raise ValueError("Adaptive HQ selector carrier needs at least one value scale")
        mask = cls._adaptive_selector_mask(selector_bits)
        restored = scale.clone().contiguous()
        if mask == 0:
            return restored
        words = restored.view(torch.int16).reshape(-1)
        word = words[0].to(torch.int32) & 0xFFFF
        words[0] = (word & (0xFFFF ^ mask)).to(torch.int16)
        return restored

    def _adaptive_code_bit_pairs(self, block_size: int) -> list[tuple[int, int]]:
        total = self._pair_code_bits_total(block_size)
        if total is None:
            raise ValueError("adaptive_bit_split needs a total K/V bit budget")
        # AdaptiveHQ is deliberately a narrow opt-in around the paper HQ
        # story: neutral symmetric allocation or the same half-bit-per-role
        # key-heavy offset used by fixed HQ.  Development froze this two-pair
        # set globally; it is not enumerated or selected per model/task.
        if total % 2:
            raise ValueError(
                "the safe AdaptiveHQ candidate policy requires an even total K/V bit budget"
            )
        margin = int(self.adaptive_bit_margin)
        symmetric = (total // 2, total // 2)
        frac = 0.5 if self.key_bit_fraction is None else float(self.key_bit_fraction)
        incumbent_key_bits = min(total - 1, max(1, int(round(total * frac))))
        requested_incumbent = (incumbent_key_bits, total - incumbent_key_bits)
        if margin == 0:
            # A singleton AdaptiveHQ control must preserve its declared prior.
            # In particular, the formal recipe's singleton is fixed HQ itself
            # (K5/V3, K7/V5, or K9/V7), not the symmetric challenger.
            pairs = [requested_incumbent]
            if any(
                role_bits < 1 or role_bits > int(self.max_codebook_bits)
                for role_bits in requested_incumbent
            ):
                raise ValueError(
                    "the singleton AdaptiveHQ incumbent is infeasible under "
                    f"total={total} and max_codebook_bits={self.max_codebook_bits}"
                )
            return pairs
        role_offset_bits = max(1, int(block_size) // 2)
        key_heavy = (symmetric[0] + role_offset_bits, symmetric[1] - role_offset_bits)
        pairs = [symmetric, key_heavy]
        if any(
            key_bits < 1
            or value_bits < 1
            or key_bits > int(self.max_codebook_bits)
            or value_bits > int(self.max_codebook_bits)
            for key_bits, value_bits in pairs
        ):
            raise ValueError(
                "the safe AdaptiveHQ pairs are infeasible under "
                f"total={total} and max_codebook_bits={self.max_codebook_bits}"
            )
        # ``key_bit_fraction`` remains only a deterministic immature-state
        # prior.  It cannot expand or otherwise alter the frozen candidate set.
        return pairs

    def _adaptive_distortion_table(
        self,
        block_size: int,
        pairs: list[tuple[int, int]],
        m: int,
    ) -> list[dict]:
        """Return candidate constants for the actual source/codebook/scales."""
        key_mode = self._scale_mode_for("k")
        value_mode = self._scale_mode_for("v")
        selector_bits = (len(pairs) - 1).bit_length()
        table = []
        for key_bits, value_bits in pairs:
            value_base = _adaptive_datafree_relative_distortion(
                int(value_bits),
                int(block_size),
                self.dist,
                int(m),
                self.codebook_init,
                value_mode,
            )
            value_all_carriers = _adaptive_datafree_relative_distortion(
                int(value_bits),
                int(block_size),
                self.dist,
                int(m),
                self.codebook_init,
                value_mode,
                selector_bits,
            )
            table.append(
                {
                    "pair": [int(key_bits), int(value_bits)],
                    "value_scale_fraction_lsb_clear_bits": int(selector_bits),
                    "key_relative_distortion": _adaptive_datafree_relative_distortion(
                        int(key_bits),
                        int(block_size),
                        self.dist,
                        int(m),
                        self.codebook_init,
                        key_mode,
                    ),
                    "value_relative_distortion_base": value_base,
                    "value_selector_carrier_relative_distortion_delta_per_vector": (
                        value_all_carriers - value_base
                    ),
                }
            )
        return table

    def _adaptive_policy_config(
        self,
        block_size: int,
        pairs: list[tuple[int, int]],
        m: int,
        table: list[dict],
    ) -> dict:
        total = int(pairs[0][0] + pairs[0][1])
        frac = 0.5 if self.key_bit_fraction is None else float(self.key_bit_fraction)
        prior_k = min(total - 1, max(1, int(round(total * frac))))
        prior = min(pairs, key=lambda pair: (abs(pair[0] - prior_k), pair[0]))
        return {
            "policy_version": ADAPTIVE_HQ_POLICY_VERSION,
            "distortion_table_version": ADAPTIVE_HQ_DISTORTION_TABLE_VERSION,
            "objective": ADAPTIVE_HQ_OBJECTIVE,
            "power_statistic": (
                "cumulative_raw_pre_rotation_sum_squares_divided_by_element_count"
            ),
            "current_update_included": True,
            "maturity_gate": "cumulative_batch_tokens",
            "head_dim": int(m),
            "block_size": int(block_size),
            "source_law": self.dist,
            "codebook_init": self.codebook_init,
            "key_scale_mode": self._scale_mode_for("k"),
            "value_scale_mode": self._scale_mode_for("v"),
            "key_error_weight": float(self.key_error_weight),
            "value_error_weight": float(self.value_error_weight),
            "stats_min_tokens": int(self.adaptive_stats_min_tokens),
            "candidate_pairs": [[int(k), int(v)] for k, v in pairs],
            "candidate_rotations": [
                {
                    "pair": [int(k), int(v)],
                    "key": self._resolved_mode(
                        int(m), int(block_size), float(k) / float(block_size)
                    ),
                    "value": self._resolved_mode(
                        int(m), int(block_size), float(v) / float(block_size)
                    ),
                }
                for k, v in pairs
            ],
            "prior_pair": [int(prior[0]), int(prior[1])],
            "incumbent_pair": [int(prior[0]), int(prior[1])],
            "switch_margin_fraction": float(ADAPTIVE_HQ_SWITCH_MARGIN_FRACTION),
            "switch_margin_basis": ADAPTIVE_HQ_SWITCH_MARGIN_BASIS,
            "hysteresis": ADAPTIVE_HQ_HYSTERESIS,
            "selector_bits_per_encode": int((len(pairs) - 1).bit_length()),
            "selector_incremental_stored_bits_per_encode": 0,
            "selector_carrier_version": ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION,
            "selector_carrier": "first_value_fp16_scale_low_fraction_bits",
            "selector_scale_reconstruction": "clear_carrier_fraction_bits",
            "selector_carrier_distortion_amortization": (
                "signed_qmc_all_scales_delta_divided_by_current_update_vectors"
            ),
            "datafree_distortion_table": table,
        }

    @staticmethod
    def _adaptive_state_for_context(ctx: dict | None, policy: dict) -> tuple[dict, str]:
        """Return cache-owned layer state, or an ephemeral direct-call state."""
        owner = None if ctx is None else ctx.get("cache_state")
        scope = "cache_layer" if owner is not None else "update_local"
        if owner is None:
            owner = {}
        if not isinstance(owner, dict):
            raise TypeError("ctx['cache_state'] must be a mutable dict")
        state = owner.setdefault("adaptive_hq", {})
        if not isinstance(state, dict):
            raise TypeError("ctx['cache_state']['adaptive_hq'] must be a dict")
        if not state:
            state.update(
                {
                    "schema_version": 1,
                    "policy_version": ADAPTIVE_HQ_POLICY_VERSION,
                    # Keep cache-owned state disjoint from the payload metadata
                    # returned by ``encode``. A caller may freely edit that
                    # metadata without mutating the future causal policy.
                    "policy": copy.deepcopy(policy),
                    "cumulative": {
                        "key_square_sum": 0.0,
                        "value_square_sum": 0.0,
                        "key_element_count": 0,
                        "value_element_count": 0,
                        "token_count": 0,
                        "vector_count": 0,
                    },
                    "encoded_updates": 0,
                    "selector_bits_total": 0,
                    "selector_incremental_stored_bits_total": 0,
                    "split_histogram": {},
                    "selection_runs": [],
                    "last_selection": None,
                }
            )
        elif (
            state.get("policy_version") != ADAPTIVE_HQ_POLICY_VERSION
            or state.get("policy") != policy
        ):
            raise ValueError(
                "Adaptive HQ cache state was created for a different policy/config; "
                "reset the cache before reconfiguring the codec"
            )
        return state, scope

    @staticmethod
    def _adaptive_next_cumulative(
        state: dict,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[dict, dict]:
        """Include the current raw update in a proposed causal accumulator."""
        if key.shape != value.shape:
            raise ValueError("adaptive_bit_split expects key/value to have the same shape")
        if key.ndim != 4:
            raise ValueError("adaptive_bit_split expects (batch, heads, seq, head_dim) tensors")
        # One stacked reduction and one host transfer keeps the CUDA path to a
        # single synchronization.  Python float accumulation then preserves
        # prior chunks without any codec-global tensor state.
        update_sums = torch.stack(
            (key.float().square().sum(), value.float().square().sum())
        ).tolist()
        previous = state["cumulative"]
        batch, heads, seq, _ = key.shape
        cumulative = {
            "key_square_sum": float(previous["key_square_sum"]) + float(update_sums[0]),
            "value_square_sum": float(previous["value_square_sum"]) + float(update_sums[1]),
            "key_element_count": int(previous["key_element_count"]) + int(key.numel()),
            "value_element_count": int(previous["value_element_count"]) + int(value.numel()),
            "token_count": int(previous["token_count"]) + int(batch * seq),
            "vector_count": int(previous["vector_count"]) + int(batch * heads * seq),
        }
        powers = {
            "key": cumulative["key_square_sum"] / max(1, cumulative["key_element_count"]),
            "value": cumulative["value_square_sum"] / max(1, cumulative["value_element_count"]),
        }
        update = {
            "tokens": int(batch * seq),
            "vectors": int(batch * heads * seq),
            "key_elements": int(key.numel()),
            "value_elements": int(value.numel()),
        }
        return cumulative, {"powers": powers, "update": update}

    @staticmethod
    def _commit_adaptive_state(
        state: dict,
        cumulative: dict,
        split_meta: dict,
    ) -> None:
        """Commit a successful encode to cache-owned provenance/state."""
        state["cumulative"] = cumulative
        state["encoded_updates"] = int(state["encoded_updates"]) + 1
        state["selector_bits_total"] = int(state["selector_bits_total"]) + int(
            split_meta["selector_bits"]
        )
        state["selector_incremental_stored_bits_total"] = int(
            state["selector_incremental_stored_bits_total"]
        ) + int(split_meta["selector_incremental_stored_bits"])
        pair = [
            int(split_meta["key_code_bits_per_block"]),
            int(split_meta["value_code_bits_per_block"]),
        ]
        pair_key = f"{pair[0]}:{pair[1]}"
        update = split_meta["update"]
        bucket = state["split_histogram"].setdefault(
            pair_key,
            {"pair": pair, "updates": 0, "tokens": 0, "vectors": 0},
        )
        bucket["updates"] += 1
        bucket["tokens"] += int(update["tokens"])
        bucket["vectors"] += int(update["vectors"])

        runs = state["selection_runs"]
        if runs and runs[-1]["pair"] == pair:
            runs[-1]["updates"] += 1
            runs[-1]["tokens"] += int(update["tokens"])
            runs[-1]["vectors"] += int(update["vectors"])
            runs[-1]["end_cumulative_tokens"] = int(cumulative["token_count"])
            runs[-1]["end_cumulative_vectors"] = int(cumulative["vector_count"])
        else:
            runs.append(
                {
                    "pair": pair,
                    "updates": 1,
                    "tokens": int(update["tokens"]),
                    "vectors": int(update["vectors"]),
                    "start_cumulative_tokens": int(cumulative["token_count"] - update["tokens"]),
                    "end_cumulative_tokens": int(cumulative["token_count"]),
                    "start_cumulative_vectors": int(
                        cumulative["vector_count"] - update["vectors"]
                    ),
                    "end_cumulative_vectors": int(cumulative["vector_count"]),
                }
            )
        candidate_objectives = copy.deepcopy(split_meta["candidate_objectives"])
        selected_objective = candidate_objectives[int(split_meta["selected_pair_index"])]
        state["last_selection"] = {
            "pair": pair,
            "selected_pair_index": int(split_meta["selected_pair_index"]),
            "key_rotation_mode": str(split_meta["key_rotation_mode"]),
            "value_rotation_mode": str(split_meta["value_rotation_mode"]),
            "used_stats": bool(split_meta["used_stats"]),
            "update_tokens": int(update["tokens"]),
            "carrier_amortization_vector_count": int(update["vectors"]),
            "cumulative_token_count": int(cumulative["token_count"]),
            "cumulative_vector_count": int(cumulative["vector_count"]),
            "cumulative_key_power": float(split_meta["cumulative_key_power"]),
            "cumulative_value_power": float(split_meta["cumulative_value_power"]),
            "incumbent_pair": copy.deepcopy(split_meta["incumbent_pair"]),
            "previous_pair": copy.deepcopy(split_meta["previous_pair"]),
            "switch_margin_fraction": float(split_meta["switch_margin_fraction"]),
            "switch_margin_basis": str(split_meta["switch_margin_basis"]),
            "hysteresis": str(split_meta["hysteresis"]),
            "decision_reason": str(split_meta["decision_reason"]),
            "incumbent_objective": float(split_meta["incumbent_objective"]),
            "challenger_objective": (
                None
                if split_meta["challenger_objective"] is None
                else float(split_meta["challenger_objective"])
            ),
            "challenger_relative_improvement": (
                None
                if split_meta["challenger_relative_improvement"] is None
                else float(split_meta["challenger_relative_improvement"])
            ),
            "value_relative_distortion_effective": float(
                selected_objective["value_relative_distortion_effective"]
            ),
            "predicted_key_error": float(split_meta["predicted_key_error"]),
            "predicted_value_error": float(split_meta["predicted_value_error"]),
            "weighted_objective": float(split_meta["weighted_objective"]),
            "candidate_objectives": candidate_objectives,
        }

    def _select_adaptive_code_bits(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        block_size: int,
        ctx: dict | None,
    ) -> tuple[int, int, dict, dict, dict]:
        """Choose from causal powers and actual data-free codebook constants."""
        pairs = self._adaptive_code_bit_pairs(block_size)
        # The selected pair is data-dependent and therefore cannot be treated
        # like a static recipe tag.  A packed decoder needs one selector for
        # every jointly encoded K/V update.  ``bit_length(n - 1)`` is exactly
        # ceil(log2(n)) and correctly gives zero bits for a singleton set.
        selector_bits = (len(pairs) - 1).bit_length()
        m = int(key.shape[-1])
        table = self._adaptive_distortion_table(block_size, pairs, m)
        policy = self._adaptive_policy_config(block_size, pairs, m, table)
        state, state_scope = self._adaptive_state_for_context(ctx, policy)
        cumulative, stats = self._adaptive_next_cumulative(state, key, value)
        key_power = float(stats["powers"]["key"])
        value_power = float(stats["powers"]["value"])
        incumbent = tuple(policy["incumbent_pair"])
        last_selection = state.get("last_selection")
        previous_pair = (
            tuple(last_selection["pair"])
            if isinstance(last_selection, dict) and isinstance(last_selection.get("pair"), list)
            else None
        )
        used_stats = int(cumulative["token_count"]) >= int(self.adaptive_stats_min_tokens)
        kw, vw = float(self.key_error_weight), float(self.value_error_weight)
        update_vectors = int(stats["update"]["vectors"])
        scored = []
        for pair, constants in zip(pairs, table):
            pred_key = key_power * float(constants["key_relative_distortion"])
            effective_value_distortion = _adaptive_value_distortion_for_update(
                constants, update_vectors
            )
            pred_value = value_power * effective_value_distortion
            scored.append(
                {
                    "pair": pair,
                    "value_relative_distortion_effective": effective_value_distortion,
                    "carrier_amortization_vector_count": update_vectors,
                    "predicted_key_error": pred_key,
                    "predicted_value_error": pred_value,
                    "weighted_objective": kw * pred_key + vw * pred_value,
                }
            )
        selected, decision = _adaptive_hysteretic_choice(
            scored,
            incumbent,
            previous_pair,
            used_stats=used_stats,
        )
        best = selected["pair"]
        meta = {
            "policy": "cache_causal_datafree_rd",
            "policy_version": ADAPTIVE_HQ_POLICY_VERSION,
            "distortion_table_version": ADAPTIVE_HQ_DISTORTION_TABLE_VERSION,
            "objective": ADAPTIVE_HQ_OBJECTIVE,
            "used_stats": bool(used_stats),
            "key_code_bits_per_block": int(best[0]),
            "value_code_bits_per_block": int(best[1]),
            "key_rotation_mode": self._resolved_mode(
                m, block_size, float(best[0]) / float(block_size)
            ),
            "value_rotation_mode": self._resolved_mode(
                m, block_size, float(best[1]) / float(block_size)
            ),
            "candidates": pairs,
            "selected_pair_index": int(pairs.index(best)),
            "selector_bits": int(selector_bits),
            "selector_information_bits": int(selector_bits),
            "selector_incremental_stored_bits": 0,
            "selector_carrier_version": ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION,
            "selector_carrier": "first_value_fp16_scale_low_fraction_bits",
            "selector_scale_reconstruction": "clear_carrier_fraction_bits",
            "selector_carrier_distortion_amortization": (
                "signed_qmc_all_scales_delta_divided_by_current_update_vectors"
            ),
            "carrier_amortization_vector_count": update_vectors,
            "selection_scope": "per_encode_kv_pair",
            "state_scope": state_scope,
            # ``prior_pair`` remains as a compatibility spelling.  In v6 it is
            # explicitly the fixed-HQ incumbent, not a neutral symmetric pair.
            "prior_pair": list(incumbent),
            **decision,
            "cumulative_token_count": int(cumulative["token_count"]),
            "cumulative_vector_count": int(cumulative["vector_count"]),
            "cumulative_key_power": key_power,
            "cumulative_value_power": value_power,
            "predicted_key_error": float(selected["predicted_key_error"]),
            "predicted_value_error": float(selected["predicted_value_error"]),
            "weighted_objective": float(selected["weighted_objective"]),
            "candidate_objectives": [
                {
                    "pair": [int(row["pair"][0]), int(row["pair"][1])],
                    "value_relative_distortion_effective": float(
                        row["value_relative_distortion_effective"]
                    ),
                    "carrier_amortization_vector_count": int(
                        row["carrier_amortization_vector_count"]
                    ),
                    "predicted_key_error": float(row["predicted_key_error"]),
                    "predicted_value_error": float(row["predicted_value_error"]),
                    "weighted_objective": float(row["weighted_objective"]),
                }
                for row in scored
            ],
            # Do not alias cache-owned ``state['policy']`` even when ``ctx``
            # was supplied by a cache. This metadata is public/mutable.
            "datafree_distortion_table": copy.deepcopy(table),
            "update": stats["update"],
        }
        return int(best[0]), int(best[1]), meta, state, cumulative

    def _encode_adaptive_pair(self, key: torch.Tensor, value: torch.Tensor, ctx=None):
        if key.shape[-1] != value.shape[-1]:
            raise ValueError("adaptive_bit_split expects key/value to share head_dim")
        layer_idx = (ctx or {}).get("layer_idx")
        m = key.shape[-1]
        kb, vb = self.bits, self._value_bits()
        # The adaptive budget is explicitly measured per matched K/V block;
        # nominal integer ``bits`` must not silently shrink that block before
        # the selected code-bit split is known.
        block_size = self._resolve_block_size_for("k", kb, m, code_bits_override=1)
        best_k, best_v, split_meta, state, cumulative = self._select_adaptive_code_bits(
            key, value, block_size, ctx
        )
        key_payload = self._quant_rotor(
            key, kb, "k", layer_idx=layer_idx, code_bits_override=best_k
        )
        value_payload = self._quant_rotor(
            value, vb, "v", layer_idx=layer_idx, code_bits_override=best_v
        )
        # The split is data-dependent and must be recoverable by a packed
        # decoder.  Store it inside the already-budgeted first fp16 value
        # scale, sacrificing only that word's low fraction precision.  This
        # avoids the former selector-rate overrun while retaining an explicit,
        # auditable carrier rather than treating a data-dependent Python tag
        # as free metadata.
        value_payload["scale"] = self._embed_adaptive_selector(
            value_payload["scale"],
            int(split_meta["selected_pair_index"]),
            int(split_meta["selector_bits"]),
        )
        kp = {"mode": "single", "main": key_payload, "head_dim": m}
        vp = {"mode": "single", "main": value_payload, "head_dim": m}
        stored = self._stored_bits_from_main_payload(
            key, key_payload
        ) + self._stored_bits_from_main_payload(value, value_payload)
        # Commit only after both role payloads were produced successfully; a
        # failed codebook/encode must not advance cache-owned causal state.
        self._commit_adaptive_state(state, cumulative, split_meta)
        meta = {
            "dtype": key.dtype,
            "stored_bits": int(stored),
            "adaptive_bit_split": split_meta,
        }
        return CompressedKV(key_q=kp, value_q=vp, meta=meta)

    def _decode_adaptive_pair(self, compressed: CompressedKV, dtype):
        """Validate and decode an Adaptive HQ payload from its in-budget carrier."""

        split = compressed.meta.get("adaptive_bit_split")
        if not isinstance(split, dict):
            raise ValueError("Adaptive HQ payload is missing split provenance")
        if split.get("policy_version") != ADAPTIVE_HQ_POLICY_VERSION:
            raise ValueError("Adaptive HQ payload policy version does not match this decoder")
        if split.get("selector_carrier_version") != ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION:
            raise ValueError("Adaptive HQ payload selector carrier version is unsupported")
        selector_bits = split.get("selector_bits")
        if not isinstance(selector_bits, int) or isinstance(selector_bits, bool):
            raise ValueError("Adaptive HQ payload has an invalid selector width")
        try:
            value_main = compressed.value_q["main"]
            key_main = compressed.key_q["main"]
            carried_scale = value_main["scale"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Adaptive HQ payload has malformed role fields") from exc
        if not isinstance(value_main, dict) or not isinstance(key_main, dict):
            raise ValueError("Adaptive HQ payload has malformed main role fields")
        try:
            block_size = int(key_main["block_size"])
            if int(value_main["block_size"]) != block_size:
                raise ValueError("Adaptive HQ key/value block sizes disagree")
            expected_candidates = self._adaptive_code_bit_pairs(block_size)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Adaptive HQ payload cannot resolve its candidate set") from exc
        candidates = split.get("candidates")
        try:
            normalized_candidates = [(int(pair[0]), int(pair[1])) for pair in candidates]
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("Adaptive HQ payload has malformed candidate pairs") from exc
        if normalized_candidates != expected_candidates:
            raise ValueError("Adaptive HQ payload candidate set does not match the decoder policy")
        expected_selector_bits = (len(expected_candidates) - 1).bit_length()
        if selector_bits != expected_selector_bits:
            raise ValueError("Adaptive HQ payload selector width does not match its candidate set")
        selected_index = self._extract_adaptive_selector(carried_scale, selector_bits)
        if selected_index >= len(normalized_candidates):
            raise ValueError("Adaptive HQ scale carrier selects an invalid candidate")
        key_bits, value_bits = normalized_candidates[selected_index]
        declared = (
            split.get("selected_pair_index"),
            split.get("key_code_bits_per_block"),
            split.get("value_code_bits_per_block"),
            key_main.get("code_bits", int(key_main["bits"]) * int(key_main["block_size"])),
            value_main.get("code_bits", int(value_main["bits"]) * int(value_main["block_size"])),
        )
        expected = (selected_index, key_bits, value_bits, key_bits, value_bits)
        if declared != expected:
            raise ValueError(
                "Adaptive HQ selector carrier, split provenance, and code streams disagree"
            )

        # Do not mutate the caller-owned payload.  Only the value role carries
        # the selector; keys decode directly from their stored scale.
        decoded_value_main = dict(value_main)
        decoded_value_main["scale"] = self._clear_adaptive_selector(carried_scale, selector_bits)
        decoded_value = dict(compressed.value_q)
        decoded_value["main"] = decoded_value_main
        return self._decode_tensor(compressed.key_q, dtype), self._decode_tensor(
            decoded_value, dtype
        )

    def _stored_bits(self, x: torch.Tensor, bits: int, role: str) -> int:
        """Fixed-width data-plane accounting at the requested logical rate.

        ``block_size`` only changes how the ``bits``-per-coordinate budget is
        allocated (jointly per block vs. independently per coordinate); the
        total number of payload bits is unchanged (``n_blocks * bits *
        block_size == d * bits``), which is what makes the block-VQ
        comparison to TurboQuant an honest equal-storage comparison.
        Enabled outlier bitmap/QJL fields are charged below.  Self-describing
        control tags, Python/container bytes, and global codebook/LUT/rotation
        state are outside this per-token data-plane count.
        """
        b, h, s, d = x.shape
        vecs = b * h * s
        block_size = self._resolve_block_size_for(role, bits, d)
        code_bits = self._code_bits_for(role, bits, block_size)
        n_outlier = self._n_outlier_for(role)
        split = 0 < n_outlier < d  # outlier split => two subsets per vector
        if not split:
            payload_bits = vecs * (d // block_size) * code_bits
            scale_bits = vecs * 16
            mask_bits = 0
        else:
            no = n_outlier
            payload_bits = vecs * (no * self._outlier_bits_for(role, bits) + (d - no) * bits)
            scale_bits = vecs * 2 * 16
            # The outlier set is selected from this update's data and is not
            # reconstructible from the codes.  Store one packed channel bitmap
            # per role payload; omitting it overstates compression.
            mask_bits = d
        # QJL residual: 1 projected-sign bit per coordinate + one fp16
        # residual L2 norm per
        # encoded subset vector (two subsets when the outlier split is active).
        qjl_bits = (x.numel() * 1 + vecs * (2 if split else 1) * 16) if self.use_qjl else 0
        return payload_bits + scale_bits + mask_bits + qjl_bits

    def calibrate(self, kv_stats, ctx=None):
        """Fit and load a SpinQuant-style calibration (learned per-role
        rotation + real-data codebook refit) from captured KV samples.

        Args:
            kv_stats: ``{"key": (N, head_dim), "value": (N, head_dim)}`` real
                KV vectors (e.g. from
                :func:`.hayatequant_calib.capture_kv_samples`).
            ctx: Optional kwargs forwarded to
                :func:`.hayatequant_calib.calibrate_hayatequant`
                (``steps``, ``lr``, ``learn_rotation``, ``learn_codebook``...).

        Returns:
            The fitted :class:`.hayatequant_calib.HayateQuantCalibration`
            (also installed on this codec; save it with ``.save(path)`` and
            reload via ``HayateQuantCodec(calibration=path)``).
        """
        from .calibration import calibrate_hayatequant

        self.calibration = calibrate_hayatequant(
            self, kv_stats["key"], kv_stats["value"], **(ctx or {})
        )
        self.flag_calibration = True
        return self.calibration

    def encode(self, key, value, ctx=None) -> CompressedKV:
        layer_idx = (ctx or {}).get("layer_idx")
        if self._adaptive_bit_split_enabled():
            return self._encode_adaptive_pair(key, value, ctx=ctx)
        kp = self._encode_tensor(key, self.bits, "k", layer_idx=layer_idx)
        vp = self._encode_tensor(value, self._value_bits(), "v", layer_idx=layer_idx)
        stored = self._stored_bits(key, self.bits, "k") + self._stored_bits(
            value, self._value_bits(), "v"
        )
        return CompressedKV(
            key_q=kp, value_q=vp, meta={"dtype": key.dtype, "stored_bits": int(stored)}
        )

    def decode(self, compressed: CompressedKV, ctx=None):
        dtype = compressed.meta.get("dtype", torch.float16)
        if "adaptive_bit_split" in compressed.meta:
            return self._decode_adaptive_pair(compressed, dtype)
        return self._decode_tensor(compressed.key_q, dtype), self._decode_tensor(
            compressed.value_q, dtype
        )

    # --- fused fake-quant round-trip (the serving hot path) -----------------
    def _rt_supported(self, m: int) -> bool:
        """Whether the fused round-trip covers this configuration.

        The fused path handles no-outlier/no-QJL configurations with a
        GEMM-expressible rotation, including calibrated rotations/codebooks.
        Anything else transparently falls back to the payload path, so
        ``fake_quant_roundtrip`` is always semantically valid.
        """
        if self._adaptive_bit_split_enabled():
            return False
        if self._n_outlier_for("k") > 0 or self._n_outlier_for("v") > 0 or self.use_qjl:
            return False
        for role, bits in (("k", self.bits), ("v", self._value_bits())):
            bs = self._resolve_block_size_for(role, bits, m)
            rotation_rate = self._rotation_rate_for(role, bits, bs)
            if self._resolved_mode(m, bs, rotation_rate) == "hadamard":
                # The sequential WHT butterfly is not a cached (m, m) GEMM;
                # it only resolves at very large head dims -- fall back.
                return False
        return True

    def _rt_rotation(
        self,
        m: int,
        block_size: int,
        bits: float,
        device,
        role: str | None = None,
        layer_idx=None,
    ) -> torch.Tensor:
        """The (m, m) fp32 rotation matrix ``_rotate`` would apply (cached)."""
        cal = self._calib()
        if cal is not None and role is not None:
            w = cal.rotation_on(role, m, device, layer_idx=layer_idx)
            if w is not None:
                return w
        mode = self._resolved_mode(m, block_size, bits)
        dev = str(device)
        if mode == "hybrid" and is_power_of_two(m):
            return hybrid_rotation_matrix(m, block_size, self.seed, dev, torch.float32)
        if mode == "hadamard_gemm" and is_power_of_two(m):
            return signed_hadamard_matrix(m, self.seed, dev, torch.float32)
        return _dense_rotation_on(m, self.seed, dev, torch.float32)

    def _rt_call(
        self, fn, x: torch.Tensor, bits: int, modes: tuple, role: str | None = None, layer_idx=None
    ) -> torch.Tensor:
        """Run the fused round-trip core on stacked roles ``x`` (R, b, h, s, m)."""
        m = x.shape[-1]
        block_size = self._resolve_block_size_for(role or "k", bits, m)
        code_bits = self._code_bits_for(role or "k", bits, block_size)
        explicit_code_bits = code_bits != int(bits) * int(block_size)
        rotation_rate = self._rotation_rate_for(
            role or "k",
            bits,
            block_size,
            code_bits_override=(code_bits if explicit_code_bits else None),
        )
        w = self._rt_rotation(m, block_size, rotation_rate, x.device, role, layer_idx=layer_idx)
        cal = self._calib()
        cb_cal = (
            cal.codebook_on(role, bits, block_size, x.device)
            if cal is not None
            and role is not None
            and m == cal.head_dim
            and not explicit_code_bits
            else None
        )
        if cb_cal is not None:
            cb32, cb_score, csq, bounds = _codebook_aux_from_tensor(cb_cal, x.device)
        elif explicit_code_bits:
            cb32, cb_score, csq, bounds = _codebook_aux_code_bits_on(
                code_bits, block_size, self.dist, m, self.codebook_init, str(x.device)
            )
        else:
            cb32, cb_score, csq, bounds = _codebook_aux_on(
                bits, block_size, self.dist, m, self.codebook_init, str(x.device)
            )
        # Per-role block count (x[0], not the stacked x) so the LUT-vs-exact
        # crossover decision is identical to the standalone encode path.
        n_total_blocks = max(1, x[0].numel() // block_size)
        spec = (
            self._use_lut_code_bits(block_size, code_bits, n_total_blocks, x.is_cuda)
            if explicit_code_bits
            else self._use_lut(block_size, bits, n_total_blocks, x.is_cuda)
        )
        # With ONECOMP_TRITON_CODEC=1 the whole normalize/assign/scale/
        # reconstruct pipeline collapses into ONE fused Triton kernel between
        # the two rotation GEMMs (see .triton_step).  It always serves the
        # exact-search (decode-step) regime; ONECOMP_TRITON_PREFILL=1
        # additionally routes the prefill (LUT) regime through it -- the
        # kernel's exact per-row search is compute-light enough that it can
        # beat the memory-bound LUT gather (measured per-GPU, see
        # papers/hayatequant_paper/experiments/run_rt_speed_v11_prefill.sh).
        if x.is_cuda and os.getenv("ONECOMP_TRITON_CODEC", "0") == "1":
            from .triton_step import triton_step_supported, roundtrip_step_triton

            if triton_step_supported(m, block_size, cb32.shape[0], modes) and (
                spec is None or os.getenv("ONECOMP_TRITON_PREFILL", "0") == "1"
            ):
                return roundtrip_step_triton(x, w, cb32, block_size, modes)
        if spec is None:
            lut, bins = None, 0
        elif cb_cal is not None:
            lut = cal.lut_on(role, bits, block_size, spec[0], spec[1], x.device)
            bins = spec[0]
        elif explicit_code_bits:
            lut = _block_lut_code_bits_on(
                code_bits,
                block_size,
                self.dist,
                m,
                self.codebook_init,
                spec[0],
                spec[1],
                str(x.device),
            )
            bins = spec[0]
        else:
            lut = _block_lut_on(
                bits, block_size, self.dist, m, self.codebook_init, spec[0], spec[1], str(x.device)
            )
            bins = spec[0]
        return fn(x, w, cb32, cb_score, csq, bounds, lut, bins, block_size, modes)

    def fake_quant_roundtrip(self, key, value, ctx=None):
        """Fused encode+decode: ``(key_hat, value_hat, stored_bits)``.

        Numerically equivalent to ``decode(encode(key, value))`` (same
        rotation, codebook, assignment and scale-fit operations in the same
        dtypes -- see ``test_hayatequant_roundtrip_matches_encode_decode``)
        but skips payload construction and the decode-time codeword gather,
        and reuses cached per-codebook constants.  K/V retain separate rotation
        GEMM shapes so the result also preserves public-encoder fp16 scale
        rounding at the rare boundary cases.
        With ``ONECOMP_COMPILE_CODEC=1`` the whole pipeline additionally
        replays as a single CUDA graph.  ``ONECOMP_COMPILE_DECODE_ONLY=1``
        restricts that graphing to serving decode steps (``seq_len == 1``),
        preserving the eager prefill path exactly.  Configurations outside
        the fused path's coverage (outliers, QJL, calibration, butterfly
        rotation) fall back to the base-class encode/decode chain.
        """
        m = key.shape[-1]
        if not self._rt_supported(m):
            return super().fake_quant_roundtrip(key, value, ctx=ctx)
        kb, vb = self.bits, self._value_bits()
        stored = int(self._stored_bits(key, kb, "k") + self._stored_bits(value, vb, "v"))
        k_mode, v_mode = self._scale_mode_for("k"), self._scale_mode_for("v")
        compile_allowed = not (
            os.getenv("ONECOMP_COMPILE_DECODE_ONLY", "0") == "1" and key.shape[-2] != 1
        )
        compile_requested = compile_allowed and os.getenv("ONECOMP_COMPILE_CODEC", "0") == "1"
        if compile_requested and (key.device.type != "cuda" or value.device.type != "cuda"):
            raise RuntimeError(
                "ONECOMP_COMPILE_CODEC=1 is a CUDA-graph serving option and "
                "cannot be used for a CPU HayateQuant round trip; unset it or "
                "run the benchmark on CUDA"
            )
        fn, compiled = _rt_dispatch(compile_allowed=compile_allowed)
        dtype = key.dtype
        layer_idx = (ctx or {}).get("layer_idx")
        # Keep each role's GEMM shape identical to the public payload encoder.
        # Stacking K/V changes the GEMM's row count and can select a different
        # reduction kernel; the resulting last-bit fp32 difference occasionally
        # crosses an fp16 scale-rounding boundary.  That made the advertised
        # fused round trip differ from ``decode(encode(...))`` by one adjacent
        # fp16 scale even though K/V shared a rotation and codebook.  Separate
        # role calls preserve the audited payload arithmetic exactly.
        khat = self._rt_call(
            fn, key.float().unsqueeze(0), kb, (k_mode,), role="k", layer_idx=layer_idx
        )[0]
        vhat = self._rt_call(
            fn, value.float().unsqueeze(0), vb, (v_mode,), role="v", layer_idx=layer_idx
        )[0]
        if compiled:
            # CUDA-graph outputs live in the graph's memory pool and are
            # overwritten by the next replay; copy them out unconditionally.
            return khat.to(dtype, copy=True), vhat.to(dtype, copy=True), stored
        return khat.to(dtype), vhat.to(dtype), stored

    def get_kv_config(self) -> dict:
        cfg = super().get_kv_config()
        cfg.update(
            {
                "dist": self.dist,
                "value_bits": self.value_bits,
                "block_size": self.block_size,
                "max_codebook_bits": self.max_codebook_bits,
                "codebook_init": self.codebook_init,
                "required_codebook_bundle": self.required_codebook_bundle,
                "rotation_mode": self.rotation_mode,
                "seed": self.seed,
                "encoder": self.encoder,
                "lut_bins": self.lut_bins,
                "lut_candidates": self.lut_candidates,
                "hadamard_gemm_max_dim": self.hadamard_gemm_max_dim,
                "ls_scale": self.ls_scale,
                "key_scale_mode": self.key_scale_mode,
                "value_scale_mode": self.value_scale_mode,
                "n_outlier": self.n_outlier,
                "outlier_bits": self.outlier_bits,
                "key_n_outlier": self.key_n_outlier,
                "value_n_outlier": self.value_n_outlier,
                "key_outlier_bits": self.key_outlier_bits,
                "value_outlier_bits": self.value_outlier_bits,
                "avg_bits": self.avg_bits,
                "code_bits_per_pair_block": self.code_bits_per_pair_block,
                "key_bit_fraction": self.key_bit_fraction,
                "adaptive_bit_split": self.adaptive_bit_split,
                "adaptive_bit_margin": self.adaptive_bit_margin,
                "adaptive_stats_min_tokens": self.adaptive_stats_min_tokens,
                "key_error_weight": self.key_error_weight,
                "value_error_weight": self.value_error_weight,
                "key_code_bits_per_block": self.key_code_bits_per_block,
                "value_code_bits_per_block": self.value_code_bits_per_block,
                "vllm_mode": self.vllm_mode,
                "use_qjl": self.use_qjl,
            }
        )
        cal = self._calib()
        if cal is not None:
            cfg["calibration"] = cal.digest
        return cfg

    def cache_runtime_provenance(self, layer_states: dict) -> dict:
        """Serialize cache-owned Adaptive HQ state without exposing mutability."""
        if not self._adaptive_bit_split_enabled():
            return {}
        layers = {}
        global_histogram = {}
        selector_bits_total = 0
        selector_incremental_stored_bits_total = 0
        encoded_updates = 0
        encoded_tokens = 0
        for raw_layer, owner in sorted(layer_states.items(), key=lambda item: int(item[0])):
            if not isinstance(owner, dict) or "adaptive_hq" not in owner:
                continue
            state = owner["adaptive_hq"]
            cumulative = copy.deepcopy(state["cumulative"])
            cumulative["key_power"] = float(cumulative["key_square_sum"]) / max(
                1, int(cumulative["key_element_count"])
            )
            cumulative["value_power"] = float(cumulative["value_square_sum"]) / max(
                1, int(cumulative["value_element_count"])
            )
            layer_record = {
                "schema_version": int(state["schema_version"]),
                "policy_version": state["policy_version"],
                "policy": copy.deepcopy(state["policy"]),
                "cumulative": cumulative,
                "encoded_updates": int(state["encoded_updates"]),
                "selector_bits_total": int(state["selector_bits_total"]),
                "selector_incremental_stored_bits_total": int(
                    state["selector_incremental_stored_bits_total"]
                ),
                "split_histogram": copy.deepcopy(state["split_histogram"]),
                "selection_runs": copy.deepcopy(state["selection_runs"]),
                "last_selection": copy.deepcopy(state["last_selection"]),
            }
            layers[str(int(raw_layer))] = layer_record
            selector_bits_total += layer_record["selector_bits_total"]
            selector_incremental_stored_bits_total += layer_record[
                "selector_incremental_stored_bits_total"
            ]
            encoded_updates += layer_record["encoded_updates"]
            encoded_tokens += int(cumulative["token_count"])
            for pair_key, bucket in layer_record["split_histogram"].items():
                merged = global_histogram.setdefault(
                    pair_key,
                    {"pair": list(bucket["pair"]), "updates": 0, "tokens": 0, "vectors": 0},
                )
                for field in ("updates", "tokens", "vectors"):
                    merged[field] += int(bucket[field])
        return {
            "kind": "adaptive_hq",
            "schema_version": 1,
            "policy_version": ADAPTIVE_HQ_POLICY_VERSION,
            "state_ownership": "compressed_dynamic_cache_instance_per_layer",
            "aggregate": {
                "encoded_updates": encoded_updates,
                "encoded_tokens": encoded_tokens,
                "selector_bits_total": selector_bits_total,
                "selector_incremental_stored_bits_total": (selector_incremental_stored_bits_total),
                "selector_carrier_version": ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION,
                "split_histogram": global_histogram,
            },
            "layers": layers,
        }

    def get_resolved_kv_config(self, head_dim: int) -> dict:
        """Return exact role-wise blocks, rotations, scales, and source law.

        The public recipe deliberately permits ``block_size=None`` and
        ``rotation_mode='auto'``.  Those declarations are resolved differently
        for K and V when their rates differ, so a paper result must retain the
        concrete choices at the evaluated model's head dimension.
        """
        d = int(head_dim)
        if d <= 0:
            raise ValueError("head_dim must be positive")
        cal = self._calib()

        def _canonical_scale(mode: str) -> str:
            return {"ub": "ip", "bsm": "ls"}.get(mode, mode)

        def _role(role: str, bits: int) -> dict:
            n_outlier = int(self._n_outlier_for(role))
            if n_outlier >= d:
                raise ValueError(f"head_dim must exceed HayateQuant {role} outlier count")
            main_dim = d - n_outlier
            block = self._resolve_block_size_for(
                role,
                int(bits),
                main_dim,
                code_bits_override=(1 if self._adaptive_bit_split_enabled() else None),
            )
            scale = self._scale_mode_for(role)
            rotation = (
                "payload_dependent_candidate_rate"
                if self._adaptive_bit_split_enabled()
                else (
                    "calibrated_artifact"
                    if cal is not None
                    else self._resolved_mode(
                        main_dim,
                        block,
                        self._rotation_rate_for(role, int(bits), block),
                    )
                )
            )
            record = {
                "bits": int(bits),
                "block_size": int(block),
                "rotation_mode": rotation,
                "rotation_seed": int(self.seed),
                "scale_mode": scale,
                "mathematical_scale_mode": _canonical_scale(scale),
                "source_law": self.dist,
                "quantized_dim": main_dim,
                "outlier_channels": n_outlier,
                "outlier_bits": (
                    int(self._outlier_bits_for(role, int(bits))) if n_outlier else None
                ),
                "outlier_selection": (
                    "top_n_dynamic_max_abs_over_tokens" if n_outlier else "none"
                ),
                "qjl_enabled": bool(self.use_qjl),
            }
            code_bits = self._code_bits_override_for(role)
            if code_bits is not None:
                record["code_bits_per_block"] = int(code_bits)
                record["payload_bits_per_coordinate"] = float(code_bits) / float(block)
            if self._adaptive_bit_split_enabled():
                record["adaptive_bit_split"] = str(self.adaptive_bit_split)
                record["pair_code_bits_per_block"] = int(self._pair_code_bits_total(block))
            if n_outlier:
                out_bits = int(self._outlier_bits_for(role, int(bits)))

                def _subset(subset_bits: int, subset_dim: int) -> dict:
                    subset_block = self._resolve_block_size_for(
                        role, int(subset_bits), int(subset_dim)
                    )
                    return {
                        "bits": int(subset_bits),
                        "block_size": int(subset_block),
                        "rotation_mode": self._resolved_mode(
                            int(subset_dim), int(subset_block), int(subset_bits)
                        ),
                        "rotation_seed": int(self.seed),
                        "scale_mode": scale,
                        "mathematical_scale_mode": _canonical_scale(scale),
                        "source_law": self.dist,
                        "quantized_dim": int(subset_dim),
                        "qjl_enabled": bool(self.use_qjl),
                    }

                record["subsets"] = {
                    "inlier": _subset(int(bits), main_dim),
                    "outlier": _subset(out_bits, n_outlier),
                }
            return record

        resolved = {
            "schema_version": 1,
            "head_dim": d,
            "codec": "hayatequant",
            "calibration": cal.digest if cal is not None else None,
            "assignment": {
                "encoder": self.encoder,
                "lut_bins": self.lut_bins,
                "lut_candidates": self.lut_candidates,
                "shape_independent": self.encoder in {"exact", "lut"},
                "tie_break": "smallest_global_codebook_index",
            },
            "roles": {
                "key": _role("k", int(self.bits)),
                "value": _role("v", int(self._value_bits())),
            },
        }
        if self.required_codebook_bundle is not None:
            resolved["codebook_artifacts"] = require_shipped_gaussian_codebook_bundle(
                self.required_codebook_bundle
            )
        else:
            specs: set[tuple[int, int]] = set()
            if self.dist == "gaussian" and cal is None:
                if self._adaptive_bit_split_enabled():
                    block = self._resolve_block_size_for(
                        "k", int(self.bits), d, code_bits_override=1
                    )
                    for key_bits, value_bits in self._adaptive_code_bit_pairs(block):
                        specs.add((int(block), 1 << int(key_bits)))
                        specs.add((int(block), 1 << int(value_bits)))
                else:
                    for role, bits in (("k", int(self.bits)), ("v", int(self._value_bits()))):
                        block = int(
                            resolved["roles"]["key" if role == "k" else "value"]["block_size"]
                        )
                        if block > 1:
                            specs.add((block, 1 << self._code_bits_for(role, bits, block)))
            resolved["codebook_artifacts"] = {
                "required": False,
                "development_override_dir": os.environ.get("KVCOMP_CODEBOOK_DIR"),
                "artifacts": [
                    gaussian_codebook_artifact_provenance(block, k) for block, k in sorted(specs)
                ],
            }
        if self._adaptive_bit_split_enabled():
            block = self._resolve_block_size_for("k", int(self.bits), d, code_bits_override=1)
            pairs = self._adaptive_code_bit_pairs(block)
            total = int(pairs[0][0] + pairs[0][1])
            frac = 0.5 if self.key_bit_fraction is None else float(self.key_bit_fraction)
            prior_k = min(total - 1, max(1, int(round(total * frac))))
            prior = min(pairs, key=lambda pair: (abs(pair[0] - prior_k), pair[0]))
            resolved["adaptive_allocation"] = {
                "policy": "cache_causal_datafree_rd",
                "policy_version": ADAPTIVE_HQ_POLICY_VERSION,
                "distortion_table_version": ADAPTIVE_HQ_DISTORTION_TABLE_VERSION,
                "objective": ADAPTIVE_HQ_OBJECTIVE,
                "state_ownership": "compressed_dynamic_cache_instance_per_layer",
                "selection_scope": "per_encode_kv_pair",
                "block_size": int(block),
                "pair_code_bits_per_block": total,
                "candidate_pairs": [[int(k), int(v)] for k, v in pairs],
                "candidate_rotations": [
                    {
                        "pair": [int(k), int(v)],
                        "key": self._resolved_mode(d, block, float(k) / float(block)),
                        "value": self._resolved_mode(d, block, float(v) / float(block)),
                    }
                    for k, v in pairs
                ],
                "prior_pair": [int(prior[0]), int(prior[1])],
                "incumbent_pair": [int(prior[0]), int(prior[1])],
                "switch_margin_fraction": float(ADAPTIVE_HQ_SWITCH_MARGIN_FRACTION),
                "switch_margin_basis": ADAPTIVE_HQ_SWITCH_MARGIN_BASIS,
                "hysteresis": ADAPTIVE_HQ_HYSTERESIS,
                "selector_bits_per_encode": int((len(pairs) - 1).bit_length()),
                "selector_incremental_stored_bits_per_encode": 0,
                "selector_carrier_version": ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION,
                "selector_carrier": "first_value_fp16_scale_low_fraction_bits",
                "selector_scale_reconstruction": "clear_carrier_fraction_bits",
                "selector_carrier_distortion_amortization": (
                    "signed_qmc_all_scales_delta_divided_by_current_update_vectors"
                ),
                "selected_pair": "payload_dependent",
                "runtime_distortion_constants": "payload_and_cache_provenance",
            }
        return resolved

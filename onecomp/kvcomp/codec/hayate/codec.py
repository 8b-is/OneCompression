"""HayateQuant: block-vector-quantized, hybrid-rotation KV codec.

HayateQuant is a **strict, data-free generalization of TurboQuant**
(arXiv:2504.19874) that closes TurboQuant's own stated distortion slack and
absorbs HALO's validated, storage-neutral levers into one codec, replacing
both ``turboquant`` and the (now removed) ``halo`` codec.

Recap: TurboQuant rotates a KV vector by a random orthogonal transform so its
coordinates are (exactly, for finite ``d``, or asymptotically as ``d -> inf``)
i.i.d. draws from a scalar law -- ``Beta((d-1)/2, (d-1)/2)`` or its Gaussian
limit -- then fits a **scalar** Lloyd-Max quantizer to that law, *entirely
data-free* (the codebook only depends on the bit-width, not on any real KV
data). The paper proves this is within a constant factor (``<= sqrt(3*pi/2)
~= 2.7x``) of the information-theoretic distortion-rate optimum. Classical
high-rate quantization theory (Gersho & Gray, *Vector Quantization and Signal
Compression*, 1992; Zamir, *Lattice Coding for Signals and Networks*, 2014)
identifies exactly where that residual factor lives: **independent scalar
quantization of a memoryless Gaussian source loses the "space-filling" and
"shape" gains available to any joint (vector/lattice) quantizer of dimension
> 1** -- a classical ``1.53`` dB (``~0.255`` bit/dim) gap at high rate that
only closes as the joint quantization dimension grows (``E8``: ``~0.65`` dB
gap remaining; this repo's own HALO analysis flags exactly this as unexplored
"Lever 5", see ``benchmark/kvcomp-longcontext/paper/HALO_ENHANCED_REPORT.md``).

HayateQuant closes this gap with two data-free, storage-neutral additions on
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
   because the codebook is *learned* on the exact source law it is always at
   least as good as any fixed lattice at the same rate. The total number of
   stored bits is unchanged (``bits`` per coordinate, same as TurboQuant --
   see ``_stored_bits``); only *how* those bits are allocated changes (jointly
   per block instead of independently per coordinate), which is exactly the
   controlled comparison needed to isolate the vector-quantization gain.
   ``block_size=1`` reduces the block codebook to the ordinary scalar
   Lloyd-Max codebook, bit-for-bit -- HayateQuant is therefore **always able
   to fall back to exactly TurboQuant's quantizer**, never worse.

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
   ``hybrid_rotate``/``hybrid_unrotate``), and the combined cost
   ``O(D log D + D*b)`` stays far below a single full ``D x D`` dense
   rotation (``O(D^2)``) while giving each block genuine Haar-random local
   isotropy -- keeping the Hadamard step's proven-good global incoherence
   against real anisotropic KV (the robustness lever) *and* the exact local
   isotropy the block-VQ codebook needs (the accuracy lever). The block size
   ``block_size`` is therefore literally the interpolation knob between "pure
   Hadamard" (``block_size=1``, cheapest, most robust, no vector-quantization
   gain) and "pure dense" (``block_size=D``, most expensive, most isotropic)
   that was raised as an open question for HALO -- HayateQuant answers it by
   only paying the dense-rotation cost on small blocks.

3. **HALO's validated, storage-neutral levers (inherited, not reinvented).**
   Role-asymmetric reconstruction scale (:mod:`._scale`): keys use the
   norm-correcting (``"nc"``) scale (preserves the attention logit, shrunk by
   ``cos(theta)`` instead of ``cos^2(theta)``), values use the least-squares
   (``"ls"``/``"ls_iter"``) scale (the correct MSE objective for values, which
   are consumed linearly). Outlier mixed-precision (``n_outlier``/
   ``outlier_bits``) for true 2.5/3.5-bit operating points. Optional QJL-style
   1-bit residual (``use_qjl``, off by default -- community evaluations agree
   it is unbiased but high-variance through softmax and can hurt attention).
   These levers are dimension-agnostic (:func:`._scale.fit_scale` reduces
   over the concatenated per-block codewords exactly as it would over
   concatenated scalar centroids), so they compose for free with block-VQ.

**HayateQuant is a strict superset of TurboQuant and HALO**: setting
``block_size=1`` and ``rotation_mode`` to ``"hadamard"``/``"hadamard_gemm"``/
``"dense"`` with ``key_scale_mode=value_scale_mode="rms"`` reproduces
TurboQuant bit-for-bit; adding the role-asymmetric scale modes reproduces
HALO bit-for-bit. The only genuinely new, previously-unexplored lever is
block-VQ (default on, via ``block_size=None`` auto-resolution), which is
exactly the "Lever 5 / future work" gap this repository's own HALO analysis
identified and left unimplemented.

See ``docs/design/kvcomp-architecture.md`` (HayateQuant section) and
``papers/hayatequant_paper/README.md`` for the full derivation, and
``benchmark/kvcomp-longcontext/paper/tests/verify_hayatequant_vq_gain.py`` /
``verify_hayatequant_isotropy.py`` for the CPU-only, data-free, in-repo proofs
that (a) block-VQ strictly reduces distortion at equal bit-rate as
``block_size`` grows from 1, matching the classical space-filling-gain curve,
and (b) the hybrid rotation's block marginals are measurably closer to
isotropic than the Hadamard-only rotation's.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

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
)
from .._scale import SCALE_MODES, resolve_scale_mode, fit_scale


@lru_cache(maxsize=16)
def _scalar_codebook(bits: int, dist: str, m: int) -> torch.Tensor:
    if dist == "beta":
        # ``lloyd_max_beta`` lives on the *unit-sphere* coordinate scale
        # ([-1, 1], TurboQuant's L2-normalisation convention); HayateQuant
        # normalises by the RMS (= L2 / sqrt(m)) everywhere, so rescale the
        # codebook by sqrt(m) to match.  Decision boundaries scale with the
        # codewords, so the assigned codes -- and, after the per-vector scale,
        # the reconstruction -- are bit-for-bit identical to TurboQuant's
        # L2-normalised beta path (and consistent with the sqrt(d)-rescaled
        # block codebook of ``lloyd_vq_beta_block``).
        return lloyd_max_beta(bits, m) * math.sqrt(m)
    return lloyd_max_gaussian(bits)


@lru_cache(maxsize=64)
def _block_codebook(
    bits: int, block_size: int, dist: str, m: int, codebook_init: str = "auto",
) -> torch.Tensor:
    """``(2**(bits*block_size), block_size)`` data-free block-VQ codebook.

    ``block_size=1`` returns the ordinary scalar Lloyd-Max codebook reshaped
    to ``(2**bits, 1)`` -- the exact TurboQuant/HALO codebook, so every
    downstream computation (nearest-codeword search, role-asymmetric scale)
    reduces to the scalar case bit-for-bit.
    """
    if block_size == 1:
        return _scalar_codebook(bits, dist, m).unsqueeze(-1)
    if dist == "beta":
        return lloyd_vq_beta_block(bits, block_size, m, init=codebook_init)
    return lloyd_vq_gaussian_block(bits, block_size, init=codebook_init, ambient_dim=m)


@lru_cache(maxsize=64)
def _block_codebook_by_code_bits(
    code_bits: int, block_size: int, dist: str, m: int, codebook_init: str = "auto",
) -> torch.Tensor:
    """Data-free block codebook with explicit stored bits per block.

    This is the fractional-rate counterpart to :func:`_block_codebook`.
    ``code_bits == bits * block_size`` reproduces the ordinary HayateQuant
    codebook, while other integer ``code_bits`` values define fixed-length
    rates ``code_bits / block_size`` without changing the random-access cache
    contract.
    """
    code_bits = int(code_bits)
    if code_bits < 1:
        raise ValueError(f"code_bits must be positive, got {code_bits}")
    if block_size == 1:
        return _scalar_codebook(code_bits, dist, m).unsqueeze(-1)
    if dist == "beta":
        return lloyd_vq_beta_codebook(code_bits, block_size, m, init=codebook_init)
    return lloyd_vq_gaussian_codebook(
        code_bits, block_size, init=codebook_init, ambient_dim=m)


@lru_cache(maxsize=16)
def _dense_rotation(dim: int, seed: int) -> torch.Tensor:
    return random_orthogonal(dim, seed)


@lru_cache(maxsize=64)
def _dense_rotation_on(dim: int, seed: int, device_str: str, dtype) -> torch.Tensor:
    """Device-resident dense orthogonal matrix, cached per (dim, seed, device, dtype)."""
    return _dense_rotation(dim, seed).to(device=torch.device(device_str), dtype=dtype)


@lru_cache(maxsize=64)
def _block_codebook_on(
    bits: int, block_size: int, dist: str, m: int, codebook_init: str, device_str: str,
) -> torch.Tensor:
    """Device-resident block codebook, cached so the host->device copy happens once."""
    return _block_codebook(bits, block_size, dist, m, codebook_init).to(torch.device(device_str))


@lru_cache(maxsize=64)
def _block_codebook_code_bits_on(
    code_bits: int, block_size: int, dist: str, m: int, codebook_init: str, device_str: str,
) -> torch.Tensor:
    """Device-resident explicit-code-bits block codebook."""
    return _block_codebook_by_code_bits(
        code_bits, block_size, dist, m, codebook_init).to(torch.device(device_str))


@lru_cache(maxsize=64)
def _block_lut_on(
    bits: int, block_size: int, dist: str, m: int, codebook_init: str, bins: int, top_t: int,
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
    code_bits: int, block_size: int, dist: str, m: int, codebook_init: str,
    bins: int, top_t: int, device_str: str,
) -> torch.Tensor:
    """Device-resident candidate-list LUT for an explicit-code-bit codebook."""
    cb = _block_codebook_code_bits_on(code_bits, block_size, dist, m, codebook_init, device_str)
    return build_vq_lut(cb, bins, top_t)


@lru_cache(maxsize=64)
def _codebook_aux_on(
    bits: int, block_size: int, dist: str, m: int, codebook_init: str, device_str: str,
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
    code_bits: int, block_size: int, dist: str, m: int, codebook_init: str, device_str: str,
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
            codes = quantize_via_lut(vb, lut, bins, cb32)
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
                num_safe = num.sign() * num.abs().clamp_min(1e-8)
                scale = (xi * xi).sum(dim=-1, keepdim=True) / num_safe
            else:
                scale = num / den
                if mode == "ls_iter":
                    chat = _assign(xi / scale.clamp_min(1e-8))
                    num2 = (xi * chat).sum(dim=-1, keepdim=True)
                    den2 = (chat * chat).sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    scale = num2 / den2
        outs.append(chat * scale)
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
        torch._dynamo.config.cache_size_limit = max(
            64, torch._dynamo.config.cache_size_limit)
        torch._dynamo.config.accumulated_cache_size_limit = max(
            256, torch._dynamo.config.accumulated_cache_size_limit)
        _RT_COMPILED = torch.compile(
            _roundtrip_tensors, mode="reduce-overhead", dynamic=False)
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
        use_qjl: Optional 1-bit QJL-style residual refinement (off by default).
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
            explicit-rate HF/research path. ``"online_mse"`` tries a small
            fixed-budget set of K/V code-bit splits on the current KV tensors
            and stores the lowest weighted reconstruction-error split in the
            payload. It is deliberately opt-in because it spends extra encode
            work to choose the split.
    """

    flag_calibration: bool = False
    bits: int = 4
    value_bits: int = None
    dist: str = "gaussian"
    block_size: int = None
    max_codebook_bits: int = 9
    codebook_init: str = "auto"
    rotation_mode: str = "auto"
    encoder: str = "auto"
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
    key_error_weight: float = 1.0
    value_error_weight: float = 1.0
    key_code_bits_per_block: int = None
    value_code_bits_per_block: int = None
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
            if self.key_code_bits_per_block is not None or self.value_code_bits_per_block is not None:
                raise ValueError(
                    "adaptive_bit_split cannot be combined with explicit "
                    "key/value_code_bits_per_block")
            frac = 0.625 if self.key_bit_fraction is None else float(self.key_bit_fraction)
            if not 0.0 < frac < 1.0:
                raise ValueError(f"key_bit_fraction must be in (0, 1), got {frac!r}")
            center = min(total - 1, max(1, int(round(total * frac))))
            self.max_codebook_bits = max(self.max_codebook_bits, center, total - center)
            return
        if self.key_code_bits_per_block is not None or self.value_code_bits_per_block is not None:
            return
        frac = 0.625 if self.key_bit_fraction is None else float(self.key_bit_fraction)
        if not 0.0 < frac < 1.0:
            raise ValueError(f"key_bit_fraction must be in (0, 1), got {frac!r}")
        key_bits = min(total - 1, max(1, int(round(total * frac))))
        value_bits = total - key_bits
        self.key_code_bits_per_block = key_bits
        self.value_code_bits_per_block = value_bits
        self.max_codebook_bits = max(self.max_codebook_bits, key_bits, value_bits)

    def validate_params(self):
        self._configure_fractional_rate_ui()
        if self.rotation_mode not in ("auto", "hybrid", "hadamard_gemm", "hadamard", "dense"):
            raise ValueError(
                "rotation_mode must be 'auto', 'hybrid', 'hadamard_gemm', 'hadamard' or 'dense'")
        if self.encoder not in ("auto", "lut", "exact"):
            raise ValueError(f"encoder must be 'auto', 'lut' or 'exact', got {self.encoder!r}")
        if self.dist not in ("gaussian", "beta"):
            raise ValueError("dist must be 'gaussian' or 'beta'")
        if self.codebook_init not in ("auto", "forgy", "fibquant"):
            raise ValueError(
                "codebook_init must be 'auto', 'forgy' or 'fibquant', "
                f"got {self.codebook_init!r}")
        if not (isinstance(self.bits, int) and 1 <= self.bits <= 8):
            raise ValueError(f"bits must be int in 1..8, got {self.bits!r}")
        if self.avg_bits is not None and float(self.avg_bits) <= 0.0:
            raise ValueError(f"avg_bits must be positive, got {self.avg_bits!r}")
        if self.code_bits_per_pair_block is not None and not (
                isinstance(self.code_bits_per_pair_block, int)
                and self.code_bits_per_pair_block >= 2):
            raise ValueError(
                "code_bits_per_pair_block must be None or an int >= 2, "
                f"got {self.code_bits_per_pair_block!r}")
        if self.adaptive_bit_split not in (None, "", "off", "none", False, "online", "online_mse"):
            raise ValueError(
                "adaptive_bit_split must be None/'off' or 'online_mse', "
                f"got {self.adaptive_bit_split!r}")
        if self._adaptive_bit_split_enabled() and not self._fractional_rate_requested():
            raise ValueError(
                "adaptive_bit_split needs avg_bits or code_bits_per_pair_block "
                "to define a fixed total K/V bit budget")
        if not (isinstance(self.adaptive_bit_margin, int) and self.adaptive_bit_margin >= 0):
            raise ValueError(
                f"adaptive_bit_margin must be a non-negative int, got {self.adaptive_bit_margin!r}")
        if float(self.key_error_weight) <= 0.0 or float(self.value_error_weight) <= 0.0:
            raise ValueError("key_error_weight and value_error_weight must be positive")
        for label, b in (
            ("value_bits", self.value_bits),
            ("outlier_bits", self.outlier_bits),
            ("key_outlier_bits", self.key_outlier_bits),
            ("value_outlier_bits", self.value_outlier_bits),
        ):
            if b is not None and not (isinstance(b, int) and 1 <= b <= 8):
                raise ValueError(f"{label} must be None or int in 1..8, got {b!r}")
        for label, b in (
            ("key_code_bits_per_block", self.key_code_bits_per_block),
            ("value_code_bits_per_block", self.value_code_bits_per_block),
        ):
            if b is not None and not (isinstance(b, int) and b >= 1):
                raise ValueError(f"{label} must be None or a positive int, got {b!r}")
            if b is not None and b > self.max_codebook_bits:
                raise ValueError(
                    f"{label}={b} exceeds max_codebook_bits={self.max_codebook_bits}; "
                    "raise max_codebook_bits explicitly if this large codebook is intended")
        if self.block_size not in self._BLOCK_SIZES:
            raise ValueError(f"block_size must be one of {self._BLOCK_SIZES}, got {self.block_size!r}")
        for mode in (self.key_scale_mode, self.value_scale_mode):
            if mode is not None and mode not in self._SCALE_MODES:
                raise ValueError(f"scale_mode must be one of {self._SCALE_MODES} or None, got {mode!r}")
        if not (isinstance(self.n_outlier, int) and self.n_outlier >= 0):
            raise ValueError(f"n_outlier must be a non-negative int, got {self.n_outlier!r}")
        for label, n in (("key_n_outlier", self.key_n_outlier),
                         ("value_n_outlier", self.value_n_outlier)):
            if n is not None and not (isinstance(n, int) and n >= 0):
                raise ValueError(f"{label} must be None or a non-negative int, got {n!r}")
        if self.calibration is not None:
            from .calibration import HayateQuantCalibration
            if not isinstance(self.calibration, (str, HayateQuantCalibration)):
                raise ValueError(
                    "calibration must be None, a path, or a HayateQuantCalibration, "
                    f"got {type(self.calibration).__name__}")
            if self._n_outlier_for("k") or self._n_outlier_for("v"):
                raise ValueError("calibration is not supported with n_outlier>0 "
                                 "(subset dims differ from the calibrated head_dim)")
        if self._has_explicit_code_bits() and (
                self._n_outlier_for("k") or self._n_outlier_for("v")):
            raise ValueError(
                "explicit code_bits_per_block is currently supported only "
                "without outlier channel splits")
        if self._has_explicit_code_bits() and self.calibration is not None:
            raise ValueError(
                "calibration is not supported with explicit "
                "code_bits_per_block yet")
        if self._adaptive_bit_split_enabled() and (
                self._n_outlier_for("k") or self._n_outlier_for("v") or self.calibration is not None):
            raise ValueError(
                "adaptive_bit_split is currently supported only without "
                "outlier splits or calibration")
        # OneComp convention: the flag (not the folder) marks calibrated codecs.
        self.flag_calibration = self.calibration is not None

    def _value_bits(self) -> int:
        return self.value_bits if self.value_bits is not None else self.bits

    def _code_bits_override_for(self, role: str) -> int | None:
        return self.key_code_bits_per_block if role == "k" else self.value_code_bits_per_block

    def _has_explicit_code_bits(self) -> bool:
        return self.key_code_bits_per_block is not None or self.value_code_bits_per_block is not None

    def _code_bits_for(self, role: str, bits: int, block_size: int) -> int:
        override = self._code_bits_override_for(role)
        return int(override) if override is not None else int(bits) * int(block_size)

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

    def _resolved_mode(self, d: int, block_size: int, bits: int) -> str:
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
            return self.rotation_mode
        if block_size > 1:
            if bits <= 2:
                return "dense"
            return "hybrid" if is_power_of_two(d) else "dense"
        if is_power_of_two(d):
            return "hadamard_gemm" if d <= self.hadamard_gemm_max_dim else "hadamard"
        return "dense"

    # --- rotation ----------------------------------------------------------
    def _rotate(self, x: torch.Tensor, m: int, block_size: int, bits: int,
                role: str = "k", layer_idx=None):
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
                    f"{'no' if cal is None else 'a different'} calibration loaded")
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
        q = _dense_rotation_on(m, seed, str(xr.device), torch.float32)
        return xr @ q.t()

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
        if self.encoder == "auto" and (
                not is_cuda or n_total_blocks * k < self._LUT_MIN_WORK):
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
        if self.encoder == "auto" and (
                not is_cuda or n_total_blocks * k < self._LUT_MIN_WORK):
            return None
        return spec

    # --- rotate + block-VQ + role-asymmetric scale for one channel subset --
    def _quant_rotor(
        self, xsub: torch.Tensor, bits: int, role: str, layer_idx=None,
        code_bits_override: int | None = None,
    ) -> dict:
        m = xsub.shape[-1]
        block_size = resolve_block_size(bits, self.block_size, self.max_codebook_bits, m)
        code_bits = (
            int(code_bits_override)
            if code_bits_override is not None
            else self._code_bits_for(role, bits, block_size)
        )
        explicit_code_bits = code_bits != int(bits) * int(block_size)
        xr, rot = self._rotate(xsub, m, block_size, bits, role, layer_idx=layer_idx)
        # One RMS scale per whole subset vector (not per block) -- matches
        # TurboQuant/HALO's storage convention exactly (a single fp16 scale
        # per vector regardless of block_size).
        rms = xr.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        device_str = str(xsub.device)
        cal = self._calib()
        cb_cal = (
            cal.codebook_on(role, bits, block_size, xsub.device)
            if cal is not None and m == cal.head_dim and not explicit_code_bits
            else None
        )
        if cb_cal is not None:
            cb = cb_cal
        elif explicit_code_bits:
            cb = _block_codebook_code_bits_on(
                code_bits, block_size, self.dist, m, self.codebook_init, device_str)
        else:
            cb = _block_codebook_on(bits, block_size, self.dist, m, self.codebook_init, device_str)
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
                code_bits, block_size, self.dist, m, self.codebook_init,
                lut_spec[0], lut_spec[1], device_str)
        else:
            lut = _block_lut_on(bits, block_size, self.dist, m, self.codebook_init,
                                lut_spec[0], lut_spec[1], device_str)

        def _lookup(vec_full):
            vb = vec_full.reshape(*vec_full.shape[:-1], n_blocks, block_size)
            if lut is not None:
                codes_ = quantize_via_lut(vb, lut, lut_spec[0], cb)
                chat_b = lookup_codewords(cb, codes_)
            else:
                codes_, chat_b = quantize_and_lookup(vb, cb)
            return codes_, chat_b.reshape(*vec_full.shape[:-1], m)

        def _requant(z_full):
            codes2, chat2 = _lookup(z_full)
            return chat2, codes2

        codes, chat = _lookup(xr / rms)
        scale, codes2 = fit_scale(xr, chat, self._scale_mode_for(role), rms, requantize_fn=_requant)
        if codes2 is not None:
            codes = codes2
        # ``dist`` is stored (as in TurboQuant) so decode picks the same
        # codebook even if the codec instance is reconfigured between
        # encode and decode.
        payload = {"codes": codes, "scale": scale, "bits": bits, "m": m,
                   "block_size": block_size, "rot": rot, "dist": self.dist,
                   "codebook_init": self.codebook_init}
        if explicit_code_bits:
            payload["code_bits"] = code_bits
        if cb_cal is not None:
            # Decode must gather from the SAME learned codebook; record the
            # role + artifact digest (same convention as the "c" rot tag).
            payload["cb_cal"] = (role, cal.digest)
        if self.use_qjl:
            recon = lookup_codewords(cb, codes).reshape(*codes.shape[:-1], m) * scale
            resid = xr - recon
            payload["qjl_sign"] = (resid >= 0)
            payload["qjl_rms"] = resid.abs().mean(dim=-1, keepdim=True)
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
                    f"{'no' if cal is None else 'a different'} calibration loaded")
            cb = cal.codebook_on(role, p["bits"], p["block_size"], device)
        else:
            codebook_init = p.get("codebook_init", self.codebook_init)
            if "code_bits" in p:
                cb = _block_codebook_code_bits_on(
                    p["code_bits"], p["block_size"], dist, p["m"], codebook_init, str(device))
            else:
                cb = _block_codebook_on(
                    p["bits"], p["block_size"], dist, p["m"], codebook_init, str(device))
        chat = lookup_codewords(cb, p["codes"]).reshape(*p["codes"].shape[:-1], p["m"])
        xr = chat * p["scale"]
        if "qjl_sign" in p:
            sign = torch.where(p["qjl_sign"], 1.0, -1.0).to(xr.dtype)
            xr = xr + sign * p["qjl_rms"]
        return self._unrotate(xr, p["m"], p["rot"])

    # --- full-tensor encode / decode (handles the outlier split) -----------
    def _encode_tensor(self, x: torch.Tensor, bits: int, role: str, layer_idx=None) -> dict:
        head_dim = x.shape[-1]
        n_outlier = self._n_outlier_for(role)
        if n_outlier <= 0 or n_outlier >= head_dim:
            return {"mode": "single", "main": self._quant_rotor(x, bits, role, layer_idx=layer_idx),
                    "head_dim": head_dim}
        # Outlier mixed-precision: top-n_outlier channels by energy over tokens.
        energy = x.float().abs().amax(dim=-2)
        energy = energy.reshape(-1, head_dim).mean(dim=0)
        out_idx = torch.argsort(energy, descending=True)[: n_outlier]
        out_idx, _ = torch.sort(out_idx)
        mask = torch.zeros(head_dim, dtype=torch.bool, device=x.device)
        mask[out_idx] = True
        return {
            "mode": "outlier", "head_dim": head_dim, "mask": mask,
            "out": self._quant_rotor(x[..., mask], self._outlier_bits_for(role, bits), role,
                                     layer_idx=layer_idx),
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
        b, h, s, d = x.shape
        vecs = b * h * s
        block_size = int(payload["block_size"])
        code_bits = int(payload.get("code_bits", int(payload["bits"]) * block_size))
        return vecs * (d // block_size) * code_bits + vecs * 16

    def _adaptive_code_bit_pairs(self, block_size: int) -> list[tuple[int, int]]:
        total = self._pair_code_bits_total(block_size)
        if total is None:
            raise ValueError("adaptive_bit_split needs a total K/V bit budget")
        frac = 0.625 if self.key_bit_fraction is None else float(self.key_bit_fraction)
        center = min(total - 1, max(1, int(round(total * frac))))
        margin = int(self.adaptive_bit_margin)
        lo = max(1, total - int(self.max_codebook_bits), center - margin)
        hi = min(int(self.max_codebook_bits), total - 1, center + margin)
        pairs = [(k, total - k) for k in range(lo, hi + 1)]
        if not pairs:
            raise ValueError(
                "adaptive_bit_split has no feasible K/V split under "
                f"total={total} and max_codebook_bits={self.max_codebook_bits}")
        return pairs

    def _score_main_payload(self, x: torch.Tensor, payload: dict) -> float:
        # Orthogonal rotations preserve MSE, but decoding here keeps the
        # selector identical to the real payload semantics, including scale
        # mode and future residual options. The scalar float intentionally
        # syncs only in opt-in adaptive mode.
        xhat = self._decode_rotor(payload, x.device, x.dtype)
        return float((xhat.float() - x.float()).pow(2).mean())

    def _encode_adaptive_pair(self, key: torch.Tensor, value: torch.Tensor, layer_idx=None):
        if key.shape[-1] != value.shape[-1]:
            raise ValueError("adaptive_bit_split expects key/value to share head_dim")
        m = key.shape[-1]
        kb, vb = self.bits, self._value_bits()
        block_size = resolve_block_size(kb, self.block_size, self.max_codebook_bits, m)
        pairs = self._adaptive_code_bit_pairs(block_size)
        key_payloads, key_errs = {}, {}
        value_payloads, value_errs = {}, {}
        for k_code in sorted({p[0] for p in pairs}):
            payload = self._quant_rotor(
                key, kb, "k", layer_idx=layer_idx, code_bits_override=k_code)
            key_payloads[k_code] = payload
            key_errs[k_code] = self._score_main_payload(key, payload)
        for v_code in sorted({p[1] for p in pairs}):
            payload = self._quant_rotor(
                value, vb, "v", layer_idx=layer_idx, code_bits_override=v_code)
            value_payloads[v_code] = payload
            value_errs[v_code] = self._score_main_payload(value, payload)
        kw, vw = float(self.key_error_weight), float(self.value_error_weight)
        best_k, best_v = min(
            pairs, key=lambda pair: kw * key_errs[pair[0]] + vw * value_errs[pair[1]])
        kp = {"mode": "single", "main": key_payloads[best_k], "head_dim": m}
        vp = {"mode": "single", "main": value_payloads[best_v], "head_dim": m}
        stored = (
            self._stored_bits_from_main_payload(key, key_payloads[best_k])
            + self._stored_bits_from_main_payload(value, value_payloads[best_v])
        )
        meta = {
            "dtype": key.dtype,
            "stored_bits": int(stored),
            "adaptive_bit_split": {
                "policy": "online_mse",
                "key_code_bits_per_block": int(best_k),
                "value_code_bits_per_block": int(best_v),
                "candidates": pairs,
                "key_error": key_errs[best_k],
                "value_error": value_errs[best_v],
            },
        }
        return CompressedKV(key_q=kp, value_q=vp, meta=meta)

    def _stored_bits(self, x: torch.Tensor, bits: int, role: str) -> int:
        """Bit accounting -- identical to TurboQuant/HALO at the same ``bits``.

        ``block_size`` only changes how the ``bits``-per-coordinate budget is
        allocated (jointly per block vs. independently per coordinate); the
        total number of payload bits is unchanged (``n_blocks * bits *
        block_size == d * bits``), which is what makes the block-VQ
        comparison to TurboQuant an honest equal-storage comparison.
        """
        b, h, s, d = x.shape
        vecs = b * h * s
        block_size = resolve_block_size(bits, self.block_size, self.max_codebook_bits, d)
        code_bits = self._code_bits_for(role, bits, block_size)
        n_outlier = self._n_outlier_for(role)
        split = 0 < n_outlier < d  # outlier split => two subsets per vector
        if not split:
            payload_bits = vecs * (d // block_size) * code_bits
            scale_bits = vecs * 16
        else:
            no = n_outlier
            payload_bits = vecs * (no * self._outlier_bits_for(role, bits) + (d - no) * bits)
            scale_bits = vecs * 2 * 16
        # QJL residual: 1 sign bit per coordinate + one fp16 residual-RMS per
        # encoded subset vector (two subsets when the outlier split is active).
        qjl_bits = (x.numel() * 1 + vecs * (2 if split else 1) * 16) if self.use_qjl else 0
        return payload_bits + scale_bits + qjl_bits

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
            self, kv_stats["key"], kv_stats["value"], **(ctx or {}))
        self.flag_calibration = True
        return self.calibration

    def encode(self, key, value, ctx=None) -> CompressedKV:
        layer_idx = (ctx or {}).get("layer_idx")
        if self._adaptive_bit_split_enabled():
            return self._encode_adaptive_pair(key, value, layer_idx=layer_idx)
        kp = self._encode_tensor(key, self.bits, "k", layer_idx=layer_idx)
        vp = self._encode_tensor(value, self._value_bits(), "v", layer_idx=layer_idx)
        stored = self._stored_bits(key, self.bits, "k") + self._stored_bits(value, self._value_bits(), "v")
        return CompressedKV(key_q=kp, value_q=vp, meta={"dtype": key.dtype, "stored_bits": int(stored)})

    def decode(self, compressed: CompressedKV, ctx=None):
        dtype = compressed.meta.get("dtype", torch.float16)
        return self._decode_tensor(compressed.key_q, dtype), self._decode_tensor(compressed.value_q, dtype)

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
        for bits in {self.bits, self._value_bits()}:
            bs = resolve_block_size(bits, self.block_size, self.max_codebook_bits, m)
            if self._resolved_mode(m, bs, bits) == "hadamard":
                # The sequential WHT butterfly is not a cached (m, m) GEMM;
                # it only resolves at very large head dims -- fall back.
                return False
        return True

    def _rt_rotation(self, m: int, block_size: int, bits: int, device, role: str | None = None,
                     layer_idx=None) -> torch.Tensor:
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

    def _rt_call(self, fn, x: torch.Tensor, bits: int, modes: tuple, role: str | None = None,
                 layer_idx=None) -> torch.Tensor:
        """Run the fused round-trip core on stacked roles ``x`` (R, b, h, s, m)."""
        m = x.shape[-1]
        block_size = resolve_block_size(bits, self.block_size, self.max_codebook_bits, m)
        code_bits = self._code_bits_for(role or "k", bits, block_size)
        explicit_code_bits = code_bits != int(bits) * int(block_size)
        w = self._rt_rotation(m, block_size, bits, x.device, role, layer_idx=layer_idx)
        cal = self._calib()
        cb_cal = (
            cal.codebook_on(role, bits, block_size, x.device)
            if cal is not None and role is not None and m == cal.head_dim and not explicit_code_bits
            else None
        )
        if cb_cal is not None:
            cb32, cb_score, csq, bounds = _codebook_aux_from_tensor(cb_cal, x.device)
        elif explicit_code_bits:
            cb32, cb_score, csq, bounds = _codebook_aux_code_bits_on(
                code_bits, block_size, self.dist, m, self.codebook_init, str(x.device))
        else:
            cb32, cb_score, csq, bounds = _codebook_aux_on(
                bits, block_size, self.dist, m, self.codebook_init, str(x.device))
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
                    spec is None or os.getenv("ONECOMP_TRITON_PREFILL", "0") == "1"):
                return roundtrip_step_triton(x, w, cb32, block_size, modes)
        if spec is None:
            lut, bins = None, 0
        elif cb_cal is not None:
            lut = cal.lut_on(role, bits, block_size, spec[0], spec[1], x.device)
            bins = spec[0]
        elif explicit_code_bits:
            lut = _block_lut_code_bits_on(
                code_bits, block_size, self.dist, m, self.codebook_init,
                spec[0], spec[1], str(x.device))
            bins = spec[0]
        else:
            lut = _block_lut_on(
                bits, block_size, self.dist, m, self.codebook_init, spec[0], spec[1], str(x.device))
            bins = spec[0]
        return fn(x, w, cb32, cb_score, csq, bounds, lut, bins, block_size, modes)

    def _rt_roles_share_codebook(self, m: int, kb: int, vb: int) -> bool:
        if self._adaptive_bit_split_enabled():
            return False
        k_bs = resolve_block_size(kb, self.block_size, self.max_codebook_bits, m)
        v_bs = resolve_block_size(vb, self.block_size, self.max_codebook_bits, m)
        return (
            k_bs == v_bs
            and self._code_bits_for("k", kb, k_bs) == self._code_bits_for("v", vb, v_bs)
        )

    def fake_quant_roundtrip(self, key, value, ctx=None):
        """Fused encode+decode: ``(key_hat, value_hat, stored_bits)``.

        Numerically equivalent to ``decode(encode(key, value))`` (same
        rotation, codebook, assignment and scale-fit operations in the same
        dtypes -- see ``test_hayatequant_roundtrip_matches_encode_decode``)
        but skips payload construction and the decode-time codeword gather,
        batches key and value through one rotation/quantization pipeline when
        their bit-widths agree, and reuses cached per-codebook constants.
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
            os.getenv("ONECOMP_COMPILE_DECODE_ONLY", "0") == "1"
            and key.shape[-2] != 1
        )
        fn, compiled = _rt_dispatch(compile_allowed=compile_allowed)
        dtype = key.dtype
        cal = self._calib()
        layer_idx = (ctx or {}).get("layer_idx")
        if kb == vb and cal is None and self._rt_roles_share_codebook(m, kb, vb):
            y = self._rt_call(fn, torch.stack((key, value)).float(), kb, (k_mode, v_mode))
            khat, vhat = y[0], y[1]
        else:
            khat = self._rt_call(fn, key.float().unsqueeze(0), kb, (k_mode,), role="k",
                                 layer_idx=layer_idx)[0]
            vhat = self._rt_call(fn, value.float().unsqueeze(0), vb, (v_mode,), role="v",
                                 layer_idx=layer_idx)[0]
        if compiled:
            # CUDA-graph outputs live in the graph's memory pool and are
            # overwritten by the next replay; copy them out unconditionally.
            return khat.to(dtype, copy=True), vhat.to(dtype, copy=True), stored
        return khat.to(dtype), vhat.to(dtype), stored

    def get_kv_config(self) -> dict:
        cfg = super().get_kv_config()
        cfg.update({
            "dist": self.dist, "value_bits": self.value_bits,
            "block_size": self.block_size, "max_codebook_bits": self.max_codebook_bits,
            "codebook_init": self.codebook_init,
            "rotation_mode": self.rotation_mode, "encoder": self.encoder,
            "hadamard_gemm_max_dim": self.hadamard_gemm_max_dim,
            "ls_scale": self.ls_scale, "key_scale_mode": self.key_scale_mode,
            "value_scale_mode": self.value_scale_mode, "n_outlier": self.n_outlier,
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
            "key_error_weight": self.key_error_weight,
            "value_error_weight": self.value_error_weight,
            "key_code_bits_per_block": self.key_code_bits_per_block,
            "value_code_bits_per_block": self.value_code_bits_per_block,
            "use_qjl": self.use_qjl,
        })
        cal = self._calib()
        if cal is not None:
            cfg["calibration"] = cal.digest
        return cfg

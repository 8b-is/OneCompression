"""Data-free block vector quantization (VQ) codebooks.

TurboQuant (arXiv:2504.19874) rotates a KV vector so its coordinates are
(exactly, Lemma 1, or asymptotically as ``d -> inf``) i.i.d. draws from a
scalar law -- ``Beta((d-1)/2, (d-1)/2)`` on ``[-1, 1]`` (exact, L2-normalised)
or ``N(0, 1)`` (the high-dimensional limit, RMS-normalised) -- and then fits a
**scalar** Lloyd-Max quantizer to that law (``rotation.lloyd_max_beta`` /
``rotation.lloyd_max_gaussian``).

Classical rate-distortion theory (Gersho & Gray, *Vector Quantization and
Signal Compression*, 1992; Zamir, *Lattice Coding for Signals and Networks*,
2014) says this leaves a *known, quantified* gap open: independent scalar
quantization of a memoryless Gaussian source loses the **space-filling
gain** (Voronoi cells are hypercubes, not the rounder cells a higher-dimension
quantizer can use) *and*, at finite rate, some of the **shape gain** (the
optimal quantizer's cell shape should follow the source density, which is
increasingly spherical/isotropic in higher joint dimension). The high-rate
loss of scalar quantization relative to the (mem oryless) Shannon
rate-distortion bound is the classical ``1.53`` dB (``~0.255`` bit/dim);
lattice/vector quantizers close part of this: the block/lattice dimension
``b`` trades quantizer complexity for how much of that gap is recovered
(``b=1``: scalar, the largest gap; ``b -> inf``: the full gap closes). This
is exactly TurboQuant's own stated slack -- the paper's distortion bound is
"near-optimal ... within a small constant factor (~2.7x)" of the
information-theoretic optimum, and a part of that constant is precisely this
scalar-vs-vector gap (independently flagged as unexplored "future work,
Lever 5" in this repo's HALO analysis, ``papers/halo_paper`` /
``benchmark/kvcomp-longcontext/paper/HALO_ENHANCED_REPORT.md``).

This module implements the **data-free** (no calibration on real data)
generalization: instead of a scalar codebook, fit a ``block_size``-dimensional
vector codebook to synthetic samples drawn *exactly* from TurboQuant's own
source law (the finite-``d`` Beta/Dirichlet joint law, or its Gaussian limit),
using Lloyd's algorithm (batched K-means) -- the same "data-free calibration"
principle TurboQuant itself uses for the scalar codebook, just in
``block_size`` dimensions instead of 1. Nearest-codeword search is a single
GEMM (``z @ cb.T``) plus an ``argmax``, so encode/decode stay O(1) matmuls per
block, exactly like the scalar path -- no explicit lattice geometry or
per-lattice decoding algorithm is required, and (because the codebook is
*learned* on the exact source law rather than a fixed lattice such as ``E8``)
the achieved distortion is always at least as good as the corresponding fixed
lattice at the same rate, while remaining just as data-free as TurboQuant's
own scalar codebook (see ``verify_hayatequant_vq_gain.py`` for the measured,
CPU-only ``b in {1,2,4,8}`` distortion curve against the theoretical
``1.53 dB -> 0 dB`` closing gap).

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache

import torch

# Precomputed high-budget Lloyd codebook artifacts (see
# ``codebooks/fit_codebooks.py``).  The online fit below uses a budget sized
# for an acceptable first-encode latency (100k samples, 25 iters, 4
# restarts); fitting offline with ~8-16x that budget measurably improves the
# codebook (block distortion -1.4% at the 2-bit ``(b=4, K=256)`` point and
# -2.6% at the 3-bit ``(b=4, K=4096)`` point, ``bench_rvq_gain.py``), is
# free at runtime, and makes the codebook bit-exactly reproducible across
# machines/BLAS builds (the artifact is committed, not refit).
_CODEBOOK_DIR = os.path.join(os.path.dirname(__file__), "codebooks")

# Candidate block sizes considered by :func:`resolve_block_size`, largest first.
_BLOCK_CANDIDATES = (8, 4, 2, 1)
_CODEBOOK_INITS = ("auto", "forgy", "fibquant")


def resolve_block_size(
    bits: int, block_size: "int | None", max_codebook_bits: int, m: int
) -> int:
    """Pick the largest feasible block size for a ``bits``-per-coordinate rate.

    The block codebook has ``2 ** (bits * block_size)`` entries; this is
    capped at ``2 ** max_codebook_bits`` for Lloyd-fit and encode/decode cost.
    A candidate must also evenly divide the subset dimension ``m`` (so the
    vector splits into whole blocks with no ragged remainder). ``block_size=1``
    is the unconditional fallback (even when ``bits > max_codebook_bits``,
    i.e. the cap constrains only the *block-size selection*, never the scalar
    codebook's own ``2**bits`` size) and reproduces the scalar TurboQuant/HALO
    codebook exactly -- HayateQuant is therefore always able to fall back to
    bit-for-bit TurboQuant behaviour, never worse.

    Args:
        bits: Bits per coordinate (the nominal codec rate).
        block_size: Requested block size, or ``None`` to auto-select the
            largest candidate in ``(8, 4, 2, 1)`` that fits under the cap.
        max_codebook_bits: Log2 cap on the number of codewords.
        m: Dimension of the subset being quantized (block size must divide it).
    """
    def _feasible(b: int) -> bool:
        return bits * b <= max_codebook_bits and m % b == 0

    if block_size is not None:
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        for b in _BLOCK_CANDIDATES:
            if b <= block_size and _feasible(b):
                return b
        return 1
    for b in _BLOCK_CANDIDATES:
        if _feasible(b):
            return b
    return 1


def _normal_icdf(p: torch.Tensor) -> torch.Tensor:
    """Inverse standard-normal CDF without adding a SciPy dependency."""
    p = p.clamp(1.0e-12, 1.0 - 1.0e-12)
    return math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)


def _regularized_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a,b), Numerical Recipes continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    def _betacf(aa: float, bb: float, xx: float) -> float:
        qab = aa + bb
        qap = aa + 1.0
        qam = aa - 1.0
        c = 1.0
        d = 1.0 - qab * xx / qap
        if abs(d) < 1.0e-300:
            d = 1.0e-300
        d = 1.0 / d
        h = d
        for m in range(1, 101):
            m2 = 2 * m
            aa_step = m * (bb - m) * xx / ((qam + m2) * (aa + m2))
            d = 1.0 + aa_step * d
            if abs(d) < 1.0e-300:
                d = 1.0e-300
            c = 1.0 + aa_step / c
            if abs(c) < 1.0e-300:
                c = 1.0e-300
            d = 1.0 / d
            h *= d * c
            aa_step = -(aa + m) * (qab + m) * xx / ((aa + m2) * (qap + m2))
            d = 1.0 + aa_step * d
            if abs(d) < 1.0e-300:
                d = 1.0e-300
            c = 1.0 + aa_step / c
            if abs(c) < 1.0e-300:
                c = 1.0e-300
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 3.0e-14:
                break
        return h

    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _beta_ppf(q: float, a: float, b: float) -> float:
    """Inverse regularized beta by monotone bisection."""
    q = min(max(float(q), 0.0), 1.0)
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(64):
        mid = (lo + hi) * 0.5
        if _regularized_beta(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


def _roberts_phi(dim: int) -> float:
    lo, hi = 1.0, 2.0
    for _ in range(96):
        mid = (lo + hi) * 0.5
        if mid ** (dim + 1) - mid - 1.0 > 0.0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) * 0.5


def _fibquant_directions(block_size: int, k: int) -> torch.Tensor:
    """Quasi-uniform FibQuant directions on S^(block_size-1)."""
    if block_size < 2:
        raise ValueError("FibQuant directions require block_size >= 2")
    idx = torch.arange(k, dtype=torch.float64)
    golden = 1.618033988749895
    theta_g = 1.0 - 1.0 / golden
    if block_size == 2:
        theta = 2.0 * math.pi * idx * theta_g
        return torch.stack((theta.cos(), theta.sin()), dim=1).to(torch.float32)
    if block_size == 3:
        z = 1.0 - (2.0 * (idx + 1.0) - 1.0) / float(k)
        theta = 2.0 * math.pi * idx * theta_g
        r = (1.0 - z.square()).clamp_min(0.0).sqrt()
        return torch.stack((r * theta.cos(), r * theta.sin(), z), dim=1).to(torch.float32)
    phi = _roberts_phi(block_size)
    cols = []
    for j in range(1, block_size + 1):
        frac = ((idx + 0.5) * (phi ** (-j))).remainder(1.0).clamp(1.0e-12, 1.0 - 1.0e-12)
        cols.append(_normal_icdf(frac))
    dirs = torch.stack(cols, dim=1)
    return (dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1.0e-12)).to(torch.float32)


def _fibquant_radii(block_size: int, ambient_dim: int, k: int) -> torch.Tensor:
    """FibQuant Bennett-Gersho radial companding, RMS-scaled for HayateQuant."""
    d = max(int(ambient_dim), block_size)
    qs = [(i + 0.5) / float(k) for i in range(k)]
    if block_size == d:
        unit = torch.ones(k, dtype=torch.float64)
    elif block_size == 2 and d > 2:
        unit = torch.tensor(
            [math.sqrt(max(0.0, 1.0 - (1.0 - q) ** (4.0 / d))) for q in qs],
            dtype=torch.float64,
        )
    else:
        alpha = block_size / 2.0
        beta = block_size / (block_size + 2.0) * ((d - block_size - 2.0) / 2.0) + 1.0
        beta = max(beta, 1.0e-6)
        unit = torch.tensor([math.sqrt(_beta_ppf(q, alpha, beta)) for q in qs], dtype=torch.float64)
    return (unit * math.sqrt(float(d))).to(torch.float32)


def fibquant_initial_codebook(bits: int, block_size: int, ambient_dim: int) -> torch.Tensor:
    """Deterministic FibQuant radial-angular initial codebook.

    It is used as a Lloyd warm-start, not as a raw replacement for the learned
    HayateQuant codebook.
    """
    if block_size < 2:
        raise ValueError("FibQuant initial codebook requires block_size >= 2")
    k = 1 << (bits * block_size)
    dirs = _fibquant_directions(block_size, k)
    radii = _fibquant_radii(block_size, ambient_dim, k)
    return (radii.unsqueeze(1) * dirs).contiguous()


def _validate_codebook_init(init: str) -> str:
    if init not in _CODEBOOK_INITS:
        raise ValueError(f"codebook_init must be one of {_CODEBOOK_INITS}, got {init!r}")
    return init


def _kmeans(samples: torch.Tensor, k: int, iters: int, seed: int, init: "torch.Tensor | None" = None):
    """Batched Lloyd's algorithm (K-means) for ``k`` centroids in R^m.

    Memory-efficient (``index_add_``-based centroid update, no ``O(N*K)``
    one-hot matrix); deterministic (seeded Forgy init from a fixed
    permutation). Dead clusters (zero assignments) are re-seeded from the
    globally worst-fit point so no codeword is ever wasted -- important at
    low bit-width where ``k`` can be a sizeable fraction of typical sample
    density in the tails.

    Returns ``(centroids, sse)`` where ``sse`` (up to an additive constant
    shared by every candidate fit) is used by :func:`_kmeans_best_of` to pick
    the best of several restarts.
    """
    n, m = samples.shape
    device = samples.device
    if init is None:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        perm = torch.randperm(n, generator=gen)[:k]
        centroids = samples[perm].clone()
    else:
        centroids = init.to(device=device, dtype=samples.dtype).clone()
        if tuple(centroids.shape) != (k, m):
            raise ValueError(f"init shape must be {(k, m)}, got {tuple(centroids.shape)}")
    sq = samples.pow(2).sum(dim=1, keepdim=True)  # (n, 1), reused every iter
    for _ in range(iters):
        csq = centroids.pow(2).sum(dim=1)  # (k,)
        # Squared distance via ||x||^2 - 2 x.c + ||c||^2 (only need argmin -> drop ||x||^2).
        scores = samples @ centroids.t() - 0.5 * csq.unsqueeze(0)  # (n, k), maximise
        assign = scores.argmax(dim=1)
        counts = torch.zeros(k, device=device, dtype=samples.dtype).index_add_(
            0, assign, torch.ones(n, device=device, dtype=samples.dtype)
        )
        sums = torch.zeros(k, m, device=device, dtype=samples.dtype).index_add_(0, assign, samples)
        empty = counts == 0
        new_centroids = torch.where(empty.unsqueeze(1), centroids, sums / counts.clamp_min(1).unsqueeze(1))
        if empty.any():
            # Re-seed each empty cluster at the point currently worst-served by
            # its own centroid (largest squared reconstruction error):
            # ||x - c||^2 = ||x||^2 - 2*(x.c - 0.5||c||^2) = sq - 2*score.
            own_dist = sq.squeeze(1) - 2 * scores.gather(1, assign.unsqueeze(1)).squeeze(1)
            n_empty = int(empty.sum())
            worst_idx = torch.topk(own_dist, k=min(n_empty, n)).indices
            new_centroids[empty] = samples[worst_idx[: int(empty.sum())]]
        if torch.allclose(new_centroids, centroids, atol=1e-6):
            centroids = new_centroids
            break
        centroids = new_centroids
    final_scores = samples @ centroids.t() - 0.5 * centroids.pow(2).sum(dim=1).unsqueeze(0)
    best = final_scores.gather(1, final_scores.argmax(dim=1, keepdim=True)).squeeze(1)
    sse = float((sq.squeeze(1) - 2 * best).sum())  # up to the const ||x||^2 term, monotone in true SSE
    return centroids, sse


def _kmeans_best_of(
    samples: torch.Tensor, k: int, iters: int, seed: int, restarts: int,
    init: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Run :func:`_kmeans` from ``restarts`` seeded inits, keep the lowest-SSE fit.

    Lloyd's algorithm only guarantees a local optimum; a handful of restarts
    (cheap relative to the fit itself) meaningfully reduces the chance of an
    unlucky init, especially for very small ``k`` (e.g. ``k=4`` at 1-2 bit)
    where a single bad Forgy draw can visibly hurt the fit.
    """
    best_centroids, best_sse = None, float("inf")
    for r in range(max(1, restarts)):
        centroids, sse = _kmeans(
            samples, k, iters, seed + r * 7919,
            init=init if r == 0 else None,
        )
        if sse < best_sse:
            best_centroids, best_sse = centroids, sse
    return best_centroids


def _beta_block_samples(block_size: int, d: int, n_samples: int, seed: int) -> torch.Tensor:
    """Exact samples of a ``block_size``-dim marginal of a random point on ``S^{d-1}``.

    A uniformly random unit vector ``u`` in ``R^d`` has, for any fixed subset
    of ``block_size`` coordinates, the exact joint law that TurboQuant's
    scalar ``Beta((d-1)/2, (d-1)/2)`` marginal generalises to blocks (the two
    coincide exactly at ``block_size=1``). Rescaling by ``sqrt(d)`` matches
    the RMS-normalised convention used elsewhere in this package (so the
    block codebook's units line up with :func:`rotation.lloyd_max_gaussian`
    in the ``d -> inf`` limit).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    g = torch.randn(n_samples, d, generator=gen, dtype=torch.float64)
    u = g / g.norm(dim=-1, keepdim=True)
    return (u[:, :block_size] * math.sqrt(d)).to(torch.float32)


def _gaussian_block_samples(block_size: int, n_samples: int, seed: int) -> torch.Tensor:
    """I.i.d. standard-normal samples (the ``d -> inf`` limit marginal)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n_samples, block_size, generator=gen, dtype=torch.float32)


@lru_cache(maxsize=32)
def _gaussian_codebook_artifact(block_size: int, k: int) -> "torch.Tensor | None":
    """Load the precomputed high-budget Gaussian codebook, if shipped.

    Artifacts are plain ``(K, block_size)`` fp32 tensors named
    ``gauss_b{block_size}_k{k}.pt`` under :data:`_CODEBOOK_DIR` (or the
    ``KVCOMP_CODEBOOK_DIR`` override, used by the fit tooling to A/B new
    artifacts). Missing artifact -> ``None`` (online fit fallback), so the
    codec never *requires* the files.
    """
    for base in (os.environ.get("KVCOMP_CODEBOOK_DIR"), _CODEBOOK_DIR):
        if not base:
            continue
        path = os.path.join(base, f"gauss_b{block_size}_k{k}.pt")
        if os.path.isfile(path):
            cb = torch.load(path, map_location="cpu", weights_only=True)
            if tuple(cb.shape) == (k, block_size):
                return cb.to(torch.float32)
    return None


@lru_cache(maxsize=32)
def lloyd_vq_gaussian_block(
    bits: int, block_size: int, iters: int = 25, n_samples: int = 100_000,
    seed: int = 0, restarts: int = 4, *, init: str = "auto",
    ambient_dim: "int | None" = None,
) -> torch.Tensor:
    """Data-free block-VQ codebook for i.i.d. ``N(0, 1)`` coordinates.

    Returns a ``(2**(bits*block_size), block_size)`` codebook fit by Lloyd's
    algorithm (best of ``restarts`` seeded inits, see :func:`_kmeans_best_of`)
    to synthetic Gaussian samples. ``block_size=1`` recovers
    :func:`rotation.lloyd_max_gaussian` (up to Lloyd-iteration noise; both
    converge to the same 1-D Lloyd-Max optimum).

    When a precomputed high-budget artifact exists for ``(block_size, K)``
    (see ``codebooks/fit_codebooks.py``) it is preferred over the online fit:
    same law, same algorithm, ~8-16x the sample/restart budget (measured
    -1.4% block distortion at 2-bit ``K=256`` and -2.6% at 3-bit ``K=4096``),
    zero runtime cost, and bit-exact reproducibility across machines.
    Callers passing non-default fit hyperparameters keep the online fit.
    """
    init = _validate_codebook_init(init)
    k = 1 << (bits * block_size)
    default_fit = (iters, n_samples, seed, restarts) == (25, 100_000, 0, 4)
    if default_fit and block_size > 1 and init != "fibquant":
        art = _gaussian_codebook_artifact(block_size, k)
        if art is not None:
            return art
    n_samples = min(n_samples, max(20_000, 2_000_000_000 // max(k, 1)))
    samples = _gaussian_block_samples(block_size, n_samples, seed)
    warm = None
    if block_size > 1 and init in ("auto", "fibquant"):
        warm = fibquant_initial_codebook(bits, block_size, ambient_dim or max(128, block_size + 3))
    return _kmeans_best_of(samples, k, iters, seed, restarts, init=warm)


@lru_cache(maxsize=32)
def lloyd_vq_beta_block(
    bits: int, block_size: int, d: int, iters: int = 25, n_samples: int = 100_000,
    seed: int = 0, restarts: int = 4, *, init: str = "auto",
) -> torch.Tensor:
    """Data-free block-VQ codebook for the exact finite-``d`` Beta/Dirichlet law.

    Returns a ``(2**(bits*block_size), block_size)`` codebook fit by Lloyd's
    algorithm (best of ``restarts`` seeded inits) to exact samples of a
    ``block_size``-dim marginal of a random point on ``S^{d-1}`` (rescaled by
    ``sqrt(d)``). ``block_size=1`` recovers :func:`rotation.lloyd_max_beta`.
    """
    init = _validate_codebook_init(init)
    k = 1 << (bits * block_size)
    n_samples = min(n_samples, max(20_000, 2_000_000_000 // max(k, 1)))
    samples = _beta_block_samples(block_size, d, n_samples, seed)
    warm = fibquant_initial_codebook(bits, block_size, d) \
        if block_size > 1 and init in ("auto", "fibquant") else None
    return _kmeans_best_of(samples, k, iters, seed, restarts, init=warm)


# Above this many (rows * codewords) products, chunk the nearest-codeword
# GEMM over rows so no single intermediate score matrix is materialised at
# once. Sized so a chunk's fp32 score matrix stays a small, fixed multiple of
# a few hundred MB regardless of how large the caller's batch/seq/heads get
# (unbounded intermediate size was measured to dominate encode latency on
# real (batch, heads, seq, n_blocks) shapes -- see
# ``benchmark/kvcomp-longcontext/codec_speed.py`` -- turning an O(N*K) GEMM
# that should take ~1ms of raw FLOPs into a 100+ms wall-clock cost purely
# from allocating and touching one gigantic (N, K) tensor).
_MAX_SCORE_ELEMENTS = 64_000_000  # ~256MB of fp32 scores per chunk


def quantize_to_vq_codebook(z: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
    """Nearest-codeword index for each ``block_size``-dim vector in ``z``.

    Args:
        z: ``(..., block_size)`` query blocks.
        cb: ``(K, block_size)`` codebook.

    Returns:
        Integer codes, ``(...,)``.

    ``block_size == 1`` (every "block" is a single scalar) is dispatched to
    :func:`onecomp.kvcomp.codec.rotation.quantize_to_codebook`'s ``O(N log K)``
    sorted-boundary ``searchsorted`` -- the same algorithm TurboQuant/HALO's
    own scalar quantizer uses -- instead of the general ``O(N*K)`` GEMM +
    argmax below, so HayateQuant's ``block_size=1`` fallback really is
    exactly as fast as TurboQuant/HALO, not just bit-for-bit identical in
    the reconstructed value. For ``block_size > 1`` (genuine vector
    quantization, no 1-D ordering exists), the GEMM nearest-neighbour search
    is run in row-chunks bounded by :data:`_MAX_SCORE_ELEMENTS` so encode
    latency scales linearly with the number of blocks instead of blowing up
    memory (and, transitively, wall-clock time) on realistic
    ``batch*heads*seq`` block counts.
    """
    cb = cb.to(device=z.device, dtype=z.dtype)
    k, block_size = cb.shape
    flat = z.reshape(-1, block_size)
    if block_size == 1:
        from ..rotation import quantize_to_codebook  # local import: avoid a cycle
        cb1 = cb.squeeze(-1)
        sorted_cb, order = cb1.sort()
        # 1-D Lloyd-Max codebooks are already sorted (Lloyd iterations preserve
        # centroid order), so the permutation is the identity for every
        # codebook this codec actually builds; skipping the full-N
        # ``order[codes]`` gather then makes this path exactly the raw
        # TurboQuant/HALO searchsorted (measured ~3x faster at real block
        # counts). The K-element equality check is a negligible sync.
        if torch.equal(sorted_cb, cb1):
            codes = quantize_to_codebook(flat.squeeze(-1), cb1)
        else:
            codes = order[quantize_to_codebook(flat.squeeze(-1), sorted_cb)]
        return codes.reshape(z.shape[:-1])

    # On CUDA, run the scoring GEMM in fp16 (fp32 accumulate on tensor
    # cores): ~2x the throughput and half the score-matrix traffic of fp32.
    # Only the *argmax comparison* sees fp16 rounding; a flip needs two
    # codewords' scores within fp16 resolution, and the excess distortion of
    # such a flip is bounded by twice that same score gap -- measured to be
    # far below 0.1% (see the LUT/exact parity tests, which cover this path).
    if flat.is_cuda and flat.dtype == torch.float32:
        flat = flat.half()
        cb = cb.half()
    csq = cb.pow(2).sum(dim=1)  # (K,)
    n = flat.shape[0]
    chunk = max(1, min(n, _MAX_SCORE_ELEMENTS // max(k, 1)))
    if chunk >= n:
        scores = flat @ cb.t() - 0.5 * csq.unsqueeze(0)  # maximise <=> minimise ||z-c||^2
        codes = scores.argmax(dim=1)
    else:
        codes = torch.empty(n, dtype=torch.long, device=z.device)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            scores = flat[start:end] @ cb.t() - 0.5 * csq.unsqueeze(0)
            codes[start:end] = scores.argmax(dim=1)
    return codes.reshape(z.shape[:-1])


def quantize_exact_pre(z: torch.Tensor, cb: torch.Tensor, csq: torch.Tensor) -> torch.Tensor:
    """Exact nearest-codeword assignment with precomputed codeword norms.

    Compile-/CUDA-graph-friendly variant of :func:`quantize_to_vq_codebook`
    for ``block_size > 1``: the caller supplies the codebook already on the
    right device and in the scoring dtype, together with its precomputed
    squared norms ``csq = cb.pow(2).sum(1)``, so the body is a pure chain of
    tensor ops -- no cache lookups, no per-call dtype casts, and no
    data-dependent host synchronisation.  Used by the fused round-trip path
    (``HayateQuantCodec.fake_quant_roundtrip``), whose per-token cost is
    launch-bound, and safe to wrap in ``torch.compile``.
    """
    k, b = cb.shape
    flat = z.reshape(-1, b)
    n = flat.shape[0]
    chunk = max(1, min(n, _MAX_SCORE_ELEMENTS // max(k, 1)))
    if chunk >= n:
        scores = flat @ cb.t() - 0.5 * csq.unsqueeze(0)
        return scores.argmax(dim=1).reshape(z.shape[:-1])
    codes = torch.empty(n, dtype=torch.long, device=z.device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        scores = flat[start:end] @ cb.t() - 0.5 * csq.unsqueeze(0)
        codes[start:end] = scores.argmax(dim=1)
    return codes.reshape(z.shape[:-1])


# Below this many gathered rows the extra index-arithmetic kernels of the
# flattened-take path cost more than they save (the pathological advanced-
# indexing kernel only dominates at large N); measured crossover is well
# below 64k rows on A100 either way, so a single conservative constant works.
_TAKE_MIN_ROWS = 65536


def lookup_codewords(cb: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """Gather codewords: ``cb[codes]`` with the fast kernel per block size.

    ``(K, b)`` row gathers hit a pathologically slow path in PyTorch's
    advanced-indexing/index_select kernels for small row sizes ``b >= 4``
    (measured 2.16ms vs 0.11ms for a *flattened 1-D* ``take`` of the same
    bytes at ``b=4, K=256``, 4.2M rows on A100 -- 19x; see
    ``paper/tests/bench_lut_spec_sweep.py``). This helper dispatches:

    * ``b >= 4`` on CUDA with a large row count: flatten to a 1-D ``take``
      of ``b`` consecutive scalars per code (19x faster at prefill sizes;
      below :data:`_TAKE_MIN_ROWS` the extra index-arithmetic kernels would
      instead add ~30us to a decode-step-sized gather, so small calls keep
      ``index_select``).
    * otherwise: ``index_select`` over the flattened codes (itself ~1.6x
      faster than ``cb[codes]``'s advanced indexing at ``b=2``).

    This is the decode path's dominant cost, and also the candidate-gather
    inside the LUT encoder.
    """
    k, b = cb.shape
    flat = codes.reshape(-1)
    if b >= 4 and flat.is_cuda and flat.numel() >= _TAKE_MIN_ROWS:
        idx = flat.unsqueeze(-1) * b + torch.arange(b, device=flat.device)
        out = cb.reshape(-1).take(idx)
    else:
        out = torch.index_select(cb, 0, flat)
    return out.reshape(*codes.shape, b)


def quantize_and_lookup(z: torch.Tensor, cb: torch.Tensor):
    """Return ``(codes, chat)``: nearest-codeword indices and their vectors.

    Convenience wrapper around :func:`quantize_to_vq_codebook` that also
    performs the codebook lookup, since almost every caller needs both.
    """
    cb = cb.to(device=z.device, dtype=z.dtype)
    codes = quantize_to_vq_codebook(z, cb)
    return codes, lookup_codewords(cb, codes)


# --- Grid-LUT (cell-indexed candidate-list) fast nearest-codeword encoder ---
#
# The exact nearest-codeword search above costs O(K) per block (a GEMM row
# against all K codewords). But the *query distribution is known and fixed*:
# after rotation and per-vector RMS normalisation every coordinate is
# (approximately at finite d, exactly in the d->inf limit) a standard normal,
# so queries live in a small, known box. That permits the classical
# encoder-complexity trick from VQ practice (cell-indexed candidate lists):
# precompute, once per codebook, a regular grid over ``[-lim, lim]^b`` whose
# cells each store the ``T`` codewords nearest to the cell centre; encoding a
# block is then (a) clamp+bin+flat-gather to fetch the cell's candidate list,
# (b) an exact distance argmin over just those ``T`` candidates. Encode cost
# becomes O(T) per block, *independent of K* (``T=8-16 << K=64-4096``), which
# (a) removes block-VQ's dominant runtime cost over scalar quantization at
# encode time, and (b) makes previously latency-prohibitive large codebooks
# (e.g. K=4096 at 3-bit x block_size=4) affordable at encode time,
# recovering their accuracy gain.
#
# Accuracy: a pure top-1 cell->codeword table was measured to lose ~3% MSE at
# (b=4, K=256, 48 bins) from Voronoi-boundary misassignment -- too much, it
# eats the block-VQ gain itself. The candidate-list variant instead only
# requires the *true* nearest codeword of a query to appear among the top-T
# neighbours of its cell's centre, which holds essentially always for
# ``T >= 8`` at these resolutions; the residual excess distortion is measured
# (not assumed) to be < 0.5% -- see the smoke/unit tests
# (``test_hayatequant_lut_encoder_matches_exact_distortion``). Queries outside
# the box are clamped, which is harmless: the outermost Voronoi regions are
# unbounded, so the clamped point's candidate list still contains the
# original's nearest codeword.

_LUT_LIM = 5.0


def lut_spec_for(block_size: int, k: int) -> "tuple[int, int] | None":
    """Default ``(bins_per_dim, candidates_T)`` for a ``(block_size, K)`` codebook.

    ``None`` means the LUT is infeasible for this block size (``bins**b``
    cells explode for ``b >= 8``; those callers use the exact chunked-GEMM
    search -- under the default ``max_codebook_bits`` cap, ``b=8`` only occurs
    at 1-bit where ``K=256`` keeps the exact search cheap anyway).

    The encode cost is dominated by the ``T``-candidate gather+distance step
    (nearly independent of ``bins``), so the specs trade grid resolution
    (memory) for a smaller ``T`` where the measured excess distortion stays
    negligible (A100 sweep: ``paper/tests/bench_lut_spec_sweep.py``):

    * ``b=2``: 256 bins -> 65k cells; cell width ``~0.04 sigma`` is far below
      the 2-D codeword spacing (``>= ~0.4 sigma`` at ``K<=512``), so ``T=2``
      already makes boundary misassignment unmeasurable.
    * ``b=4``, ``K <= 512`` (the default 2-bit point): ``bins=64, T=2`` --
      measured 5.0x faster than the previous ``(32, 8)`` (0.96ms vs 4.76ms
      per 4.2M-block pass) at +0.19% excess block distortion (vs -0.001%);
      the LUT is 134 MB but built once per codebook and cached.
    * ``b=4``, large ``K`` (the ``max_codebook_bits=12`` quality-ceiling
      points, ``K`` up to 4096): accuracy comes from the candidate list --
      keep ``T=16`` (excess -0.02%), still ~11x faster than the exact GEMM.
    """
    if block_size == 2:
        return 256, min(2, k)
    if block_size == 4:
        return (64, min(2, k)) if k <= 512 else (32, min(16, k))
    return None


def build_vq_lut(
    cb: torch.Tensor, bins: int, top_t: int, lim: float = _LUT_LIM
) -> torch.Tensor:
    """Precompute the ``(bins**b, T)`` integer cell -> candidate-codewords table.

    Every cell stores the ``T`` codewords nearest to its centre (exact,
    chunked scoring). The table build is the only place an O(cells * K) cost
    is ever paid -- once per (codebook, resolution); thereafter every encode
    is O(T) per block regardless of ``K``.

    Indices are stored as int16 when they fit (``K <= 32767`` -- always true
    under ``max_codebook_bits <= 15``): the table's random-access gather is
    the encoder's dominant memory traffic, so halving the element size halves
    both the resident table size (e.g. 67MB instead of 134MB at
    ``bins=64, T=2, b=4``) and the per-encode gather bytes.
    """
    k, b = cb.shape
    axis = ((torch.arange(bins, dtype=torch.float32, device=cb.device) + 0.5)
            / bins * (2.0 * lim) - lim)
    grids = torch.meshgrid(*([axis] * b), indexing="ij")
    centers = torch.stack([g.reshape(-1) for g in grids], dim=-1)  # (bins**b, b)
    csq = cb.pow(2).sum(dim=1)
    n = centers.shape[0]
    chunk = max(1, min(n, _MAX_SCORE_ELEMENTS // max(k, 1)))
    idx_dtype = torch.int16 if k <= 32767 else torch.int32
    out = torch.empty(n, top_t, dtype=idx_dtype, device=cb.device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        scores = centers[start:end] @ cb.t() - 0.5 * csq.unsqueeze(0)
        out[start:end] = scores.topk(top_t, dim=1).indices.to(idx_dtype)
    return out


def quantize_via_lut(
    z: torch.Tensor, lut: torch.Tensor, bins: int, cb: torch.Tensor,
    lim: float = _LUT_LIM,
) -> torch.Tensor:
    """O(T)-per-block nearest-codeword assignment via the candidate-list LUT.

    Args:
        z: ``(..., block_size)`` query blocks (RMS-normalised coordinates).
        lut: ``(bins**block_size, T)`` table from :func:`build_vq_lut`.
        bins: Per-dimension resolution the table was built with.
        cb: ``(K, block_size)`` codebook (for the exact argmin over the ``T``
            candidates).
        lim: Box half-width the table was built with.

    Returns:
        Integer codes, ``(...,)`` (int64, same convention as the exact search).
    """
    b = z.shape[-1]
    step = bins / (2.0 * lim)
    zi = ((z.clamp(-lim, lim) + lim) * step).long().clamp_(0, bins - 1)
    flat = zi[..., 0]
    for j in range(1, b):
        flat = flat * bins + zi[..., j]
    cands = lut[flat].long()                       # (..., T)
    if cands.shape[-1] == 1:
        return cands.squeeze(-1)
    # On CUDA, compute the T-candidate distances in fp16: halves the gather /
    # arithmetic traffic of the dominant candidate-gather + reduction step
    # (measured 4.5x at (b=4, K=256, T=8) and 5.7x at (b=4, K=4096, T=16) on
    # A100 -- see paper/tests/bench_encoder_paths.py). Same argument as the
    # fp16 exact-GEMM scoring: the candidate list itself is exact, only the
    # argmin over T sees fp16 rounding, and the measured excess distortion is
    # noise-level (<0.02%).
    if z.is_cuda and z.dtype == torch.float32:
        z = z.half()
        cb = cb.half()
    cw = lookup_codewords(cb, cands)               # (..., T, b)
    dist = (cw - z.unsqueeze(-2)).pow(2).sum(-1)   # (..., T) exact distances
    best = dist.argmin(dim=-1, keepdim=True)
    return cands.gather(-1, best).squeeze(-1)

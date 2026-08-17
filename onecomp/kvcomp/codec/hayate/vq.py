"""Canonical data-free vector quantization primitives for HayateQuant.

HayateQuant fixes two-coordinate Gaussian Lloyd codebooks.  The five shipped
tables cover the key/value widths needed by average rates 2, 3, and 4.  They
are manifest-pinned so inference never fits or selects a codebook at runtime.

This module also contains the exact and candidate-LUT assignment primitives
shared by the research codec and the packed vLLM backend.  The offline fitting
helpers remain private and are used only by :mod:`.codebooks.fit_codebooks`.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache

import torch

# Committed Lloyd codebook artifacts (see ``codebooks/manifest.json`` and
# ``codebooks/fit_codebooks.py``).  The manifest-pinned tensors prevent a
# machine- or BLAS-dependent first-use refit in a canonical HayateQuant run.
_CODEBOOK_DIR = os.path.join(os.path.dirname(__file__), "codebooks")
_CODEBOOK_MANIFEST_NAME = "manifest.json"
_SHIPPED_CODEBOOK_MANIFEST_SHA256 = (
    "f144f4a066c786f9fafef66ed3b2cb09cdc9bf6c804df84c32dcc15423d0a6dc"
)


def _kmeans(samples: torch.Tensor, k: int, iters: int, seed: int) -> tuple[torch.Tensor, float]:
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
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    if k > n:
        # Forgy needs a deterministic initial centroid for every codeword even
        # when a bounded synthetic fit set is smaller than K.
        perm = perm.repeat((k + n - 1) // n)
    centroids = samples[perm[:k]].clone()
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
        new_centroids = torch.where(
            empty.unsqueeze(1), centroids, sums / counts.clamp_min(1).unsqueeze(1)
        )
        if empty.any():
            # Re-seed each empty cluster at the point currently worst-served by
            # its own centroid (largest squared reconstruction error):
            # ||x - c||^2 = ||x||^2 - 2*(x.c - 0.5||c||^2) = sq - 2*score.
            own_dist = sq.squeeze(1) - 2 * scores.gather(1, assign.unsqueeze(1)).squeeze(1)
            n_empty = int(empty.sum())
            # A large codebook can legitimately have more dead clusters than
            # available fit samples (the production paper jobs hit this for
            # high-dimensional model activations).  Reuse the worst points
            # cyclically instead of attempting an impossible one-to-one
            # assignment, which previously raised a shape-mismatch error.
            worst_idx = torch.topk(own_dist, k=min(n_empty, n), largest=True).indices
            if n_empty > n:
                worst_idx = worst_idx.repeat((n_empty + n - 1) // n)
            new_centroids[empty] = samples[worst_idx[:n_empty]]
        if torch.allclose(new_centroids, centroids, atol=1e-6):
            centroids = new_centroids
            break
        centroids = new_centroids
    final_scores = samples @ centroids.t() - 0.5 * centroids.pow(2).sum(dim=1).unsqueeze(0)
    best = final_scores.gather(1, final_scores.argmax(dim=1, keepdim=True)).squeeze(1)
    sse = float(
        (sq.squeeze(1) - 2 * best).sum()
    )  # up to the const ||x||^2 term, monotone in true SSE
    return centroids, sse


def _kmeans_best_of(
    samples: torch.Tensor,
    k: int,
    iters: int,
    seed: int,
    restarts: int,
) -> torch.Tensor:
    """Run :func:`_kmeans` from ``restarts`` seeded inits, keep the lowest-SSE fit.

    Lloyd's algorithm only guarantees a local optimum; a handful of restarts
    (cheap relative to the fit itself) meaningfully reduces the chance of an
    unlucky init, especially for very small ``k`` (e.g. ``k=4`` at 1-2 bit)
    where a single bad Forgy draw can visibly hurt the fit.
    """
    best_centroids, best_sse = None, float("inf")
    for r in range(max(1, restarts)):
        centroids, sse = _kmeans(samples, k, iters, seed + r * 7919)
        if sse < best_sse:
            best_centroids, best_sse = centroids, sse
    return best_centroids


def _gaussian_block_samples(block_size: int, n_samples: int, seed: int) -> torch.Tensor:
    """I.i.d. standard-normal samples (the ``d -> inf`` limit marginal)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n_samples, block_size, generator=gen, dtype=torch.float32)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


@lru_cache(maxsize=8)
def _load_codebook_manifest(path: str) -> dict:
    """Load one immutable artifact manifest, failing closed on malformed JSON."""

    if not os.path.isfile(path):
        raise FileNotFoundError(f"HayateQuant codebook manifest is missing: {path}")
    try:
        with open(path, encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid HayateQuant codebook manifest: {path}") from exc
    if (
        not isinstance(document, dict)
        or document.get("schema") != "hayatequant.shipped_gaussian_codebooks.v1"
        or not isinstance(document.get("artifacts"), dict)
        or not isinstance(document.get("required_artifacts"), list)
    ):
        raise RuntimeError(f"unsupported HayateQuant codebook manifest schema: {path}")
    return document


def _load_shipped_codebook_manifest() -> tuple[str, dict]:
    """Load the committed manifest only after checking its root-of-trust SHA."""

    manifest_path = os.path.join(_CODEBOOK_DIR, _CODEBOOK_MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"HayateQuant codebook manifest is missing: {manifest_path}")
    actual_sha = _sha256_file(manifest_path)
    if actual_sha != _SHIPPED_CODEBOOK_MANIFEST_SHA256:
        raise RuntimeError(
            "shipped HayateQuant codebook manifest SHA-256 mismatch: " f"{manifest_path}"
        )
    return manifest_path, _load_codebook_manifest(manifest_path)


def _validate_gaussian_artifact(
    path: str,
    block_size: int,
    k: int,
    *,
    expected: "dict | None",
    source: str,
) -> tuple[torch.Tensor, dict]:
    """Strictly validate one tensor before it can enter a HayateQuant payload."""

    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # torch serialization exposes several exception types
        raise RuntimeError(f"cannot load HayateQuant {source} artifact: {path}") from exc
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"HayateQuant artifact must be a tensor: {path}")
    if tensor.dtype != torch.float32:
        raise TypeError(
            f"HayateQuant artifact {path} must have original dtype torch.float32, "
            f"got {tensor.dtype}"
        )
    if tuple(tensor.shape) != (int(k), int(block_size)):
        raise ValueError(
            f"HayateQuant artifact {path} has shape {tuple(tensor.shape)}, "
            f"expected {(int(k), int(block_size))}"
        )
    tensor = tensor.contiguous()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"HayateQuant artifact contains non-finite values: {path}")
    if int(torch.unique(tensor, dim=0).shape[0]) != int(k):
        raise ValueError(f"HayateQuant artifact contains duplicate codewords: {path}")
    file_sha = _sha256_file(path)
    tensor_sha = _sha256_tensor(tensor)
    if expected is not None:
        exact_contract = {
            "block_size": int(block_size),
            "k": int(k),
            "shape": [int(k), int(block_size)],
            "dtype": "torch.float32",
        }
        for field, value in exact_contract.items():
            if expected.get(field) != value:
                raise RuntimeError(
                    f"HayateQuant manifest contract mismatch for {path}/{field}: "
                    f"{expected.get(field)!r} != {value!r}"
                )
        if expected.get("file_sha256") != file_sha:
            raise RuntimeError(f"HayateQuant artifact file SHA-256 mismatch: {path}")
        if expected.get("tensor_sha256") != tensor_sha:
            raise RuntimeError(f"HayateQuant artifact tensor SHA-256 mismatch: {path}")
    return tensor, {
        "source": source,
        "path": os.path.abspath(path),
        "file_sha256": file_sha,
        "tensor_sha256": tensor_sha,
        "manifest_pinned": expected is not None,
        "block_size": int(block_size),
        "k": int(k),
    }


@lru_cache(maxsize=5)
def _gaussian_codebook_artifact_record(
    block_size: int,
    k: int,
) -> tuple[torch.Tensor, dict]:
    """Load one manifest-pinned codebook; no runtime override or fit exists."""

    filename = f"gauss_b{int(block_size)}_k{int(k)}.pt"
    manifest_path, manifest = _load_shipped_codebook_manifest()
    expected = manifest["artifacts"].get(filename)
    shipped_path = os.path.join(_CODEBOOK_DIR, filename)
    if not isinstance(expected, dict):
        raise RuntimeError(f"HayateQuant codebook is not manifest-pinned: {filename}")
    if not os.path.isfile(shipped_path):
        raise FileNotFoundError(f"manifest-pinned HayateQuant artifact is missing: {shipped_path}")
    tensor, record = _validate_gaussian_artifact(
        shipped_path,
        block_size,
        k,
        expected=expected,
        source="shipped_manifest",
    )
    record["manifest_path"] = os.path.abspath(manifest_path)
    record["manifest_sha256"] = _sha256_file(manifest_path)
    record["origin"] = expected.get("origin")
    return tensor, record


def require_hayatequant_codebooks() -> dict:
    """Validate and describe the complete canonical codebook set."""

    manifest_path, manifest = _load_shipped_codebook_manifest()
    filenames = manifest["required_artifacts"]
    if not isinstance(filenames, list) or not filenames:
        raise RuntimeError("HayateQuant required_artifacts must be a non-empty list")
    if len(filenames) != len(set(filenames)):
        raise RuntimeError("HayateQuant required_artifacts contains duplicates")
    artifacts = []
    for filename in filenames:
        expected = manifest["artifacts"].get(filename)
        if not isinstance(expected, dict):
            raise RuntimeError(f"required HayateQuant artifact is unpinned: {filename!r}")
        record = _gaussian_codebook_artifact_record(
            int(expected["block_size"]), int(expected["k"])
        )
        artifacts.append(dict(record[1]))
    return {
        "required": True,
        "manifest_path": os.path.abspath(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "artifacts": artifacts,
    }


@lru_cache(maxsize=5)
def load_hayatequant_codebook(code_bits: int) -> torch.Tensor:
    """Load the canonical two-coordinate codebook for one role width."""

    if isinstance(code_bits, bool) or not isinstance(code_bits, int):
        raise ValueError("HayateQuant code_bits must be an integer")
    if code_bits not in {2, 4, 6, 8, 10}:
        raise ValueError("HayateQuant code_bits must be one of 2, 4, 6, 8, or 10")
    return _gaussian_codebook_artifact_record(2, 1 << code_bits)[0]


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
    sorted-boundary ``searchsorted`` -- the same algorithm TurboQuant's
    own scalar quantizer uses -- instead of the general ``O(N*K)`` GEMM +
    argmax below, so HayateQuant's ``block_size=1`` fallback really is
    exactly as fast as TurboQuant while retaining the same scalar
    assignments. For ``block_size > 1`` (genuine vector
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
        # TurboQuant searchsorted (measured ~3x faster at real block
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


def build_vq_lut(cb: torch.Tensor, bins: int, top_t: int, lim: float = _LUT_LIM) -> torch.Tensor:
    """Precompute the ``(bins**b, T)`` integer cell -> candidate-codewords table.

    Every cell stores the ``T`` codewords nearest to its centre (exact,
    chunked scoring), ordered by ascending global codebook index.  The index
    order makes :func:`quantize_via_lut`'s first-minimum reduction use the same
    smallest-index tie break as the exact encoder without adding work to the
    encode hot path.  The table build is the only place an O(cells * K) cost
    is ever paid -- once per (codebook, resolution); thereafter every encode
    is O(T) per block regardless of ``K``.

    Indices are stored as int16 when they fit (``K <= 32767`` -- always true
    under ``max_codebook_bits <= 15``): the table's random-access gather is
    the encoder's dominant memory traffic, so halving the element size halves
    both the resident table size (e.g. 67MB instead of 134MB at
    ``bins=64, T=2, b=4``) and the per-encode gather bytes.
    """
    k, b = cb.shape
    axis = (torch.arange(bins, dtype=torch.float32, device=cb.device) + 0.5) / bins * (
        2.0 * lim
    ) - lim
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
        # ``topk`` does not specify which index survives a score tie at the
        # T-th boundary.  A formal table encoder needs the table itself to be
        # reproducible, not merely its subsequent candidate-distance argmin.
        # Stable descending sort retains the codebook's ascending global index
        # for equal scores; the final ascending-index sort below also fixes the
        # hot-path first-minimum tie contract.
        candidates = scores.argsort(dim=1, descending=True, stable=True)[:, :top_t]
        out[start:end] = candidates.sort(dim=1).values.to(idx_dtype)
    return out


def quantize_via_lut(
    z: torch.Tensor,
    lut: torch.Tensor,
    bins: int,
    cb: torch.Tensor,
    lim: float = _LUT_LIM,
    *,
    scoring_cb: torch.Tensor | None = None,
    scoring_csq: torch.Tensor | None = None,
) -> torch.Tensor:
    """O(T)-per-block nearest-codeword assignment via the candidate-list LUT.

    Args:
        z: ``(..., block_size)`` query blocks (RMS-normalised coordinates).
        lut: ``(bins**block_size, T)`` table from :func:`build_vq_lut`, whose
            candidate indices are sorted in ascending global-codebook order.
        bins: Per-dimension resolution the table was built with.
        cb: ``(K, block_size)`` reconstruction codebook.
        lim: Box half-width the table was built with.
        scoring_cb / scoring_csq: Optional canonical exact-search scoring
            codebook and its precomputed squared norms.  When supplied, the
            final candidate ranking uses the same
            ``z @ scoring_cb.T - 0.5 * scoring_csq`` arithmetic and dtype as
            :func:`quantize_exact_pre`.  Both tensors must be supplied
            together.  Omitting them retains the historical explicit
            subtract/square candidate-distance implementation.

    Returns:
        Integer codes, ``(...,)`` (int64, same convention as the exact search).
        Equal candidate distances resolve to the smallest global codebook
        index because internally built LUT rows are index-sorted.
    """
    if (scoring_cb is None) != (scoring_csq is None):
        raise ValueError("scoring_cb and scoring_csq must be supplied together")
    b = z.shape[-1]
    # Canonical CUDA prefills use two candidates with b=2/4.  Fuse their
    # cell arithmetic, LUT gather, fp16 candidate distances, first-minimum
    # tie break, and int64 code write into one kernel.  The kernel preserves
    # the eager path's explicit fp16 subtraction/square rounding and fp32
    # reduction accumulation; large-fixture parity tests cover both layouts.
    # Keep torch.compile graphs on the pure-tensor body below.
    if (
        z.is_cuda
        and z.dtype == torch.float32
        and cb.dtype == torch.float32
        and lut.dtype in (torch.int16, torch.int32)
        and b in (2, 4)
        and int(lut.shape[-1]) == 2
        and scoring_cb is None
        and not torch.compiler.is_compiling()
    ):
        from .triton_step import HAS_TRITON, quantize_lut_t2_triton

        if HAS_TRITON:
            return quantize_lut_t2_triton(z, lut, bins, cb, lim=lim)
    if (
        scoring_cb is not None
        and z.is_cuda
        and z.dtype == torch.float32
        and scoring_cb.is_cuda
        and scoring_cb.dtype == torch.float16
        and scoring_csq.dtype == torch.float16
        and lut.dtype in (torch.int16, torch.int32)
        and b == 2
        and int(lut.shape[-1]) == 2
        and not torch.compiler.is_compiling()
    ):
        from .triton_step import HAS_TRITON, quantize_lut_canonical_t2_triton

        if HAS_TRITON:
            return quantize_lut_canonical_t2_triton(
                z,
                lut,
                bins,
                scoring_cb,
                scoring_csq,
                lim=lim,
            )
    step = bins / (2.0 * lim)
    zi = ((z.clamp(-lim, lim) + lim) * step).long().clamp_(0, bins - 1)
    flat = zi[..., 0]
    for j in range(1, b):
        flat = flat * bins + zi[..., j]
    cands = lut[flat].long()  # (..., T)
    if cands.shape[-1] == 1:
        return cands.squeeze(-1)
    if scoring_cb is not None:
        # Canonical exact HayateQuant scores a half-precision query against the cached
        # half-precision codebook and subtracts the cached half-precision
        # squared norm.  Restricting that same score to the cell candidates
        # removes a subtle implementation mismatch in the former LUT path:
        # explicit fp16 subtract/square distances can rank near-boundary
        # codewords differently even when the full-search winner is present.
        score_z = z
        if score_z.dtype != scoring_cb.dtype:
            score_z = score_z.to(dtype=scoring_cb.dtype)
        flat_z = score_z.reshape(-1, b)
        flat_cands = cands.reshape(-1, cands.shape[-1])
        candidate_cb = lookup_codewords(scoring_cb, flat_cands)
        # Strided-batched GEMM keeps fp16 inputs/fp32 accumulation, matching
        # the canonical all-codeword GEMM's scoring precision while doing
        # only O(T) work per query block.
        scores = torch.bmm(candidate_cb, flat_z.unsqueeze(-1)).squeeze(-1)
        candidate_csq = torch.index_select(scoring_csq, 0, flat_cands.reshape(-1)).reshape_as(
            flat_cands
        )
        scores = scores - 0.5 * candidate_csq
        best = scores.argmax(dim=-1, keepdim=True)
        return flat_cands.gather(-1, best).squeeze(-1).reshape(z.shape[:-1])
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
    cw = lookup_codewords(cb, cands)  # (..., T, b)
    dist = (cw - z.unsqueeze(-2)).pow(2).sum(-1)  # (..., T) exact distances
    best = dist.argmin(dim=-1, keepdim=True)
    return cands.gather(-1, best).squeeze(-1)

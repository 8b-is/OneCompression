"""Random orthogonal rotation and Lloyd-Max codebook utilities.

Used by :class:`onecomp.kvcomp.codec.turboquant.TurboQuantCodec`.  A random
orthogonal rotation makes the per-coordinate marginal of a KV vector close to
Gaussian, after which a scalar Lloyd-Max quantizer (fit *data-free* to the
standard normal) is near-optimal.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache

import torch

try:  # Dao-AILab CUDA kernel (github.com/Dao-AILab/fast-hadamard-transform)
    from fast_hadamard_transform import hadamard_transform as _fht_cuda
except ImportError:  # pragma: no cover - exercised on hosts without the wheel
    _fht_cuda = None

# Escape hatch for parity tests / debugging: set ONECOMP_DISABLE_FHT=1 to force
# the pure-PyTorch butterfly even when the CUDA kernel is importable.
_FHT_DISABLED = os.getenv("ONECOMP_DISABLE_FHT", "0") == "1"

# The kernel supports power-of-two dims up to 2**15 and fp16/bf16/fp32 inputs.
_FHT_MAX_DIM = 32768
_FHT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def is_power_of_two(n: int) -> bool:
    """Return True if ``n`` is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0


def _wht_butterfly(x: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch orthonormal WHT butterfly (fallback / CPU path)."""
    d = x.shape[-1]
    orig_shape = x.shape
    y = x.reshape(-1, d).clone().float()
    h = 1
    while h < d:
        y = y.reshape(y.shape[0], d // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack([a + b, a - b], dim=2)
        y = y.reshape(y.shape[0], d)
        h *= 2
    y = y / math.sqrt(d)
    return y.reshape(orig_shape).to(x.dtype)


def walsh_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Walsh-Hadamard transform along the last dimension.

    Normalised by ``1/sqrt(d)`` so the transform is its own inverse:
    ``walsh_hadamard_transform(walsh_hadamard_transform(x)) == x``.
    Requires the last dimension to be a power of two.

    On CUDA tensors this dispatches to the Dao-AILab ``fast_hadamard_transform``
    kernel when the wheel is installed (a single fused FWHT kernel instead of
    ``log2(d)`` memory-bound butterfly stages); otherwise it falls back to the
    pure-PyTorch butterfly.  Both paths compute the same orthonormal transform
    (parity is enforced by ``test_fast_hadamard_parity``).
    """
    d = x.shape[-1]
    if not is_power_of_two(d):
        raise ValueError(f"WHT requires power-of-two last dim, got {d}")
    if (
        _fht_cuda is not None
        and not _FHT_DISABLED
        and x.is_cuda
        and d <= _FHT_MAX_DIM
        and x.dtype in _FHT_DTYPES
    ):
        return _fht_cuda(x.contiguous(), scale=1.0 / math.sqrt(d))
    return _wht_butterfly(x)


@lru_cache(maxsize=32)
def _hadamard_cpu(dim: int) -> torch.Tensor:
    """Orthonormal Sylvester-Hadamard matrix ``H / sqrt(dim)`` (power-of-two dim).

    ``H`` has entries ``+/-1`` and ``H H^T = dim * I``; the returned matrix is
    symmetric and orthonormal (its own inverse).  Built once per ``dim`` and
    cached.
    """
    if not is_power_of_two(dim):
        raise ValueError(f"Hadamard matrix requires power-of-two dim, got {dim}")
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(dim)


@lru_cache(maxsize=64)
def hadamard_matrix(dim: int, device_str: str, dtype) -> torch.Tensor:
    """Device-resident orthonormal Hadamard matrix, cached per (dim, device, dtype).

    Applying the randomized Hadamard rotation as a single dense GEMM
    ``(x * sign) @ hadamard_matrix(d)`` is mathematically identical to the
    sequential Walsh-Hadamard butterfly :func:`walsh_hadamard_transform` but maps
    to one matmul -- GEMM-friendly (saturates tensor cores / BLAS) instead of
    ``log2(d)`` memory-bound butterfly stages, so it is faster while keeping the
    Hadamard transform's superior incoherence (and thus quantization quality on
    real, anisotropic KV) over a single Haar-random dense rotation.
    """
    return _hadamard_cpu(dim).to(device=torch.device(device_str), dtype=dtype)


@lru_cache(maxsize=64)
def _cached_sign(dim: int, seed: int, device_str: str, dtype) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.randint(0, 2, (dim,), generator=gen, dtype=torch.int8) * 2 - 1
    return s.to(device=torch.device(device_str), dtype=dtype)


@lru_cache(maxsize=64)
def signed_hadamard_matrix(dim: int, seed: int, device_str: str, dtype) -> torch.Tensor:
    """Folded randomized-Hadamard rotation ``HS = diag(sign) @ H`` (cached).

    The randomized-Hadamard rotation ``(x * sign) @ H`` is *identical* to
    ``x @ HS`` with ``HS = diag(sign) @ H`` (scaling column ``j`` of ``x`` by
    ``sign[j]`` then right-multiplying by ``H`` equals right-multiplying by the
    row-scaled matrix ``HS``).  Precomputing ``HS`` turns the rotation into a
    **single GEMM with no separate elementwise sign pass** -- the GEMM is the
    same cost as a dense random rotation, but keeps the Hadamard transform's
    superior incoherence.  ``HS`` is orthonormal (``HS HS^T = D_sign H H^T
    D_sign = I``), so the inverse rotation is ``xr @ HS.t()``.

    Verified bit-for-bit identical to the ``(x*sign) @ H`` path by
    ``tests/onecomp/kvcomp/test_kvcomp.py`` (the ``hadamard_gemm`` round-trip and
    speed-ablation tests).
    """
    sign = _cached_sign(int(dim), int(seed), str(device_str), dtype)   # (d,)
    h = hadamard_matrix(int(dim), str(device_str), dtype)              # (d, d)
    return sign.unsqueeze(1) * h     # row j scaled by sign[j]  ==  diag(sign) @ H


def random_sign(dim: int, seed: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Deterministic ``+/-1`` diagonal (as a 1-D vector) for a random rotation.

    Cached per ``(dim, seed, device, dtype)``: the sign vector is reused on every
    encode/decode, so recomputing it (CPU generator + host->device copy) each
    call was pure online overhead in the Hadamard rotation path.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    return _cached_sign(int(dim), int(seed), str(dev), dtype)


def rotate(x: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """Apply rotation ``R = diag(sign) @ H`` along the last dim (row vectors)."""
    return walsh_hadamard_transform(x * sign)


def unrotate(x_rot: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`rotate`."""
    return walsh_hadamard_transform(x_rot) * sign


def lloyd_max_gaussian(bits: int, iters: int = 64, grid: int = 4096) -> torch.Tensor:
    """Compute Lloyd-Max centroids for the standard normal N(0, 1).

    Data-free: the codebook depends only on ``bits``.  Returns a sorted 1-D
    tensor of ``2**bits`` centroids.
    """
    levels = 1 << bits
    xs = torch.linspace(-6.0, 6.0, grid, dtype=torch.float64)
    pdf = torch.exp(-0.5 * xs * xs) / math.sqrt(2 * math.pi)
    pdf = pdf / pdf.sum()

    # Initialise centroids at equiprobable quantiles of the grid.
    cdf = torch.cumsum(pdf, dim=0)
    qs = (torch.arange(levels, dtype=torch.float64) + 0.5) / levels
    idx = torch.searchsorted(cdf, qs).clamp(0, grid - 1)
    centroids = xs[idx].clone()

    for _ in range(iters):
        boundaries = (centroids[1:] + centroids[:-1]) / 2.0
        region = torch.searchsorted(boundaries, xs)  # in [0, levels-1]
        # Vectorized centroid update: scatter-sum xs*pdf and pdf per region,
        # then divide.  Avoids a Python loop over all 2**bits levels.
        one_hot = (region.unsqueeze(0) == torch.arange(levels).unsqueeze(1))  # (L, grid)
        pdf_sum = (one_hot * pdf.unsqueeze(0)).sum(-1)  # (L,)
        xs_sum = (one_hot * (xs * pdf).unsqueeze(0)).sum(-1)  # (L,)
        new = torch.where(pdf_sum > 0, xs_sum / pdf_sum.clamp_min(1e-30), centroids)
        if torch.allclose(new, centroids, atol=1e-7):
            centroids = new
            break
        centroids = new
    return centroids.to(torch.float32)


def lloyd_max_beta(bits: int, d: int, iters: int = 64, grid: int = 4096) -> torch.Tensor:
    """Lloyd-Max centroids for the Beta((d-1)/2, (d-1)/2) law on ``[-1, 1]``.

    This is the *exact* marginal of a coordinate of a random unit vector after a
    random orthogonal rotation (TurboQuant, Lemma 1):
    ``f_X(x) propto (1 - x^2)^((d-3)/2)``.  As ``d`` grows this tends to a
    Gaussian, so :func:`lloyd_max_gaussian` is the high-dimensional limit.
    """
    levels = 1 << bits
    xs = torch.linspace(-1.0, 1.0, grid, dtype=torch.float64)
    # log density for numerical stability; (d-3)/2 can be large.
    logpdf = ((d - 3) / 2.0) * torch.log1p(-xs.clamp(-0.999999, 0.999999) ** 2)
    pdf = torch.exp(logpdf - logpdf.max())
    pdf = pdf / pdf.sum()
    cdf = torch.cumsum(pdf, dim=0)
    qs = (torch.arange(levels, dtype=torch.float64) + 0.5) / levels
    idx = torch.searchsorted(cdf, qs).clamp(0, grid - 1)
    centroids = xs[idx].clone()
    for _ in range(iters):
        boundaries = (centroids[1:] + centroids[:-1]) / 2.0
        region = torch.searchsorted(boundaries, xs)
        # Vectorized centroid update (same scatter-sum pattern as lloyd_max_gaussian).
        one_hot = (region.unsqueeze(0) == torch.arange(levels).unsqueeze(1))
        pdf_sum = (one_hot * pdf.unsqueeze(0)).sum(-1)
        xs_sum = (one_hot * (xs * pdf).unsqueeze(0)).sum(-1)
        new = torch.where(pdf_sum > 0, xs_sum / pdf_sum.clamp_min(1e-30), centroids)
        if torch.allclose(new, centroids, atol=1e-7):
            centroids = new
            break
        centroids = new
    return centroids.to(torch.float32)


def random_orthogonal(dim: int, seed: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """A deterministic random orthogonal matrix ``(dim, dim)`` via QR.

    Used for the paper's *dense* random rotation (Algorithm 1/2) and for
    rotating channel subsets whose size is not a power of two (outlier
    mixed-precision), where the fast Walsh-Hadamard transform does not apply.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(dim, dim, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)  # fix signs -> Haar measure
    return q.to(device=device, dtype=dtype)


def quantize_to_codebook(z: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Map each element of ``z`` to the index of the nearest centroid."""
    boundaries = (centroids[1:] + centroids[:-1]) / 2.0
    return torch.searchsorted(boundaries.to(z.device), z.contiguous())


# --- Block-dense rotation (HayateQuant) -----------------------------------
#
# Motivation (see ``hayatequant.py`` module docstring for the full derivation):
# a single dense random-orthogonal D x D rotation gives the best incoherence
# (hence the best per-coordinate Gaussianisation) but costs O(D^2); the
# Walsh-Hadamard / signed-Hadamard-GEMM rotation costs only O(D log D) /
# O(D^2) BLAS-friendly GEMM but is a *fixed* (non-Haar-random beyond the sign
# flip) structured transform, so on small blocks its output is less
# isotropic than a true Haar-random rotation.  HayateQuant's block-VQ needs
# each ``block_size``-dim block of the rotated vector to be close to an
# *isotropic* Gaussian/Beta block (that is what the offline codebook in
# ``_vq.py`` is fit to) -- so we compose:
#
#   R = HS_D  followed by  (Q_b_1 (+) Q_b_2 (+) ... (+) Q_b_{D/b})   ((+) = block-diagonal)
#
# i.e. the cheap global signed-Hadamard-GEMM (for full-vector decorrelation,
# O(D^2) as one GEMM, or O(D log D) as a butterfly) *then* a distinct small
# dense Haar-random rotation applied independently to each contiguous block
# of ``block_size`` coordinates (O(D * block_size), block_size << D).  Both
# factors are orthogonal, so the composition is orthogonal (norm-preserving,
# invertible by applying the transposes in reverse order) and the two-stage
# cost O(D log D + D * b) is far below the O(D^2) of a single dense D x D
# rotation while giving each block much better local isotropy than the
# Hadamard-only rotation alone (verified by
# ``benchmark/kvcomp-longcontext/paper/tests/verify_hayatequant_isotropy.py``).
#
# This directly answers the trade-off discussed for HALO/TurboQuant ("Hadamard
# is faster, Dense is more accurate -- can we interpolate?"): the block size
# ``b`` is exactly that interpolation knob.  ``b=1`` recovers the pure
# Hadamard rotation (no block-dense factor applied); ``b=D`` recovers a full
# dense rotation (up to the harmless extra Hadamard pre-mix).


@lru_cache(maxsize=64)
def block_dense_matrices(
    n_blocks: int, block_size: int, seed: int, device_str: str, dtype
) -> torch.Tensor:
    """Return ``(n_blocks, block_size, block_size)`` independent Haar-random
    orthogonal matrices (one per contiguous block), cached per call signature.

    Each block gets its own seed (``seed * 100003 + block_index``, matching
    :func:`_hybrid_matrix_cpu`) so that the ``n_blocks`` factors are
    independent draws, maximising the incoherence gained per bit of extra
    compute versus reusing a single shared matrix.

    Reference (two-stage) API used by the equivalence tests; the runtime hot
    path uses the folded single-GEMM :func:`hybrid_rotation_matrix`.
    """
    mats = torch.stack(
        [random_orthogonal(block_size, seed * 100003 + i) for i in range(n_blocks)],
        dim=0,
    )
    return mats.to(device=torch.device(device_str), dtype=dtype)


def apply_block_rotation(x: torch.Tensor, block_mats: torch.Tensor) -> torch.Tensor:
    """Apply per-block dense rotations ``block_mats`` (n_blocks, b, b) to the
    last dimension of ``x`` (``..., n_blocks * b``), block ``i`` transformed
    by ``block_mats[i]`` (row-vector convention: ``y_i = x_i @ block_mats[i]``).
    """
    n_blocks, b, _ = block_mats.shape
    orig_shape = x.shape
    xb = x.reshape(*orig_shape[:-1], n_blocks, b)
    yb = torch.einsum("...nb,nbc->...nc", xb, block_mats)
    return yb.reshape(orig_shape)


def unapply_block_rotation(y: torch.Tensor, block_mats: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`apply_block_rotation` (each block's matrix is
    orthogonal, so the inverse is the transpose)."""
    n_blocks, b, _ = block_mats.shape
    orig_shape = y.shape
    yb = y.reshape(*orig_shape[:-1], n_blocks, b)
    xb = torch.einsum("...nc,nbc->...nb", yb, block_mats)
    return xb.reshape(orig_shape)


def hybrid_rotate(
    x: torch.Tensor, sign: torch.Tensor, block_mats: torch.Tensor
) -> torch.Tensor:
    """Global signed-Hadamard rotation followed by per-block dense rotation.

    ``x``'s last dimension must be a power of two matching ``sign`` and equal
    to ``block_mats.shape[0] * block_mats.shape[1]``.

    Reference (two-stage) implementation; the runtime hot path uses the
    mathematically identical single-GEMM :func:`hybrid_rotation_matrix`
    instead (see that function's docstring).
    """
    xr = rotate(x, sign)
    return apply_block_rotation(xr, block_mats)


def hybrid_unrotate(
    xr: torch.Tensor, sign: torch.Tensor, block_mats: torch.Tensor
) -> torch.Tensor:
    """Inverse of :func:`hybrid_rotate`."""
    xh = unapply_block_rotation(xr, block_mats)
    return unrotate(xh, sign)


@lru_cache(maxsize=32)
def _hybrid_matrix_cpu(dim: int, block_size: int, seed: int) -> torch.Tensor:
    """Fold the two-stage hybrid rotation into one ``(dim, dim)`` matrix.

    The hybrid rotation is ``y = (x @ HS) @ blockdiag(Q_1, ..., Q_{dim/b})``
    (row-vector convention), so precomputing ``W = HS @ blockdiag(Q_i)`` --
    an orthogonal matrix, being a product of orthogonals -- turns the whole
    two-stage transform into **one GEMM**, mathematically identical to
    :func:`hybrid_rotate` (verified by
    ``test_hybrid_rotation_matrix_matches_two_stage``).

    This is the "folded rotation" trick that makes the hybrid rotation
    *free* at runtime: the butterfly WHT (``log2(d)`` memory-bound passes)
    plus the per-block einsum are replaced by a single BLAS GEMM with
    exactly the same cost as a plain dense random rotation -- i.e. the
    Hadamard-vs-Dense interpolation no longer costs anything beyond either
    endpoint. Composed in float64 for accuracy, cast to float32 once.

    Block seeds match :func:`block_dense_matrices` (``seed * 100003 + i``)
    so the folded and two-stage paths use the *same* random rotations.
    """
    h = hadamard_matrix(dim, "cpu", torch.float64)
    sign = _cached_sign(dim, seed, "cpu", torch.float64)
    hs = sign.unsqueeze(1) * h  # diag(sign) @ H, orthonormal
    w = hs.clone()
    n_blocks = dim // block_size
    for i in range(n_blocks):
        q = random_orthogonal(block_size, seed * 100003 + i, dtype=torch.float64)
        cols = slice(i * block_size, (i + 1) * block_size)
        # (HS @ blockdiag(Q))[:, cols] = HS[:, cols] @ Q_i
        w[:, cols] = hs[:, cols] @ q
    return w.to(torch.float32)


@lru_cache(maxsize=64)
def hybrid_rotation_matrix(
    dim: int, block_size: int, seed: int, device_str: str, dtype
) -> torch.Tensor:
    """Device-resident folded hybrid-rotation matrix (see :func:`_hybrid_matrix_cpu`)."""
    return _hybrid_matrix_cpu(int(dim), int(block_size), int(seed)).to(
        device=torch.device(device_str), dtype=dtype)

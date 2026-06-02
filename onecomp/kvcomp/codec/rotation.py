"""Random orthogonal rotation and Lloyd-Max codebook utilities.

Used by :class:`onecomp.kvcomp.codec.turboquant.TurboQuantCodec`.  A random
orthogonal rotation makes the per-coordinate marginal of a KV vector close to
Gaussian, after which a scalar Lloyd-Max quantizer (fit *data-free* to the
standard normal) is near-optimal.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch


def is_power_of_two(n: int) -> bool:
    """Return True if ``n`` is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0


def walsh_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Walsh-Hadamard transform along the last dimension.

    Normalised by ``1/sqrt(d)`` so the transform is its own inverse:
    ``walsh_hadamard_transform(walsh_hadamard_transform(x)) == x``.
    Requires the last dimension to be a power of two.
    """
    d = x.shape[-1]
    if not is_power_of_two(d):
        raise ValueError(f"WHT requires power-of-two last dim, got {d}")
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
        new = centroids.clone()
        for k in range(levels):
            mask = region == k
            w = pdf[mask].sum()
            if w > 0:
                new[k] = (xs[mask] * pdf[mask]).sum() / w
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
        new = centroids.clone()
        for k in range(levels):
            mask = region == k
            w = pdf[mask].sum()
            if w > 0:
                new[k] = (xs[mask] * pdf[mask]).sum() / w
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

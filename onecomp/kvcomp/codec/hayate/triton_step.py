"""Fused Triton kernel for HayateQuant's decode-step fake-quant round trip.

The per-token serving cost of the research codec is launch-bound: the eager
round trip issues dozens of micro-kernels (reductions, divisions, the scoring
GEMM, gathers) whose combined *launch* time dwarfs their GPU-busy time.  The
CUDA-graph mode (``ONECOMP_COMPILE_CODEC=1``) removes the launch overhead but
still replays every micro-kernel; this module removes the micro-kernels
themselves.  One Triton program per (role, head, token) row performs, entirely
in registers:

    RMS normalization -> exact nearest-codeword block assignment ->
    role-asymmetric scale fit (including the ``ls_iter`` re-quantization)
    -> reconstruction ``chat * s``

so a decode step costs exactly four device operations: the rotation GEMM,
this kernel, the inverse-rotation GEMM, and the dtype cast.  The two GEMMs
stay on cuBLAS/tensor cores where they are already optimal.

Scope (checked by the dispatcher in ``codec.py``): the exact-search regime
(the LUT never wins at decode-step sizes -- see ``HayateQuantCodec._use_lut``),
``block_size`` in {1, 2, 4}, codebooks up to ``K * n_blocks <= 65536`` score
elements per row, and power-of-two head dims up to 256.  Everything else
falls back to the fused PyTorch path, which is semantically identical.

Numerical notes: all arithmetic is fp32, like the eager path, except that the
eager exact search scores in fp16 on CUDA; scoring here is fp32, so the very
rare fp16 argmax tie-flips of the eager path do not occur.  Parity with
``decode(encode(x))`` is enforced by unit tests at the reconstruction-quality
level (identical up to tie-breaking).

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import torch

try:  # triton ships with the CUDA builds of torch; guard for CPU-only installs
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # pragma: no cover - exercised only on CPU-only installs
    HAS_TRITON = False

# Scale-mode encoding shared with the kernel (compile-time constants).
_MODE_ID = {
    "rms": 0, "nc": 1, "ls": 2, "ip": 3, "ls_iter": 4,
    "bsm": 2, "ub": 3,
}

# Register-pressure guard: per-row score matrix is (n_blocks, K) fp32.
_MAX_SCORE_ELEMS_PER_ROW = 65536
_MAX_DIM = 256

if HAS_TRITON:

    @triton.jit
    def _assign(zb, CB, K: tl.constexpr, NB: tl.constexpr, B: tl.constexpr):
        """Exact nearest-codeword lookup: zb (NB, B) -> chat (NB, B)."""
        offs_k = tl.arange(0, K)
        offs_b = tl.arange(0, B)
        cb = tl.load(CB + offs_k[:, None] * B + offs_b[None, :])   # (K, B)
        csq = tl.sum(cb * cb, axis=1)                              # (K,)
        # argmax of <z, c> - ||c||^2/2  ==  argmin of ||z - c||^2
        scores = tl.sum(zb[:, None, :] * cb[None, :, :], axis=2) - 0.5 * csq[None, :]
        codes = tl.argmax(scores, axis=1)                          # (NB,)
        return tl.load(CB + codes[:, None] * B + offs_b[None, :])  # (NB, B)

    @triton.jit
    def _quant_row(x, xb, rms, CB,
                   D: tl.constexpr, B: tl.constexpr, K: tl.constexpr,
                   NB: tl.constexpr, MODE: tl.constexpr, EPS: tl.constexpr):
        """One row: assign + role scale + reconstruct (mirrors fit_scale)."""
        chat_b = _assign(xb / rms, CB, K, NB, B)
        chat = tl.reshape(chat_b, (D,))
        if MODE == 0:  # rms
            s = rms
        elif MODE == 1:  # nc: ||x|| / max(||chat||, eps)
            s = tl.sqrt(tl.sum(x * x)) / tl.maximum(tl.sqrt(tl.sum(chat * chat)), EPS)
        else:
            num = tl.sum(x * chat)
            den = tl.maximum(tl.sum(chat * chat), EPS)
            if MODE == 3:  # ip: ||x||^2 / sign-preserving-clamped <x, chat>
                num_safe = tl.where(num >= 0, tl.maximum(num, EPS), tl.minimum(num, -EPS))
                s = tl.sum(x * x) / num_safe
            else:
                s = num / den                       # ls
                if MODE == 4:                       # ls_iter: requantize + refit
                    chat_b = _assign(xb / tl.maximum(s, EPS), CB, K, NB, B)
                    chat = tl.reshape(chat_b, (D,))
                    num2 = tl.sum(x * chat)
                    den2 = tl.maximum(tl.sum(chat * chat), EPS)
                    s = num2 / den2
        return chat * s

    @triton.jit
    def _hq_step_kernel(XR, CB, OUT, SPLIT,
                        D: tl.constexpr, B: tl.constexpr, K: tl.constexpr,
                        NB: tl.constexpr, MODE_K: tl.constexpr,
                        MODE_V: tl.constexpr, EPS: tl.constexpr):
        row = tl.program_id(0)
        offs_d = tl.arange(0, D)
        x = tl.load(XR + row * D + offs_d).to(tl.float32)          # (D,)
        xb = tl.reshape(x, (NB, B))
        rms = tl.maximum(tl.sqrt(tl.sum(x * x) / D), EPS)
        if row < SPLIT:
            y = _quant_row(x, xb, rms, CB, D, B, K, NB, MODE_K, EPS)
        else:
            y = _quant_row(x, xb, rms, CB, D, B, K, NB, MODE_V, EPS)
        tl.store(OUT + row * D + offs_d, y)


def triton_step_supported(d: int, block_size: int, k: int, modes) -> bool:
    """Whether the fused kernel covers this (shape, codebook, modes) call."""
    if not HAS_TRITON:
        return False
    n_blocks = d // block_size
    return (
        block_size in (1, 2, 4)
        and d <= _MAX_DIM and (d & (d - 1)) == 0
        and n_blocks * k <= _MAX_SCORE_ELEMS_PER_ROW
        and all(m in _MODE_ID for m in modes)
    )


def roundtrip_step_triton(x: torch.Tensor, w: torch.Tensor, cb32: torch.Tensor,
                          block_size: int, modes) -> torch.Tensor:
    """Fused rotate -> quantize -> scale -> reconstruct -> unrotate.

    Args:
        x: ``(R, batch, heads, seq, d)`` fp32 CUDA tensor; ``R`` stacked roles
            (role-major, so the first ``n/R`` flattened rows are keys).
        w: ``(d, d)`` fp32 rotation matrix.
        cb32: ``(K, block_size)`` fp32 contiguous codebook.
        block_size: Resolved VQ block size.
        modes: Per-role scale modes, e.g. ``("nc", "ls_iter")``.

    Returns:
        The reconstructed tensor, same shape and dtype as ``x``.
    """
    r = x.shape[0]
    d = x.shape[-1]
    cb32 = cb32.contiguous()
    flat = x.reshape(-1, d)
    xr = flat @ w
    out = torch.empty_like(xr)
    n = xr.shape[0]
    split = n // r if r == 2 else n
    k, b = cb32.shape
    _hq_step_kernel[(n,)](
        xr, cb32, out, split,
        D=d, B=b, K=k, NB=d // b,
        MODE_K=_MODE_ID[modes[0]], MODE_V=_MODE_ID[modes[-1]],
        EPS=1e-8, num_warps=4)
    return (out @ w.t()).reshape(x.shape)

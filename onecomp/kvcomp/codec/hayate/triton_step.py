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
``block_size`` in {1, 2, 4, 8}, codebooks up to ``K * n_blocks <= 65536`` score
elements per row, and power-of-two head dims up to 256.  Everything else
falls back to the fused PyTorch path, which is semantically identical.

Numerical notes: the serving round-trip arithmetic is fp32 and the fitted
scale is rounded through signed fp16 to model stored metadata.  The payload
decoder is also exact because it is only a codebook gather and scale multiply.
The monolithic round-trip kernel is therefore not used as a public payload
encoder: its reduction order does not generally reproduce the established
tensor encoder's half-GEMM assignments.  This module does expose two narrower
public-assignment kernels whose eager rounding points can be reproduced
exactly: the historical T=2 distance LUT path, the b=2/T=4 canonical-score
LUT candidate path, and the half-GEMM score path for b=8.
Large canonical fixtures, boundary rows, and first-minimum ties require exact
code parity before either dispatcher is enabled.

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
    "rms": 0,
    "nc": 1,
    "ls": 2,
    "ip": 3,
    "ls_iter": 4,
    "bsm": 2,
    "ub": 3,
}

# Register-pressure guard: per-row score matrix is (n_blocks, K) fp32.
_MAX_SCORE_ELEMS_PER_ROW = 65536
_MAX_DIM = 256

if HAS_TRITON:

    @triton.jit
    def _sub_rn_f16x2(a, b):
        """Materialise CUDA fp16 subtraction (prevent fp32 reassociation)."""
        return tl.inline_asm_elementwise(
            asm="sub.rn.f16x2 $0, $1, $2;",
            constraints="=r,r,r",
            args=[a, b],
            dtype=tl.float16,
            is_pure=True,
            pack=2,
        )

    @triton.jit
    def _mul_rn_f16x2(a, b):
        """Materialise CUDA fp16 multiplication (prevent fp32 reassociation)."""
        return tl.inline_asm_elementwise(
            asm="mul.rn.f16x2 $0, $1, $2;",
            constraints="=r,r,r",
            args=[a, b],
            dtype=tl.float16,
            is_pure=True,
            pack=2,
        )

    @triton.jit
    def _assign(zb, CB, K: tl.constexpr, NB: tl.constexpr, B: tl.constexpr):
        """Exact nearest-codeword lookup: zb (NB, B) -> chat (NB, B)."""
        offs_k = tl.arange(0, K)
        offs_b = tl.arange(0, B)
        cb = tl.load(CB + offs_k[:, None] * B + offs_b[None, :])  # (K, B)
        csq = tl.sum(cb * cb, axis=1)  # (K,)
        # argmax of <z, c> - ||c||^2/2  ==  argmin of ||z - c||^2
        scores = tl.sum(zb[:, None, :] * cb[None, :, :], axis=2) - 0.5 * csq[None, :]
        codes = tl.argmax(scores, axis=1)  # (NB,)
        return tl.load(CB + codes[:, None] * B + offs_b[None, :])  # (NB, B)

    @triton.jit
    def _quant_row(
        x,
        xb,
        rms,
        CB,
        D: tl.constexpr,
        B: tl.constexpr,
        K: tl.constexpr,
        NB: tl.constexpr,
        MODE: tl.constexpr,
        EPS: tl.constexpr,
    ):
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
                s = num / den  # ls
                if MODE == 4:  # ls_iter: requantize + refit
                    s_safe = tl.where(s >= 0, tl.maximum(s, EPS), tl.minimum(s, -EPS))
                    chat_b = _assign(xb / s_safe, CB, K, NB, B)
                    chat = tl.reshape(chat_b, (D,))
                    num2 = tl.sum(x * chat)
                    den2 = tl.maximum(tl.sum(chat * chat), EPS)
                    s = num2 / den2
        # The native cache and logical accounting store a signed fp16 scale.
        # Round here even though this fake-quant kernel has no payload.
        s = s.to(tl.float16).to(tl.float32)
        return chat * s

    @triton.jit
    def _hq_step_kernel(
        XR,
        CB,
        OUT,
        SPLIT,
        D: tl.constexpr,
        B: tl.constexpr,
        K: tl.constexpr,
        NB: tl.constexpr,
        MODE_K: tl.constexpr,
        MODE_V: tl.constexpr,
        EPS: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs_d = tl.arange(0, D)
        x = tl.load(XR + row * D + offs_d).to(tl.float32)  # (D,)
        xb = tl.reshape(x, (NB, B))
        rms = tl.maximum(tl.sqrt(tl.sum(x * x) / D), EPS)
        if row < SPLIT:
            y = _quant_row(x, xb, rms, CB, D, B, K, NB, MODE_K, EPS)
        else:
            y = _quant_row(x, xb, rms, CB, D, B, K, NB, MODE_V, EPS)
        tl.store(OUT + row * D + offs_d, y)

    @triton.jit
    def _hq_payload_decode_kernel(
        CODES,
        SCALE,
        CB,
        OUT,
        D: tl.constexpr,
        B: tl.constexpr,
        NB: tl.constexpr,
    ):
        """Gather and rescale one public payload row in one kernel."""
        row = tl.program_id(0)
        offs_nb = tl.arange(0, NB)
        offs_b = tl.arange(0, B)
        codes = tl.load(CODES + row * NB + offs_nb)
        chat_b = tl.load(CB + codes[:, None] * B + offs_b[None, :])
        chat = tl.reshape(chat_b, (D,))
        scale = tl.load(SCALE + row).to(tl.float32)
        offs_d = tl.arange(0, D)
        tl.store(OUT + row * D + offs_d, chat * scale)

    @triton.jit
    def _hq_lut_t2_kernel(
        Z,
        ZH,
        LUT,
        CBH,
        CODES,
        N: tl.constexpr,
        B: tl.constexpr,
        BINS: tl.constexpr,
        LIM: tl.constexpr,
        STEP: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Cell lookup plus exact eager-compatible T=2 candidate scoring."""
        rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < N
        offs_b = tl.arange(0, B)
        z = tl.load(Z + rows[:, None] * B + offs_b[None, :], mask=row_mask[:, None])
        z_clamped = tl.maximum(tl.minimum(z, LIM), -LIM)
        zi = ((z_clamped + LIM) * STEP).to(tl.int32)
        zi = tl.maximum(0, tl.minimum(zi, BINS - 1))
        if B == 2:
            weights = tl.where(offs_b == 0, BINS, 1)
        else:
            weights = tl.where(
                offs_b == 0,
                BINS * BINS * BINS,
                tl.where(offs_b == 1, BINS * BINS, tl.where(offs_b == 2, BINS, 1)),
            )
        flat = tl.sum(zi * weights[None, :], axis=1)

        offs_t = tl.arange(0, 2)
        cands = tl.load(LUT + flat[:, None] * 2 + offs_t[None, :], mask=row_mask[:, None])

        # ``quantize_via_lut`` rounds both operands to fp16, then materialises
        # the fp16 subtraction and square before its trailing-dimension sum.
        # Keep those rounding points explicit; the final comparison retains
        # candidate zero on equality, matching argmin's first-minimum rule.
        zh = tl.load(ZH + rows[:, None] * B + offs_b[None, :], mask=row_mask[:, None])
        cw = tl.load(
            CBH + cands[:, :, None] * B + offs_b[None, None, :],
            mask=row_mask[:, None, None],
        )
        diff = _sub_rn_f16x2(cw, zh[:, None, :])
        sq = _mul_rn_f16x2(diff, diff)
        # ``aten::sum`` promotes fp16 reduction accumulation to fp32 on CUDA
        # and casts the returned tensor back to fp16.
        dist = tl.sum(sq.to(tl.float32), axis=2).to(tl.float16)
        d0 = tl.sum(tl.where(offs_t[None, :] == 0, dist, 0.0), axis=1)
        d1 = tl.sum(tl.where(offs_t[None, :] == 1, dist, 0.0), axis=1)
        c0 = tl.sum(tl.where(offs_t[None, :] == 0, cands, 0), axis=1)
        c1 = tl.sum(tl.where(offs_t[None, :] == 1, cands, 0), axis=1)
        best = tl.where(d1 < d0, c1, c0)
        tl.store(CODES + rows, best.to(tl.int64), mask=row_mask)

    @triton.jit
    def _hq_lut_canonical_t4_kernel(
        Z,
        ZH,
        LUT,
        CBH,
        CSQH,
        CODES,
        N: tl.constexpr,
        BINS: tl.constexpr,
        LIM: tl.constexpr,
        STEP: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fused b=2,T=4 lookup with canonical half-GEMM score rounding."""
        rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < N
        offs_b = tl.arange(0, 2)
        z = tl.load(Z + rows[:, None] * 2 + offs_b[None, :], mask=row_mask[:, None])
        z_clamped = tl.maximum(tl.minimum(z, LIM), -LIM)
        zi = ((z_clamped + LIM) * STEP).to(tl.int32)
        zi = tl.maximum(0, tl.minimum(zi, BINS - 1))
        weights = tl.where(offs_b == 0, BINS, 1)
        flat = tl.sum(zi * weights[None, :], axis=1)
        offs_t = tl.arange(0, 4)
        cands = tl.load(
            LUT + flat[:, None] * 4 + offs_t[None, :],
            mask=row_mask[:, None],
        )

        zh = tl.load(ZH + rows[:, None] * 2 + offs_b[None, :], mask=row_mask[:, None])
        cbh = tl.load(
            CBH + cands[:, :, None] * 2 + offs_b[None, None, :],
            mask=row_mask[:, None, None],
        )
        # Match ``half_query @ half_codebook.T``: fp32 accumulation followed
        # by an fp16 output rounding before the pointwise norm subtraction.
        dot = tl.sum(zh[:, None, :].to(tl.float32) * cbh.to(tl.float32), axis=2).to(tl.float16)
        csq = tl.load(CSQH + cands, mask=row_mask[:, None])
        penalty = _mul_rn_f16x2(csq, tl.full((BLOCK_N, 4), 0.5, tl.float16))
        score = _sub_rn_f16x2(dot, penalty)
        best_pos = tl.argmax(score, axis=1)
        best = tl.sum(tl.where(offs_t[None, :] == best_pos[:, None], cands, 0), axis=1)
        tl.store(CODES + rows, best.to(tl.int64), mask=row_mask)

    @triton.jit
    def _hq_exact_b8_kernel(
        ZH,
        CBH,
        CSQH,
        CODES,
        B: tl.constexpr,
        K: tl.constexpr,
    ):
        """One b=8 query row, matching eager half-GEMM score rounding."""
        row = tl.program_id(0)
        offs_b = tl.arange(0, B)
        offs_k = tl.arange(0, K)
        z = tl.load(ZH + row * B + offs_b)
        cb = tl.load(CBH + offs_k[:, None] * B + offs_b[None, :])
        # Half GEMM multiplies half inputs and accumulates in fp32, then writes
        # a half output before the following pointwise penalty subtraction.
        dot = tl.sum(z[None, :].to(tl.float32) * cb.to(tl.float32), axis=1).to(tl.float16)
        csq = tl.load(CSQH + offs_k)
        penalty = _mul_rn_f16x2(csq, tl.full((K,), 0.5, tl.float16))
        score = _sub_rn_f16x2(dot, penalty)
        code = tl.argmax(score, axis=0)
        tl.store(CODES + row, code.to(tl.int64))


def triton_step_supported(d: int, block_size: int, k: int, modes) -> bool:
    """Whether the fused kernel covers this (shape, codebook, modes) call."""
    if not HAS_TRITON:
        return False
    n_blocks = d // block_size
    return (
        block_size in (1, 2, 4, 8)
        and d <= _MAX_DIM
        and (d & (d - 1)) == 0
        and n_blocks * k <= _MAX_SCORE_ELEMS_PER_ROW
        and all(m in _MODE_ID for m in modes)
    )


def triton_payload_decode_supported(d: int, block_size: int, k: int) -> bool:
    """Whether the public payload decoder covers this row layout."""
    return triton_step_supported(d, block_size, k, ("rms",))


def decode_payload_triton(
    codes: torch.Tensor,
    scale: torch.Tensor,
    cb32: torch.Tensor,
    d: int,
    block_size: int,
) -> torch.Tensor:
    """Fuse public-payload codeword gather and fp16-scale application."""
    if not triton_payload_decode_supported(d, block_size, cb32.shape[0]):
        raise ValueError("unsupported HayateQuant Triton payload decode configuration")
    b = int(block_size)
    nb = int(d) // b
    flat_codes = codes.reshape(-1, nb).contiguous()
    flat_scale = scale.reshape(-1).contiguous()
    rows = int(flat_codes.shape[0])
    out = torch.empty((rows, int(d)), dtype=torch.float32, device=codes.device)
    _hq_payload_decode_kernel[(rows,)](
        flat_codes,
        flat_scale,
        cb32.contiguous(),
        out,
        D=int(d),
        B=b,
        NB=nb,
        num_warps=4,
    )
    leading = tuple(int(size) for size in codes.shape[:-1])
    return out.reshape(*leading, int(d))


def quantize_lut_t2_triton(
    z: torch.Tensor,
    lut: torch.Tensor,
    bins: int,
    cb32: torch.Tensor,
    lim: float = 5.0,
) -> torch.Tensor:
    """Fuse the CUDA ``T=2`` LUT assignment pipeline into one kernel.

    This helper is intentionally narrow: HayateQuant's canonical prefill LUTs
    use power-of-two block sizes 2/4 and two sorted candidates.  Other layouts
    stay on the audited tensor implementation.
    """
    if not HAS_TRITON or not z.is_cuda:
        raise ValueError("HayateQuant LUT Triton assignment requires CUDA Triton")
    b = int(z.shape[-1])
    if b not in (2, 4) or int(lut.shape[-1]) != 2:
        raise ValueError("HayateQuant LUT Triton assignment supports only b in {2,4}, T=2")
    flat_z = z.reshape(-1, b).contiguous()
    rows = int(flat_z.shape[0])
    codes = torch.empty(rows, dtype=torch.long, device=z.device)
    block_n = 256
    _hq_lut_t2_kernel[(triton.cdiv(rows, block_n),)](
        flat_z,
        flat_z.half(),
        lut.contiguous(),
        cb32.half().contiguous(),
        codes,
        N=rows,
        B=b,
        BINS=int(bins),
        LIM=float(lim),
        STEP=float(bins) / (2.0 * float(lim)),
        BLOCK_N=block_n,
        num_warps=4,
    )
    return codes.reshape(z.shape[:-1])


def quantize_lut_canonical_t4_triton(
    z: torch.Tensor,
    lut: torch.Tensor,
    bins: int,
    scoring_cb: torch.Tensor,
    scoring_csq: torch.Tensor,
    lim: float = 5.0,
) -> torch.Tensor:
    """Fuse b=2,T=4 cell lookup and canonical exact candidate scoring."""
    if not HAS_TRITON or not z.is_cuda:
        raise ValueError("HayateQuant canonical LUT Triton assignment requires CUDA")
    if int(z.shape[-1]) != 2 or int(lut.shape[-1]) != 4:
        raise ValueError("canonical LUT Triton assignment supports only b=2,T=4")
    if scoring_cb.dtype != torch.float16 or scoring_csq.dtype != torch.float16:
        raise ValueError("canonical LUT Triton scoring tensors must be fp16")
    flat_z = z.reshape(-1, 2).contiguous()
    rows = int(flat_z.shape[0])
    codes = torch.empty(rows, dtype=torch.long, device=z.device)
    block_n = 256
    _hq_lut_canonical_t4_kernel[(triton.cdiv(rows, block_n),)](
        flat_z,
        flat_z.half(),
        lut.contiguous(),
        scoring_cb.contiguous(),
        scoring_csq.contiguous(),
        codes,
        N=rows,
        BINS=int(bins),
        LIM=float(lim),
        STEP=float(bins) / (2.0 * float(lim)),
        BLOCK_N=block_n,
        num_warps=4,
    )
    return codes.reshape(z.shape[:-1])


def triton_exact_b8_supported(
    cb16: torch.Tensor,
    csq16: torch.Tensor,
    device,
) -> bool:
    """Whether the audited SM100 b=8 public-assignment kernel is safe."""
    if not HAS_TRITON or not cb16.is_cuda or not csq16.is_cuda:
        return False
    k = int(cb16.shape[0]) if cb16.ndim == 2 else 0
    return (
        tuple(cb16.shape[1:]) == (8,)
        and cb16.dtype == torch.float16
        and cb16.device == torch.device(device)
        and tuple(csq16.shape) == (k,)
        and csq16.dtype == torch.float16
        and csq16.device == torch.device(device)
        and 0 < k <= 512
        and (k & (k - 1)) == 0
        # Half-GEMM reduction parity was audited against cuBLAS on B200/SM100.
        # Other architectures retain the generic tensor encoder.
        and torch.cuda.get_device_capability(device) == (10, 0)
    )


def quantize_exact_b8_triton(
    z: torch.Tensor,
    cb16: torch.Tensor,
    csq16: torch.Tensor,
) -> torch.Tensor:
    """Bit-exact b=8 assignment for the public tensor encoder."""
    if (
        not z.is_cuda
        or z.dtype != torch.float32
        or int(z.shape[-1]) != 8
        or not triton_exact_b8_supported(cb16, csq16, z.device)
    ):
        raise ValueError(
            "HayateQuant exact-b8 Triton assignment requires SM100 CUDA fp32 "
            "queries and a power-of-two fp16 codebook with K<=512"
        )
    flat_z = z.reshape(-1, 8).contiguous().half()
    rows = int(flat_z.shape[0])
    k = int(cb16.shape[0])
    codes = torch.empty(rows, dtype=torch.long, device=z.device)
    _hq_exact_b8_kernel[(rows,)](
        flat_z,
        cb16.contiguous(),
        csq16.contiguous(),
        codes,
        B=8,
        K=k,
        num_warps=2,
    )
    return codes.reshape(z.shape[:-1])


def roundtrip_step_triton(
    x: torch.Tensor, w: torch.Tensor, cb32: torch.Tensor, block_size: int, modes
) -> torch.Tensor:
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
        xr,
        cb32,
        out,
        split,
        D=d,
        B=b,
        K=k,
        NB=d // b,
        MODE_K=_MODE_ID[modes[0]],
        MODE_V=_MODE_ID[modes[-1]],
        EPS=1e-8,
        num_warps=4,
    )
    return (out @ w.t()).reshape(x.shape)

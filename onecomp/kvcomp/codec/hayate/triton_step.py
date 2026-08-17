"""Audited Triton primitives for canonical HayateQuant assignment and decode.

The payload decoder is exact because it is only a codebook gather and scale
multiply. The assignment kernels reproduce the public tensor encoder's eager
rounding points: the generic T=2 distance LUT path and the canonical b=2/T=2
score path. Large canonical fixtures, boundary rows, and first-minimum ties
require exact code parity before a dispatcher is enabled.

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
    def _hq_lut_canonical_t2_kernel(
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
        """Fused b=2,T=2 lookup with canonical half-GEMM score rounding."""
        rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < N
        offs_b = tl.arange(0, 2)
        z = tl.load(Z + rows[:, None] * 2 + offs_b[None, :], mask=row_mask[:, None])
        z_clamped = tl.maximum(tl.minimum(z, LIM), -LIM)
        zi = ((z_clamped + LIM) * STEP).to(tl.int32)
        zi = tl.maximum(0, tl.minimum(zi, BINS - 1))
        weights = tl.where(offs_b == 0, BINS, 1)
        flat = tl.sum(zi * weights[None, :], axis=1)
        offs_t = tl.arange(0, 2)
        cands = tl.load(
            LUT + flat[:, None] * 2 + offs_t[None, :],
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
        penalty = _mul_rn_f16x2(csq, tl.full((BLOCK_N, 2), 0.5, tl.float16))
        score = _sub_rn_f16x2(dot, penalty)
        best_pos = tl.argmax(score, axis=1)
        best = tl.sum(tl.where(offs_t[None, :] == best_pos[:, None], cands, 0), axis=1)
        tl.store(CODES + rows, best.to(tl.int64), mask=row_mask)


def triton_payload_decode_supported(d: int, block_size: int, k: int) -> bool:
    """Whether the public payload decoder covers this row layout."""
    return (
        HAS_TRITON
        and block_size in (1, 2, 4, 8)
        and 0 < d <= 256
        and (d & (d - 1)) == 0
        and d % block_size == 0
        and 0 < k <= 1024
    )


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


def quantize_lut_canonical_t2_triton(
    z: torch.Tensor,
    lut: torch.Tensor,
    bins: int,
    scoring_cb: torch.Tensor,
    scoring_csq: torch.Tensor,
    lim: float = 5.0,
) -> torch.Tensor:
    """Fuse b=2,T=2 cell lookup and canonical exact candidate scoring."""
    if not HAS_TRITON or not z.is_cuda:
        raise ValueError("HayateQuant canonical LUT Triton assignment requires CUDA")
    if int(z.shape[-1]) != 2 or int(lut.shape[-1]) != 2:
        raise ValueError("canonical LUT Triton assignment supports only b=2,T=2")
    if scoring_cb.dtype != torch.float16 or scoring_csq.dtype != torch.float16:
        raise ValueError("canonical LUT Triton scoring tensors must be fp16")
    flat_z = z.reshape(-1, 2).contiguous()
    rows = int(flat_z.shape[0])
    codes = torch.empty(rows, dtype=torch.long, device=z.device)
    block_n = 256
    _hq_lut_canonical_t2_kernel[(triton.cdiv(rows, block_n),)](
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

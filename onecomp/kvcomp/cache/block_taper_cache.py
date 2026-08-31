"""Block-uniform, paged-layout TaperKV simulated cache.

This module deliberately models the layout that a packed paged cache needs:
all tokens in a full block share a bit-width, and block mode has one affine
scale/zero pair per *batch-head-block*.  It is still a fake-quant research
cache; its reporting counts the corresponding packed payload and metadata.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import torch

from ..codec._quant_primitives import fake_quantize_affine
from .taper_cache import TaperKVCache, taper_bits


def block_taper_bits(
    total: int,
    block_size: int,
    sink_tokens: int,
    recent_tokens: int,
    hi_bits: int,
    b_min: int,
    ladder=None,
    residual_fp16: bool = True,
    device=None,
) -> torch.Tensor:
    """Return a conservative block-uniform age schedule of length ``total``.

    Every full page takes the maximum of its token schedule, preserving any
    sink/recent protection that intersects the page.  The partial write page
    is fp16 by default and is counted as such by the caller.
    """
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("total must be a non-negative integer")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if not isinstance(residual_fp16, bool):
        raise ValueError("residual_fp16 must be a bool")
    age = torch.arange(total - 1, -1, -1, device=device)
    bits = taper_bits(age, total, sink_tokens, recent_tokens, hi_bits, b_min, ladder)
    if total == 0:
        return bits
    n_full, tail = divmod(total, block_size)
    parts: list[torch.Tensor] = []
    if n_full:
        parts.append(
            bits[: n_full * block_size]
            .view(n_full, block_size)
            .amax(dim=1)
            .repeat_interleave(block_size)
        )
    if tail:
        tail_bits = bits[n_full * block_size :]
        parts.append(
            torch.full_like(tail_bits, 16) if residual_fp16 else tail_bits.amax().expand(tail)
        )
    return parts[0] if len(parts) == 1 else torch.cat(parts)


def fake_quant_blocks(
    x: torch.Tensor,
    bits: torch.Tensor,
    block_size: int,
    *,
    known_bit_widths: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Fake-quantize each batch-head-block with one affine scale/zero pair.

    ``x`` must be ``(batch, heads, tokens, head_dim)``.  Full blocks must have
    uniform widths; rejecting a non-uniform vector prevents a caller from
    accidentally reporting a paged layout while quantizing a different one.
    ``known_bit_widths`` is an optional static policy width set used by the
    live cache to avoid a CUDA-to-host ``unique().tolist()`` synchronisation.
    """
    if x.ndim != 4:
        raise ValueError("x must have shape (batch, heads, tokens, head_dim)")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if bits.ndim != 1 or bits.shape[0] != x.shape[2]:
        raise ValueError("bits must be a one-dimensional tensor with one entry per token")
    if bits.device != x.device:
        bits = bits.to(x.device)
    if torch.is_floating_point(bits) or bits.dtype == torch.bool:
        raise ValueError("bits must be an integer tensor")
    if bool((bits < 1).any()):
        raise ValueError("bits must be positive")

    batch, heads, tokens, dim = x.shape
    out = x.clone().float()
    n_full, tail = divmod(tokens, block_size)
    if n_full:
        full = bits[: n_full * block_size].view(n_full, block_size)
        if bool((full != full[:, :1]).any()):
            raise ValueError("bits must be uniform within every full block")
        block_bits = full[:, 0]
        # ``torch.unique(...).tolist()`` synchronises CUDA for every cache
        # regrade.  Policy schedules are static, so the live cache supplies
        # their finite CPU-side width set and avoids that device->host round
        # trip.  Keep the generic fallback for external callers with arbitrary
        # per-block bit vectors.
        widths = (
            torch.unique(block_bits).tolist() if known_bit_widths is None else known_bit_widths
        )
        for bit_width in widths:
            bit_width = int(bit_width)
            if bit_width < 1 or bit_width > 16:
                raise ValueError("known_bit_widths entries must lie in [1, 16]")
            if bit_width >= 16:
                continue
            block_indices = torch.nonzero(block_bits == bit_width, as_tuple=False).flatten()
            if block_indices.numel() == 0:
                continue
            selected = (
                block_indices[:, None] * block_size
                + torch.arange(block_size, device=x.device)[None, :]
            ).reshape(-1)
            groups = (
                x[:, :, selected, :]
                .float()
                .reshape(batch, heads, block_indices.numel(), block_size * dim)
            )
            # ``axis='channel'`` reduces the final flattened block dimension,
            # yielding one (scale, zero) pair for each batch/head/block.
            quantized = fake_quantize_affine(groups, bit_width, axis="channel")
            out[:, :, selected, :] = quantized.reshape(
                batch, heads, block_indices.numel() * block_size, dim
            )
    if tail:
        bit_width = int(bits[n_full * block_size])
        if bit_width < 16:
            groups = x[:, :, n_full * block_size :, :].float().reshape(batch, heads, 1, tail * dim)
            quantized = fake_quantize_affine(groups, bit_width, axis="channel")
            out[:, :, n_full * block_size :, :] = quantized.reshape(batch, heads, tail, dim)
    return out


class BlockTaperKVCache(TaperKVCache):
    """TaperKV with a page-uniform schedule and page-level affine metadata."""

    def __init__(self, policy, config=None, requant_stride: int | None = None, **kwargs):
        if requant_stride is None:
            requant_stride = policy.requant_stride or policy.block_size
        super().__init__(policy, config=config, requant_stride=requant_stride, **kwargs)

    def token_bits(self, total, device, layer_idx):
        p = self.policy
        return (
            block_taper_bits(
                total,
                p.block_size,
                p.sink_tokens,
                p.recent_tokens,
                p.key_bits,
                p.b_min,
                p.key_ladder,
                p.residual_fp16,
                device,
            ),
            block_taper_bits(
                total,
                p.block_size,
                p.sink_tokens,
                p.recent_tokens,
                p.value_bits,
                p.b_min_value,
                p.value_ladder,
                p.residual_fp16,
                device,
            ),
        )

    def _should_regrade(self, q_len: int, total: int) -> bool:
        """Keep each completed page physically and logically uniform.

        A prompt need not end on a page boundary.  Waiting only for the
        decode stride in that case used to quantize the token that completed
        the page while leaving its older page-mates at fp16.  The next full
        regrade then received a non-uniform work vector and either failed or
        misrepresented the page layout.  Regrade exactly when an update turns
        a partial page into one or more full pages.
        """

        if super()._should_regrade(q_len, total):
            return True
        previous = total - q_len
        return previous // self.policy.block_size < total // self.policy.block_size

    def _meta_bits(self, bits_vec, batch_heads):
        if self.policy.quant_granularity != "block":
            return self._affine_meta_bits(bits_vec, batch_heads)
        block_size = self.policy.block_size
        total = bits_vec.shape[0]
        n_full, tail = divmod(total, block_size)
        metadata = 0
        if n_full:
            block_bits = bits_vec[: n_full * block_size].view(n_full, block_size)[:, 0]
            quantized = block_bits < 16
            metadata += int(batch_heads) * (
                16 * int(quantized.sum()) + int(block_bits[quantized].sum())
            )
        if tail:
            tail_bits = int(bits_vec[n_full * block_size])
            if tail_bits < 16:
                metadata += int(batch_heads) * (16 + tail_bits)
        return metadata

    def _grade_layer(self, keys, values, layer_idx):
        if self.policy.quant_granularity != "block":
            return super()._grade_layer(keys, values, layer_idx)
        total = keys.shape[2]
        requested_k, requested_v = self.token_bits(total, keys.device, layer_idx)
        previous_k, previous_v = self._previous_bits(layer_idx, total, keys.device)
        kbits, kwork = self._lowering_work_bits(requested_k, previous_k)
        vbits, vwork = self._lowering_work_bits(requested_v, previous_v)
        # The schedule is policy-defined; pass its possible widths so the CUDA
        # hot path does not materialise ``torch.unique(...).tolist()`` on the
        # host once per layer/update.  The inclusive ranges also cover clipped
        # user ladders and preserve the old generic operator's exact result.
        key_widths = tuple(range(self.policy.b_min, self.policy.key_bits + 1))
        value_widths = tuple(range(self.policy.b_min_value, self.policy.value_bits + 1))
        key_quantized = fake_quant_blocks(
            keys, kwork, self.policy.block_size, known_bit_widths=key_widths
        )
        value_quantized = fake_quant_blocks(
            values, vwork, self.policy.block_size, known_bit_widths=value_widths
        )
        self._record_token_bits(layer_idx, kbits, vbits)
        batch_heads = keys.shape[0] * keys.shape[1]
        per_token_role = batch_heads * keys.shape[-1]
        stored_bits = int(
            kbits.sum() * per_token_role
            + vbits.sum() * per_token_role
            + self._meta_bits_for_role(kbits, batch_heads, "k")
            + self._meta_bits_for_role(vbits, batch_heads, "v")
        )
        return key_quantized, value_quantized, stored_bits

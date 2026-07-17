"""TaperKV: retrieval-preserving graded KV quantization (no eviction).

Motivation (onecomp-ref/260528.md): ThinKV compresses by *progressive
eviction* -- it permanently drops tokens.  That is strong for reasoning but
fragile for **long-context retrieval** (Needle / RULER): if the evicted token
*is* the needle, the answer is unrecoverable.

TaperKV keeps **every** token but assigns each a bit-width that *tapers* with
age: a full-precision sink + recent window, then a precision ladder that decays
toward a floor ``b_min > 0`` for old tokens.  No token is ever dropped, so the
needle survives -- lossily, but recoverably.  This trades ThinKV's hard
information loss (eviction) for graceful precision loss (PM-KVQ-style progressive
quantization), and additionally exploits the KVTuner observation that **keys
deserve more bits than values** (``key_bits >= value_bits``).

Length-preserving and RoPE-safe: like :class:`CompressedDynamicCache` we keep the
dequantized tensor (simulated quantization) and never change positions, so the
cache is simply passed to ``model.generate(past_key_values=...)``.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from logging import getLogger

import torch

from ._graded_base import GradedQuantCache, fake_quant_tokens

logger = getLogger(__name__)


def taper_bits(age: torch.Tensor, total: int, sink_tokens: int, recent_tokens: int,
               hi_bits: int, b_min: int, ladder=None) -> torch.Tensor:
    """Per-token bit-width as a function of token *age* (0 = newest).

    Sink (oldest ``sink_tokens``) and the ``recent_tokens`` newest stay at
    ``hi_bits``.  The middle region steps down ``hi_bits -> b_min`` across
    ``ladder`` equal bands (default: every integer bit-width in between).
    """
    bits = torch.full_like(age, hi_bits, dtype=torch.int64)
    is_sink = age >= (total - sink_tokens)
    is_recent = age < recent_tokens
    mid = ~(is_sink | is_recent)
    if ladder is None:
        ladder = list(range(hi_bits, b_min - 1, -1)) or [b_min]
    # Do not branch on ``mid.any()``: on CUDA that scalar conversion inserts a
    # host synchronisation into every cache update.  Computing the candidate
    # bands for the protected sink/recent entries too is cheap and ``where``
    # retains their high precision without a device-to-host round trip.
    nb = len(ladder)
    a = age.float()
    lo, hi = recent_tokens, max(recent_tokens + 1, total - sink_tokens)
    frac = ((a - lo) / max(1.0, (hi - lo))).clamp(0, 1 - 1e-6)
    band = (frac * nb).long().clamp(0, nb - 1)
    ladder_t = torch.tensor(ladder, device=age.device, dtype=torch.int64)
    bits = torch.where(mid, ladder_t[band], bits)
    # An explicitly-supplied ``ladder`` may hold values outside this role's
    # [b_min, hi_bits] range (e.g. a key ladder reused for values).  Clamp so no
    # token exceeds this role's high bit-width (preserving KVTuner K>=V) or drops
    # below its floor.  With the default ladder this is a no-op.
    return bits.clamp(min=b_min, max=hi_bits)


class TaperKVCache(GradedQuantCache):
    """Graded cache with a *static* age-indexed precision ladder; never evicts.

    The bit schedule (:meth:`token_bits`) is a fixed function of token *age*:
    sink + recent tokens stay at high precision, older tokens step down a
    ladder toward a non-zero floor.  All cache mechanics, re-quantization
    cadence and bit accounting are inherited from
    :class:`~onecomp.kvcomp.cache._graded_base.GradedQuantCache`.

    Args:
        policy: a :class:`onecomp.kvcomp.recipe.taperkv.TaperKVPolicy`.
        config: model config (preserves sliding-window layer typing).
        requant_stride: re-apply the ladder to the full cache every this many
            decode steps (1 = every step; larger = cheaper, staler).
    """

    def __init__(self, policy, config=None, requant_stride: int = 16, **kwargs):
        super().__init__(config=config, skip_layers=policy.skip_layers,
                         requant_stride=requant_stride, **kwargs)
        self.policy = policy

    def token_bits(self, total, device, layer_idx):
        p = self.policy
        age = torch.arange(total - 1, -1, -1, device=device)  # 0 = newest
        kbits = taper_bits(age, total, p.sink_tokens, p.recent_tokens,
                           p.key_bits, p.b_min, p.ladder)
        vbits = taper_bits(age, total, p.sink_tokens, p.recent_tokens,
                           p.value_bits, p.b_min_value, p.ladder)
        return kbits, vbits


def _resolve_ladder(hi_bits: int, b_min: int, ladder) -> list:
    return list(ladder) if ladder is not None else (list(range(hi_bits, b_min - 1, -1)) or [b_min])


def step_down_bits(bits: torch.Tensor, hi_bits: int, b_min: int, ladder=None) -> torch.Tensor:
    """Move every width down one rung of the same descending TaperKV ladder."""
    if bits.ndim != 1:
        raise ValueError("bits must be a one-dimensional tensor")
    ordered = _resolve_ladder(hi_bits, b_min, ladder)
    ascending = torch.tensor(sorted(ordered), device=bits.device, dtype=torch.int64)
    index = torch.searchsorted(ascending, bits.to(torch.int64))
    lower = ascending[(index - 1).clamp(min=0)]
    return torch.where(index > 0, lower, torch.full_like(bits, b_min)).clamp(
        min=b_min, max=hi_bits
    )


def apply_byte_budget(
    kbits: torch.Tensor,
    vbits: torch.Tensor,
    total: int,
    sink_tokens: int,
    recent_tokens: int,
    key_bits: int,
    value_bits: int,
    b_min: int,
    b_min_value: int,
    ladder,
    batch_heads: int,
    dim: int,
    max_bits: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one oldest-first ladder step toward a payload-only bit budget.

    Metadata is intentionally absent from both the budget and this helper:
    the policy documents a payload cap and the cache reports metadata
    separately.  ``batch_heads`` is the product of batch and KV heads, so the
    same cap means the same whole-layer payload even for batched generation.
    """
    if kbits.ndim != 1 or vbits.ndim != 1 or kbits.shape != vbits.shape:
        raise ValueError("kbits and vbits must be same-length one-dimensional tensors")
    if kbits.numel() != total:
        raise ValueError("total must equal the bit-vector length")
    if batch_heads < 1 or dim < 1:
        raise ValueError("batch_heads and dim must be positive")
    per_token_role = batch_heads * dim
    total_bits = int((kbits.sum() + vbits.sum()) * per_token_role)
    if max_bits is None or total_bits <= max_bits:
        return kbits, vbits
    start = min(sink_tokens, total)
    stop = max(start, total - recent_tokens)
    if stop <= start:
        return kbits, vbits
    indices = torch.arange(start, stop, device=kbits.device)
    next_k = step_down_bits(kbits[indices], key_bits, b_min, ladder)
    next_v = step_down_bits(vbits[indices], value_bits, b_min_value, ladder)
    savings = ((kbits[indices] - next_k) + (vbits[indices] - next_v)) * per_token_role
    if int(savings.sum()) <= 0:
        return kbits, vbits
    excess = total_bits - max_bits
    cumulative = torch.cumsum(savings, dim=0)
    enough = cumulative >= excess
    count = int(torch.nonzero(enough, as_tuple=False)[0]) + 1 if bool(enough.any()) else indices.numel()
    out_k, out_v = kbits.clone(), vbits.clone()
    chosen = indices[:count]
    out_k[chosen] = next_k[:count]
    out_v[chosen] = next_v[:count]
    return out_k, out_v


class BudgetTaperKVCache(TaperKVCache):
    """Budget-driven no-eviction TaperKV with exact per-update payload checks.

    A true byte cap cannot wait for a later regrade.  This cache consequently
    requires a stride of one; callers wanting a stale/cheaper age schedule
    should use plain ``TaperKVCache`` instead of claiming a hard budget.
    """

    def __init__(self, policy, config=None, requant_stride: int = 1, **kwargs):
        if requant_stride != 1:
            raise ValueError("BudgetTaperKVCache requires requant_stride=1 for its payload budget")
        super().__init__(policy, config=config, requant_stride=1, **kwargs)
        self._prev_bits: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def reset(self):
        super().reset()
        self._prev_bits.clear()

    def token_bits(self, total, device, layer_idx):
        p = self.policy
        return (
            torch.full((total,), p.key_bits, dtype=torch.int64, device=device),
            torch.full((total,), p.value_bits, dtype=torch.int64, device=device),
        )

    def _grade_layer(self, keys, values, layer_idx):
        p = self.policy
        total = keys.shape[2]
        batch_heads = keys.shape[0] * keys.shape[1]
        dim = keys.shape[-1]
        kbits, vbits = self.token_bits(total, keys.device, layer_idx)
        previous = self._prev_bits.get(layer_idx)
        if previous is not None:
            previous_k, previous_v = previous
            count = min(total, previous_k.numel(), previous_v.numel())
            if count:
                kbits[:count] = torch.minimum(
                    kbits[:count], previous_k[:count].to(device=keys.device, dtype=kbits.dtype)
                )
                vbits[:count] = torch.minimum(
                    vbits[:count], previous_v[:count].to(device=keys.device, dtype=vbits.dtype)
                )

        if p.max_bytes is not None:
            while True:
                next_k, next_v = apply_byte_budget(
                    kbits,
                    vbits,
                    total,
                    p.sink_tokens,
                    p.recent_tokens,
                    p.key_bits,
                    p.value_bits,
                    p.b_min,
                    p.b_min_value,
                    p.ladder,
                    batch_heads,
                    dim,
                    p.max_bytes * 8,
                )
                if next_k is kbits and next_v is vbits:
                    break
                kbits, vbits = next_k, next_v

        self._prev_bits[layer_idx] = (kbits.detach().clone(), vbits.detach().clone())
        key_quantized = fake_quant_tokens(keys, kbits)
        value_quantized = fake_quant_tokens(values, vbits)
        per_token_role = batch_heads * dim
        stored_bits = int(
            kbits.sum() * per_token_role
            + vbits.sum() * per_token_role
            + self._meta_bits(kbits, batch_heads)
            + self._meta_bits(vbits, batch_heads)
        )
        return key_quantized, value_quantized, stored_bits

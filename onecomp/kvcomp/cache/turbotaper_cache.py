"""TurboTaper cache: age-graded schedule quantized with rotated Lloyd-Max.

Reuses :class:`TaperKVCache`'s static age-indexed bit schedule (``token_bits``)
and all of :class:`GradedQuantCache`'s HF integration / accounting, but swaps the
per-token *affine* quantizer for TurboQuant's random-rotation + data-free
scalar Lloyd-Max codec (:class:`onecomp.kvcomp.codec.turboquant.TurboQuantCodec`).

This is the single behavioural difference between TaperKV and TurboTaper, so the
two are directly comparable at iso-bit; the rotated codec's MSE-optimality makes
TurboTaper the stronger method (see ``recipe/turbotaper.py``).

Design note -- delta quantization
----------------------------------
The base :class:`GradedQuantCache` re-applies the full bit ladder every
``requant_stride`` decode steps (default 16).  For the *affine* TaperKV codec
this is harmless because affine quantize→dequantize is idempotent.  The rotated
Lloyd-Max codec is *not* perfectly idempotent: the per-vector RMS normalisation
means that repeated re-quantisation of an already-quantised vector accumulates a
small drift.  For long reasoning sequences (QwQ-32B: ~4400 tokens → ~275 requant
passes on floor tokens), this drift compounds catastrophically.

Fix: **delta quantisation**.  :meth:`_grade_layer` tracks the bit assignment
from the previous call.  Only tokens whose bit-width *changed* (i.e. just
transitioned from the recent window into the floor band) and genuinely new tokens
are (re-)quantised.  Floor tokens whose bit assignment is unchanged are left in
place, guaranteeing that each token is rotated and quantised at most once per
bit-level transition.  Bit accounting still uses the full ``kbits``/``vbits``
vectors so the reported compression ratio is unaffected.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from logging import getLogger

import torch

from .taper_cache import TaperKVCache
from ..codec.turboquant import TurboQuantCodec
from ..codec._quant_primitives import fake_quantize_affine

logger = getLogger(__name__)


class TurboTaperCache(TaperKVCache):
    """Graded cache with TaperKV's age ladder + rotated Lloyd-Max quantization.

    Args:
        policy: a :class:`onecomp.kvcomp.recipe.turbotaper.TurboTaperPolicy`.
        config: model config (preserves sliding-window layer typing).
        requant_stride: re-grade cadence (inherited from the graded base).
    """

    def __init__(self, policy, config=None, requant_stride: int = 16, **kwargs):
        super().__init__(policy, config=config, requant_stride=requant_stride, **kwargs)
        self._codecs: dict = {}          # (bits, is_floor) -> TurboQuantCodec
        self._prev_kbits: dict = {}      # layer_idx -> kbits at last _grade_layer call
        self._prev_vbits: dict = {}      # layer_idx -> vbits at last _grade_layer call

    def _codec_for(self, bits: int, is_floor: bool) -> TurboQuantCodec:
        key = (bits, is_floor)
        codec = self._codecs.get(key)
        if codec is None:
            p = self.policy
            n_out = p.n_outlier_floor if is_floor else 0
            codec = TurboQuantCodec(
                bits=bits, dist=p.dist, rotation_mode=p.rotation_mode, seed=p.seed,
                n_outlier=n_out, outlier_bits=(bits + 1 if n_out else None))
            self._codecs[key] = codec
        return codec

    def _rot_quant_group(self, x: torch.Tensor, bits: torch.Tensor,
                         floor_bit: int) -> torch.Tensor:
        """Rotated Lloyd-Max fake-quantization for a group of tokens.

        ``floor_bit`` is the global floor bit-width (from policy).  Tokens whose
        ``bits == floor_bit`` use the floor codec (with ``n_outlier_floor``
        protection); all others use the standard codec.
        """
        out = x.clone().float()
        uniq = [int(b) for b in torch.unique(bits).tolist()]
        quant = [b for b in uniq if b < 16]
        for bv in quant:
            mask = bits == bv
            sub = x[:, :, mask, :].float()
            codec = self._codec_for(bv, is_floor=(bv == floor_bit))
            payload = codec._encode_tensor(sub, bv)
            if payload is None:  # non-power-of-two head_dim + hadamard -> affine fallback
                out[:, :, mask, :] = fake_quantize_affine(sub, bv, axis="token")
            else:
                out[:, :, mask, :] = codec._decode_tensor(payload, torch.float32)
        return out

    def _rot_quant(self, x: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        """Legacy full-cache rotated quantization (floor = min of bits)."""
        quant = [int(b) for b in torch.unique(bits).tolist() if b < 16]
        floor_bit = min(quant) if quant else None
        return self._rot_quant_group(x, bits, floor_bit)

    def _meta_bits(self, bits_vec, heads):
        """Rotated Lloyd-Max metadata: one fp16 per-vector norm per quantized
        token (no zero point, unlike affine).  The outlier-protected *floor* band
        stores a second per-vector norm (inner + outlier subset), matching
        :meth:`TurboQuantCodec._stored_bits` (``vecs * 2 * 16`` for the split).
        """
        q = bits_vec < 16
        n_quant = int(q.sum())
        if n_quant == 0:
            return 0
        meta = int(heads) * 16 * n_quant
        if self.policy.n_outlier_floor:
            quant = [int(b) for b in torch.unique(bits_vec).tolist() if b < 16]
            if quant:
                floor_bit = min(quant)
                n_floor = int((bits_vec == floor_bit).sum())
                meta += int(heads) * 16 * n_floor
        return meta

    def _grade_layer(self, keys, values, layer_idx):
        kbits, vbits = self.token_bits(keys.shape[2], keys.device, layer_idx)

        # --- delta quantisation: only (re-)quantise tokens that changed ---
        # Floor bits are fixed by policy; they determine the is_floor codec.
        k_floor = self.policy.b_min
        v_floor = self.policy.b_min_value

        prev_k = self._prev_kbits.get(layer_idx)
        prev_v = self._prev_vbits.get(layer_idx)
        n = keys.shape[2]

        kq = keys.float().clone()
        vq = values.float().clone()

        if prev_k is not None and prev_v is not None:
            n_prev = min(prev_k.shape[0], n)
            # Tokens 0..n_prev-1: re-quantise only if bit-width changed.
            k_need = torch.zeros(n, dtype=torch.bool, device=keys.device)
            v_need = torch.zeros(n, dtype=torch.bool, device=keys.device)
            k_need[:n_prev] = kbits[:n_prev] != prev_k[:n_prev]
            v_need[:n_prev] = vbits[:n_prev] != prev_v[:n_prev]
            # New tokens (n_prev..n-1): always quantise for the first time.
            k_need[n_prev:] = True
            v_need[n_prev:] = True
        else:
            # First call for this layer: quantise everything.
            k_need = torch.ones(n, dtype=torch.bool, device=keys.device)
            v_need = torch.ones(n, dtype=torch.bool, device=keys.device)

        if k_need.any():
            kq[:, :, k_need, :] = self._rot_quant_group(
                keys[:, :, k_need, :], kbits[k_need], k_floor)
        if v_need.any():
            vq[:, :, v_need, :] = self._rot_quant_group(
                values[:, :, v_need, :], vbits[v_need], v_floor)

        # Persist bit assignments for next call's delta check.
        self._prev_kbits[layer_idx] = kbits
        self._prev_vbits[layer_idx] = vbits

        per = keys.shape[1] * keys.shape[-1]
        heads = keys.shape[1]
        # Payload + per-vector rotated-codec norm metadata (see _meta_bits) so the
        # reported ratio is an honest whole-cache figure, comparable to the
        # TurboQuant codec baseline which also counts its per-vector norm.
        bits = int(kbits.sum() * per + vbits.sum() * per
                   + self._meta_bits(kbits, heads) + self._meta_bits(vbits, heads))
        n_out = self.policy.n_outlier_floor
        if n_out:
            for bt in (kbits, vbits):
                quant = [int(b) for b in torch.unique(bt).tolist() if b < 16]
                if not quant:
                    continue
                floor_bit = min(quant)
                n_floor = int((bt == floor_bit).sum())
                # +1 bit on n_out of d channels for each floor token (x heads).
                bits += n_floor * heads * n_out
        return kq, vq, bits

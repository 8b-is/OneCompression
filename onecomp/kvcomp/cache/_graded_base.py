"""Shared base for *graded* KV caches: keep every token, vary its bit-width.

Both proposed/baseline recipes in this package that compress by **per-token
precision instead of eviction** share the same mechanics:

* subclass :class:`transformers.DynamicCache` and override :meth:`update`,
* never drop a token (length-preserving, RoPE / causal-mask safe under vanilla
  ``model.generate``); compression is *simulated* (fake-quantize, keep float),
* re-quantize the materialised layer cache by a per-token bit-width vector,
* keep a small set of layers in full precision (``skip_layers``),
* account stored vs fp16 bits for an honest compression ratio.

The only thing that differs between recipes is the **bit schedule** -- the map
from token position to (key_bits, value_bits).  Subclasses implement
:meth:`token_bits`; everything else lives here.  Factoring the shared machinery
into this base keeps the recipes themselves minimal and directly comparable.

* :class:`~onecomp.kvcomp.cache.taper_cache.TaperKVCache` -- bits as a *static*
  function of token *age* (proposed; no eviction).
* :class:`~onecomp.kvcomp.cache.pmkvq_cache.PMKVQCache` -- bits as a *dynamic*
  function of decode progress / memory saturation + per-layer sensitivity
  (PM-KVQ baseline, arXiv:2505.18610).

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from logging import getLogger
from typing import Tuple

import torch
from transformers import DynamicCache

from ..codec._quant_primitives import fake_quantize_affine

logger = getLogger(__name__)


def fake_quant_tokens(x: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    """Per-token (variable bit-width) simulated quantization along ``head_dim``.

    ``x`` is ``(b, h, s, d)`` and ``bits`` is ``(s,)``.  Tokens sharing a
    bit-width are quantized together (per-token affine, ``axis='token'``);
    ``bits >= 16`` is left in full precision.
    """
    out = x.clone().float()
    for bv in torch.unique(bits).tolist():
        bv = int(bv)
        if bv >= 16:
            continue
        mask = bits == bv
        out[:, :, mask, :] = fake_quantize_affine(x[:, :, mask, :].float(), bv, axis="token")
    return out


class GradedQuantCache(DynamicCache):
    """``DynamicCache`` that keeps all tokens and applies a per-token bit ladder.

    Subclasses implement :meth:`token_bits` (the bit schedule).  This base owns
    the HF cache integration, re-quantization cadence, ``skip_layers`` handling,
    and bit accounting.

    Args:
        config: model config (preserves sliding-window layer typing).
        skip_layers: layer indices kept in full precision (outlier layers).
        requant_stride: re-apply the ladder to the full layer cache every this
            many decode steps (1 = every step; larger = cheaper, slightly staler
            for the *aging* tokens; prefill is always graded).
    """

    def __init__(self, config=None, skip_layers=None, requant_stride: int = 16, **kwargs):
        super().__init__(config=config, **kwargs)
        self.skip_layers = set(skip_layers or [])
        self.requant_stride = max(1, int(requant_stride))
        self._since = 0
        self._layer_bits: dict = {}    # layer_idx -> stored bits (graded, or FP for skipped)
        self._layer_fp16: dict = {}    # layer_idx -> fp16 reference bits
        self.fp16_bits_layer = 0

    # --- subclass hook -----------------------------------------------------
    def token_bits(self, total: int, device, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(key_bits, value_bits)`` int64 tensors of shape ``(total,)``.

        ``total`` is the current sequence length of the layer cache; index 0 is
        the oldest token, index ``total-1`` the newest.  Abstract.
        """
        raise NotImplementedError

    # --- internals ---------------------------------------------------------
    @staticmethod
    def _affine_meta_bits(bits_vec, heads):
        """Per-(token, head) quantization metadata for an affine codec.

        Each quantized vector (per-token bit-width ``b < 16``) stores an fp16
        ``scale`` (16 bits) and an integer ``zero`` (``b`` bits) -- exactly the
        overhead the baseline path accounts via
        :func:`onecomp.kvcomp.codec._quant_primitives.effective_bits`.  Counting
        it keeps the graded compression ratio an honest whole-cache figure that is
        directly comparable to the per-token codec baselines
        (``CompressedDynamicCache``), which already include it.
        """
        q = bits_vec < 16
        n_quant = int(q.sum())
        if n_quant == 0:
            return 0
        return int(heads) * (16 * n_quant + int(bits_vec[q].sum()))

    def _meta_bits(self, bits_vec, heads):
        """Quantization-metadata bits for one role (key or value).

        Defaults to affine ``scale + zero`` accounting; codecs whose per-vector
        metadata differs (e.g. TurboTaper's rotated Lloyd-Max, a per-vector norm
        and no zero point) override this.
        """
        return self._affine_meta_bits(bits_vec, heads)

    def _grade_layer(self, keys, values, layer_idx):
        kbits, vbits = self.token_bits(keys.shape[2], keys.device, layer_idx)
        kq = fake_quant_tokens(keys, kbits)
        vq = fake_quant_tokens(values, vbits)
        per = keys.shape[1] * keys.shape[-1]
        heads = keys.shape[1]
        bits = int(kbits.sum() * per + vbits.sum() * per
                   + self._meta_bits(kbits, heads) + self._meta_bits(vbits, heads))
        return kq, vq, bits

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        out_k, out_v = super().update(key_states, value_states, layer_idx, *args, **kwargs)
        q_len = key_states.shape[2]
        if layer_idx == 0:
            self._since += 1
        # Outlier-skip layers stay in full precision, but are still counted (at
        # fp16) so the reported memory is the honest *whole-cache* figure and is
        # directly comparable to the eviction baseline.
        if layer_idx in self.skip_layers:
            fp = (out_k.numel() + out_v.numel()) * 16
            self._layer_bits[layer_idx] = fp
            self._layer_fp16[layer_idx] = fp
            return out_k, out_v
        regrade = q_len > 1 or (self._since % self.requant_stride == 0)
        if not regrade:
            return out_k, out_v
        # Re-grade the materialised layer cache and return the graded tensors so
        # the *current* forward attends to quantized KV (faithful; makes both
        # the prefill/PPL path and the decode path reflect the precision ladder).
        layer = self.layers[layer_idx]
        kq, vq, bits = self._grade_layer(layer.keys, layer.values, layer_idx)
        layer.keys = kq.to(layer.keys.dtype)
        layer.values = vq.to(layer.values.dtype)
        self._layer_bits[layer_idx] = bits
        self._layer_fp16[layer_idx] = (layer.keys.numel() + layer.values.numel()) * 16
        self.fp16_bits_layer = self._layer_fp16[layer_idx]
        return layer.keys, layer.values

    # --- reporting ---------------------------------------------------------
    def compression_ratio(self) -> float:
        """Whole-cache compression ratio (fp16 bits / stored bits).

        Includes the full-precision outlier-skip layers, so the figure is an
        honest whole-cache ratio that is directly comparable across methods
        (quantized, eviction, and FullKV) rather than a quantized-layers-only
        ratio that would understate memory.
        """
        lb = self._layer_bits
        if not lb:
            return 1.0
        stored = sum(lb.values())
        fp16 = sum(self._layer_fp16.get(i, 0) for i in lb)
        return float(fp16 / max(1, stored)) if stored else 1.0

    def avg_bits(self) -> float:
        return 16.0 / max(1e-9, self.compression_ratio())

    def mem_fraction(self) -> float:
        """Stored / fp16 (the iso-memory comparison knob)."""
        return 1.0 / max(1e-9, self.compression_ratio())

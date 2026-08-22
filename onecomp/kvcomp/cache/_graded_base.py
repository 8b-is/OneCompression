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
        sliding_layers = [
            index for index, layer in enumerate(self.layers) if getattr(layer, "is_sliding", False)
        ]
        if sliding_layers:
            # DynamicSlidingWindowLayer.update() returns the transient full
            # attention states but retains only window-1 past states.  The
            # graded path rewrites/returns the retained layer tensor, so using
            # it here would silently drop current-prefill positions from the
            # very forward being scored.
            raise NotImplementedError(
                "GradedQuantCache does not support model-side sliding-window "
                f"layers {sliding_layers}; use a full-attention model/backend"
            )
        self.skip_layers = set()
        for raw_layer_idx in skip_layers or []:
            if isinstance(raw_layer_idx, bool) or not isinstance(raw_layer_idx, int):
                raise ValueError("skip_layers must contain integer layer indices")
            layer_idx = raw_layer_idx
            if layer_idx < 0:
                if not self.layers:
                    raise ValueError("negative skip_layers require a model config")
                layer_idx += len(self.layers)
            if layer_idx < 0 or (self.layers and layer_idx >= len(self.layers)):
                raise ValueError(f"skip layer index {raw_layer_idx} is out of range")
            self.skip_layers.add(layer_idx)
        if (
            isinstance(requant_stride, bool)
            or not isinstance(requant_stride, int)
            or requant_stride <= 0
        ):
            raise ValueError("requant_stride must be a positive integer")
        self.requant_stride = requant_stride
        self._since = 0
        self._layer_bits: dict = {}  # layer_idx -> stored bits (graded, or FP for skipped)
        self._layer_fp16: dict = {}  # layer_idx -> fp16 reference bits
        # The tensors in ``DynamicCache`` are dequantized research carriers, so
        # their dtype cannot tell us which precision is actually resident.  Keep
        # the applied per-token widths explicitly.  Besides making lifecycle
        # accounting exact, this prevents repeatedly quantizing an already
        # quantized token at the same width (which otherwise accumulates drift).
        self._layer_token_bits: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.fp16_bits_layer = 0

    def reset(self):
        """Reset tensors, cadence, and all resident bit-accounting state."""
        super().reset()
        self._since = 0
        self._layer_bits.clear()
        self._layer_fp16.clear()
        self._layer_token_bits.clear()
        self.fp16_bits_layer = 0

    # --- subclass hook -----------------------------------------------------
    def token_bits(self, total: int, device, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(key_bits, value_bits)`` int64 tensors of shape ``(total,)``.

        ``total`` is the current sequence length of the layer cache; index 0 is
        the oldest token, index ``total-1`` the newest.  Abstract.
        """
        raise NotImplementedError

    def teacher_forced_temporal_dependency(self):
        """Age/recency schedules must not see future teacher-forced targets."""
        return "age_dependent_precision_schedule_over_update_tokens"

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

        Defaults to affine ``scale + zero`` accounting. Specialized caches can
        override this when their per-vector metadata differs.
        """
        return self._affine_meta_bits(bits_vec, heads)

    def _meta_bits_for_role(self, bits_vec, heads, role: str):
        """Role-aware metadata hook; symmetric codecs reuse ``_meta_bits``."""

        return self._meta_bits(bits_vec, heads)

    def _extra_payload_bits(self, key_bits, value_bits, batch_heads, head_dim):
        """Additional packed payload not represented by the primary widths."""

        return 0

    def _previous_bits(
        self,
        layer_idx: int,
        total: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return applied widths padded with fp16 for newly appended tokens."""

        previous = self._layer_token_bits.get(layer_idx)
        result = []
        for role_bits in previous or ():
            role_bits = role_bits.to(device=device, dtype=torch.int64)
            if role_bits.numel() > total:
                role_bits = role_bits[:total]
            elif role_bits.numel() < total:
                role_bits = torch.cat(
                    (
                        role_bits,
                        torch.full(
                            (total - role_bits.numel(),),
                            16,
                            dtype=torch.int64,
                            device=device,
                        ),
                    )
                )
            result.append(role_bits)
        if not result:
            fresh = torch.full((total,), 16, dtype=torch.int64, device=device)
            return fresh, fresh.clone()
        return result[0], result[1]

    def _record_token_bits(
        self,
        layer_idx: int,
        key_bits: torch.Tensor,
        value_bits: torch.Tensor,
    ) -> None:
        if key_bits.ndim != 1 or value_bits.ndim != 1 or key_bits.shape != value_bits.shape:
            raise ValueError(
                "applied key/value bit vectors must be same-length one-dimensional tensors"
            )
        self._layer_token_bits[layer_idx] = (
            key_bits.detach().clone(),
            value_bits.detach().clone(),
        )

    def _refresh_layer_accounting(self, layer_idx: int) -> None:
        """Recompute one layer ledger from its live shape and applied widths."""

        if layer_idx >= len(self.layers):
            self._layer_bits.pop(layer_idx, None)
            self._layer_fp16.pop(layer_idx, None)
            self._layer_token_bits.pop(layer_idx, None)
            return
        layer = self.layers[layer_idx]
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            self._layer_bits.pop(layer_idx, None)
            self._layer_fp16.pop(layer_idx, None)
            self._layer_token_bits.pop(layer_idx, None)
            return
        fp16_bits = (keys.numel() + values.numel()) * 16
        self._layer_fp16[layer_idx] = fp16_bits
        self.fp16_bits_layer = fp16_bits
        if layer_idx in self.skip_layers:
            self._layer_bits[layer_idx] = fp16_bits
            self._layer_token_bits.pop(layer_idx, None)
            return
        applied = self._layer_token_bits.get(layer_idx)
        if applied is None:
            raise RuntimeError(f"missing applied precision ledger for graded layer {layer_idx}")
        key_bits, value_bits = applied
        total = keys.shape[2]
        if key_bits.numel() != total or value_bits.numel() != total:
            raise RuntimeError(
                f"precision ledger length does not match layer {layer_idx} cache length "
                f"({key_bits.numel()}/{value_bits.numel()} vs {total})"
            )
        batch_heads = keys.shape[0] * keys.shape[1]
        per_token_role = batch_heads * keys.shape[-1]
        self._layer_bits[layer_idx] = int(
            key_bits.sum() * per_token_role
            + value_bits.sum() * per_token_role
            + self._meta_bits_for_role(key_bits, batch_heads, "k")
            + self._meta_bits_for_role(value_bits, batch_heads, "v")
            + self._extra_payload_bits(key_bits, value_bits, batch_heads, keys.shape[-1])
        )

    def _refresh_all_accounting(self) -> None:
        for layer_idx in range(len(self.layers)):
            self._refresh_layer_accounting(layer_idx)

    @staticmethod
    def _lowering_work_bits(
        requested: torch.Tensor,
        previous: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return effective widths and a work vector that skips unchanged data."""

        effective = torch.minimum(requested.to(torch.int64), previous.to(torch.int64))
        work = torch.where(effective < previous, effective, torch.full_like(effective, 16))
        return effective, work

    def _grade_layer(self, keys, values, layer_idx):
        total = keys.shape[2]
        requested_k, requested_v = self.token_bits(total, keys.device, layer_idx)
        previous_k, previous_v = self._previous_bits(layer_idx, total, keys.device)
        kbits, kwork = self._lowering_work_bits(requested_k, previous_k)
        vbits, vwork = self._lowering_work_bits(requested_v, previous_v)
        kq = fake_quant_tokens(keys, kwork)
        vq = fake_quant_tokens(values, vwork)
        self._record_token_bits(layer_idx, kbits, vbits)
        batch_heads = keys.shape[0] * keys.shape[1]
        per = batch_heads * keys.shape[-1]
        bits = int(
            kbits.sum() * per
            + vbits.sum() * per
            + self._meta_bits_for_role(kbits, batch_heads, "k")
            + self._meta_bits_for_role(vbits, batch_heads, "v")
            + self._extra_payload_bits(kbits, vbits, batch_heads, keys.shape[-1])
        )
        return kq, vq, bits

    def _should_regrade(self, q_len: int, total: int) -> bool:
        """Return whether this update must regrade the full resident layer.

        Subclasses with an additional storage invariant can force an earlier
        regrade.  For example, a paged cache must regrade when a partial page
        becomes full even if the ordinary decode cadence has not elapsed.
        """

        del total
        return q_len > 1 or (self._since % self.requant_stride == 0)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        out_k, out_v = super().update(key_states, value_states, layer_idx, *args, **kwargs)
        q_len = key_states.shape[2]
        if layer_idx == 0:
            self._since += 1
        # Outlier-skip layers stay in full precision, but are still counted (at
        # fp16) so the reported memory is the honest *whole-cache* figure and is
        # directly comparable to the eviction baseline.
        if layer_idx in self.skip_layers:
            self._refresh_layer_accounting(layer_idx)
            return out_k, out_v
        total = self.layers[layer_idx].keys.shape[2]
        regrade = self._should_regrade(q_len, total)
        if not regrade:
            # The age ladder for older tokens may intentionally wait until the
            # next stride, but newly appended tokens must never remain unquantized
            # or disappear from memory accounting during that interval.
            layer = self.layers[layer_idx]
            total = layer.keys.shape[2]
            kbits, vbits = self.token_bits(total, layer.keys.device, layer_idx)
            k_new_bits, v_new_bits = kbits[-q_len:], vbits[-q_len:]
            kq = fake_quant_tokens(layer.keys[:, :, -q_len:], k_new_bits)
            vq = fake_quant_tokens(layer.values[:, :, -q_len:], v_new_bits)
            layer.keys[:, :, -q_len:] = kq.to(layer.keys.dtype)
            layer.values[:, :, -q_len:] = vq.to(layer.values.dtype)
            previous_k, previous_v = self._previous_bits(
                layer_idx, total - q_len, layer.keys.device
            )
            self._record_token_bits(
                layer_idx,
                torch.cat((previous_k, k_new_bits.to(torch.int64))),
                torch.cat((previous_v, v_new_bits.to(torch.int64))),
            )
            self._refresh_layer_accounting(layer_idx)
            return layer.keys, layer.values
        # Re-grade the materialised layer cache and return the graded tensors so
        # the *current* forward attends to quantized KV (faithful; makes both
        # the prefill/PPL path and the decode path reflect the precision ladder).
        layer = self.layers[layer_idx]
        kq, vq, bits = self._grade_layer(layer.keys, layer.values, layer_idx)
        layer.keys = kq.to(layer.keys.dtype)
        layer.values = vq.to(layer.values.dtype)
        self._refresh_layer_accounting(layer_idx)
        if self._layer_bits[layer_idx] != bits:
            raise RuntimeError("graded cache bit ledger disagrees with quantizer accounting")
        return layer.keys, layer.values

    # ``DynamicCache`` mutates resident shapes in-place for generation helpers.
    # Mirror every such operation in the precision ledger; inherited behavior
    # alone leaves stale memory reports and can reuse widths for removed tokens.
    def crop(self, max_length: int):
        old_lengths = {
            index: int(layer.get_seq_length()) for index, layer in enumerate(self.layers)
        }
        super().crop(max_length)
        for layer_idx, previous in list(self._layer_token_bits.items()):
            old_length = old_lengths.get(layer_idx, previous[0].numel())
            target = old_length - abs(max_length) if max_length < 0 else max_length
            target = max(0, min(old_length, target))
            self._layer_token_bits[layer_idx] = (
                previous[0][:target].clone(),
                previous[1][:target].clone(),
            )
        self._refresh_all_accounting()

    def batch_repeat_interleave(self, repeats: int):
        super().batch_repeat_interleave(repeats)
        self._refresh_all_accounting()

    def batch_select_indices(self, indices: torch.Tensor):
        super().batch_select_indices(indices)
        self._refresh_all_accounting()

    def reorder_cache(self, beam_idx: torch.LongTensor):
        super().reorder_cache(beam_idx)
        self._refresh_all_accounting()

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

    @property
    def fp16_bits(self) -> int:
        """Exact whole-cache fp16 reference bits for formal evaluation.

        ``compression_ratio()`` remains a convenient display metric, but the
        evaluator must not reconstruct memory from a rounded/legacy ratio.
        Exposing the integer layer ledger lets
        :func:`cache_compression_sufficient_statistics` aggregate TaperKV (and
        its block/budget subclasses) exactly across generations.
        """
        return int(sum(self._layer_fp16.values()))

    @property
    def stored_bits(self) -> int:
        """Exact whole-cache stored payload plus quantization metadata bits."""
        return int(sum(self._layer_bits.values()))

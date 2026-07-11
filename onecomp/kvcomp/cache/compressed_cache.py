"""HuggingFace cache integration for KV codecs.

We deliberately avoid KVCache-Factory's global ``replace_llama`` monkeypatching.
Instead, :class:`CompressedDynamicCache` subclasses HF's
``transformers.DynamicCache`` and overrides :meth:`update` to fake-quantize the
*incoming* key/value states before they are stored.  Because the override lives
at the cache level (not the layer level) it works unchanged for full-attention
and sliding-window architectures alike (Llama, Qwen3, Gemma, ...), and the
cache is simply passed to ``model.generate(past_key_values=...)`` -- the model
itself is never patched.

This is the research-layer path: ``encode`` then ``decode`` keeps the
dequantized tensor (simulated quantization), mirroring the weight side's
``compute_dequantized_weight``.  Real low-bit storage / Triton dequant kernels
belong to :mod:`onecomp.kvcomp.kernels`.

Per-token codecs (FP8, INT-token, TurboQuant, NVFP4, Ternary) quantize each
arriving block correctly and incrementally.  Per-channel-key codecs such as
KIVI need block (prefill) granularity and are intended for the offline /
kernel path; ``CompressedDynamicCache`` warns when given one.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from logging import getLogger

import torch
from transformers import DynamicCache

logger = getLogger(__name__)


class CompressedDynamicCache(DynamicCache):
    """``DynamicCache`` that applies a :class:`~onecomp.kvcomp.KVCodec` on update.

    Args:
        codec: The KV codec to apply (``None`` disables compression).
        skip_layers: Layer indices left uncompressed (e.g. first/last layers).
        config: Model config forwarded to ``DynamicCache`` (preserves
            sliding-window layer typing).
    """

    def __init__(self, codec=None, skip_layers=None, config=None, sink_tokens=0, recent_tokens=0, **kwargs):
        super().__init__(config=config, **kwargs)
        self.codec = codec
        self.skip_layers = set(skip_layers or (codec.skip_layers if codec else []))
        # Attention-sink tokens kept in full precision (StreamingLLM / KIVI).
        # Quantizing the first prompt tokens (and outlier layers) corrupts
        # generation on outlier-heavy models (e.g. small Qwen); preserving the
        # sinks is the standard, paper-aligned remedy.
        self.sink_tokens = int(sink_tokens)
        # Block-local trailing tokens kept in full precision.  This mirrors the
        # recent-window protection used by KV quantization serving systems for
        # prefill-sized updates.  Decode-step re-quantization of tokens that
        # leave the recent window is handled by dedicated graded caches.
        self.recent_tokens = int(recent_tokens)
        self._seen = {}  # layer_idx -> tokens seen so far
        self.stored_bits = 0
        self.fp16_bits = 0
        if codec is not None and (codec.key_axis == "channel"):
            logger.warning(
                "Codec '%s' uses per-channel key quantization; incremental cache "
                "updates quantize per arriving token. Use the block/offline path "
                "for faithful per-channel results.",
                codec.name,
            )

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if self.codec is not None and layer_idx not in self.skip_layers:
            seen = self._seen.get(layer_idx, 0)
            q_len = key_states.shape[2]
            if self.recent_tokens > 0 and q_len <= self.recent_tokens and seen >= self.sink_tokens:
                self._count_fp((key_states.numel() + value_states.numel()) * 16)
                out = super().update(key_states, value_states, layer_idx, *args, **kwargs)
                self._seen[layer_idx] = seen + q_len
                self._requantize_retired_recent(layer_idx, q_len)
                return out
            # Indices (within this block) that fall in the sink region stay FP.
            sink_in_block = max(0, min(self.sink_tokens - seen, q_len))
            recent_in_block = 0
            if self.recent_tokens > 0 and q_len > self.recent_tokens:
                recent_in_block = min(self.recent_tokens, q_len - sink_in_block)
            quant_end = q_len - recent_in_block
            kq_in, vq_in = key_states, value_states
            if sink_in_block < quant_end:
                k_quant = key_states[:, :, sink_in_block:quant_end]
                v_quant = value_states[:, :, sink_in_block:quant_end]
                # Fused encode+decode: the fake-quant cache never inspects the
                # payload, so codecs may skip building it entirely (HayateQuant
                # fuses the whole round-trip into one launch-lean pipeline; the
                # base-class default is the plain encode->decode chain).
                key_q, value_q, stored = self.codec.fake_quant_roundtrip(
                    k_quant, v_quant, ctx={"layer_idx": layer_idx})
                fp16q = (k_quant.numel() + v_quant.numel()) * 16
                self.stored_bits += stored if stored else fp16q
                self.fp16_bits += fp16q
                parts_k, parts_v = [], []
                if sink_in_block > 0:
                    parts_k.append(key_states[:, :, :sink_in_block])
                    parts_v.append(value_states[:, :, :sink_in_block])
                    sink_bits = (parts_k[-1].numel() + parts_v[-1].numel()) * 16
                    self._count_fp(sink_bits)
                parts_k.append(key_q.to(key_states.dtype))
                parts_v.append(value_q.to(value_states.dtype))
                if recent_in_block > 0:
                    parts_k.append(key_states[:, :, quant_end:])
                    parts_v.append(value_states[:, :, quant_end:])
                    recent_bits = (parts_k[-1].numel() + parts_v[-1].numel()) * 16
                    self._count_fp(recent_bits)
                kq_in = torch.cat(parts_k, dim=2)
                vq_in = torch.cat(parts_v, dim=2)
            else:
                # entire block falls in the protected region: kept in full precision
                self._count_fp((key_states.numel() + value_states.numel()) * 16)
            key_states, value_states = kq_in, vq_in
            self._seen[layer_idx] = seen + q_len
        elif self.codec is not None:
            # outlier-skip layer kept in full precision; count it in the ratio
            self._count_fp((key_states.numel() + value_states.numel()) * 16)
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)

    def _count_fp(self, bits: int) -> None:
        """Account full-precision (sink / skip-layer) KV equally in stored & fp16."""
        self.stored_bits += bits
        self.fp16_bits += bits

    def _requantize_retired_recent(self, layer_idx: int, q_len: int) -> None:
        """Quantize tokens that just moved out of the FP recent window."""
        if self.codec is None or self.recent_tokens <= 0:
            return
        if layer_idx >= len(self.layers):
            return
        layer = self.layers[layer_idx]
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            return
        total = keys.shape[2]
        retire_end = max(self.sink_tokens, total - self.recent_tokens)
        retire_start = max(self.sink_tokens, retire_end - q_len)
        if retire_end <= retire_start:
            return
        k_old = keys[:, :, retire_start:retire_end]
        v_old = values[:, :, retire_start:retire_end]
        key_q, value_q, stored = self.codec.fake_quant_roundtrip(
            k_old, v_old, ctx={"layer_idx": layer_idx})
        fp16q = (k_old.numel() + v_old.numel()) * 16
        self.stored_bits += (stored if stored else fp16q) - fp16q
        keys[:, :, retire_start:retire_end].copy_(key_q.to(keys.dtype))
        values[:, :, retire_start:retire_end].copy_(value_q.to(values.dtype))

    def compression_ratio(self) -> float:
        """Whole-cache compression ratio (fp16 bits / stored bits) so far.

        Includes the full-precision sink tokens and outlier-skip layers, so the
        figure is an honest whole-cache ratio that is directly comparable to the
        eviction and FullKV baselines (not a quantized-portion-only ratio).
        """
        if self.stored_bits == 0:
            return 1.0
        return self.fp16_bits / self.stored_bits

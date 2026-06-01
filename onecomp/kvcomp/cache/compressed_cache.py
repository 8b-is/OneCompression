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

    def __init__(self, codec=None, skip_layers=None, config=None, sink_tokens=0, **kwargs):
        super().__init__(config=config, **kwargs)
        self.codec = codec
        self.skip_layers = set(skip_layers or (codec.skip_layers if codec else []))
        # Attention-sink tokens kept in full precision (StreamingLLM / KIVI).
        # Quantizing the first prompt tokens (and outlier layers) corrupts
        # generation on outlier-heavy models (e.g. small Qwen); preserving the
        # sinks is the standard, paper-aligned remedy.
        self.sink_tokens = int(sink_tokens)
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
            # Indices (within this block) that fall in the sink region stay FP.
            sink_in_block = max(0, min(self.sink_tokens - seen, q_len))
            kq_in, vq_in = key_states, value_states
            if sink_in_block < q_len:
                k_quant = key_states[:, :, sink_in_block:]
                v_quant = value_states[:, :, sink_in_block:]
                compressed = self.codec.encode(k_quant, v_quant)
                key_q, value_q = self.codec.decode(compressed)
                stored = compressed.numel_bits()
                fp16q = (k_quant.numel() + v_quant.numel()) * 16
                self.stored_bits += stored if stored else fp16q
                self.fp16_bits += fp16q
                if sink_in_block > 0:
                    # sink tokens kept in FP: counted equally in stored and fp16
                    sink_bits = (key_states[:, :, :sink_in_block].numel()
                                 + value_states[:, :, :sink_in_block].numel()) * 16
                    self._count_fp(sink_bits)
                    kq_in = torch.cat([key_states[:, :, :sink_in_block], key_q.to(key_states.dtype)], dim=2)
                    vq_in = torch.cat([value_states[:, :, :sink_in_block], value_q.to(value_states.dtype)], dim=2)
                else:
                    kq_in = key_q.to(key_states.dtype)
                    vq_in = value_q.to(value_states.dtype)
            else:
                # entire block falls in the sink region: kept in full precision
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

    def compression_ratio(self) -> float:
        """Whole-cache compression ratio (fp16 bits / stored bits) so far.

        Includes the full-precision sink tokens and outlier-skip layers, so the
        figure is an honest whole-cache ratio that is directly comparable to the
        eviction and FullKV baselines (not a quantized-portion-only ratio).
        """
        if self.stored_bits == 0:
            return 1.0
        return self.fp16_bits / self.stored_bits

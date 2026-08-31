"""HuggingFace cache integration for KV codecs.

We deliberately avoid KVCache-Factory's global ``replace_llama`` monkeypatching.
Instead, :class:`CompressedDynamicCache` subclasses HF's
``transformers.DynamicCache`` and overrides :meth:`update` to fake-quantize the
*incoming* key/value states before they are stored.  The cache is simply passed
to ``model.generate(past_key_values=...)`` -- the model itself is never patched.
Full-attention layers are supported. A genuinely truncating model-side sliding
layer can only be kept full precision via an explicit/automatic skip; attempting
to quantize it fails closed at construction. Aggregate codec payload bits cannot
be decremented exactly when a suffix eviction removes only part of an encoded
block (which may share scale/mask metadata), while the skipped layer's resident
fp16 tensor is accounted exactly by the underlying sliding cache.

This is the research-layer path: ``encode`` then ``decode`` keeps the
dequantized tensor (simulated quantization), mirroring the weight side's
``compute_dequantized_weight``.  Real low-bit storage / Triton dequant kernels
belong to :mod:`onecomp.kvcomp.kernels`.

Per-token codecs (FP8, INT-token, TurboQuant, HayateQuant) quantize each
arriving block incrementally. Codecs with a per-channel axis on either role
(KIVI, NVFP4, Ternary, or a custom value-channel affine codec) buffer only the
unfinished sequence group in full precision and quantize each retired group
once, preserving the requested axis without a hidden full-history FP shadow.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import copy
import numbers
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
        skip_sliding_layers: Automatically keep every model-side sliding layer
            in full precision. Without this opt-in, all such layers must be
            listed explicitly in ``skip_layers`` or construction fails closed.
    """

    def __init__(
        self,
        codec=None,
        skip_layers=None,
        config=None,
        sink_tokens=0,
        recent_tokens=0,
        skip_sliding_layers=False,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.codec = codec
        if not isinstance(skip_sliding_layers, bool):
            raise ValueError("skip_sliding_layers must be a bool")
        sliding_layers = {
            index
            for index, layer in enumerate(self.layers)
            if getattr(layer, "sliding_window", None) is not None
        }
        raw_skip_layers = (
            skip_layers
            if skip_layers is not None
            else (codec.skip_layers if codec is not None else [])
        )
        self.skip_layers = set()
        for raw_layer_idx in raw_skip_layers:
            if isinstance(raw_layer_idx, bool) or not isinstance(raw_layer_idx, numbers.Integral):
                raise ValueError("skip_layers must contain integer layer indices")
            layer_idx = int(raw_layer_idx)
            if layer_idx < 0:
                # Serialized configs and the public docs allow Python-style
                # boundary indices such as -1. Cache updates, however, always
                # receive non-negative layer indices. Resolve against the
                # concrete cache layer list instead of silently missing the
                # requested protection. A lazy cache has no safe way to know
                # which future layer will be last, so fail closed there.
                if not self.layers:
                    raise ValueError(
                        "negative skip_layers require a model config (or initialized "
                        "cache data) so the number of cache layers is known"
                    )
                layer_idx += len(self.layers)
                if layer_idx < 0:
                    raise ValueError(
                        f"skip layer index {raw_layer_idx} is out of range for "
                        f"{len(self.layers)} cache layers"
                    )
            if self.layers and layer_idx >= len(self.layers):
                raise ValueError(
                    f"skip layer index {raw_layer_idx} is out of range for "
                    f"{len(self.layers)} cache layers"
                )
            self.skip_layers.add(layer_idx)
        self.sliding_window_layers = set(sliding_layers)
        self.auto_skipped_sliding_layers = (
            set(sliding_layers) if self.codec is not None and skip_sliding_layers else set()
        )
        self.skip_layers.update(self.auto_skipped_sliding_layers)
        unskipped_sliding = self.sliding_window_layers - self.skip_layers
        if self.codec is not None and unskipped_sliding:
            raise NotImplementedError(
                "CompressedDynamicCache cannot quantize model-side sliding-window "
                f"layers {sorted(unskipped_sliding)}: exact logical data-plane stored_bits "
                "for the resident token set cannot split shared scale/mask fields "
                "when old tokens are evicted. Include every sliding layer in skip_layers "
                "(full-precision, exactly accounted), pass skip_sliding_layers=True, "
                "or use a packed backend with resident-payload slicing."
            )
        # Attention-sink tokens kept in full precision (StreamingLLM / KIVI).
        # Quantizing the first prompt tokens (and outlier layers) corrupts
        # generation on outlier-heavy models (e.g. small Qwen); preserving the
        # sinks is the standard, paper-aligned remedy.
        if (
            isinstance(sink_tokens, bool)
            or not isinstance(sink_tokens, numbers.Integral)
            or sink_tokens < 0
        ):
            raise ValueError(f"sink_tokens must be a non-negative integer, got {sink_tokens!r}")
        self.sink_tokens = int(sink_tokens)
        # Block-local trailing tokens kept in full precision.  This mirrors the
        # recent-window protection used by KV quantization serving systems for
        # prefill-sized updates.  Decode-step re-quantization of tokens that
        # leave the recent window is handled by dedicated graded caches.
        if (
            isinstance(recent_tokens, bool)
            or not isinstance(recent_tokens, numbers.Integral)
            or recent_tokens < 0
        ):
            raise ValueError(
                f"recent_tokens must be a non-negative integer, got {recent_tokens!r}"
            )
        self.recent_tokens = int(recent_tokens)
        self._seen = {}  # layer_idx -> tokens seen so far
        self.stored_bits = 0
        self.fp16_bits = 0
        self._channel_quantized_until = {}
        # Mutable online codec state belongs to this cache instance, never to
        # the codec object (the same configured codec is commonly reused by a
        # cache factory).  Each layer gets an isolated namespace passed through
        # ``ctx``; codecs that are entirely static simply ignore it.
        self._codec_runtime_state = {}

    def _codec_context(self, layer_idx: int) -> dict:
        return {
            "layer_idx": int(layer_idx),
            "cache_state": self._codec_runtime_state.setdefault(int(layer_idx), {}),
        }

    def _fake_quant_roundtrip(self, key, value, layer_idx: int):
        """Run the codec and validate the cache-facing round-trip contract."""
        result = self.codec.fake_quant_roundtrip(key, value, ctx=self._codec_context(layer_idx))
        if not isinstance(result, (tuple, list)) or len(result) != 3:
            raise TypeError("KVCodec.fake_quant_roundtrip() must return (key, value, stored_bits)")
        key_q, value_q, stored_bits = result
        for role, decoded, source in (
            ("key", key_q, key),
            ("value", value_q, value),
        ):
            if not isinstance(decoded, torch.Tensor):
                raise TypeError(f"decoded {role} must be a torch.Tensor")
            if decoded.shape != source.shape:
                raise RuntimeError(
                    f"decoded {role} shape {tuple(decoded.shape)} does not match "
                    f"input shape {tuple(source.shape)}"
                )
            if decoded.device != source.device:
                raise RuntimeError(
                    f"decoded {role} device {decoded.device} does not match "
                    f"input device {source.device}"
                )
            if not torch.is_floating_point(decoded):
                raise TypeError(f"decoded {role} must have a floating-point dtype")
        if stored_bits is not None and (
            isinstance(stored_bits, bool) or not isinstance(stored_bits, int) or stored_bits < 0
        ):
            raise ValueError("codec stored_bits must be a non-negative integer or None")
        return key_q, value_q, stored_bits

    def reset(self):
        """Reset tensors, accounting, and every cache-owned codec statistic."""
        super().reset()
        self._seen.clear()
        self.stored_bits = 0
        self.fp16_bits = 0
        self._channel_quantized_until.clear()
        self._codec_runtime_state.clear()

    def get_codec_runtime_provenance(self) -> dict:
        """Return a detached, JSON-serialisable snapshot of online state."""
        callback = getattr(self.codec, "cache_runtime_provenance", None)
        runtime = callback(self._codec_runtime_state) if callable(callback) else {}
        return {
            "schema_version": 1,
            "codec": getattr(self.codec, "name", None),
            "cache_layout": {
                "sliding_window_layers": sorted(self.sliding_window_layers),
                "auto_skipped_sliding_layers": sorted(self.auto_skipped_sliding_layers),
                "effective_skip_layers": sorted(self.skip_layers),
            },
            "runtime": copy.deepcopy(runtime),
        }

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        # Defensive guard for callers that replace/append cache layers after
        # construction. The normal configured path is rejected in ``__init__``
        # before any model layer can mutate the cache.
        if (
            self.codec is not None
            and layer_idx not in self.skip_layers
            and layer_idx < len(self.layers)
            and getattr(self.layers[layer_idx], "sliding_window", None) is not None
        ):
            raise NotImplementedError(
                "CompressedDynamicCache cannot report exact logical data-plane "
                "stored_bits for the resident token set "
                "for a model-side sliding-window layer"
            )
        skipped_sliding = bool(
            self.codec is not None
            and layer_idx in self.skip_layers
            and layer_idx < len(self.layers)
            and getattr(self.layers[layer_idx], "sliding_window", None) is not None
        )
        old_sliding_fp_bits = 0
        if skipped_sliding:
            layer = self.layers[layer_idx]
            old_keys = getattr(layer, "keys", None)
            old_values = getattr(layer, "values", None)
            old_sliding_fp_bits = (
                (old_keys.numel() if old_keys is not None else 0)
                + (old_values.numel() if old_values is not None else 0)
            ) * 16
        if (
            self.codec is not None
            and not skipped_sliding
            and "channel"
            in (
                getattr(self.codec, "key_axis", "token"),
                getattr(self.codec, "value_axis", "token"),
            )
        ):
            return self._update_channel_codec(key_states, value_states, layer_idx, *args, **kwargs)
        if self.codec is not None and layer_idx not in self.skip_layers:
            seen = self._seen.get(layer_idx, 0)
            q_len = key_states.shape[2]
            if self.recent_tokens > 0 and q_len <= self.recent_tokens and seen >= self.sink_tokens:
                self._count_fp((key_states.numel() + value_states.numel()) * 16)
                out = super().update(key_states, value_states, layer_idx, *args, **kwargs)
                self._seen[layer_idx] = seen + q_len
                self._requantize_retired_recent(layer_idx, seen)
                return out
            # Indices (within this block) that fall in the sink region stay FP.
            sink_in_block = max(0, min(self.sink_tokens - seen, q_len))
            recent_in_block = 0
            if self.recent_tokens > 0:
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
                key_q, value_q, stored = self._fake_quant_roundtrip(k_quant, v_quant, layer_idx)
                fp16q = (k_quant.numel() + v_quant.numel()) * 16
                self.stored_bits += stored if stored is not None else fp16q
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
                # The common decode path has no protected prefix/suffix, so
                # each list contains only the freshly quantized tensor.  A
                # one-element ``cat`` still allocates and copies the full KV
                # block; retain concatenation only when sink/recent pieces
                # actually have to be joined.
                kq_in = parts_k[0] if len(parts_k) == 1 else torch.cat(parts_k, dim=2)
                vq_in = parts_v[0] if len(parts_v) == 1 else torch.cat(parts_v, dim=2)
            else:
                # entire block falls in the protected region: kept in full precision
                self._count_fp((key_states.numel() + value_states.numel()) * 16)
            key_states, value_states = kq_in, vq_in
            self._seen[layer_idx] = seen + q_len
        elif self.codec is not None:
            # outlier-skip layer kept in full precision; count it in the ratio
            if not skipped_sliding:
                self._count_fp((key_states.numel() + value_states.numel()) * 16)
        out = super().update(key_states, value_states, layer_idx, *args, **kwargs)
        if skipped_sliding:
            # The layer is full precision, so unlike a packed quantized block
            # its exact resident cost is directly recoverable after HF applies
            # the model's sliding-window eviction. Track the resident delta,
            # not cumulative arrivals.
            layer = self.layers[layer_idx]
            new_keys = getattr(layer, "keys", None)
            new_values = getattr(layer, "values", None)
            new_sliding_fp_bits = (
                (new_keys.numel() if new_keys is not None else 0)
                + (new_values.numel() if new_values is not None else 0)
            ) * 16
            self._count_fp(new_sliding_fp_bits - old_sliding_fp_bits)
        if self.codec is not None and layer_idx not in self.skip_layers:
            self._requantize_retired_recent(layer_idx, seen)
        return out

    def teacher_forced_temporal_dependency(self):
        """Return the codec's cross-token dependency for causal evaluators.

        its class name in each benchmark.  It keeps teacher-forced PPL and
        continuation likelihood aligned with the codec actually installed in
        the cache.
        """
        if self.codec is None:
            return None
        dependency = getattr(self.codec, "teacher_forced_temporal_dependency", None)
        if not callable(dependency):
            raise TypeError(
                "CompressedDynamicCache codec must expose " "teacher_forced_temporal_dependency()"
            )
        codec_dependency = dependency()
        if codec_dependency is not None:
            return codec_dependency
        # A trailing FP window is a function of *current* sequence length. In a
        # full teacher-forced update only the final N tokens stay FP, so an
        # early logit sees its token quantized according to future arrivals.
        # Causal incremental scoring lets each token retire only after N later
        # tokens have actually arrived. Formal HQ/TQ use recent_tokens=0.
        if self.recent_tokens > 0:
            return "recent_window_status_changes_over_update_tokens"
        return None

    def _update_channel_codec(self, key_states, value_states, layer_idx, *args, **kwargs):
        """Faithful block update when either K or V uses a per-channel axis.

        A one-token per-channel affine group has min=max and reconstructs exactly.
        Accumulate an FP residual until a complete sequence-axis group retires,
        then quantize that group exactly once. This matches KIVI's streaming
        layout and avoids keeping an unreported full-history FP shadow.
        """
        # A skipped layer never invokes the codec, so it does not require a
        # finite channel group. Check this before validating codec-only state;
        # otherwise an intentionally full-precision boundary layer can fail on
        # an option that is irrelevant to that layer.
        if layer_idx in self.skip_layers:
            out = super().update(key_states, value_states, layer_idx, *args, **kwargs)
            self._count_fp((key_states.numel() + value_states.numel()) * 16)
            self._seen[layer_idx] = self._seen.get(layer_idx, 0) + key_states.shape[2]
            return out

        group_size = int(getattr(self.codec, "group_size", -1))
        if group_size <= 0:
            raise ValueError(
                "incremental per-channel KV quantization requires group_size>0; "
                "an unbounded sequence-axis group cannot be finalized online"
            )

        out = super().update(key_states, value_states, layer_idx, *args, **kwargs)
        layer = self.layers[layer_idx]
        appended_fp_bits = (key_states.numel() + value_states.numel()) * 16
        self._count_fp(appended_fp_bits)
        self._seen[layer_idx] = self._seen.get(layer_idx, 0) + key_states.shape[2]
        total = int(layer.keys.shape[2])
        # A sliding-window DynamicLayer may discard positions. The KIVI block
        # cursor is defined over a growing full cache, so fail rather than
        # silently applying global offsets to a truncated local tensor.
        if total != self._seen[layer_idx]:
            raise NotImplementedError(
                "per-channel compressed cache does not yet support a model-side "
                "sliding-window DynamicLayer"
            )
        sink_end = min(self.sink_tokens, total)
        start = max(sink_end, self._channel_quantized_until.get(layer_idx, sink_end))
        retire_before = max(sink_end, total - self.recent_tokens)
        n_complete = max(0, retire_before - start) // group_size
        end = start + n_complete * group_size
        if end > start:
            raw_k = layer.keys[:, :, start:end]
            raw_v = layer.values[:, :, start:end]
            kq, vq, quant_bits = self._fake_quant_roundtrip(raw_k, raw_v, layer_idx)
            fp_bits = (raw_k.numel() + raw_v.numel()) * 16
            self.stored_bits += (quant_bits if quant_bits is not None else fp_bits) - fp_bits
            layer.keys[:, :, start:end].copy_(kq.to(layer.keys.dtype))
            layer.values[:, :, start:end].copy_(vq.to(layer.values.dtype))
            self._channel_quantized_until[layer_idx] = end
        else:
            self._channel_quantized_until.setdefault(layer_idx, start)
        return layer.keys, layer.values

    def _count_fp(self, bits: int) -> None:
        """Account full-precision (sink / skip-layer) KV equally in stored & fp16."""
        self.stored_bits += bits
        self.fp16_bits += bits

    def _requantize_retired_recent(self, layer_idx: int, old_total: int) -> None:
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
        # Only re-quantize tokens from the *previous* cache that used to be in
        # its FP recent window.  A large incoming chunk already quantized its
        # own non-recent prefix above; including it again would double-quantize
        # data and subtract FP accounting for bits that were never FP.
        retire_start = max(self.sink_tokens, old_total - self.recent_tokens)
        retire_end = min(old_total, max(self.sink_tokens, total - self.recent_tokens))
        if retire_end <= retire_start:
            return
        k_old = keys[:, :, retire_start:retire_end]
        v_old = values[:, :, retire_start:retire_end]
        key_q, value_q, stored = self._fake_quant_roundtrip(k_old, v_old, layer_idx)
        fp16q = (k_old.numel() + v_old.numel()) * 16
        self.stored_bits += (stored if stored is not None else fp16q) - fp16q
        keys[:, :, retire_start:retire_end].copy_(key_q.to(keys.dtype))
        values[:, :, retire_start:retire_end].copy_(value_q.to(values.dtype))

    def _count_fp(self, bits: int) -> None:
        """Account full-precision (sink / skip-layer) KV equally in stored & fp16."""
        self.stored_bits += bits
        self.fp16_bits += bits

    def compression_ratio(self) -> float:
        """Logical whole-cache data-plane ratio (fp16 bits / stored bits).

        Includes the full-precision sink tokens and outlier-skip layers, so the
        denominator covers the full logical KV data plane rather than only the
        quantized portion.  It excludes Python/container bytes and persistent
        codec state and therefore is not a physical resident-memory ratio.
        """
        if self.fp16_bits == 0:
            return 1.0
        if self.stored_bits < 0:
            raise RuntimeError("compressed cache stored_bits became negative")
        if self.stored_bits == 0:
            return float("inf")
        return self.fp16_bits / self.stored_bits

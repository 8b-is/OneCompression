"""Live ThinKV KV cache for HuggingFace ``model.generate``.

This applies the thought-adaptive **TBQ** (Think-Before-you-Quantize) scheme
*online* during autoregressive decoding:

* prompt tokens are treated as a single Reasoning block and group-quantized,
* each decode token is buffered in full precision and, once a group of
  ``group_size`` same-thought tokens accumulates, group-quantized per the
  current thought (R/E -> 4-bit NVFP4, T -> 2-bit ternary; the formal scheme
  also allows R -> 8-bit FP8),
* the most-recent ``< group_size`` tokens stay in full precision (the residual
  window the paper keeps un-quantized for the active group).

Thought classification uses attention **sparsity** (the paper's
``softmax(qK^T)`` signal).  HF's cache ``update`` does not see attention
probabilities, so :func:`thinkv_attention_capture` registers reversible forward
hooks on the attention modules that record per-layer sparsity; the cache
averages it over the calibrated optimal layers ``L*`` every ``refresh_interval``
steps and shares one thought across all layers, per the generation loop.

Fidelity / scope
----------------
TBQ is **length-preserving** and therefore RoPE/causal-mask-safe under vanilla
HF generate, so it faithfully reproduces ThinKV's *quantization* on the paper's
reasoning benchmarks.  Decode-time **TBE eviction** physically shrinks the
cache, which breaks HF's position/mask bookkeeping (precisely why the paper
implemented it as a vLLM paged-attention backend, PR #16160); eviction is
therefore handled by the engine for offline verification and by the vLLM plugin
for serving, not in this HF generate path.

This is a tensor-based, O(n)-per-step implementation (matching the standard
``DynamicCache`` cost); the list-based :class:`ThinKVEngine` remains the
reference for unit tests and injected-K/V verification.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import contextlib
import math
from functools import partial
from logging import getLogger
from typing import List, Optional

import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer

from ..codec.fp8 import FP8Codec
from ..codec.lowbit_fp import NVFP4Codec, TernaryCodec
from ..recipe.thinkv import (
    ThinKVPolicy,
    ThoughtClassifier,
    attention_sparsity,
    T_REASONING,
    _ID_TO_NAME,
)

logger = getLogger(__name__)


class _LayerTBQ:
    """Per-layer tensor state for the TBQ buffer + committed quantized cache."""

    __slots__ = (
        "sink_k",
        "sink_v",
        "committed_k",
        "committed_v",
        "committed_bits",
        "buf_k",
        "buf_v",
        "buf_thought",
        "heads",
        "dim",
        "n_committed",
        "skip",
    )

    def __init__(self):
        self.sink_k = None  # (heads, ns, dim) full-precision attention sinks
        self.sink_v = None
        self.committed_k = None  # (heads, nc, dim)
        self.committed_v = None
        self.committed_bits = 0.0
        self.buf_k = None  # (heads, nb, dim) full precision
        self.buf_v = None
        self.buf_thought = None
        self.heads = self.dim = 0
        self.n_committed = 0
        self.skip = False  # layer left fully un-quantized (outlier layer)

    @property
    def n_sink(self) -> int:
        return 0 if self.sink_k is None else self.sink_k.shape[1]

    @property
    def n_buffer(self) -> int:
        return 0 if self.buf_k is None else self.buf_k.shape[1]

    @property
    def total(self) -> int:
        return self.n_sink + self.n_committed + self.n_buffer


class ThinKVCache(DynamicCache):
    """``DynamicCache`` applying live ThinKV TBQ quantization.

    Args:
        policy: A :class:`ThinKVPolicy` (thresholds, precision, group_size,
            refresh_interval, L*).
        config: Model config (forwarded for layer typing).
    """

    def __init__(
        self,
        policy: Optional[ThinKVPolicy] = None,
        config=None,
        sink_tokens: int = 4,
        skip_layers=None,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        sliding_layers = [
            index for index, layer in enumerate(self.layers) if getattr(layer, "is_sliding", False)
        ]
        if sliding_layers:
            # ThinKV materializes its own complete tensor and assigns it to the
            # HF layer directly, bypassing DynamicSlidingWindowLayer's
            # cumulative-length/offset lifecycle.  Pretending that combination
            # works changes the model's attention semantics.
            raise NotImplementedError(
                "ThinKVCache does not support model-side sliding-window "
                f"layers {sliding_layers}; use a full-attention/native backend"
            )
        self.policy = policy or ThinKVPolicy()
        if isinstance(sink_tokens, bool) or not isinstance(sink_tokens, int) or sink_tokens < 0:
            raise ValueError("sink_tokens must be a non-negative integer")
        self.sink_tokens = sink_tokens
        self.skip_layers = set()
        raw_skip_layers = (
            skip_layers
            if skip_layers is not None
            else getattr(self.policy, "skip_layers", None) or []
        )
        for raw_layer_idx in raw_skip_layers:
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
        self.classifier = ThoughtClassifier(thresholds=tuple(self.policy.thresholds))
        self.calib_layers = list(self.policy.calib_layers) if self.policy.calib_layers else None
        self._codecs = {
            2: TernaryCodec(group_size=self.policy.group_size),
            4: NVFP4Codec(group_size=self.policy.group_size),
            8: FP8Codec(),
        }
        self._states: List[_LayerTBQ] = []
        self._pending_sparsity: dict = {}
        self._step = 0
        self._step_thought = T_REASONING
        self._first_layer: Optional[int] = None

    def reset(self):
        """Reset HF tensors and every live TBQ/classifier state variable."""
        super().reset()
        self._states.clear()
        self._pending_sparsity.clear()
        self._step = 0
        self._step_thought = T_REASONING
        self._first_layer = None
        for warning_flag in ("_warned_batch", "_warned_no_attn_weights"):
            if hasattr(self, warning_flag):
                delattr(self, warning_flag)

    # --- helpers -----------------------------------------------------------
    def _codec_for(self, thought: int):
        bits = self.policy.precision.get(_ID_TO_NAME[thought], 4)
        return (
            self._codecs[2] if bits <= 2 else (self._codecs[4] if bits <= 4 else self._codecs[8])
        )

    def teacher_forced_temporal_dependency(self):
        """TBQ prefill groups fit scales across the incoming token block."""
        return "thin_kv_group_statistics_over_update_tokens"

    def _ensure_state(self, layer_idx: int) -> _LayerTBQ:
        while len(self._states) <= layer_idx:
            self._states.append(_LayerTBQ())
        return self._states[layer_idx]

    def _quant_block(self, k_block, v_block, thought):
        """Fake-quantize a ``(heads, n, dim)`` block under ``thought``'s codec."""
        codec = self._codec_for(thought)
        encoded = codec.encode(k_block[None], v_block[None])
        kq, vq = codec.decode(encoded)
        return kq[0], vq[0], int(encoded.numel_bits())

    def _commit_buffer(self, st: _LayerTBQ):
        if st.buf_k is None:
            return
        n = st.buf_k.shape[1]
        kq, vq, stored_bits = self._quant_block(st.buf_k, st.buf_v, st.buf_thought)
        st.committed_k = kq if st.committed_k is None else torch.cat([st.committed_k, kq], dim=1)
        st.committed_v = vq if st.committed_v is None else torch.cat([st.committed_v, vq], dim=1)
        st.committed_bits += stored_bits
        st.n_committed += n
        st.buf_k = st.buf_v = None
        st.buf_thought = None

    # --- sparsity feed -----------------------------------------------------
    def feed_sparsity(self, layer_idx: int, sparsity: float):
        if isinstance(layer_idx, bool) or not isinstance(layer_idx, int) or layer_idx < 0:
            raise ValueError("ThinKV sparsity layer_idx must be a non-negative integer")
        try:
            sparsity = float(sparsity)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("ThinKV sparsity must be a finite value in [0, 1]") from exc
        if not math.isfinite(sparsity) or not 0.0 <= sparsity <= 1.0:
            raise ValueError("ThinKV sparsity must be a finite value in [0, 1]")
        self._pending_sparsity[layer_idx] = sparsity

    def _decide_thought(self) -> int:
        if not self._pending_sparsity:
            return self._step_thought
        layers = (
            self.calib_layers or sorted(self._pending_sparsity)[: self.policy.max_calib_layers]
        )
        vals = [
            self._pending_sparsity[layer_idx]
            for layer_idx in layers
            if layer_idx in self._pending_sparsity
        ]
        if not vals:
            return self._step_thought
        return self.classifier.classify(sum(vals) / len(vals))

    # --- HF Cache API ------------------------------------------------------
    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if key_states.shape[0] != 1:
            raise NotImplementedError(
                "ThinKVCache is single-sequence; batch size >1 cannot silently "
                "fall back to an uncompressed DynamicCache"
            )

        while len(self.layers) <= layer_idx:
            self.layers.append(DynamicLayer())
        st = self._ensure_state(layer_idx)
        if self._first_layer is None:
            self._first_layer = layer_idx

        q_len = key_states.shape[2]
        k0 = key_states[0]  # (heads, q_len, dim)
        v0 = value_states[0]
        st.heads, st.dim = k0.shape[0], k0.shape[-1]

        if layer_idx in self.skip_layers:
            st.skip = True
            st.committed_k = (
                k0 if st.committed_k is None else torch.cat([st.committed_k, k0], dim=1)
            )
            st.committed_v = (
                v0 if st.committed_v is None else torch.cat([st.committed_v, v0], dim=1)
            )
            st.committed_bits += q_len * st.heads * st.dim * 2 * 16.0
            st.n_committed += q_len
        elif q_len > 1 or st.total == 0:
            # Prefill: keep sink tokens FP, group-quantize the rest as Reasoning.
            ns = max(0, min(self.sink_tokens - st.n_sink, q_len))
            if ns > 0:
                sk, sv = k0[:, :ns], v0[:, :ns]
                st.sink_k = sk if st.sink_k is None else torch.cat([st.sink_k, sk], dim=1)
                st.sink_v = sv if st.sink_v is None else torch.cat([st.sink_v, sv], dim=1)
            if ns < q_len:
                kq, vq, stored_bits = self._quant_block(k0[:, ns:], v0[:, ns:], T_REASONING)
                st.committed_k = (
                    kq if st.committed_k is None else torch.cat([st.committed_k, kq], dim=1)
                )
                st.committed_v = (
                    vq if st.committed_v is None else torch.cat([st.committed_v, vq], dim=1)
                )
                st.committed_bits += stored_bits
                st.n_committed += q_len - ns
        else:
            if layer_idx == self._first_layer:
                self._step += 1
                if self._step % self.policy.refresh_interval == 0:
                    self._step_thought = self._decide_thought()
            thought = self._step_thought
            if st.buf_thought is not None and thought != st.buf_thought:
                self._commit_buffer(st)  # never mix thoughts within a group
            if st.buf_k is None:
                st.buf_k, st.buf_v, st.buf_thought = k0.clone(), v0.clone(), thought
            else:
                st.buf_k = torch.cat([st.buf_k, k0], dim=1)
                st.buf_v = torch.cat([st.buf_v, v0], dim=1)
            if st.buf_k.shape[1] >= self.policy.group_size:
                self._commit_buffer(st)

        keys = (
            self._materialize(st.sink_k, st.committed_k, st.buf_k)
            .unsqueeze(0)
            .to(key_states.dtype)
        )
        values = (
            self._materialize(st.sink_v, st.committed_v, st.buf_v)
            .unsqueeze(0)
            .to(value_states.dtype)
        )
        layer = self.layers[layer_idx]
        if not layer.is_initialized:
            layer.lazy_initialization(key_states, value_states)
        layer.keys = keys
        layer.values = values
        return keys, values

    @staticmethod
    def _materialize(*parts):
        tensors = [p for p in parts if p is not None]
        if not tensors:
            return None
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=1)

    # --- reporting ---------------------------------------------------------
    def compression_ratio(self) -> float:
        stored = fp16 = 0.0
        for st in self._states:
            if st.total == 0:
                continue
            per_vec = st.heads * st.dim
            fp_tokens = st.n_buffer + st.n_sink + (st.n_committed if st.skip else 0)
            committed_bits = 0.0 if st.skip else st.committed_bits
            stored += committed_bits + fp_tokens * per_vec * 2 * 16.0
            fp16 += st.total * per_vec * 2 * 16.0
        return fp16 / stored if stored else 1.0

    def avg_bits(self) -> float:
        num = den = 0.0
        for st in self._states:
            per_vec = st.heads * st.dim * 2
            fp_tokens = st.n_buffer + st.n_sink + (st.n_committed if st.skip else 0)
            committed_bits = 0.0 if st.skip else st.committed_bits
            num += committed_bits + fp_tokens * per_vec * 16.0
            den += st.total * per_vec
        return num / den if den else 16.0


def _attn_hook(cache: ThinKVCache, layer_idx: int, rel_threshold: float, module, inputs, output):
    """Forward hook: read attn_weights (eager + output_attentions) -> sparsity."""
    attn_weights = None
    if isinstance(output, tuple):
        for o in output[1:]:
            if torch.is_tensor(o) and o.dim() == 4:
                attn_weights = o
                break
    if attn_weights is None:
        # Attention weights are only materialised when the model is called with
        # ``output_attentions=True`` and ``attn_implementation="eager"``.  If
        # either condition is missing (e.g. sdpa/flash or attn_capture=False),
        # the hook silently returns and thought classification stays disabled.
        # Warn once per cache instance so the user knows why ThinKV falls back
        # to uniform quantisation.
        raise RuntimeError(
            "ThinKV thought classification requested but no attention-weight "
            f"tensor was returned by layer {layer_idx}; load the model with "
            "attn_implementation='eager' and output_attentions=True"
        )
    last = attn_weights[:, :, -1, :]  # (batch, heads, kv_len)
    sp = attention_sparsity(last.float(), rel_threshold).mean()
    cache.feed_sparsity(layer_idx, float(sp))


def _find_decoder_layers(model):
    """Locate the decoder ``layers`` ModuleList across common HF layouts.

    Handles ``model.model.layers`` (most CausalLMs), bare ``model.layers``, and
    multimodal nestings like ``model.model.language_model.layers`` /
    ``model.language_model.model.layers``.  Returns ``None`` if not found (caller
    treats capture as a no-op rather than crashing).
    """
    seen = set()
    stack = [model]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        layers = getattr(obj, "layers", None)
        if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
            return layers
        for attr in ("model", "language_model", "transformer", "text_model"):
            child = getattr(obj, attr, None)
            if child is not None:
                stack.append(child)
    return None


def _iter_attn_modules(model):
    layers = _find_decoder_layers(model)
    if layers is None:
        return
    for i, layer in enumerate(layers):
        yield i, getattr(layer, "self_attn", layer)


@contextlib.contextmanager
def thinkv_attention_capture(model, cache: ThinKVCache, rel_threshold: float = 0.01):
    """Register reversible hooks feeding attention sparsity into ``cache``.

    Requires ``attn_implementation="eager"`` and
    ``model.generate(..., output_attentions=True)`` so attention modules
    materialize ``attn_weights``.
    """
    if (
        isinstance(rel_threshold, bool)
        or not isinstance(rel_threshold, (int, float))
        or not math.isfinite(float(rel_threshold))
        or not 0.0 <= float(rel_threshold) <= 1.0
    ):
        raise ValueError("rel_threshold must be a finite value in [0, 1]")
    modules = list(_iter_attn_modules(model))
    if not modules:
        raise RuntimeError("ThinKV attention capture found no decoder attention modules")
    handles = [
        attn.register_forward_hook(partial(_attn_hook, cache, i, rel_threshold))
        for i, attn in modules
    ]
    try:
        yield cache
    finally:
        for h in handles:
            h.remove()

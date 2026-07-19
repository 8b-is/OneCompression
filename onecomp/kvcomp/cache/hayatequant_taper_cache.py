"""Hugging Face reference caches composing HayateQuant with TaperKV.

The composition is deliberately sequential and unambiguous: each arriving KV
vector is reconstructed by HayateQuant exactly once, then the resident cache is
managed by TaperKV's age-dependent affine schedule.  This mirrors the vLLM
semantic backend; it is not a packed representation containing both payloads.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from .block_taper_cache import BlockTaperKVCache
from .taper_cache import TaperKVCache


class _HayateQuantTaperMixin:
    """Apply the HayateQuant prefilter once before TaperKV stores an update."""

    def __init__(self, policy, codec, *args, **kwargs):
        if getattr(codec, "name", None) != "hayatequant":
            raise ValueError("HayateQuant-TaperKV composition requires a HayateQuant codec")
        codec_skip = list(getattr(codec, "skip_layers", ()) or ())
        combined_skip = list(dict.fromkeys([*policy.skip_layers, *codec_skip]))
        combined_policy = replace(policy, skip_layers=combined_skip)
        super().__init__(combined_policy, *args, **kwargs)
        self.hayate_codec = codec
        self._hayate_runtime_state: dict[int, dict] = {}

    def reset(self):
        """Clear both the resident cache and per-layer HayateQuant state."""

        super().reset()
        self._hayate_runtime_state.clear()

    def _hayate_roundtrip(self, key_states, value_states, layer_idx):
        context = {
            "layer_idx": int(layer_idx),
            "cache_state": self._hayate_runtime_state.setdefault(int(layer_idx), {}),
        }
        result = self.hayate_codec.fake_quant_roundtrip(
            key_states,
            value_states,
            ctx=context,
        )
        if not isinstance(result, (tuple, list)) or len(result) != 3:
            raise TypeError("HayateQuant fake_quant_roundtrip must return (key, value, bits)")
        key_hat, value_hat, _ = result
        for name, reconstructed, source in (
            ("key", key_hat, key_states),
            ("value", value_hat, value_states),
        ):
            if not isinstance(reconstructed, torch.Tensor):
                raise TypeError(f"HayateQuant reconstructed {name} must be a tensor")
            if reconstructed.shape != source.shape or reconstructed.device != source.device:
                raise RuntimeError(
                    f"HayateQuant reconstructed {name} must preserve shape and device"
                )
        return key_hat.to(key_states.dtype), value_hat.to(value_states.dtype)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        """Apply HayateQuant once, then store through the selected TaperKV cache."""

        if layer_idx not in self.skip_layers:
            key_states, value_states = self._hayate_roundtrip(
                key_states,
                value_states,
                layer_idx,
            )
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)


class HayateQuantTaperKVCache(_HayateQuantTaperMixin, TaperKVCache):
    """Token-granular HayateQuant-then-TaperKV reference cache."""


class HayateQuantBlockTaperKVCache(_HayateQuantTaperMixin, BlockTaperKVCache):
    """Block-granular HayateQuant-then-TaperKV reference cache."""


__all__ = ["HayateQuantTaperKVCache", "HayateQuantBlockTaperKVCache"]

"""TaperKV semantic reference vLLM backend.

This backend is intentionally conservative.  It regrades the logical paged
cache immediately before FlashAttention, which reproduces the fake-quant
quality path but retains the ordinary vLLM FP16/BF16 allocation.  Consequently
it is valid for vLLM inference and correctness smoke tests, not for memory or
throughput comparisons with packed TaperKV kernels.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import fields
from typing import Any

import torch
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl

from onecomp.kvcomp.cache._graded_base import fake_quant_tokens
from onecomp.kvcomp.cache.block_taper_cache import block_taper_bits, fake_quant_blocks
from onecomp.kvcomp.cache.taper_cache import taper_bits
from vllm_plugins.kvcomp.runtime_evidence import emit_runtime_evidence_once
from vllm_plugins.kvcomp.taper_config import TaperKVVLLMRecipe

TAPER_CONFIG_ENV = "KVCOMP_VLLM_TAPER_CONFIG"
_RECIPE_MARKER: str | None = None
_RECIPE: TaperKVVLLMRecipe | None = None


def _upstream_kv_axis() -> int:
    """Resolve the K/V axis from the installed upstream backend contract."""

    probe = FlashAttentionBackend.get_kv_cache_shape(3, 16, 1, 16)
    if probe[:2] == (3, 2):
        return 1
    if probe[:2] == (2, 3):
        return 0
    raise RuntimeError(f"unsupported vLLM FlashAttention KV cache shape: {probe}")


_UPSTREAM_KV_AXIS = _upstream_kv_axis()


def taper_reference_recipe() -> TaperKVVLLMRecipe:
    """Load and cache the validated process-local reference recipe."""

    global _RECIPE_MARKER, _RECIPE
    raw = os.environ.get(TAPER_CONFIG_ENV)
    if not raw:
        raise RuntimeError("TaperKV backend selected without KVCOMP_VLLM_TAPER_CONFIG")
    if raw == _RECIPE_MARKER and _RECIPE is not None:
        return _RECIPE
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("invalid JSON in KVCOMP_VLLM_TAPER_CONFIG") from exc
    _RECIPE = TaperKVVLLMRecipe.from_mapping(payload)
    _RECIPE_MARKER = raw
    return _RECIPE


class TaperKVReferenceBackend(FlashAttentionBackend):
    """FlashAttention data plane with explicit TaperKV semantic regrading."""

    @staticmethod
    def get_impl_cls() -> type["TaperKVReferenceImpl"]:
        return TaperKVReferenceImpl


class TaperKVReferenceImpl(FlashAttentionImpl):
    """Reference-only implementation; physical cache layout remains vLLM native."""

    @staticmethod
    def _split_kv_cache(kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate and split vLLM FlashAttention's canonical paged cache.

        vLLM has changed this axis across releases.  Resolve it from the
        installed upstream backend rather than guessing from a version string,
        then use that same contract next to every cache mutation.
        """

        if kv_cache.ndim != 5 or kv_cache.shape[_UPSTREAM_KV_AXIS] != 2:
            expected = (
                "[blocks, 2, block_size, kv_heads, head_size]"
                if _UPSTREAM_KV_AXIS == 1
                else "[2, blocks, block_size, kv_heads, head_size]"
            )
            raise RuntimeError(
                "TaperKV vLLM reference backend requires FlashAttention cache shape " + expected
            )
        return kv_cache.unbind(_UPSTREAM_KV_AXIS)

    @staticmethod
    def _layer_index(layer: Any) -> int | None:
        for name in ("layer_idx", "layer_id", "layer_number"):
            value = getattr(layer, name, None)
            if isinstance(value, int):
                return value
        layer_name = getattr(layer, "layer_name", None)
        if isinstance(layer_name, str):
            matches = re.findall(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)", layer_name)
            if matches:
                return int(matches[-1])
        return None

    @classmethod
    def _layer_selection(cls, layer: Any, recipe: TaperKVVLLMRecipe) -> tuple[int | None, bool]:
        layer_idx = cls._layer_index(layer)
        if layer_idx is None and recipe.skip_layers:
            raise RuntimeError(
                "TaperKV cannot honor skip_layers because the vLLM layer index "
                "could not be resolved"
            )
        return layer_idx, layer_idx in recipe.skip_layers

    def _hayate_codec(self, recipe: TaperKVVLLMRecipe):
        """Lazily cache the immutable HQ reference codec per attention impl."""
        if recipe.hayate_config is None:
            return None
        marker = json.dumps(recipe.hayate_config, sort_keys=True)
        if getattr(self, "_taper_hayate_marker", None) != marker:
            from onecomp.kvcomp.codec.hayate import HayateQuantCodec

            allowed = {field.name for field in fields(HayateQuantCodec)}
            kwargs = {key: value for key, value in recipe.hayate_config.items() if key in allowed}
            self._taper_hayate_codec = HayateQuantCodec(**kwargs)
            self._taper_hayate_marker = marker
        return self._taper_hayate_codec

    def _hayate_roundtrip(self, key, value, recipe, layer_idx):
        codec = self._hayate_codec(recipe)
        if codec is None:
            return key, value
        state = getattr(self, "_taper_hayate_state", None)
        if state is None:
            state = {}
            self._taper_hayate_state = state
        context = {
            "layer_idx": layer_idx,
            "cache_state": state.setdefault(layer_idx, {}),
        }
        key_hat, value_hat, _ = codec.fake_quant_roundtrip(
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            ctx=context,
        )
        return (
            key_hat.squeeze(0).transpose(0, 1).to(dtype=key.dtype),
            value_hat.squeeze(0).transpose(0, 1).to(dtype=value.dtype),
        )

    @staticmethod
    def _quantize_new_tokens(key, value, recipe):
        if recipe.policy_name == "block_taperkv" and recipe.residual_fp16:
            return key, value
        key_bits = torch.full(
            (key.shape[0],), recipe.key_bits, dtype=torch.int64, device=key.device
        )
        value_bits = torch.full(
            (value.shape[0],), recipe.value_bits, dtype=torch.int64, device=value.device
        )
        key_hat = fake_quant_tokens(key.transpose(0, 1).unsqueeze(0), key_bits)
        value_hat = fake_quant_tokens(value.transpose(0, 1).unsqueeze(0), value_bits)
        return (
            key_hat.squeeze(0).transpose(0, 1).to(dtype=key.dtype),
            value_hat.squeeze(0).transpose(0, 1).to(dtype=value.dtype),
        )

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        """Apply HQ once to arriving KV, then let upstream scatter it."""
        if not os.environ.get(TAPER_CONFIG_ENV):
            return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
        recipe = taper_reference_recipe()
        layer_idx, skipped = self._layer_selection(layer, recipe)
        if skipped:
            self._taper_pending_slots = None
            return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

        key, value = self._hayate_roundtrip(key, value, recipe, layer_idx)
        self._taper_pending_slots = slot_mapping
        return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

    def _quantize_pending_slots(self, kv_cache: torch.Tensor, recipe: TaperKVVLLMRecipe) -> None:
        slots = getattr(self, "_taper_pending_slots", None)
        if slots is None or slots.numel() == 0:
            return
        slots = slots[slots >= 0]
        if slots.numel() == 0:
            return
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        flat_keys = key_cache.flatten(0, 1)
        flat_values = value_cache.flatten(0, 1)
        slots = slots.to(device=flat_keys.device, dtype=torch.long)
        keys = flat_keys.index_select(0, slots)
        values = flat_values.index_select(0, slots)
        keys, values = self._quantize_new_tokens(keys, values, recipe)
        flat_keys.index_copy_(0, slots, keys)
        flat_values.index_copy_(0, slots, values)

    def _requantize_cache(self, kv_cache: torch.Tensor, attn_metadata, layer: Any) -> None:
        recipe = taper_reference_recipe()
        layer_idx, skipped = self._layer_selection(layer, recipe)
        if skipped:
            return
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        if attn_metadata.use_cascade:
            raise RuntimeError("TaperKV vLLM reference backend does not support cascade attention")
        block_table = attn_metadata.block_table
        seq_lens = attn_metadata.seq_lens
        if block_table is None or seq_lens is None:
            raise RuntimeError(
                "TaperKV vLLM reference backend needs block_table and seq_lens metadata"
            )
        page_size = kv_cache.shape[2]
        # seq_lens are tiny metadata tensors.  Bringing only this metadata to
        # CPU is intentional: it avoids data-dependent GPU control flow and
        # makes the reference path deterministic.  This backend is not used in
        # throughput claims; packed TaperKV needs a separate data plane.
        lengths = [int(length) for length in seq_lens.detach().to("cpu").tolist()]
        max_pages = 0
        regraded_requests = 0
        for request_index, total in enumerate(lengths):
            if total <= 0:
                continue
            n_pages = (total + page_size - 1) // page_size
            max_pages = max(max_pages, n_pages)
            regraded_requests += 1
            page_ids = block_table[request_index, :n_pages]
            if page_ids.numel() != n_pages or bool((page_ids < 0).any().item()):
                raise RuntimeError("invalid paged KV block table in TaperKV reference backend")
            page_ids = page_ids.to(device=key_cache.device, dtype=torch.long)
            key_pages = key_cache.index_select(0, page_ids)
            value_pages = value_cache.index_select(0, page_ids)
            keys = key_pages.reshape(-1, key_cache.shape[-2], key_cache.shape[-1])[:total]
            values = value_pages.reshape(-1, value_cache.shape[-2], value_cache.shape[-1])[:total]
            # Cache is [tokens, heads, dim]; the OneComp cache helpers use
            # [batch, heads, tokens, dim].  Reuse those canonical operators so
            # policy semantics cannot drift between HF and vLLM validation.
            source_k = keys.transpose(0, 1).unsqueeze(0)
            source_v = values.transpose(0, 1).unsqueeze(0)
            if recipe.policy_name == "block_taperkv" and recipe.quant_granularity == "block":
                kbits = block_taper_bits(
                    total,
                    recipe.block_size,
                    recipe.sink_tokens,
                    recipe.recent_tokens,
                    recipe.key_bits,
                    recipe.b_min,
                    recipe.ladder,
                    recipe.residual_fp16,
                    device=keys.device,
                )
                vbits = block_taper_bits(
                    total,
                    recipe.block_size,
                    recipe.sink_tokens,
                    recipe.recent_tokens,
                    recipe.value_bits,
                    recipe.b_min_value,
                    recipe.ladder,
                    recipe.residual_fp16,
                    device=values.device,
                )
                quant_k = fake_quant_blocks(
                    source_k,
                    kbits,
                    recipe.block_size,
                    known_bit_widths=tuple(range(recipe.b_min, recipe.key_bits + 1)),
                )
                quant_v = fake_quant_blocks(
                    source_v,
                    vbits,
                    recipe.block_size,
                    known_bit_widths=tuple(range(recipe.b_min_value, recipe.value_bits + 1)),
                )
            else:
                age = torch.arange(total - 1, -1, -1, device=keys.device)
                kbits = taper_bits(
                    age,
                    total,
                    recipe.sink_tokens,
                    recipe.recent_tokens,
                    recipe.key_bits,
                    recipe.b_min,
                    recipe.ladder,
                )
                vbits = taper_bits(
                    age,
                    total,
                    recipe.sink_tokens,
                    recipe.recent_tokens,
                    recipe.value_bits,
                    recipe.b_min_value,
                    recipe.ladder,
                )
                quant_k = fake_quant_tokens(source_k, kbits)
                quant_v = fake_quant_tokens(source_v, vbits)
            key_pages.reshape(-1, key_cache.shape[-2], key_cache.shape[-1])[:total].copy_(
                quant_k.squeeze(0).transpose(0, 1).to(key_pages.dtype)
            )
            value_pages.reshape(-1, value_cache.shape[-2], value_cache.shape[-1])[:total].copy_(
                quant_v.squeeze(0).transpose(0, 1).to(value_pages.dtype)
            )
            key_cache.index_copy_(0, page_ids, key_pages)
            value_cache.index_copy_(0, page_ids, value_pages)
        if regraded_requests:
            evidence = {
                "backend": ("vllm_plugins.kvcomp.taper_reference.TaperKVReferenceBackend"),
                "implementation": recipe.implementation,
                "event": "logical_paged_cache_regraded",
                "policy_name": recipe.policy_name,
                "hayate_then_taper": recipe.hayate_config is not None,
                "layer_index": layer_idx,
                "physical_cache_shape": list(kv_cache.shape),
                "physical_cache_dtype": str(kv_cache.dtype).removeprefix("torch."),
                "physical_cache_device": str(kv_cache.device),
                "page_size": int(page_size),
                "request_count": regraded_requests,
                "maximum_pages_per_request": max_pages,
                "maximum_sequence_length": max(lengths),
            }
            emit_runtime_evidence_once(
                self,
                "taper_regrade",
                evidence,
            )
            # The first decode can be shorter than one page even when a
            # later decode has crossed a page boundary.  Preserve one further
            # evidence record for the first multi-page mutation, otherwise a
            # one-shot receipt cannot prove the paged-cache path it claims to
            # validate.  This does not change the fake-quant data plane.
            if max_pages > 1:
                emit_runtime_evidence_once(self, "taper_regrade_multi_page", evidence)

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        # The FlashAttention registry is process-global.  A later ordinary
        # engine in the same interpreter may clear the TaperKV payload; in that
        # case preserve upstream FlashAttention rather than failing or applying
        # an old recipe to the wrong model.
        if attn_metadata is not None and os.environ.get(TAPER_CONFIG_ENV):
            if attn_metadata.use_cascade:
                raise RuntimeError(
                    "TaperKV vLLM reference backend does not support cascade attention"
                )
            if getattr(self, "dcp_world_size", 1) > 1:
                raise RuntimeError(
                    "TaperKV vLLM reference backend does not support decode context parallelism"
                )
            recipe = taper_reference_recipe()
            _layer_idx, skipped = self._layer_selection(layer, recipe)
            since = getattr(self, "_taper_since", 0) + 1
            self._taper_since = since
            max_query_len = int(attn_metadata.max_query_len)
            regrade = max_query_len > 1 or since % recipe.requant_stride == 0
            if not skipped and regrade:
                self._requantize_cache(kv_cache, attn_metadata, layer)
            elif not skipped:
                self._quantize_pending_slots(kv_cache, recipe)
            self._taper_pending_slots = None
        return super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )


# Backward-compatible aliases.  These names predate the explicit distinction
# between a semantic reference path and a physically packed native backend.
taper_native_recipe = taper_reference_recipe
TaperKVNativeBackend = TaperKVReferenceBackend
TaperKVNativeImpl = TaperKVReferenceImpl

__all__ = [
    "TAPER_CONFIG_ENV",
    "TaperKVReferenceBackend",
    "TaperKVReferenceImpl",
    "taper_reference_recipe",
    "TaperKVNativeBackend",
    "TaperKVNativeImpl",
    "taper_native_recipe",
]

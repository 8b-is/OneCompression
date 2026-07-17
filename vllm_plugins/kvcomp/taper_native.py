"""TapeKV semantic vLLM backend.

This backend is intentionally conservative.  It regrades the logical paged
cache immediately before FlashAttention, which reproduces the fake-quant
quality path but retains the ordinary vLLM FP16/BF16 allocation.  Consequently
it is valid for vLLM inference and correctness smoke tests, not for memory or
throughput comparisons with packed TapeKV kernels.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from onecomp.kvcomp.cache._graded_base import fake_quant_tokens
from onecomp.kvcomp.cache.block_taper_cache import block_taper_bits, fake_quant_blocks
from onecomp.kvcomp.cache.taper_cache import taper_bits
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl

from vllm_plugins.kvcomp.taper_config import TaperVLLMRecipe


TAPER_CONFIG_ENV = "KVCOMP_VLLM_TAPER_CONFIG"


def taper_native_recipe() -> TaperVLLMRecipe:
    raw = os.environ.get(TAPER_CONFIG_ENV)
    if not raw:
        raise RuntimeError("TapeKV backend selected without KVCOMP_VLLM_TAPER_CONFIG")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("invalid JSON in KVCOMP_VLLM_TAPER_CONFIG") from exc
    return TaperVLLMRecipe.from_mapping(payload)


class TaperKVNativeBackend(FlashAttentionBackend):
    """FlashAttention data plane with explicit TapeKV semantic regrading."""

    @staticmethod
    def get_impl_cls() -> type["TaperKVNativeImpl"]:
        return TaperKVNativeImpl


class TaperKVNativeImpl(FlashAttentionImpl):
    """Reference-only implementation; physical cache layout remains vLLM native."""

    @staticmethod
    def _layer_index(layer: Any) -> int | None:
        for name in ("layer_idx", "layer_id", "layer_number"):
            value = getattr(layer, name, None)
            if isinstance(value, int):
                return value
        return None

    def _hayate_codec(self, recipe: TaperVLLMRecipe):
        """Lazily cache the immutable HQ reference codec per attention impl."""
        if recipe.hayate_config is None:
            return None
        marker = json.dumps(recipe.hayate_config, sort_keys=True)
        if getattr(self, "_taper_hayate_marker", None) != marker:
            from onecomp.kvcomp.codec.hayate import HayateQuantCodec

            allowed = HayateQuantCodec.__dataclass_fields__
            kwargs = {key: value for key, value in recipe.hayate_config.items() if key in allowed}
            self._taper_hayate_codec = HayateQuantCodec(**kwargs)
            self._taper_hayate_marker = marker
        return self._taper_hayate_codec

    def _requantize_cache(self, kv_cache: torch.Tensor, attn_metadata, layer: Any) -> None:
        recipe = taper_native_recipe()
        layer_idx = self._layer_index(layer)
        if layer_idx is not None and layer_idx in recipe.skip_layers:
            return
        if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
            raise RuntimeError(
                "TapeKV vLLM reference backend requires FlashAttention cache shape "
                "[blocks, 2, block_size, kv_heads, head_size]"
            )
        if attn_metadata.use_cascade:
            raise RuntimeError("TapeKV vLLM reference backend does not support cascade attention")
        block_table = attn_metadata.block_table
        seq_lens = attn_metadata.seq_lens
        if block_table is None or seq_lens is None:
            raise RuntimeError("TapeKV vLLM reference backend needs block_table and seq_lens metadata")
        page_size = kv_cache.shape[2]
        # seq_lens are tiny metadata tensors.  Bringing only this metadata to
        # CPU is intentional: it avoids data-dependent GPU control flow and
        # makes the reference path deterministic.  This backend is not used in
        # throughput claims; packed TapeKV needs a separate data plane.
        lengths = [int(length) for length in seq_lens.detach().to("cpu").tolist()]
        key_cache, value_cache = kv_cache.unbind(1)
        for request_index, total in enumerate(lengths):
            if total <= 0:
                continue
            n_pages = (total + page_size - 1) // page_size
            page_ids = block_table[request_index, :n_pages]
            if page_ids.numel() != n_pages or bool((page_ids < 0).any().item()):
                raise RuntimeError("invalid paged KV block table in TapeKV reference backend")
            keys = key_cache[page_ids].reshape(-1, key_cache.shape[-2], key_cache.shape[-1])[:total]
            values = value_cache[page_ids].reshape(-1, value_cache.shape[-2], value_cache.shape[-1])[:total]
            # Cache is [tokens, heads, dim]; the OneComp cache helpers use
            # [batch, heads, tokens, dim].  Reuse those canonical operators so
            # policy semantics cannot drift between HF and vLLM validation.
            source_k = keys.transpose(0, 1).unsqueeze(0)
            source_v = values.transpose(0, 1).unsqueeze(0)
            codec = self._hayate_codec(recipe)
            if codec is not None:
                # Explicit composition order: HQ reconstructs each logical KV
                # vector first; TapeKV then applies its age-dependent affine
                # regrade.  This is a quality/reference path only--the paged
                # allocation remains model dtype and is never a packed-HQ claim.
                source_k, source_v, _ = codec.fake_quant_roundtrip(
                    source_k, source_v, ctx={"layer_idx": layer_idx}
                )
            if recipe.policy_name == "block_taperkv" and recipe.quant_granularity == "block":
                kbits = block_taper_bits(total, recipe.block_size, recipe.sink_tokens,
                    recipe.recent_tokens, recipe.key_bits, recipe.b_min, recipe.ladder,
                    recipe.residual_fp16, device=keys.device)
                vbits = block_taper_bits(total, recipe.block_size, recipe.sink_tokens,
                    recipe.recent_tokens, recipe.value_bits, recipe.b_min_value, recipe.ladder,
                    recipe.residual_fp16, device=values.device)
                quant_k = fake_quant_blocks(source_k, kbits, recipe.block_size)
                quant_v = fake_quant_blocks(source_v, vbits, recipe.block_size)
            else:
                age = torch.arange(total - 1, -1, -1, device=keys.device)
                kbits = taper_bits(age, total, recipe.sink_tokens, recipe.recent_tokens,
                    recipe.key_bits, recipe.b_min, recipe.ladder)
                vbits = taper_bits(age, total, recipe.sink_tokens, recipe.recent_tokens,
                    recipe.value_bits, recipe.b_min_value, recipe.ladder)
                quant_k = fake_quant_tokens(source_k, kbits)
                quant_v = fake_quant_tokens(source_v, vbits)
            key_pages = key_cache[page_ids].clone()
            value_pages = value_cache[page_ids].clone()
            key_pages.reshape(-1, key_cache.shape[-2], key_cache.shape[-1])[:total].copy_(
                quant_k.squeeze(0).transpose(0, 1).to(key_pages.dtype)
            )
            value_pages.reshape(-1, value_cache.shape[-2], value_cache.shape[-1])[:total].copy_(
                quant_v.squeeze(0).transpose(0, 1).to(value_pages.dtype)
            )
            key_cache[page_ids] = key_pages
            value_cache[page_ids] = value_pages

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        # The FlashAttention registry is process-global.  A later ordinary
        # engine in the same interpreter may clear the TapeKV payload; in that
        # case preserve upstream FlashAttention rather than failing or applying
        # an old recipe to the wrong model.
        if attn_metadata is not None and os.environ.get(TAPER_CONFIG_ENV):
            self._requantize_cache(kv_cache, attn_metadata, layer)
        return super().forward(
            layer, query, key, value, kv_cache, attn_metadata, output,
            output_scale=output_scale, output_block_scale=output_block_scale,
        )

# Copyright 2026 Fujitsu Ltd.
"""Packed K2/V2 TurboQuant backend for vLLM.

vLLM 0.24 exposes only K/V widths 3 and 4 in its TurboQuant kernels.  This
plugin keeps the upstream scalar quantizer semantics while replacing cache
allocation, store, decode, and continuation dequantization for rate two.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import torch
import torch.nn.functional as F

from vllm.v1.attention.backends.turboquant_attn import (
    _HAS_FLASH_ATTN,
    TurboQuantAttentionBackend,
    TurboQuantMetadata,
    _build_hadamard,
)
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

from vllm_plugins.kvcomp.hayate_native import (
    HayateQuantNativeImpl,
    _layer_index,
)
from vllm_plugins.kvcomp.native_turboquant import (
    disable_vllm_implicit_turboquant_boundary_skip,
)
from vllm_plugins.kvcomp.runtime_evidence import emit_runtime_evidence_once
from vllm_plugins.kvcomp.turboquant2_triton import (
    triton_turboquant2_decode_attention,
    triton_turboquant2_dequant_kv,
    triton_turboquant2_store,
    turboquant2_layout,
)

TURBOQUANT2_CONFIG_ENV = "KVCOMP_VLLM_TURBOQUANT2_CONFIG"


def _native_config() -> dict[str, int]:
    raw = os.environ.get(TURBOQUANT2_CONFIG_ENV)
    if not raw:
        return {}
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {TURBOQUANT2_CONFIG_ENV}") from exc
    if config != {"rate": 2}:
        raise ValueError(f"{TURBOQUANT2_CONFIG_ENV} must contain exactly {{'rate': 2}}")
    return config


def configure_turboquant2_native_backend() -> None:
    """Install K2/V2 in the launcher and propagate it to spawn children."""

    disable_vllm_implicit_turboquant_boundary_skip()
    os.environ[TURBOQUANT2_CONFIG_ENV] = json.dumps({"rate": 2}, sort_keys=True)
    from vllm_plugins.kvcomp.vllm_plugin import (  # pylint: disable=import-outside-toplevel
        _get_logger,
        _install_turboquant2_cache_spec_hook,
        _register_turboquant2_backend,
    )

    _register_turboquant2_backend(_get_logger())
    _install_turboquant2_cache_spec_hook()


class TurboQuant2NativeBackend(TurboQuantAttentionBackend):
    """Attention backend owning the packed 70-byte K2/V2 slot at D=128."""

    @staticmethod
    def get_impl_cls() -> type["TurboQuant2NativeImpl"]:
        return TurboQuant2NativeImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "turboquant_3bit_nc",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        slot_size = turboquant2_layout(head_size)["slot_size_aligned"]
        return (num_blocks, block_size, num_kv_heads, slot_size)


class TurboQuant2NativeImpl(HayateQuantNativeImpl):
    """Specialized scalar TurboQuant implementation for two-bit K and V."""

    @staticmethod
    def _onecomp_native_enabled(layer: Any) -> bool:
        return bool(getattr(layer, "_turboquant2_native_enabled", False))

    def _ensure_on_device(self, layer: Any, device: torch.device | str) -> None:
        if hasattr(layer, "_turboquant2_cached"):
            return
        if not _native_config():
            layer._turboquant2_native_enabled = False
            super()._ensure_on_device(layer, device)
            layer._turboquant2_cached = True
            return

        head_dim = self.head_size
        turboquant2_layout(head_dim)
        from vllm.model_executor.layers.quantization.turboquant.centroids import (  # pylint: disable=import-outside-toplevel
            get_centroids,
        )

        dev = torch.device(device)
        rotation = _build_hadamard(head_dim, str(dev)).contiguous()
        centroids = get_centroids(head_dim, 2).to(device=dev, dtype=torch.float32)
        centroids, _ = centroids.sort()
        layer._tq_PiT = rotation
        layer._tq_Pi = rotation
        layer._tq_centroids = centroids
        layer._tq_midpoints = ((centroids[:-1] + centroids[1:]) / 2).contiguous()
        layer._tq2_layer_index = _layer_index(layer)
        layer._turboquant2_native_enabled = True
        layer._tq_cached = True
        layer._turboquant2_cached = True

    def _store_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer: Any,
    ) -> None:
        if not self._onecomp_native_enabled(layer):
            super()._store_kv(key, value, kv_cache, slot_mapping, layer)
            return
        triton_turboquant2_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            layer._tq_PiT,
            layer._tq_midpoints,
        )
        emit_runtime_evidence_once(
            self,
            "turboquant2_store",
            {
                "backend": ("vllm_plugins.kvcomp.turboquant2_native.TurboQuant2NativeBackend"),
                "implementation": "native_packed_turboquant2",
                "event": "packed_turboquant2_store_launched",
                "store_encoder": "triton_fused_scalar_k2v2",
                "layer_index": layer._tq2_layer_index,
                "method": "turboquant",
                "rate": 2,
                "key_bits": 2,
                "value_bits": 2,
                "layout": turboquant2_layout(int(key.shape[-1])),
                "cache_shape": list(kv_cache.shape),
                "cache_dtype": str(kv_cache.dtype).removeprefix("torch."),
                "cache_device": str(kv_cache.device),
            },
        )

    def _decode_attention(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        if layer is None or not self._onecomp_native_enabled(layer):
            return super()._decode_attention(
                query, kv_cache, attn_metadata, Pi, centroids, PiT, layer
            )
        batch, query_heads, head_dim = query.shape
        splits = self.max_num_kv_splits
        mid_output = output = lse = None
        if is_workspace_manager_initialized():
            mid_output, output, lse = current_workspace_manager().get_simultaneous(
                ((batch, query_heads, splits, head_dim + 1), torch.float32),
                ((batch, query_heads, head_dim), query.dtype),
                ((batch, query_heads), torch.float32),
            )
        result = triton_turboquant2_decode_attention(
            query,
            kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            layer._tq_PiT,
            layer._tq_centroids,
            self.scale,
            mid_o_buf=mid_output,
            output_buf=output,
            lse_buf=lse,
            buf_holder=layer,
            max_num_kv_splits=splits,
        )
        emit_runtime_evidence_once(
            self,
            "turboquant2_decode",
            {
                "backend": ("vllm_plugins.kvcomp.turboquant2_native.TurboQuant2NativeBackend"),
                "implementation": "native_packed_turboquant2",
                "event": "packed_turboquant2_decode_attention_launched",
                "layer_index": layer._tq2_layer_index,
                "method": "turboquant",
                "rate": 2,
                "key_bits": 2,
                "value_bits": 2,
                "layout": turboquant2_layout(head_dim),
                "cache_shape": list(kv_cache.shape),
                "cache_dtype": str(kv_cache.dtype).removeprefix("torch."),
                "cache_device": str(kv_cache.device),
                "query_shape": list(query.shape),
                "kv_splits": int(splits),
            },
        )
        return result

    def _continuation_prefill(
        self,
        layer: Any,
        query: torch.Tensor,
        key_chunk: torch.Tensor,
        val_chunk: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cached_len: int,
        seq_len: int,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        if not self._onecomp_native_enabled(layer):
            return super()._continuation_prefill(
                layer,
                query,
                key_chunk,
                val_chunk,
                kv_cache,
                block_table,
                cached_len,
                seq_len,
                Pi,
                centroids,
            )
        query_len, query_heads, head_dim = query.shape
        kv_heads = key_chunk.shape[1]
        page_size = kv_cache.shape[1]
        allocated_len = math.ceil(cached_len / page_size) * page_size
        shape = (1, kv_heads, allocated_len, head_dim)
        key_buffer, value_buffer = current_workspace_manager().get_simultaneous(
            (shape, query.dtype), (shape, query.dtype)
        )
        triton_turboquant2_dequant_kv(
            kv_cache,
            block_table,
            layer._tq_centroids,
            key_buffer,
            value_buffer,
        )
        rotated_key = key_buffer[0, :, :cached_len].reshape(-1, head_dim).float()
        cached_key = (
            (rotated_key @ layer._tq_Pi)
            .to(query.dtype)
            .reshape(kv_heads, cached_len, head_dim)
            .transpose(0, 1)
        )
        cached_value = value_buffer[0, :, :cached_len].transpose(0, 1)
        full_key = torch.empty(seq_len, kv_heads, head_dim, device=query.device, dtype=query.dtype)
        full_value = torch.empty_like(full_key)
        full_key[:cached_len], full_key[cached_len:] = cached_key, key_chunk
        full_value[:cached_len], full_value[cached_len:] = cached_value, val_chunk

        if _HAS_FLASH_ATTN:
            if not hasattr(self, "_tq2_cu2q"):
                self._tq2_cu2q = torch.zeros(2, device=query.device, dtype=torch.int32)
                self._tq2_cu2k = torch.zeros(2, device=query.device, dtype=torch.int32)
            self._tq2_cu2q[1:2] = query_len
            self._tq2_cu2k[1:2] = seq_len
            return self._flash_attn_varlen(
                q=query,
                k=full_key,
                v=full_value,
                cu_seqlens_q=self._tq2_cu2q,
                cu_seqlens_k=self._tq2_cu2k,
                max_seqlen_q=query_len,
                max_seqlen_k=seq_len,
            )
        query_positions = torch.arange(query_len, device=query.device).unsqueeze(1) + cached_len
        key_positions = torch.arange(seq_len, device=query.device).unsqueeze(0)
        output = F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            full_key.transpose(0, 1).unsqueeze(0),
            full_value.transpose(0, 1).unsqueeze(0),
            attn_mask=key_positions <= query_positions,
            scale=self.scale,
            enable_gqa=(kv_heads < query_heads),
        )
        return output[0].transpose(0, 1)


__all__ = [
    "TURBOQUANT2_CONFIG_ENV",
    "TurboQuant2NativeBackend",
    "TurboQuant2NativeImpl",
    "configure_turboquant2_native_backend",
]

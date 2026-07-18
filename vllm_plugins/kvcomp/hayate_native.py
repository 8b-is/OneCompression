# Copyright 2026 Fujitsu Ltd.
"""Experimental vLLM key-block-VQ/TurboQuant-value backend.

vLLM exposes TurboQuant through fixed ``kv_cache_dtype`` strings.  Rather than
adding a new dtype to vLLM's pydantic config surface, the OneComp plugin maps a
HayateQuant model onto the byte-compatible TurboQuant dtype and overrides the
TurboQuant attention backend in-process.  The cache size and scheduler behavior
therefore stay identical to upstream vLLM, while the key quantizer uses
HayateQuant block-VQ codebooks. Values still use upstream TurboQuant's
uniform scalar quantizer, so this backend is not a faithful HayateQuant
implementation and is never selected by default.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import torch
import torch.nn.functional as F

from vllm.v1.attention.backends.turboquant_attn import (
    _CONTINUATION_DECODE_THRESHOLD,
    _HAS_FLASH_ATTN,
    TurboQuantAttentionBackend,
    TurboQuantAttentionImpl,
    TurboQuantMetadata,
)
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

from vllm_plugins.kvcomp.hayate_triton import (
    triton_hayatequant_decode_attention,
    triton_hayatequant_full_dequant_kv,
    triton_hayatequant_store,
)

_CONFIG_ENV = "KVCOMP_VLLM_HAYATE_NATIVE_CONFIG"


def _native_config() -> dict[str, Any]:
    raw = os.environ.get(_CONFIG_ENV)
    if not raw:
        return {}
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return cfg if isinstance(cfg, dict) else {}


class HayateQuantNativeBackend(TurboQuantAttentionBackend):
    """TurboQuant-layout vLLM backend with HayateQuant key block-VQ."""

    @staticmethod
    def get_impl_cls() -> type["HayateQuantNativeImpl"]:
        return HayateQuantNativeImpl


class HayateQuantNativeImpl(TurboQuantAttentionImpl):
    """Explicit opt-in hybrid replacement for vLLM's TurboQuant implementation.

    The implementation keeps the upstream TurboQuant value path and cache
    layout, then swaps the key scalar MSE indexer for a HayateQuant block-VQ
    indexer when the model config requests the supported 3/4-bit block_size=2
    operating point. Values remain uniform scalar-quantized. Unsupported or
    non-block settings delegate to upstream TurboQuant so the engine remains
    serveable; benchmark output must label this hybrid distinctly.
    """

    def _ensure_on_device(self, layer, device):
        if hasattr(layer, "_tq_cached"):
            return

        if self.tq_config.key_fp8 or self.tq_config.key_quant_bits not in (3, 4):
            layer._hq_native_enabled = False
            return super()._ensure_on_device(layer, device)

        cfg = _native_config()
        # This backend class can remain registered after a previous Hayate
        # engine because vLLM's attention registry is process-global. An empty
        # config is therefore an explicit upstream-TurboQuant state, not an
        # invitation to synthesize Hayate defaults for an unrelated engine.
        if not cfg:
            layer._hq_native_enabled = False
            return super()._ensure_on_device(layer, device)
        key_bits = int(cfg.get("bits") or self.tq_config.key_quant_bits)
        value_bits = cfg.get("value_bits")
        value_bits = int(value_bits) if value_bits is not None else self.tq_config.value_quant_bits
        if key_bits != self.tq_config.key_quant_bits:
            layer._hq_native_enabled = False
            return super()._ensure_on_device(layer, device)

        dist = str(cfg.get("dist") or "gaussian")
        codebook_init = str(cfg.get("codebook_init") or "auto")
        requested_block = cfg.get("block_size")
        requested_block = int(requested_block) if requested_block is not None else None
        max_codebook_bits = int(cfg.get("max_codebook_bits") or 9)
        rotation_mode = str(cfg.get("rotation_mode") or "auto")
        key_scale_mode = str(cfg.get("key_scale_mode") or "nc").lower()
        if key_scale_mode == "ub":
            key_scale_mode = "ip"
        if key_scale_mode not in {"nc", "ip"}:
            layer._hq_native_enabled = False
            return super()._ensure_on_device(layer, device)
        seed = int(cfg.get("seed") or 0)
        D = self.head_size

        from onecomp.kvcomp.codec.hayate.codec import (  # pylint: disable=import-outside-toplevel
            _block_codebook,
            _dense_rotation_on,
        )
        from onecomp.kvcomp.codec.hayate.vq import (
            resolve_block_size,
        )  # pylint: disable=import-outside-toplevel
        from onecomp.kvcomp.codec.rotation import (  # pylint: disable=import-outside-toplevel
            hybrid_rotation_matrix,
            is_power_of_two,
            signed_hadamard_matrix,
        )

        block_size = resolve_block_size(key_bits, requested_block, max_codebook_bits, D)
        if block_size != 2 or value_bits not in (3, 4):
            layer._hq_native_enabled = False
            return super()._ensure_on_device(layer, device)

        if rotation_mode == "auto":
            mode = "hybrid" if key_bits >= 3 and is_power_of_two(D) else "dense"
        else:
            mode = rotation_mode
        device_str = str(torch.device(device))
        if mode == "hybrid" and is_power_of_two(D):
            rotation = hybrid_rotation_matrix(D, block_size, seed, device_str, torch.float32)
        elif mode in ("hadamard", "hadamard_gemm") and is_power_of_two(D):
            rotation = signed_hadamard_matrix(D, seed, device_str, torch.float32)
        else:
            rotation = _dense_rotation_on(D, seed, device_str, torch.float32)

        codebook = _block_codebook(key_bits, block_size, dist, D, codebook_init).to(
            device=torch.device(device), dtype=torch.float32
        )
        layer._tq_PiT = rotation.contiguous()
        # Inverse rotation for continuation-prefill bulk dequant.
        layer._tq_Pi = rotation.t().contiguous()
        layer._tq_Pi_half = layer._tq_Pi.to(torch.float16)
        layer._tq_centroids = codebook.contiguous()
        layer._hq_block_size = block_size
        layer._hq_key_bits = key_bits
        layer._hq_key_scale_mode = key_scale_mode
        layer._hq_native_enabled = True
        layer._tq_cached = True

    def _store_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer: Any,
    ):
        if not getattr(layer, "_hq_native_enabled", False):
            triton_turboquant_store(
                key,
                value,
                kv_cache,
                slot_mapping,
                layer._tq_PiT,
                layer._tq_midpoints,
                mse_bits=self.tq_config.key_mse_bits,
                key_packed_size=self.tq_config.key_packed_size,
                value_quant_bits=self.tq_config.effective_value_quant_bits,
                key_fp8=self.tq_config.key_fp8,
            )
            return
        triton_hayatequant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            layer._tq_PiT,
            layer._tq_centroids,
            layer._hq_block_size,
            layer._hq_key_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            key_scale_mode=layer._hq_key_scale_mode,
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
        if layer is None or not getattr(layer, "_hq_native_enabled", False):
            return super()._decode_attention(
                query, kv_cache, attn_metadata, Pi, centroids, PiT, layer
            )

        B = query.shape[0]
        D = self.head_size
        S = self.max_num_kv_splits
        Hq = self.num_heads
        mid_o_buf = output_buf = lse_buf = None
        if is_workspace_manager_initialized():
            mid_o_buf, output_buf, lse_buf = current_workspace_manager().get_simultaneous(
                ((B, Hq, S, D + 1), torch.float32),
                ((B, Hq, D), query.dtype),
                ((B, Hq), torch.float32),
            )
        return triton_hayatequant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            rotation=PiT if PiT is not None else layer._tq_PiT,
            codebook=centroids,
            scale=self.scale,
            key_bits=layer._hq_key_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            hayate_block_size=layer._hq_block_size,
            norm_correction=self.tq_config.norm_correction,
            key_scale_mode=layer._hq_key_scale_mode,
            mid_o_buf=mid_o_buf,
            output_buf=output_buf,
            lse_buf=lse_buf,
            buf_holder=layer,
            max_num_kv_splits=self.max_num_kv_splits,
        )

    def _prefill_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: Any = None,
    ) -> torch.Tensor:
        if layer is None or not getattr(layer, "_hq_native_enabled", False):
            return super()._prefill_attention(
                query, key, value, kv_cache, attn_metadata, Pi, centroids, PiT, layer
            )

        N, Hq, D = query.shape
        if _HAS_FLASH_ATTN and attn_metadata.max_query_len == attn_metadata.max_seq_len:
            return self._flash_attn_varlen(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
            )

        Hk = key.shape[1]
        use_gqa = Hk < Hq
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1
        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)

        qsl = (
            attn_metadata.query_start_loc_cpu.tolist()
            if attn_metadata.query_start_loc_cpu is not None
            else query_start_loc.tolist()
        )
        seq_lens_list = (
            attn_metadata.seq_lens_cpu.tolist()
            if attn_metadata.seq_lens_cpu is not None
            else attn_metadata.seq_lens.tolist()
        )

        if not hasattr(self, "_cu_2"):
            self._cu_2 = torch.zeros(2, device=query.device, dtype=torch.int32)
        max_seq = attn_metadata.max_seq_len
        arange_cache = getattr(self, "_arange_cache", None)
        if arange_cache is None or arange_cache.shape[0] <= max_seq:
            arange_cache = torch.arange(
                0, max_seq + 1, device=query.device, dtype=attn_metadata.seq_lens.dtype
            )
            self._arange_cache = arange_cache

        for i in range(num_reqs):
            q_start = qsl[i]
            q_end = qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue
            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]
            k_seq = key[q_start:q_end]
            v_seq = value[q_start:q_end]

            if q_len == seq_len:
                if _HAS_FLASH_ATTN:
                    self._cu_2[1:2] = q_len
                    out = self._flash_attn_varlen(
                        q=q_seq,
                        k=k_seq,
                        v=v_seq,
                        cu_seqlens_q=self._cu_2,
                        cu_seqlens_k=self._cu_2,
                        max_seqlen_q=q_len,
                        max_seqlen_k=q_len,
                    )
                else:
                    out = F.scaled_dot_product_attention(
                        q_seq.transpose(0, 1).contiguous(),
                        k_seq.transpose(0, 1).contiguous(),
                        v_seq.transpose(0, 1).contiguous(),
                        is_causal=True,
                        scale=self.scale,
                        enable_gqa=use_gqa,
                    ).transpose(0, 1)
                output[q_start:q_end] = out.to(query.dtype)
                continue

            cached_len = seq_len - q_len
            if q_len <= _CONTINUATION_DECODE_THRESHOLD:
                synth_seq_lens = arange_cache[cached_len + 1 : seq_len + 1]
                synth_bt = attn_metadata.block_table[i : i + 1].expand(q_len, -1)
                decode_meta = TurboQuantMetadata(
                    seq_lens=synth_seq_lens,
                    slot_mapping=attn_metadata.slot_mapping[q_start:q_end],
                    block_table=synth_bt,
                    query_start_loc=torch.arange(
                        0,
                        q_len + 1,
                        device=query.device,
                        dtype=attn_metadata.query_start_loc.dtype,
                    ),
                    num_actual_tokens=q_len,
                    max_query_len=1,
                    max_seq_len=seq_len,
                    is_prefill=False,
                    num_decodes=q_len,
                    num_decode_tokens=q_len,
                )
                out = self._decode_attention(
                    q_seq, kv_cache, decode_meta, Pi, centroids, PiT, layer
                )
            else:
                out = self._continuation_prefill(
                    layer,
                    q_seq,
                    k_seq,
                    v_seq,
                    kv_cache,
                    attn_metadata.block_table[i : i + 1],
                    cached_len,
                    seq_len,
                    Pi,
                    centroids,
                )
            output[q_start:q_end] = out.to(query.dtype)

        return output

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
        if not getattr(layer, "_hq_native_enabled", False):
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

        q_len, Hq, D = query.shape
        Hk = key_chunk.shape[1]
        device = query.device
        block_size = kv_cache.shape[1]
        alloc_len = math.ceil(cached_len / block_size) * block_size
        buf_shape = (1, Hk, alloc_len, D)
        k_buf, v_buf = current_workspace_manager().get_simultaneous(
            (buf_shape, torch.float16),
            (buf_shape, torch.float16),
        )
        k_cached = k_buf[:, :, :alloc_len, :]
        v_cached = v_buf[:, :, :alloc_len, :]
        triton_hayatequant_full_dequant_kv(
            kv_cache=kv_cache,
            block_table=block_table,
            codebook=centroids,
            k_out=k_cached,
            v_out=v_cached,
            key_bits=layer._hq_key_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            hayate_block_size=layer._hq_block_size,
            norm_correction=self.tq_config.norm_correction,
            key_scale_mode=layer._hq_key_scale_mode,
        )

        k_flat = k_cached[0, :, :cached_len, :].reshape(-1, D)
        k_flat = k_flat @ layer._tq_Pi_half
        k_cached_trim = k_flat.reshape(Hk, cached_len, D).transpose(0, 1)
        v_cached_trim = v_cached[0, :, :cached_len, :].transpose(0, 1)

        qdtype = query.dtype
        k_full = torch.empty(seq_len, Hk, D, dtype=qdtype, device=device)
        v_full = torch.empty(seq_len, Hk, D, dtype=qdtype, device=device)
        k_full[:cached_len] = k_cached_trim.to(qdtype)
        k_full[cached_len:] = key_chunk
        v_full[:cached_len] = v_cached_trim.to(qdtype)
        v_full[cached_len:] = val_chunk

        if _HAS_FLASH_ATTN:
            if not hasattr(self, "_cu_2_q"):
                self._cu_2_q = torch.zeros(2, device=device, dtype=torch.int32)
                self._cu_2_k = torch.zeros(2, device=device, dtype=torch.int32)
            self._cu_2_q[1:2] = q_len
            self._cu_2_k[1:2] = seq_len
            return self._flash_attn_varlen(
                q=query,
                k=k_full,
                v=v_full,
                cu_seqlens_q=self._cu_2_q,
                cu_seqlens_k=self._cu_2_k,
                max_seqlen_q=q_len,
                max_seqlen_k=seq_len,
            )

        q_t = query.transpose(0, 1).unsqueeze(0)
        k_t = k_full.transpose(0, 1).unsqueeze(0)
        v_t = v_full.transpose(0, 1).unsqueeze(0)
        q_pos = torch.arange(q_len, device=device).unsqueeze(1) + cached_len
        k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
        mask = k_pos <= q_pos
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=mask,
            scale=self.scale,
            enable_gqa=(Hk < Hq),
        )
        return out[0].transpose(0, 1)

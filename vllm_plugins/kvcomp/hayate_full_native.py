# Copyright 2026 Fujitsu Ltd.
"""Faithful packed fixed-recipe HayateQuant backend for vLLM 0.20.

This backend is separate from the historical key-HQ/value-TQ hybrid.  It is
registered explicitly over vLLM's TURBOQUANT enum in a fresh process, but
overrides cache allocation as well as store/decode, so its physical data plane
is the HQ layout rather than a byte-compatible TurboQuant approximation.
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

from vllm_plugins.kvcomp.hayate_full_config import (
    FullHQNativeRecipe,
    full_hq_native_recipe_from_mapping,
)
from vllm_plugins.kvcomp.hayate_full_triton import (
    triton_full_hayatequant_decode_attention,
    triton_full_hayatequant_dequant_kv,
    triton_full_hayatequant_store,
)

FULL_CONFIG_ENV = "KVCOMP_VLLM_HAYATE_FULL_NATIVE_CONFIG"


def _full_native_config() -> dict[str, Any]:
    raw = os.environ.get(FULL_CONFIG_ENV)
    if not raw:
        return {}
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {FULL_CONFIG_ENV}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{FULL_CONFIG_ENV} must contain a JSON object")
    return config


def _full_native_recipe() -> FullHQNativeRecipe | None:
    config = _full_native_config()
    return full_hq_native_recipe_from_mapping(config) if config else None


def configure_full_hq_native_backend(config: dict[str, Any]) -> FullHQNativeRecipe:
    """Validate, export, and register full HQ before constructing ``LLM``.

    The environment payload propagates to vLLM EngineCore spawn children,
    where the general OneComp plugin re-registers the same backend.  Call this
    only in a fresh process; backend registries are process-global.
    """

    recipe = full_hq_native_recipe_from_mapping(config)
    if os.environ.get("KVCOMP_VLLM_HAYATE_NATIVE_CONFIG"):
        raise RuntimeError("historical Hayate hybrid config is already set")
    os.environ[FULL_CONFIG_ENV] = json.dumps(config, sort_keys=True)
    from vllm.v1.attention.backends.registry import (  # pylint: disable=import-outside-toplevel
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.TURBOQUANT,
        "vllm_plugins.kvcomp.hayate_full_native.FullHayateQuantNativeBackend",
    )
    # vLLM 0.20's cache-memory spec is independent of backend shape; install
    # the explicit-process hook in the launcher as well as in spawn children.
    from vllm_plugins.kvcomp.vllm_plugin import (  # pylint: disable=import-outside-toplevel
        _install_hayate_full_cache_spec_hook,
    )

    _install_hayate_full_cache_spec_hook()
    return recipe


class FullHayateQuantNativeBackend(TurboQuantAttentionBackend):
    """vLLM backend owning the faithful packed fixed-HQ cache layout."""

    @staticmethod
    def get_impl_cls() -> type["FullHayateQuantNativeImpl"]:
        return FullHayateQuantNativeImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "turboquant_3bit_nc",
    ) -> tuple[int, ...]:
        recipe = _full_native_recipe()
        if recipe is None:
            return TurboQuantAttentionBackend.get_kv_cache_shape(
                num_blocks,
                block_size,
                num_kv_heads,
                head_size,
                cache_dtype_str,
            )
        slot = recipe.layout(head_size)["slot_size_aligned"]
        return (num_blocks, block_size, num_kv_heads, slot)


class FullHayateQuantNativeImpl(TurboQuantAttentionImpl):
    """Fixed HQ native store/decode implementation for both K and V."""

    def _ensure_on_device(self, layer, device):
        if hasattr(layer, "_hq_full_cached"):
            return
        recipe = _full_native_recipe()
        if recipe is None:
            layer._hq_full_native_enabled = False
            super()._ensure_on_device(layer, device)
            layer._hq_full_cached = True
            return

        d = self.head_size
        recipe.validate_head_dim(d)
        config = _full_native_config()
        if config.get("adaptive_bit_split") not in (None, "", "off", "none", False):
            raise ValueError(
                "FullHayateQuantNativeImpl is the fixed HQ backend; Adaptive HQ "
                "requires a cache-local selector and is not silently approximated"
            )
        if config.get("calibration"):
            raise ValueError("full HQ native currently supports data-free rotations only")
        if (
            int(config.get("n_outlier") or 0)
            or int(config.get("key_n_outlier") or 0)
            or int(config.get("value_n_outlier") or 0)
        ):
            raise ValueError("full HQ native fixed recipe does not support dynamic outlier masks")

        from onecomp.kvcomp.codec.hayate.codec import (  # pylint: disable=import-outside-toplevel
            HayateQuantCodec,
            _block_codebook_by_code_bits,
            _block_lut_code_bits_on,
        )

        # Use the public codec's resolver itself so "auto" keeps exactly the
        # paper's source/rate-matched dense/hybrid/Hadamard choice.
        resolver = HayateQuantCodec(
            bits=max(1, min(8, int(round(recipe.key_rate)))),
            value_bits=max(1, min(8, int(round(recipe.value_rate)))),
            dist=recipe.dist,
            block_size=None,
            max_codebook_bits=recipe.max_codebook_bits,
            codebook_init=recipe.codebook_init,
            rotation_mode=recipe.rotation_mode,
            encoder=recipe.assignment_mode,
            key_scale_mode="nc",
            value_scale_mode="ls_iter",
            seed=recipe.seed,
        )
        dev = torch.device(device)
        key_rotation = resolver._rt_rotation(  # pylint: disable=protected-access
            d, recipe.key_block_size, recipe.key_rate, dev, role="k"
        ).contiguous()
        value_rotation = resolver._rt_rotation(  # pylint: disable=protected-access
            d, recipe.value_block_size, recipe.value_rate, dev, role="v"
        ).contiguous()
        key_codebook = (
            _block_codebook_by_code_bits(
                recipe.key_code_bits_per_block,
                recipe.key_block_size,
                recipe.dist,
                d,
                recipe.codebook_init,
            )
            .to(device=dev, dtype=torch.float32)
            .contiguous()
        )
        value_codebook = (
            _block_codebook_by_code_bits(
                recipe.value_code_bits_per_block,
                recipe.value_block_size,
                recipe.dist,
                d,
                recipe.codebook_init,
            )
            .to(device=dev, dtype=torch.float32)
            .contiguous()
        )
        if recipe.assignment_mode == "lut":
            key_lut = _block_lut_code_bits_on(
                recipe.key_code_bits_per_block,
                recipe.key_block_size,
                recipe.dist,
                d,
                recipe.codebook_init,
                recipe.lut_bins,
                recipe.lut_candidates,
                str(dev),
            )
            value_lut = _block_lut_code_bits_on(
                recipe.value_code_bits_per_block,
                recipe.value_block_size,
                recipe.dist,
                d,
                recipe.codebook_init,
                recipe.lut_bins,
                recipe.lut_candidates,
                str(dev),
            )
        else:
            key_lut = value_lut = None
        # LUT candidate ranking deliberately uses the same cached scoring
        # dtype/norm arithmetic as the canonical all-codeword search.
        key_scoring_codebook = key_codebook.half()
        value_scoring_codebook = value_codebook.half()
        key_scoring_csq = key_scoring_codebook.pow(2).sum(dim=1)
        value_scoring_csq = value_scoring_codebook.pow(2).sum(dim=1)

        layer._hq_full_recipe = recipe
        layer._hq_full_key_rotation = key_rotation
        layer._hq_full_value_rotation = value_rotation
        layer._hq_full_key_codebook = key_codebook
        layer._hq_full_value_codebook = value_codebook
        layer._hq_full_key_lut = key_lut
        layer._hq_full_value_lut = value_lut
        layer._hq_full_key_scoring_codebook = key_scoring_codebook
        layer._hq_full_key_scoring_csq = key_scoring_csq
        layer._hq_full_value_scoring_codebook = value_scoring_codebook
        layer._hq_full_value_scoring_csq = value_scoring_csq
        layer._hq_full_native_enabled = True
        # Base forward fetches these three fields before dispatching into our
        # override.  Aliasing them avoids changing upstream control flow.
        layer._tq_PiT = key_rotation
        layer._tq_Pi = key_rotation.t().contiguous()
        layer._tq_centroids = key_codebook
        layer._tq_cached = True
        layer._hq_full_cached = True

    def _store_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer: Any,
    ):
        if not getattr(layer, "_hq_full_native_enabled", False):
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
        triton_full_hayatequant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            layer._hq_full_key_rotation,
            layer._hq_full_value_rotation,
            layer._hq_full_key_codebook,
            layer._hq_full_value_codebook,
            layer._hq_full_recipe,
            key_lut=layer._hq_full_key_lut,
            value_lut=layer._hq_full_value_lut,
            key_scoring_codebook=layer._hq_full_key_scoring_codebook,
            key_scoring_csq=layer._hq_full_key_scoring_csq,
            value_scoring_codebook=layer._hq_full_value_scoring_codebook,
            value_scoring_csq=layer._hq_full_value_scoring_csq,
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
        if layer is None or not getattr(layer, "_hq_full_native_enabled", False):
            return super()._decode_attention(
                query, kv_cache, attn_metadata, Pi, centroids, PiT, layer
            )
        bsz, hq, d = query.shape
        # The upstream TurboQuant backend keeps a fixed 32-way split for
        # CUDA-graph compatibility.  HQ's packed decode is also correct with
        # fewer reduction tiles, and short-context decode otherwise launches
        # many programs that immediately return.  vLLM exposes a CPU mirror of
        # sequence lengths, so selecting a bounded power-of-work split here
        # costs no device synchronisation.  Long contexts retain the upstream
        # cap; a missing CPU mirror conservatively falls back to the fixed
        # value.
        splits = self.max_num_kv_splits
        dynamic_splits = os.environ.get("KVCOMP_HQ_DYNAMIC_SPLITS", "1").lower()
        seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        if (
            dynamic_splits not in {"0", "false", "off", "no"}
            and seq_lens_cpu is not None
            and int(seq_lens_cpu.numel()) > 0
        ):
            max_seq_len = int(seq_lens_cpu.max().item())
            splits = min(self.max_num_kv_splits, max(1, math.ceil(max_seq_len / 256)))
        mid = rotated = lse = None
        if is_workspace_manager_initialized():
            mid, rotated, lse = current_workspace_manager().get_simultaneous(
                ((bsz, hq, splits, d + 1), torch.float32),
                ((bsz, hq, d), torch.float32),
                ((bsz, hq), torch.float32),
            )
        return triton_full_hayatequant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            key_rotation=layer._hq_full_key_rotation,
            value_rotation=layer._hq_full_value_rotation,
            key_codebook=layer._hq_full_key_codebook,
            value_codebook=layer._hq_full_value_codebook,
            recipe=layer._hq_full_recipe,
            scale=self.scale,
            mid_o_buf=mid,
            rotated_output_buf=rotated,
            lse_buf=lse,
            buf_holder=layer,
            max_num_kv_splits=splits,
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
        if layer is None or not getattr(layer, "_hq_full_native_enabled", False):
            return super()._prefill_attention(
                query, key, value, kv_cache, attn_metadata, Pi, centroids, PiT, layer
            )
        n, hq, d = query.shape
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
        hk = key.shape[1]
        use_gqa = hk < hq
        qloc = attn_metadata.query_start_loc
        nreq = qloc.shape[0] - 1
        output = torch.zeros(n, hq, d, device=query.device, dtype=query.dtype)
        qsl = (
            attn_metadata.query_start_loc_cpu.tolist()
            if attn_metadata.query_start_loc_cpu is not None
            else qloc.tolist()
        )
        slens = (
            attn_metadata.seq_lens_cpu.tolist()
            if attn_metadata.seq_lens_cpu is not None
            else attn_metadata.seq_lens.tolist()
        )
        if not hasattr(self, "_hq_full_cu2"):
            self._hq_full_cu2 = torch.zeros(2, device=query.device, dtype=torch.int32)
        max_seq = attn_metadata.max_seq_len
        arange = getattr(self, "_hq_full_arange", None)
        if arange is None or arange.shape[0] <= max_seq:
            arange = torch.arange(
                max_seq + 1, device=query.device, dtype=attn_metadata.seq_lens.dtype
            )
            self._hq_full_arange = arange
        for i in range(nreq):
            qs, qe = qsl[i], qsl[i + 1]
            qlen = qe - qs
            if qlen <= 0:
                continue
            seqlen = slens[i]
            qseq, kseq, vseq = query[qs:qe], key[qs:qe], value[qs:qe]
            if qlen == seqlen:
                if _HAS_FLASH_ATTN:
                    self._hq_full_cu2[1:2] = qlen
                    out = self._flash_attn_varlen(
                        q=qseq,
                        k=kseq,
                        v=vseq,
                        cu_seqlens_q=self._hq_full_cu2,
                        cu_seqlens_k=self._hq_full_cu2,
                        max_seqlen_q=qlen,
                        max_seqlen_k=qlen,
                    )
                else:
                    out = F.scaled_dot_product_attention(
                        qseq.transpose(0, 1).contiguous(),
                        kseq.transpose(0, 1).contiguous(),
                        vseq.transpose(0, 1).contiguous(),
                        is_causal=True,
                        scale=self.scale,
                        enable_gqa=use_gqa,
                    ).transpose(0, 1)
                output[qs:qe] = out.to(query.dtype)
                continue
            cached = seqlen - qlen
            if qlen <= _CONTINUATION_DECODE_THRESHOLD:
                synth_lens = arange[cached + 1 : seqlen + 1]
                synth_bt = attn_metadata.block_table[i : i + 1].expand(qlen, -1)
                meta = TurboQuantMetadata(
                    seq_lens=synth_lens,
                    slot_mapping=attn_metadata.slot_mapping[qs:qe],
                    block_table=synth_bt,
                    query_start_loc=torch.arange(qlen + 1, device=query.device, dtype=qloc.dtype),
                    num_actual_tokens=qlen,
                    max_query_len=1,
                    max_seq_len=seqlen,
                    is_prefill=False,
                    num_decodes=qlen,
                    num_decode_tokens=qlen,
                )
                out = self._decode_attention(qseq, kv_cache, meta, Pi, centroids, PiT, layer)
            else:
                out = self._continuation_prefill(
                    layer,
                    qseq,
                    kseq,
                    vseq,
                    kv_cache,
                    attn_metadata.block_table[i : i + 1],
                    cached,
                    seqlen,
                    Pi,
                    centroids,
                )
            output[qs:qe] = out.to(query.dtype)
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
        if not getattr(layer, "_hq_full_native_enabled", False):
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
        qlen, hq, d = query.shape
        hk = key_chunk.shape[1]
        page = kv_cache.shape[1]
        alloc_len = math.ceil(cached_len / page) * page
        shape = (1, hk, alloc_len, d)
        kbuf, vbuf = current_workspace_manager().get_simultaneous(
            (shape, query.dtype), (shape, query.dtype)
        )
        triton_full_hayatequant_dequant_kv(
            kv_cache=kv_cache,
            block_table=block_table,
            key_codebook=layer._hq_full_key_codebook,
            value_codebook=layer._hq_full_value_codebook,
            key_rotation=layer._hq_full_key_rotation,
            value_rotation=layer._hq_full_value_rotation,
            recipe=layer._hq_full_recipe,
            k_out=kbuf,
            v_out=vbuf,
        )
        kcached = kbuf[0, :, :cached_len, :].transpose(0, 1)
        vcached = vbuf[0, :, :cached_len, :].transpose(0, 1)
        kfull = torch.empty(seq_len, hk, d, device=query.device, dtype=query.dtype)
        vfull = torch.empty_like(kfull)
        kfull[:cached_len], kfull[cached_len:] = kcached, key_chunk
        vfull[:cached_len], vfull[cached_len:] = vcached, val_chunk
        if _HAS_FLASH_ATTN:
            if not hasattr(self, "_hq_full_cu2q"):
                self._hq_full_cu2q = torch.zeros(2, device=query.device, dtype=torch.int32)
                self._hq_full_cu2k = torch.zeros(2, device=query.device, dtype=torch.int32)
            self._hq_full_cu2q[1:2] = qlen
            self._hq_full_cu2k[1:2] = seq_len
            return self._flash_attn_varlen(
                q=query,
                k=kfull,
                v=vfull,
                cu_seqlens_q=self._hq_full_cu2q,
                cu_seqlens_k=self._hq_full_cu2k,
                max_seqlen_q=qlen,
                max_seqlen_k=seq_len,
            )
        qpos = torch.arange(qlen, device=query.device).unsqueeze(1) + cached_len
        kpos = torch.arange(seq_len, device=query.device).unsqueeze(0)
        out = F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            kfull.transpose(0, 1).unsqueeze(0),
            vfull.transpose(0, 1).unsqueeze(0),
            attn_mask=kpos <= qpos,
            scale=self.scale,
            enable_gqa=(hk < hq),
        )
        return out[0].transpose(0, 1)


__all__ = [
    "FULL_CONFIG_ENV",
    "configure_full_hq_native_backend",
    "FullHayateQuantNativeBackend",
    "FullHayateQuantNativeImpl",
]

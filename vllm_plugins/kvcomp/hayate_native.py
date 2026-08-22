# Copyright 2026 Fujitsu Ltd.
"""Packed canonical HayateQuant backend for vLLM.

The plugin uses vLLM's TURBOQUANT enum only as a carrier.  This backend replaces
cache allocation, store, prefill, and decode, so the physical data plane and
both K/V quantizers are canonical HayateQuant.
"""

from __future__ import annotations

import json
import math
import os
import re
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

from vllm_plugins.kvcomp.hayate_config import (
    HAYATEQUANT_LUT_BINS,
    HAYATEQUANT_LUT_CANDIDATES,
    HayateQuantNativeRecipe,
    hayatequant_native_runtime_fields,
    hayatequant_native_recipe_from_mapping,
)
from vllm_plugins.kvcomp.hayate_triton import (
    triton_hayatequant_decode_attention,
    triton_hayatequant_dequant_kv,
    triton_hayatequant_store,
)
from vllm_plugins.kvcomp.runtime_evidence import emit_runtime_evidence_once

HAYATEQUANT_CONFIG_ENV = "KVCOMP_VLLM_HAYATEQUANT_CONFIG"
HAYATEQUANT_DISABLE_PROJECTION_FUSION_ENV = "KVCOMP_VLLM_HAYATEQUANT_DISABLE_PROJECTION_FUSION"
_HAYATE_PROJECTION_CLASSES_PATCHED = False


def install_hayatequant_projection_fusion() -> bool:
    """Link supported attention modules to their output projection.

    HQ reconstructs values in a rotated basis.  For an unquantized row-
    parallel output projection, ``(x R^T) W^T == x (W R)^T``.  Linking the
    sibling modules lets the backend fold ``R`` into ``W`` once after weight
    loading and remove one small GEMM from every layer and decode token.
    Unsupported architectures and quantized projections retain the ordinary
    inverse-rotation path.
    """

    global _HAYATE_PROJECTION_CLASSES_PATCHED
    if _HAYATE_PROJECTION_CLASSES_PATCHED:
        return True

    candidates = (
        ("vllm.model_executor.models.llama", "LlamaAttention"),
        ("vllm.model_executor.models.qwen2", "Qwen2Attention"),
        ("vllm.model_executor.models.qwen3", "Qwen3Attention"),
    )
    patched = 0
    for module_name, class_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            owner_cls = getattr(module, class_name)
        except (ImportError, AttributeError):  # pragma: no cover - vLLM version dependent
            continue
        if getattr(owner_cls, "_onecomp_hayate_projection_linked", False):
            patched += 1
            continue
        original_init = owner_cls.__init__

        def _linked_init(self, *args, __original=original_init, __name=class_name, **kwargs):
            __original(self, *args, **kwargs)
            attention = getattr(self, "attn", None)
            projection = getattr(self, "o_proj", None)
            if attention is not None and projection is not None:
                attention._hayate_output_projection = projection
                attention._hayate_qkv_projection = getattr(self, "qkv_proj", None)
                attention._hayate_q_size = getattr(self, "q_size", None)
                attention._hayate_kv_size = getattr(self, "kv_size", None)
                attention._hayate_projection_owner = __name
                implementation = getattr(attention, "impl", None)
                if implementation is not None:
                    implementation._hayate_layer = attention

        _linked_init._onecomp_hayate_original = original_init
        owner_cls.__init__ = _linked_init
        owner_cls._onecomp_hayate_projection_linked = True
        patched += 1
    _HAYATE_PROJECTION_CLASSES_PATCHED = patched > 0
    return _HAYATE_PROJECTION_CLASSES_PATCHED


def _fold_value_rotation_into_projection(
    layer: Any,
    value_rotation: torch.Tensor,
    head_dim: int,
) -> bool:
    """Fold one per-head value rotation into a safe dense output weight."""

    if os.environ.get(HAYATEQUANT_DISABLE_PROJECTION_FUSION_ENV) == "1":
        return False
    projection = getattr(layer, "_hayate_output_projection", None)
    if projection is None:
        return False
    if getattr(projection, "_hayate_value_rotation_folded", False):
        return True
    quant_method = getattr(projection, "quant_method", None)
    if quant_method is None or quant_method.__class__.__name__ != "UnquantizedLinearMethod":
        return False
    weight = getattr(projection, "weight", None)
    if (
        weight is None
        or weight.ndim != 2
        or not weight.is_cuda
        or not weight.dtype.is_floating_point
        or weight.shape[1] % head_dim
        or value_rotation.shape != (head_dim, head_dim)
    ):
        return False
    with torch.no_grad():
        output_size, local_input_size = weight.shape
        local_heads = local_input_size // head_dim
        folded = (
            weight.detach().reshape(output_size * local_heads, head_dim).float() @ value_rotation
        )
        weight.copy_(folded.reshape_as(weight).to(weight.dtype))
    projection._hayate_value_rotation_folded = True
    projection._hayate_value_rotation_head_dim = int(head_dim)
    return True


def _fold_value_rotation_into_qkv_projection(
    layer: Any,
    value_rotation: torch.Tensor,
    head_dim: int,
) -> bool:
    """Make the fused QKV projection emit values in HQ's rotated basis.

    Values do not pass through RoPE, so ``v R`` can be produced once by
    replacing each value-head weight ``W_v`` with ``R^T W_v``.  Query and key
    weights deliberately remain untouched because their Hayate rotation is
    applied after position-dependent RoPE and cannot be folded safely.
    """

    if os.environ.get(HAYATEQUANT_DISABLE_PROJECTION_FUSION_ENV) == "1":
        return False
    projection = getattr(layer, "_hayate_qkv_projection", None)
    if projection is None:
        return False
    if getattr(projection, "_hayate_value_input_rotation_folded", False):
        return True
    quant_method = getattr(projection, "quant_method", None)
    if quant_method is None or quant_method.__class__.__name__ != "UnquantizedLinearMethod":
        return False
    q_size = getattr(layer, "_hayate_q_size", None)
    kv_size = getattr(layer, "_hayate_kv_size", None)
    weight = getattr(projection, "weight", None)
    if (
        not isinstance(q_size, int)
        or not isinstance(kv_size, int)
        or q_size <= 0
        or kv_size <= 0
        or kv_size % head_dim
        or weight is None
        or weight.ndim != 2
        or not weight.is_cuda
        or not weight.dtype.is_floating_point
        or weight.shape[0] != q_size + 2 * kv_size
        or value_rotation.shape != (head_dim, head_dim)
    ):
        return False

    value_start = q_size + kv_size
    local_value_heads = kv_size // head_dim
    bias = getattr(projection, "bias", None)
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        return False
    with torch.no_grad():
        value_weight = (
            weight[value_start : value_start + kv_size]
            .detach()
            .reshape(local_value_heads, head_dim, weight.shape[1])
        )
        folded_weight = torch.matmul(value_rotation.t(), value_weight.float())
        weight[value_start : value_start + kv_size].copy_(
            folded_weight.reshape(kv_size, weight.shape[1]).to(weight.dtype)
        )
        if bias is not None:
            value_bias = (
                bias[value_start : value_start + kv_size]
                .detach()
                .reshape(local_value_heads, head_dim)
            )
            bias[value_start : value_start + kv_size].copy_(
                (value_bias.float() @ value_rotation).reshape(kv_size).to(bias.dtype)
            )
    projection._hayate_value_input_rotation_folded = True
    projection._hayate_value_input_rotation_head_dim = int(head_dim)
    return True


def _to_folded_value_basis(output: torch.Tensor, layer: Any) -> torch.Tensor:
    """Convert an ordinary prefill result to the projection-folded basis."""

    if not getattr(layer, "_hayate_projection_folded", False):
        return output
    if getattr(layer, "_hayate_value_input_folded", False):
        return output
    return (output.float() @ layer._hayate_value_rotation).to(output.dtype)


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


def _all_first_chunk_prefills(
    query_start_locations: list[int],
    sequence_lengths: list[int],
) -> bool:
    """Return true only when every request has no previously cached tokens."""

    if len(query_start_locations) != len(sequence_lengths) + 1:
        raise ValueError("query starts and sequence lengths have incompatible sizes")
    for index, sequence_length in enumerate(sequence_lengths):
        query_length = query_start_locations[index + 1] - query_start_locations[index]
        if query_length < 0 or sequence_length < query_length:
            raise ValueError("invalid query/sequence lengths in HayateQuant metadata")
        if query_length != sequence_length:
            return False
    return True


def _native_config() -> dict[str, Any]:
    raw = os.environ.get(HAYATEQUANT_CONFIG_ENV)
    if not raw:
        return {}
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {HAYATEQUANT_CONFIG_ENV}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{HAYATEQUANT_CONFIG_ENV} must contain a JSON object")
    return config


def _native_recipe() -> HayateQuantNativeRecipe | None:
    config = _native_config()
    return hayatequant_native_recipe_from_mapping(config) if config else None


def configure_hayatequant_native_backend(config: dict[str, Any]) -> HayateQuantNativeRecipe:
    """Validate, export, and register HayateQuant before constructing ``LLM``.

    The environment payload propagates to vLLM EngineCore spawn children,
    where the general OneComp plugin re-registers the same backend.  Call this
    only in a fresh process; backend registries are process-global.
    """

    from vllm_plugins.kvcomp.native_turboquant import (  # pylint: disable=import-outside-toplevel
        disable_vllm_implicit_turboquant_boundary_skip,
    )

    recipe = hayatequant_native_recipe_from_mapping(config)
    disable_vllm_implicit_turboquant_boundary_skip()
    install_hayatequant_projection_fusion()
    os.environ[HAYATEQUANT_CONFIG_ENV] = json.dumps(config, sort_keys=True)
    from vllm.v1.attention.backends.registry import (  # pylint: disable=import-outside-toplevel
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.TURBOQUANT,
        "vllm_plugins.kvcomp.hayate_native.HayateQuantNativeBackend",
    )
    # vLLM 0.20's cache-memory spec is independent of backend shape; install
    # the explicit-process hook in the launcher as well as in spawn children.
    from vllm_plugins.kvcomp.vllm_plugin import (  # pylint: disable=import-outside-toplevel
        _install_hayatequant_cache_spec_hook,
    )

    _install_hayatequant_cache_spec_hook()
    return recipe


class HayateQuantNativeBackend(TurboQuantAttentionBackend):
    """vLLM backend owning the canonical packed HayateQuant cache layout."""

    @staticmethod
    def get_impl_cls() -> type["HayateQuantNativeImpl"]:
        return HayateQuantNativeImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "turboquant_3bit_nc",
    ) -> tuple[int, ...]:
        recipe = _native_recipe()
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


class HayateQuantNativeImpl(TurboQuantAttentionImpl):
    """Canonical HayateQuant store/decode implementation for both K and V."""

    @staticmethod
    def _onecomp_native_enabled(layer: Any) -> bool:
        """Allow sibling packed backends to reuse the robust prefill path."""

        return bool(getattr(layer, "_hayate_native_enabled", False))

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """Fold linked dense projections after loading and before first QKV."""

        super().process_weights_after_loading(act_dtype)
        layer = getattr(self, "_hayate_layer", None)
        projection = getattr(layer, "_hayate_output_projection", None)
        weight = getattr(projection, "weight", None)
        if layer is not None and weight is not None and weight.is_cuda:
            self._ensure_on_device(layer, weight.device)

    def _ensure_on_device(self, layer, device):
        if hasattr(layer, "_hayate_cached"):
            return
        recipe = _native_recipe()
        if recipe is None:
            layer._hayate_native_enabled = False
            super()._ensure_on_device(layer, device)
            layer._hayate_cached = True
            return

        d = self.head_size
        recipe.validate_head_dim(d)

        from onecomp.kvcomp.codec.hayate.codec import (  # pylint: disable=import-outside-toplevel
            HayateQuantCodec,
            _block_codebook_by_code_bits,
            _block_lut_code_bits_on,
        )

        # Use the public codec's resolver itself so "auto" keeps exactly the
        # paper's source/rate-matched dense/hybrid/Hadamard choice.
        resolver = HayateQuantCodec(bits=int(recipe.average_rate))
        dev = torch.device(device)
        key_rotation = resolver._rt_rotation(  # pylint: disable=protected-access
            d, recipe.key_block_size, recipe.key_rate, dev, role="k"
        ).contiguous()
        value_rotation = resolver._rt_rotation(  # pylint: disable=protected-access
            d, recipe.value_block_size, recipe.value_rate, dev, role="v"
        ).contiguous()
        key_codebook = (
            _block_codebook_by_code_bits(recipe.key_code_bits_per_block)
            .to(device=dev, dtype=torch.float32)
            .contiguous()
        )
        value_codebook = (
            _block_codebook_by_code_bits(recipe.value_code_bits_per_block)
            .to(device=dev, dtype=torch.float32)
            .contiguous()
        )
        key_lut = _block_lut_code_bits_on(
            recipe.key_code_bits_per_block,
            HAYATEQUANT_LUT_BINS,
            HAYATEQUANT_LUT_CANDIDATES,
            str(dev),
        )
        value_lut = _block_lut_code_bits_on(
            recipe.value_code_bits_per_block,
            HAYATEQUANT_LUT_BINS,
            HAYATEQUANT_LUT_CANDIDATES,
            str(dev),
        )
        # LUT candidate ranking deliberately uses the same cached scoring
        # dtype/norm arithmetic as the canonical all-codeword search.
        key_scoring_codebook = key_codebook.half()
        value_scoring_codebook = value_codebook.half()
        key_scoring_csq = key_scoring_codebook.pow(2).sum(dim=1)
        value_scoring_csq = value_scoring_codebook.pow(2).sum(dim=1)

        layer._hayate_recipe = recipe
        layer._hayate_layer_index = _layer_index(layer)
        layer._hayate_key_rotation = key_rotation
        layer._hayate_value_rotation = value_rotation
        layer._hayate_key_codebook = key_codebook
        layer._hayate_value_codebook = value_codebook
        layer._hayate_key_lut = key_lut
        layer._hayate_value_lut = value_lut
        layer._hayate_key_scoring_codebook = key_scoring_codebook
        layer._hayate_key_scoring_csq = key_scoring_csq
        layer._hayate_value_scoring_codebook = value_scoring_codebook
        layer._hayate_value_scoring_csq = value_scoring_csq
        # The store hot path must only share a rotation when the two immutable
        # role tables literally alias.  Resolve the inexpensive identity check
        # once while materialising a layer, never by comparing GPU contents.
        layer._hayate_shared_rotation = bool(
            key_rotation.device == value_rotation.device
            and key_rotation.dtype == value_rotation.dtype
            and key_rotation.shape == value_rotation.shape
            and key_rotation.data_ptr() == value_rotation.data_ptr()
        )
        layer._hayate_projection_folded = _fold_value_rotation_into_projection(
            layer,
            value_rotation,
            d,
        )
        layer._hayate_value_input_folded = bool(
            layer._hayate_projection_folded
            and _fold_value_rotation_into_qkv_projection(layer, value_rotation, d)
        )
        layer._hayate_native_enabled = True
        # Base forward fetches these three fields before dispatching into our
        # override.  Aliasing them avoids changing upstream control flow.
        layer._tq_PiT = key_rotation
        layer._tq_Pi = key_rotation.t().contiguous()
        layer._tq_centroids = key_codebook
        layer._tq_cached = True
        layer._hayate_cached = True

    def _store_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer: Any,
    ):
        if not getattr(layer, "_hayate_native_enabled", False):
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
            layer._hayate_key_rotation,
            layer._hayate_value_rotation,
            layer._hayate_key_codebook,
            layer._hayate_value_codebook,
            layer._hayate_recipe,
            key_lut=layer._hayate_key_lut,
            value_lut=layer._hayate_value_lut,
            key_scoring_codebook=layer._hayate_key_scoring_codebook,
            key_scoring_csq=layer._hayate_key_scoring_csq,
            value_scoring_codebook=layer._hayate_value_scoring_codebook,
            value_scoring_csq=layer._hayate_value_scoring_csq,
            shared_rotation=layer._hayate_shared_rotation,
            value_in_rotated_basis=layer._hayate_value_input_folded,
            return_payload=False,
            require_fused=True,
        )
        emit_runtime_evidence_once(
            self,
            "hayatequant_store",
            {
                "backend": ("vllm_plugins.kvcomp.hayate_native." "HayateQuantNativeBackend"),
                "implementation": "native_packed_hayatequant",
                "event": "packed_hayatequant_store_launched",
                "store_encoder": "triton_fused_b2_k_nc_v_ls_iter_encode_pack",
                "layer_index": layer._hayate_layer_index,
                **hayatequant_native_runtime_fields(layer._hayate_recipe, int(key.shape[-1])),
                "cache_shape": list(kv_cache.shape),
                "cache_dtype": str(kv_cache.dtype).removeprefix("torch."),
                "cache_device": str(kv_cache.device),
                "shared_rotation_fast_path": bool(
                    layer._hayate_shared_rotation and not layer._hayate_value_input_folded
                ),
                "value_projection_fused": bool(layer._hayate_value_input_folded),
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
        bsz, hq, d = query.shape
        # Keep the split grid fixed for CUDA-graph replay. On B200, the full
        # 32-way grid is also fastest at the formal 4K serving context; dynamic
        # sequence-dependent grids create extra specializations and can force
        # graph recapture after the prompt boundary.
        splits = self.max_num_kv_splits
        projection_folded = bool(getattr(layer, "_hayate_projection_folded", False))
        mid = qrot = rotated = lse = None
        if is_workspace_manager_initialized():
            mid, qrot, rotated, lse = current_workspace_manager().get_simultaneous(
                ((bsz, hq, splits, d + 1), torch.float32),
                ((bsz, hq, d), torch.float32),
                ((bsz, hq, d), query.dtype if projection_folded else torch.float32),
                ((bsz, hq), torch.float32),
            )
        result = triton_hayatequant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            key_rotation=layer._hayate_key_rotation,
            value_rotation=layer._hayate_value_rotation,
            key_codebook=layer._hayate_key_codebook,
            value_codebook=layer._hayate_value_codebook,
            recipe=layer._hayate_recipe,
            scale=self.scale,
            mid_o_buf=mid,
            query_rotation_buf=qrot,
            rotated_output_buf=rotated,
            lse_buf=lse,
            buf_holder=layer,
            max_num_kv_splits=splits,
            output_in_rotated_basis=projection_folded,
        )
        emit_runtime_evidence_once(
            self,
            "hayatequant_decode",
            {
                "backend": ("vllm_plugins.kvcomp.hayate_native." "HayateQuantNativeBackend"),
                "implementation": "native_packed_hayatequant",
                "event": "packed_hayatequant_decode_attention_launched",
                "layer_index": layer._hayate_layer_index,
                **hayatequant_native_runtime_fields(layer._hayate_recipe, int(query.shape[-1])),
                "cache_shape": list(kv_cache.shape),
                "cache_dtype": str(kv_cache.dtype).removeprefix("torch."),
                "cache_device": str(kv_cache.device),
                "query_shape": list(query.shape),
                "kv_splits": int(splits),
                "output_projection_fused": projection_folded,
                "value_projection_fused": bool(layer._hayate_value_input_folded),
            },
        )
        return result

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
        if layer is None or not self._onecomp_native_enabled(layer):
            return super()._prefill_attention(
                query, key, value, kv_cache, attn_metadata, Pi, centroids, PiT, layer
            )
        n, hq, d = query.shape
        qloc = attn_metadata.query_start_loc
        # TurboQuantMetadata owns only the device tensors in vLLM's public
        # backend contract. Convert each vector once, matching the upstream
        # continuation-prefill implementation.
        qsl = qloc.tolist()
        slens = attn_metadata.seq_lens.tolist()
        # Equality of only the two maxima is insufficient for a mixed batch:
        # a long fresh prompt can hide a shorter continuation request. Check
        # every CPU-resident request length before ignoring the packed cache.
        if _HAS_FLASH_ATTN and _all_first_chunk_prefills(qsl, slens):
            out = self._flash_attn_varlen(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
            )
            return _to_folded_value_basis(out, layer)
        hk = key.shape[1]
        use_gqa = hk < hq
        nreq = qloc.shape[0] - 1
        output = torch.zeros(n, hq, d, device=query.device, dtype=query.dtype)
        if not hasattr(self, "_hayate_cu2"):
            self._hayate_cu2 = torch.zeros(2, device=query.device, dtype=torch.int32)
        max_seq = attn_metadata.max_seq_len
        arange = getattr(self, "_hayate_arange", None)
        if arange is None or arange.shape[0] <= max_seq:
            arange = torch.arange(
                max_seq + 1, device=query.device, dtype=attn_metadata.seq_lens.dtype
            )
            self._hayate_arange = arange
        for i in range(nreq):
            qs, qe = qsl[i], qsl[i + 1]
            qlen = qe - qs
            if qlen <= 0:
                continue
            seqlen = slens[i]
            qseq, kseq, vseq = query[qs:qe], key[qs:qe], value[qs:qe]
            if qlen == seqlen:
                if _HAS_FLASH_ATTN:
                    self._hayate_cu2[1:2] = qlen
                    out = self._flash_attn_varlen(
                        q=qseq,
                        k=kseq,
                        v=vseq,
                        cu_seqlens_q=self._hayate_cu2,
                        cu_seqlens_k=self._hayate_cu2,
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
                output[qs:qe] = _to_folded_value_basis(out, layer).to(query.dtype)
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
        qlen, hq, d = query.shape
        hk = key_chunk.shape[1]
        page = kv_cache.shape[1]
        alloc_len = math.ceil(cached_len / page) * page
        shape = (1, hk, alloc_len, d)
        kbuf, vbuf = current_workspace_manager().get_simultaneous(
            (shape, query.dtype), (shape, query.dtype)
        )
        triton_hayatequant_dequant_kv(
            kv_cache=kv_cache,
            block_table=block_table,
            key_codebook=layer._hayate_key_codebook,
            value_codebook=layer._hayate_value_codebook,
            key_rotation=layer._hayate_key_rotation,
            value_rotation=layer._hayate_value_rotation,
            recipe=layer._hayate_recipe,
            k_out=kbuf,
            v_out=vbuf,
            value_in_rotated_basis=layer._hayate_value_input_folded,
        )
        kcached = kbuf[0, :, :cached_len, :].transpose(0, 1)
        vcached = vbuf[0, :, :cached_len, :].transpose(0, 1)
        kfull = torch.empty(seq_len, hk, d, device=query.device, dtype=query.dtype)
        vfull = torch.empty_like(kfull)
        kfull[:cached_len], kfull[cached_len:] = kcached, key_chunk
        vfull[:cached_len], vfull[cached_len:] = vcached, val_chunk
        if _HAS_FLASH_ATTN:
            if not hasattr(self, "_hayate_cu2q"):
                self._hayate_cu2q = torch.zeros(2, device=query.device, dtype=torch.int32)
                self._hayate_cu2k = torch.zeros(2, device=query.device, dtype=torch.int32)
            self._hayate_cu2q[1:2] = qlen
            self._hayate_cu2k[1:2] = seq_len
            out = self._flash_attn_varlen(
                q=query,
                k=kfull,
                v=vfull,
                cu_seqlens_q=self._hayate_cu2q,
                cu_seqlens_k=self._hayate_cu2k,
                max_seqlen_q=qlen,
                max_seqlen_k=seq_len,
            )
            return _to_folded_value_basis(out, layer)
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
        return _to_folded_value_basis(out[0].transpose(0, 1), layer)


__all__ = [
    "HAYATEQUANT_CONFIG_ENV",
    "HAYATEQUANT_DISABLE_PROJECTION_FUSION_ENV",
    "_all_first_chunk_prefills",
    "configure_hayatequant_native_backend",
    "install_hayatequant_projection_fusion",
    "HayateQuantNativeBackend",
    "HayateQuantNativeImpl",
]

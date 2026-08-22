"""Standalone GPU TaperKV backend for vLLM.

The backend applies TaperKV's age schedule directly to vLLM paged-cache values
on GPU. Precision only decreases for a resident slot/page, preventing repeated
same-width fake quantization drift. The current allocator remains model dtype;
runtime evidence therefore labels this a semantic, not packed-memory, backend.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import torch
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
)

from vllm_plugins.kvcomp.runtime_evidence import emit_runtime_evidence_once
from vllm_plugins.kvcomp.taper_config import TaperKVVLLMRecipe

TAPER_CONFIG_ENV = "KVCOMP_VLLM_TAPER_CONFIG"
_RECIPE_MARKER: str | None = None
_RECIPE: TaperKVVLLMRecipe | None = None


def _upstream_kv_axis() -> int:
    probe = FlashAttentionBackend.get_kv_cache_shape(3, 16, 1, 16)
    if probe[:2] == (3, 2):
        return 1
    if probe[:2] == (2, 3):
        return 0
    raise RuntimeError(f"unsupported vLLM FlashAttention cache shape: {probe}")


_UPSTREAM_KV_AXIS = _upstream_kv_axis()


def taperkv_recipe() -> TaperKVVLLMRecipe:
    """Load and cache the validated process-local TaperKV recipe."""

    global _RECIPE_MARKER, _RECIPE
    raw = os.environ.get(TAPER_CONFIG_ENV)
    if not raw:
        raise RuntimeError("TaperKV backend selected without a recipe payload")
    if raw == _RECIPE_MARKER and _RECIPE is not None:
        return _RECIPE
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("invalid JSON in KVCOMP_VLLM_TAPER_CONFIG") from exc
    _RECIPE = TaperKVVLLMRecipe.from_mapping(payload)
    _RECIPE_MARKER = raw
    return _RECIPE


def _batched_taper_bits(
    positions: torch.Tensor,
    lengths: torch.Tensor,
    high_bits: int,
    floor_bits: int,
    sink_tokens: int,
    recent_tokens: int,
    ladder: tuple[int, ...] | torch.Tensor,
) -> torch.Tensor:
    """Vectorized age schedule for positions shaped [batch, logical_tokens]."""

    if positions.ndim != 2 or lengths.ndim != 1 or positions.shape[0] != lengths.shape[0]:
        raise ValueError("positions/lengths have incompatible shapes")
    age = lengths[:, None] - 1 - positions
    protected = (positions < sink_tokens) | (age < recent_tokens)
    low = torch.full_like(lengths[:, None], recent_tokens)
    high = torch.maximum(low + 1, lengths[:, None] - sink_tokens)
    fraction = ((age - low).float() / (high - low).clamp_min(1).float()).clamp(
        0.0,
        1.0 - 1e-6,
    )
    bands = (fraction * len(ladder)).to(torch.int64).clamp(0, len(ladder) - 1)
    ladder_tensor = (
        ladder
        if isinstance(ladder, torch.Tensor)
        else torch.as_tensor(ladder, dtype=torch.int64, device=positions.device)
    )
    if ladder_tensor.device != positions.device or ladder_tensor.dtype != torch.int64:
        raise ValueError("ladder tensor must be int64 on the schedule device")
    widths = ladder_tensor[bands]
    return torch.where(
        protected,
        torch.full_like(widths, high_bits),
        widths,
    ).clamp(min=floor_bits, max=high_bits)


def _fake_quantize_variable_tokens(
    rows: torch.Tensor,
    bits: torch.Tensor,
) -> torch.Tensor:
    """Per-token/head affine fake quantization with vector-valued widths."""

    if rows.ndim != 3 or bits.ndim != 1 or rows.shape[0] != bits.shape[0]:
        raise ValueError("rows must be [tokens, heads, dim] with one width per token")
    source = rows.float()
    qmax = torch.pow(2.0, bits.float()).sub_(1.0)[:, None, None]
    minimum = source.amin(dim=-1, keepdim=True)
    maximum = source.amax(dim=-1, keepdim=True)
    scale = ((maximum - minimum) / qmax).clamp_min_(1e-8)
    codes = torch.round((source - minimum) / scale)
    codes = torch.minimum(torch.maximum(codes, torch.zeros_like(codes)), qmax)
    return (codes * scale + minimum).to(rows.dtype)


def _fake_quantize_variable_pages(
    pages: torch.Tensor,
    bits: torch.Tensor,
) -> torch.Tensor:
    """One affine scale/zero per physical page and KV head."""

    if pages.ndim != 4 or bits.ndim != 1 or pages.shape[0] != bits.shape[0]:
        raise ValueError("pages must be [pages, page_size, heads, dim]")
    page_count, page_size, heads, dim = pages.shape
    source = (
        pages.permute(0, 2, 1, 3)
        .float()
        .reshape(
            page_count,
            heads,
            page_size * dim,
        )
    )
    qmax = torch.pow(2.0, bits.float()).sub_(1.0)[:, None, None]
    minimum = source.amin(dim=-1, keepdim=True)
    maximum = source.amax(dim=-1, keepdim=True)
    scale = ((maximum - minimum) / qmax).clamp_min_(1e-8)
    codes = torch.round((source - minimum) / scale)
    codes = torch.minimum(torch.maximum(codes, torch.zeros_like(codes)), qmax)
    quantized = (codes * scale + minimum).reshape(
        page_count,
        heads,
        page_size,
        dim,
    )
    return quantized.permute(0, 2, 1, 3).to(pages.dtype)


def _assert_nonnegative(indices: torch.Tensor, message: str) -> None:
    condition = torch.all(indices >= 0)
    if hasattr(torch, "_assert_async"):
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise RuntimeError(message)


def _assert_unique_indices(indices: torch.Tensor, size: int, message: str) -> None:
    """Fail when one mutable physical entry is shared by multiple requests."""

    del size
    if indices.numel() < 2:
        return
    # Decode only mutates O(batch * ladder-bands) entries. Sorting that short
    # list avoids allocating and clearing one counter per physical cache entry.
    ordered = torch.sort(indices).values
    condition = torch.all(ordered[1:] != ordered[:-1])
    if hasattr(torch, "_assert_async"):
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise RuntimeError(message)


def _transition_candidate_positions(
    lengths: torch.Tensor,
    *,
    ladder_count: int,
    sink_tokens: int,
    recent_tokens: int,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the only logical positions that can cross ladder boundaries.

    For a fixed token, TaperKV's middle-region fraction is monotone in the
    sequence length. During ``stride`` one-token decode updates, each of the
    ``ladder_count - 1`` boundaries can move by at most ``stride`` positions.
    No other resident token can therefore require a lower width.
    """

    if lengths.ndim != 1:
        raise ValueError("lengths must be one-dimensional")
    if ladder_count < 1 or stride < 1:
        raise ValueError("ladder_count and stride must be positive")
    batch = int(lengths.shape[0])
    if ladder_count == 1:
        empty = torch.empty((batch, 0), dtype=torch.long, device=lengths.device)
        return empty, torch.empty_like(empty, dtype=torch.bool)
    boundaries = torch.arange(
        1,
        ladder_count,
        dtype=torch.long,
        device=lengths.device,
    )[None, :, None]
    offsets = torch.arange(stride, dtype=torch.long, device=lengths.device)[None, None, :]
    denominator = (lengths - sink_tokens - recent_tokens).clamp_min(1)[:, None, None]
    # ceil(boundary * denominator / ladder_count), expressed with integers so
    # this candidate set exactly matches _batched_taper_bits' band edges.
    threshold_age = recent_tokens + torch.div(
        boundaries * denominator + ladder_count - 1,
        ladder_count,
        rounding_mode="floor",
    )
    positions = lengths[:, None, None] - 1 - threshold_age - offsets
    valid = (
        (positions >= sink_tokens)
        & (positions >= 0)
        & (positions < (lengths - recent_tokens)[:, None, None])
    )
    return positions.reshape(batch, -1), valid.reshape(batch, -1)


class TaperKVBackend(FlashAttentionBackend):
    """FlashAttention data plane with standalone TaperKV semantic regrading."""

    @staticmethod
    def get_impl_cls() -> type["TaperKVImpl"]:
        return TaperKVImpl


class TaperKVImpl(FlashAttentionImpl):
    """GPU-vectorized model-dtype TaperKV implementation."""

    @staticmethod
    def _split_kv_cache(
        kv_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if kv_cache.ndim != 5 or kv_cache.shape[_UPSTREAM_KV_AXIS] != 2:
            raise RuntimeError("TaperKV requires vLLM's five-dimensional paged cache")
        return kv_cache.unbind(_UPSTREAM_KV_AXIS)

    @staticmethod
    def _layer_index(layer: Any) -> int | None:
        for name in ("layer_idx", "layer_id", "layer_number"):
            value = getattr(layer, name, None)
            if isinstance(value, int):
                return value
        layer_name = getattr(layer, "layer_name", None)
        if isinstance(layer_name, str):
            matches = re.findall(
                r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)",
                layer_name,
            )
            if matches:
                return int(matches[-1])
        return None

    @classmethod
    def _layer_selection(
        cls,
        layer: Any,
        recipe: TaperKVVLLMRecipe,
    ) -> tuple[int | None, bool]:
        layer_idx = cls._layer_index(layer)
        if layer_idx is None and recipe.skip_layers:
            raise RuntimeError("TaperKV could not resolve layer index for skip_layers")
        return layer_idx, layer_idx in recipe.skip_layers

    @staticmethod
    def _is_block_mode(recipe: TaperKVVLLMRecipe) -> bool:
        return recipe.policy_name == "block_taperkv" and recipe.quant_granularity == "block"

    def _ensure_precision_state(
        self,
        kv_cache: torch.Tensor,
        recipe: TaperKVVLLMRecipe,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_cache, _ = self._split_kv_cache(kv_cache)
        page_size = int(key_cache.shape[1])
        if self._is_block_mode(recipe):
            if recipe.block_size != page_size:
                raise RuntimeError(
                    f"TaperKV block_size={recipe.block_size} must equal "
                    f"vLLM page_size={page_size}"
                )
            size = int(key_cache.shape[0])
        else:
            size = int(key_cache.shape[0]) * page_size
        key_bits = getattr(self, "_taper_key_bits", None)
        value_bits = getattr(self, "_taper_value_bits", None)
        if (
            key_bits is None
            or value_bits is None
            or key_bits.shape != (size,)
            or key_bits.device != kv_cache.device
        ):
            key_bits = torch.full(
                (size,),
                16,
                dtype=torch.int8,
                device=kv_cache.device,
            )
            value_bits = key_bits.clone()
            self._taper_key_bits = key_bits
            self._taper_value_bits = value_bits
        return key_bits, value_bits

    def _device_ladders(
        self,
        device: torch.device,
        recipe: TaperKVVLLMRecipe,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize immutable ladder constants before CUDA graph capture."""

        marker = (recipe.key_ladder, recipe.value_ladder, str(device))
        if getattr(self, "_taper_ladder_marker", None) != marker:
            self._taper_key_ladder = torch.tensor(
                recipe.key_ladder,
                dtype=torch.int64,
                device=device,
            )
            self._taper_value_ladder = torch.tensor(
                recipe.value_ladder,
                dtype=torch.int64,
                device=device,
            )
            self._taper_ladder_marker = marker
        return self._taper_key_ladder, self._taper_value_ladder

    def _pending_slots(self, kv_cache: torch.Tensor) -> torch.Tensor:
        pending = getattr(self, "_taper_pending_slots", None)
        if pending is None:
            return torch.empty(0, dtype=torch.long, device=kv_cache.device)
        # Keep CUDA-graph capture shapes fixed.  PAD_SLOT_ID entries are
        # masked inside the Triton kernel; boolean indexing here would produce
        # a data-dependent tensor shape and capture an empty warm-up path.
        pending = pending.to(device=kv_cache.device, dtype=torch.long)
        return pending if kv_cache.is_cuda else pending[pending >= 0]

    def _reset_pending_precision(
        self,
        kv_cache: torch.Tensor,
        recipe: TaperKVVLLMRecipe,
    ) -> torch.Tensor:
        slots = self._pending_slots(kv_cache)
        slots = slots[slots >= 0]
        if slots.numel() == 0:
            return slots
        key_bits, value_bits = self._ensure_precision_state(kv_cache, recipe)
        if self._is_block_mode(recipe):
            page_size = self._split_kv_cache(kv_cache)[0].shape[1]
            indices = torch.div(slots, page_size, rounding_mode="floor")
        else:
            indices = slots
        key_bits.index_fill_(0, indices, 16)
        value_bits.index_fill_(0, indices, 16)
        return slots

    @staticmethod
    def _lower_tokens(
        cache: torch.Tensor,
        ledger: torch.Tensor,
        slots: torch.Tensor,
        requested: torch.Tensor,
        *,
        force: bool = False,
    ) -> None:
        if slots.numel() == 0:
            return
        if cache.is_cuda:
            from vllm_plugins.kvcomp.taper_triton import (
                triton_taper_fake_quantize_tokens,
            )

            triton_taper_fake_quantize_tokens(
                cache,
                ledger,
                slots,
                requested,
                force=force,
            )
            return
        current = ledger.index_select(0, slots).to(torch.int64)
        effective = (
            requested.to(torch.int64)
            if force
            else torch.minimum(current, requested.to(torch.int64))
        )
        changed = torch.nonzero(effective < current, as_tuple=False).flatten()
        if force:
            changed = torch.arange(slots.numel(), device=slots.device)
        if changed.numel() == 0:
            return
        changed_slots = slots.index_select(0, changed)
        changed_bits = effective.index_select(0, changed)
        page_size = int(cache.shape[1])
        pages = torch.div(changed_slots, page_size, rounding_mode="floor")
        positions = changed_slots.remainder(page_size)
        rows = cache[pages, positions]
        quantized = _fake_quantize_variable_tokens(rows, changed_bits)
        cache[pages, positions] = quantized
        ledger.index_copy_(0, slots, effective.to(torch.int8))

    @classmethod
    def _lower_kv_tokens(
        cls,
        kv_cache: torch.Tensor,
        key_ledger: torch.Tensor,
        value_ledger: torch.Tensor,
        slots: torch.Tensor,
        key_requested: torch.Tensor,
        value_requested: torch.Tensor,
        *,
        force: bool = False,
    ) -> None:
        """Lower both roles with one CUDA launch and an exact CPU fallback."""

        if slots.numel() == 0:
            return
        if kv_cache.is_cuda:
            from vllm_plugins.kvcomp.taper_triton import (
                triton_taper_fake_quantize_kv_tokens,
            )

            triton_taper_fake_quantize_kv_tokens(
                kv_cache,
                key_ledger,
                value_ledger,
                slots,
                key_requested,
                value_requested,
                kv_axis=_UPSTREAM_KV_AXIS,
                force=force,
            )
            return
        key_cache, value_cache = cls._split_kv_cache(kv_cache)
        cls._lower_tokens(key_cache, key_ledger, slots, key_requested, force=force)
        cls._lower_tokens(value_cache, value_ledger, slots, value_requested, force=force)

    @staticmethod
    def _lower_pages(
        cache: torch.Tensor,
        ledger: torch.Tensor,
        pages: torch.Tensor,
        requested: torch.Tensor,
    ) -> None:
        current = ledger.index_select(0, pages).to(torch.int64)
        effective = torch.minimum(current, requested.to(torch.int64))
        changed = torch.nonzero(effective < current, as_tuple=False).flatten()
        if changed.numel() == 0:
            return
        changed_pages = pages.index_select(0, changed)
        changed_bits = effective.index_select(0, changed)
        source = cache.index_select(0, changed_pages)
        quantized = _fake_quantize_variable_pages(source, changed_bits)
        cache.index_copy_(0, changed_pages, quantized)
        ledger.index_copy_(0, pages, effective.to(torch.int8))

    @staticmethod
    def _logical_token_slots(
        block_table: torch.Tensor,
        lengths: torch.Tensor,
        page_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, max_pages = block_table.shape
        max_tokens = max_pages * page_size
        positions = torch.arange(
            max_tokens,
            device=block_table.device,
            dtype=torch.long,
        ).expand(batch, -1)
        page_offsets = torch.div(positions, page_size, rounding_mode="floor")
        page_ids = torch.gather(block_table.to(torch.long), 1, page_offsets)
        valid = positions < lengths[:, None]
        selected_pages = page_ids[valid]
        _assert_nonnegative(selected_pages, "invalid TaperKV vLLM block table")
        slots = selected_pages * page_size + positions[valid].remainder(page_size)
        return slots, positions, valid

    def _quantize_pending_tokens(
        self,
        kv_cache: torch.Tensor,
        recipe: TaperKVVLLMRecipe,
        slots: torch.Tensor,
    ) -> None:
        if not kv_cache.is_cuda:
            slots = slots[slots >= 0]
            if slots.numel() == 0:
                return
        if self._is_block_mode(recipe):
            # A residual page stays fp16 while partial, then becomes one
            # high-width affine block as soon as its final slot is written.
            # This path runs between full-cache regrade strides and prevents
            # an unaligned prompt page from remaining fp16 for extra steps.
            key_cache, value_cache = self._split_kv_cache(kv_cache)
            page_size = int(key_cache.shape[1])
            pages = torch.div(slots, page_size, rounding_mode="floor")
            _assert_unique_indices(
                pages,
                key_cache.shape[0],
                "TaperKV cannot mutate a physical page shared by multiple requests",
            )
            completed = slots.remainder(page_size) == page_size - 1
            fp16 = torch.full_like(pages, 16)
            key_widths = torch.where(
                completed,
                torch.full_like(pages, recipe.key_bits),
                fp16,
            )
            value_widths = torch.where(
                completed,
                torch.full_like(pages, recipe.value_bits),
                fp16,
            )
            key_ledger, value_ledger = self._ensure_precision_state(kv_cache, recipe)
            self._lower_pages(key_cache, key_ledger, pages, key_widths)
            self._lower_pages(value_cache, value_ledger, pages, value_widths)
            return
        key_ledger, value_ledger = self._ensure_precision_state(kv_cache, recipe)
        key_widths = torch.full(
            (slots.numel(),),
            recipe.key_bits,
            dtype=torch.int64,
            device=slots.device,
        )
        value_widths = torch.full(
            (slots.numel(),),
            recipe.value_bits,
            dtype=torch.int64,
            device=slots.device,
        )
        self._lower_kv_tokens(
            kv_cache,
            key_ledger,
            value_ledger,
            slots,
            key_widths,
            value_widths,
            force=True,
        )

    def _regrade_tokens(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        lengths: torch.Tensor,
        recipe: TaperKVVLLMRecipe,
    ) -> None:
        page_size = int(key_cache.shape[1])
        slots, positions, valid = self._logical_token_slots(
            block_table,
            lengths,
            page_size,
        )
        key_ladder, value_ladder = self._device_ladders(kv_cache.device, recipe)
        key_schedule = _batched_taper_bits(
            positions,
            lengths,
            recipe.key_bits,
            recipe.b_min,
            recipe.sink_tokens,
            recipe.recent_tokens,
            key_ladder,
        )[valid]
        value_schedule = _batched_taper_bits(
            positions,
            lengths,
            recipe.value_bits,
            recipe.b_min_value,
            recipe.sink_tokens,
            recipe.recent_tokens,
            value_ladder,
        )[valid]
        key_ledger, value_ledger = self._ensure_precision_state(kv_cache, recipe)
        _assert_unique_indices(
            slots,
            key_ledger.numel(),
            "TaperKV cannot mutate a physical token shared by multiple requests",
        )
        self._lower_kv_tokens(
            kv_cache,
            key_ledger,
            value_ledger,
            slots,
            key_schedule,
            value_schedule,
        )

    def _regrade_token_boundaries(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        lengths: torch.Tensor,
        recipe: TaperKVVLLMRecipe,
    ) -> None:
        """Lower only tokens reachable by a decode-stride boundary move."""

        key_positions, key_valid = _transition_candidate_positions(
            lengths,
            ladder_count=len(recipe.key_ladder),
            sink_tokens=recipe.sink_tokens,
            recent_tokens=recipe.recent_tokens,
            stride=recipe.requant_stride,
        )
        value_positions, value_valid = _transition_candidate_positions(
            lengths,
            ladder_count=len(recipe.value_ladder),
            sink_tokens=recipe.sink_tokens,
            recent_tokens=recipe.recent_tokens,
            stride=recipe.requant_stride,
        )
        positions = torch.cat((key_positions, value_positions), dim=1)
        valid = torch.cat((key_valid, value_valid), dim=1)
        if positions.shape[1] == 0:
            return

        # Key and value boundaries can identify the same logical token. Keep
        # one occurrence so concurrent programs never requantize the same row.
        same = positions[:, :, None] == positions[:, None, :]
        later = torch.triu(
            torch.ones(
                (positions.shape[1], positions.shape[1]),
                dtype=torch.bool,
                device=positions.device,
            ),
            diagonal=1,
        )
        selected = valid & ~(same & later[None]).any(dim=-1)

        page_size = int(self._split_kv_cache(kv_cache)[0].shape[1])
        page_offsets = torch.div(positions, page_size, rounding_mode="floor")
        safe_offsets = page_offsets.clamp(min=0, max=block_table.shape[1] - 1)
        physical_pages = torch.gather(block_table.to(torch.long), 1, safe_offsets)
        slots = physical_pages * page_size + positions.remainder(page_size)
        # Align the cadence to sequence length rather than a Python counter.
        # seq_lens is a replay-updated CUDA-graph input, so the same captured
        # graph applies the transition on exactly the intended decode steps.
        cadence = lengths.remainder(recipe.requant_stride) == 0
        selected &= cadence[:, None]
        slots = torch.where(selected, slots, torch.full_like(slots, -1)).reshape(-1)
        key_ledger, value_ledger = self._ensure_precision_state(kv_cache, recipe)
        # Prefix caching is disabled by the engine contract.  Logical
        # duplicates within a request were removed above, and PAD slots remain
        # negative, so every live program owns one physical cache row.
        key_ladder, value_ladder = self._device_ladders(kv_cache.device, recipe)
        key_schedule = _batched_taper_bits(
            positions,
            lengths,
            recipe.key_bits,
            recipe.b_min,
            recipe.sink_tokens,
            recipe.recent_tokens,
            key_ladder,
        ).reshape(-1)
        value_schedule = _batched_taper_bits(
            positions,
            lengths,
            recipe.value_bits,
            recipe.b_min_value,
            recipe.sink_tokens,
            recipe.recent_tokens,
            value_ladder,
        ).reshape(-1)
        if not kv_cache.is_cuda:
            live = slots >= 0
            slots = slots[live]
            key_schedule = key_schedule[live]
            value_schedule = value_schedule[live]
            _assert_unique_indices(
                slots,
                key_ledger.numel(),
                "TaperKV cannot mutate a physical token shared by multiple requests",
            )
        self._lower_kv_tokens(
            kv_cache,
            key_ledger,
            value_ledger,
            slots,
            key_schedule,
            value_schedule,
        )

    def _regrade_pages(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        lengths: torch.Tensor,
        kv_cache: torch.Tensor,
        recipe: TaperKVVLLMRecipe,
    ) -> None:
        page_size = int(key_cache.shape[1])
        if recipe.block_size != page_size:
            raise RuntimeError(
                f"TaperKV block_size={recipe.block_size} must equal " f"vLLM page_size={page_size}"
            )
        batch, max_pages = block_table.shape
        logical_pages = torch.arange(
            max_pages,
            device=block_table.device,
            dtype=torch.long,
        ).expand(batch, -1)
        valid_pages = logical_pages * page_size < lengths[:, None]
        physical_pages = block_table.to(torch.long)[valid_pages]
        _assert_nonnegative(physical_pages, "invalid TaperKV vLLM block table")

        positions = torch.arange(
            max_pages * page_size,
            device=block_table.device,
            dtype=torch.long,
        ).expand(batch, -1)
        key_ladder, value_ladder = self._device_ladders(kv_cache.device, recipe)
        key_tokens = _batched_taper_bits(
            positions,
            lengths,
            recipe.key_bits,
            recipe.b_min,
            recipe.sink_tokens,
            recipe.recent_tokens,
            key_ladder,
        ).view(batch, max_pages, page_size)
        value_tokens = _batched_taper_bits(
            positions,
            lengths,
            recipe.value_bits,
            recipe.b_min_value,
            recipe.sink_tokens,
            recipe.recent_tokens,
            value_ladder,
        ).view(batch, max_pages, page_size)
        key_pages = key_tokens.amax(dim=-1)
        value_pages = value_tokens.amax(dim=-1)
        full_pages = (logical_pages + 1) * page_size <= lengths[:, None]
        key_pages = torch.where(full_pages, key_pages, 16)[valid_pages]
        value_pages = torch.where(full_pages, value_pages, 16)[valid_pages]

        key_ledger, value_ledger = self._ensure_precision_state(kv_cache, recipe)
        _assert_unique_indices(
            physical_pages,
            key_ledger.numel(),
            "TaperKV cannot mutate a physical page shared by multiple requests",
        )
        self._lower_pages(key_cache, key_ledger, physical_pages, key_pages)
        self._lower_pages(value_cache, value_ledger, physical_pages, value_pages)

    def _requantize_cache(
        self,
        kv_cache: torch.Tensor,
        attn_metadata,
        layer: Any,
        recipe: TaperKVVLLMRecipe,
    ) -> None:
        layer_idx, skipped = self._layer_selection(layer, recipe)
        if skipped:
            return
        if attn_metadata.use_cascade:
            raise RuntimeError("TaperKV does not support cascade attention")
        block_table = attn_metadata.block_table
        lengths = attn_metadata.seq_lens
        if block_table is None or lengths is None:
            raise RuntimeError("TaperKV requires block_table and seq_lens")
        block_table = block_table.to(device=kv_cache.device)
        lengths = lengths.to(device=kv_cache.device, dtype=torch.long)
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        if self._is_block_mode(recipe):
            self._regrade_pages(
                key_cache,
                value_cache,
                block_table,
                lengths,
                kv_cache,
                recipe,
            )
        else:
            self._regrade_tokens(
                key_cache,
                value_cache,
                kv_cache,
                block_table,
                lengths,
                recipe,
            )
        emit_runtime_evidence_once(
            self,
            "taperkv_regrade",
            {
                "backend": "vllm_plugins.kvcomp.taper_backend.TaperKVBackend",
                "implementation": recipe.implementation,
                "event": "gpu_paged_cache_regraded",
                "token_quantizer": "triton_stride_aware_fused_kv_affine",
                "new_token_cache_update": "triton_fused_affine_quantize_and_paged_scatter",
                "decode_regrade_strategy": ("sequence_length_aligned_ladder_boundary_incremental"),
                "cuda_graph_safe_decode": not self._is_block_mode(recipe),
                "policy_name": recipe.policy_name,
                "layer_index": layer_idx,
                "physical_cache_shape": list(kv_cache.shape),
                "physical_cache_dtype": str(kv_cache.dtype).removeprefix("torch."),
                "physical_cache_device": str(kv_cache.device),
                "page_size": int(key_cache.shape[1]),
                "request_count": int(lengths.shape[0]),
                "maximum_sequence_length": int(attn_metadata.max_seq_len),
            },
        )

    def do_kv_cache_update(
        self,
        layer,
        key,
        value,
        kv_cache,
        slot_mapping,
    ):
        if not os.environ.get(TAPER_CONFIG_ENV):
            return super().do_kv_cache_update(
                layer,
                key,
                value,
                kv_cache,
                slot_mapping,
            )
        self._taper_pending_slots = slot_mapping
        recipe = taperkv_recipe()
        _layer_idx, skipped = self._layer_selection(layer, recipe)
        if skipped or self._is_block_mode(recipe) or not kv_cache.is_cuda:
            return super().do_kv_cache_update(
                layer,
                key,
                value,
                kv_cache,
                slot_mapping,
            )
        from vllm_plugins.kvcomp.taper_triton import (
            triton_taper_quantize_and_store_kv,
        )

        key_ledger, value_ledger = self._ensure_precision_state(kv_cache, recipe)
        triton_taper_quantize_and_store_kv(
            key,
            value,
            kv_cache,
            key_ledger,
            value_ledger,
            slot_mapping.to(device=kv_cache.device, dtype=torch.long),
            key_bits=recipe.key_bits,
            value_bits=recipe.value_bits,
            kv_axis=_UPSTREAM_KV_AXIS,
        )
        return None

    def _process_cache_update(
        self,
        kv_cache: torch.Tensor,
        attn_metadata,
        layer: Any,
        recipe: TaperKVVLLMRecipe,
    ) -> None:
        # vLLM's kernel warm-up uses only PAD_SLOT_ID entries.  Those dummy
        # passes may also carry repeated physical pages in their persistent
        # block tables, so they must not mutate cache contents or policy state.
        slots = self._pending_slots(kv_cache)
        if slots.numel() == 0:
            return

        is_prefill = int(attn_metadata.max_query_len) > 1
        if not self._is_block_mode(recipe):
            if is_prefill:
                # The fused store has overwritten reused slots and set their
                # ledgers to the high widths. Establish the complete age
                # schedule without requantizing protected/recent rows.
                # Prefill is outside the one-token CUDA graph, so filtering is
                # also the safe way to reject vLLM's all-PAD warm-up batch
                # before inspecting its deliberately duplicated block table.
                live_slots = slots[slots >= 0]
                if live_slots.numel() == 0:
                    return
                self._requantize_cache(
                    kv_cache,
                    attn_metadata,
                    layer,
                    recipe,
                )
                return
            # The fused cache update already quantized every new row. This
            # fixed-shape boundary pass is masked by seq_len % requant_stride
            # entirely on device and is therefore safe to capture and replay.
            block_table = attn_metadata.block_table
            lengths = attn_metadata.seq_lens
            if block_table is None or lengths is None:
                raise RuntimeError("TaperKV requires block_table and seq_lens")
            self._regrade_token_boundaries(
                kv_cache,
                block_table.to(device=kv_cache.device),
                lengths.to(device=kv_cache.device, dtype=torch.long),
                recipe,
            )
            return

        # Block TaperKV still changes whole pages and retains its eager cadence.
        slots = slots[slots >= 0]
        if slots.numel() == 0:
            return
        since = getattr(self, "_taper_since", 0) + 1
        self._taper_since = since
        regrade = is_prefill or (since % recipe.requant_stride == 0)
        if regrade:
            slots = self._reset_pending_precision(kv_cache, recipe)
            self._requantize_cache(
                kv_cache,
                attn_metadata,
                layer,
                recipe,
            )
        else:
            self._quantize_pending_tokens(kv_cache, recipe, slots)

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
        if attn_metadata is not None and os.environ.get(TAPER_CONFIG_ENV):
            if attn_metadata.use_cascade:
                raise RuntimeError("TaperKV does not support cascade attention")
            if getattr(self, "dcp_world_size", 1) > 1:
                raise RuntimeError("TaperKV does not support decode context parallelism")
            recipe = taperkv_recipe()
            _layer_idx, skipped = self._layer_selection(layer, recipe)
            if not skipped:
                self._process_cache_update(
                    kv_cache,
                    attn_metadata,
                    layer,
                    recipe,
                )
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


__all__ = [
    "TAPER_CONFIG_ENV",
    "TaperKVBackend",
    "TaperKVImpl",
    "taperkv_recipe",
]

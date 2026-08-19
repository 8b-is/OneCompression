from __future__ import annotations

import pytest
import torch

pytest.importorskip("vllm")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

from vllm.model_executor.layers.quantization.turboquant.centroids import (  # noqa: E402
    get_centroids,
)
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard  # noqa: E402
from vllm_plugins.kvcomp.turboquant2_triton import (  # noqa: E402
    triton_turboquant2_decode_attention,
    triton_turboquant2_dequant_kv,
    triton_turboquant2_store,
    turboquant2_layout,
)


def _unpack_2bit(raw: torch.Tensor, offset: int, count: int) -> torch.Tensor:
    byte_index = torch.arange(count, device=raw.device) // 4
    shift = (torch.arange(count, device=raw.device) % 4) * 2
    return ((raw[offset + byte_index].to(torch.int32) >> shift) & 0x3).long()


def test_turboquant2_packed_store_dequant_and_attention_match_scalar_reference():
    torch.manual_seed(2202)
    device = torch.device("cuda")
    dtype = torch.float16
    tokens, kv_heads, query_heads, head_dim, page_size = 19, 2, 4, 128, 16
    pages = (tokens + page_size - 1) // page_size
    layout = turboquant2_layout(head_dim)
    assert layout["slot_size_aligned"] == 70
    cache = torch.full(
        (pages, page_size, kv_heads, layout["slot_size_aligned"]),
        0xA5,
        dtype=torch.uint8,
        device=device,
    )
    slots = torch.arange(tokens, dtype=torch.int32, device=device)
    key = torch.randn(tokens, kv_heads, head_dim, dtype=dtype, device=device)
    value = torch.randn_like(key)
    rotation = _build_hadamard(head_dim, str(device)).contiguous()
    centroids = get_centroids(head_dim, 2).to(device).sort().values
    midpoints = (centroids[:-1] + centroids[1:]) / 2

    triton_turboquant2_store(key, value, cache, slots, rotation, midpoints.contiguous())
    torch.cuda.synchronize()

    key_flat = key.float().reshape(-1, head_dim)
    key_norm = key_flat.norm(dim=1)
    rotated = (key_flat / (key_norm[:, None] + 1e-8)) @ rotation
    key_codes = torch.bucketize(rotated, midpoints)
    value_flat = value.float().reshape(-1, head_dim)
    value_min = value_flat.amin(dim=1)
    value_scale = ((value_flat.amax(dim=1) - value_min) / 3).clamp_min(1e-8)
    value_codes = (
        torch.floor((value_flat - value_min[:, None]) / value_scale[:, None] + 0.5)
        .clamp(0, 3)
        .long()
    )
    for token in range(tokens):
        page, position = divmod(token, page_size)
        for head in range(kv_heads):
            row = token * kv_heads + head
            raw = cache[page, position, head]
            assert torch.equal(_unpack_2bit(raw, 0, head_dim), key_codes[row])
            packed_value_codes = _unpack_2bit(raw, layout["value_codes_offset"], head_dim)
            mismatched = packed_value_codes != value_codes[row]
            # Triton and torch reductions may straddle a half-step by one ULP;
            # only such mathematically tied assignments are allowed to differ.
            if mismatched.any():
                normalized = (value_flat[row] - value_min[row]) / value_scale[row]
                half_step_distance = ((normalized - torch.floor(normalized)) - 0.5).abs()
                assert torch.all(half_step_distance[mismatched] < 2e-3)
                assert torch.all(
                    (packed_value_codes[mismatched] - value_codes[row][mismatched]).abs() == 1
                )

    block_table = torch.arange(pages, dtype=torch.int32, device=device).view(1, -1)
    allocated = pages * page_size
    rotated_key_out = torch.empty(1, kv_heads, allocated, head_dim, dtype=dtype, device=device)
    value_out = torch.empty_like(rotated_key_out)
    triton_turboquant2_dequant_kv(cache, block_table, centroids, rotated_key_out, value_out)
    torch.cuda.synchronize()

    selected = centroids[key_codes]
    selected = selected / selected.norm(dim=1, keepdim=True)
    reference_key = ((selected * key_norm[:, None]) @ rotation).reshape(tokens, kv_heads, head_dim)
    reference_value = (value_codes.float() * value_scale[:, None] + value_min[:, None]).reshape(
        tokens, kv_heads, head_dim
    )
    materialized_key = (
        rotated_key_out[0, :, :tokens].transpose(0, 1).reshape(-1, head_dim).float() @ rotation
    ).reshape(tokens, kv_heads, head_dim)
    torch.testing.assert_close(materialized_key, reference_key, rtol=2e-3, atol=3e-3)
    torch.testing.assert_close(
        value_out[0, :, :tokens].transpose(0, 1).float(),
        reference_value,
        rtol=2e-3,
        atol=3e-3,
    )

    query = torch.randn(1, query_heads, head_dim, dtype=dtype, device=device)
    seq_lens = torch.tensor([tokens], dtype=torch.int32, device=device)
    output = triton_turboquant2_decode_attention(
        query,
        cache,
        block_table,
        seq_lens,
        rotation,
        centroids,
        head_dim**-0.5,
        max_num_kv_splits=1,
    )
    groups = query_heads // kv_heads
    expanded_key = reference_key.repeat_interleave(groups, dim=1)
    expanded_value = reference_value.repeat_interleave(groups, dim=1)
    scores = torch.einsum("bhd,thd->bht", query.float(), expanded_key.float()) * head_dim**-0.5
    reference_output = torch.einsum(
        "bht,thd->bhd", scores.softmax(dim=-1), expanded_value.float()
    ).to(dtype)
    torch.testing.assert_close(output, reference_output, rtol=3e-3, atol=2e-2)

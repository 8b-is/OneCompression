from __future__ import annotations

import torch
import pytest

pytest.importorskip("vllm")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

from onecomp.kvcomp.codec.hayate.codec import (  # noqa: E402
    HayateQuantCodec,
    _block_codebook_by_code_bits,
)
from onecomp.kvcomp.codec.hayate.vq import (  # noqa: E402
    lookup_codewords,
    quantize_exact_pre,
    quantize_via_lut,
)
from vllm_plugins.kvcomp.hayate_config import hayatequant_native_recipe  # noqa: E402
from vllm_plugins.kvcomp.hayate_triton import (  # noqa: E402
    triton_hayatequant_decode_attention,
    triton_hayatequant_dequant_kv,
    triton_hayatequant_store,
)
from onecomp.kvcomp.codec.hayate.codec import _block_lut_code_bits_on  # noqa: E402
from onecomp.kvcomp.codec.hayate import triton_step  # noqa: E402


def _tables(rate: int, head_dim: int, device: torch.device):
    recipe = hayatequant_native_recipe(rate)
    codec = HayateQuantCodec(bits=rate)
    key_rotation = codec._rt_rotation(  # pylint: disable=protected-access
        head_dim,
        recipe.key_block_size,
        recipe.key_rate,
        device,
        role="k",
    ).contiguous()
    value_rotation = codec._rt_rotation(  # pylint: disable=protected-access
        head_dim,
        recipe.value_block_size,
        recipe.value_rate,
        device,
        role="v",
    ).contiguous()
    key_codebook = _block_codebook_by_code_bits(recipe.key_code_bits_per_block).to(
        device=device, dtype=torch.float32
    )
    value_codebook = _block_codebook_by_code_bits(recipe.value_code_bits_per_block).to(
        device=device, dtype=torch.float32
    )
    return recipe, codec, key_rotation, value_rotation, key_codebook, value_codebook


def _unpack_codes(
    cache: torch.Tensor,
    slots: torch.Tensor,
    heads: int,
    offset: int,
    code_bits: int,
    blocks: int,
) -> torch.Tensor:
    rows = []
    page_size = cache.shape[1]
    for slot in slots.cpu().tolist():
        page, position = divmod(int(slot), page_size)
        for head in range(heads):
            raw = cache[page, position, head].cpu()
            row = []
            for block in range(blocks):
                bit = block * code_bits
                byte, shift = divmod(bit, 8)
                word = int(raw[offset + byte])
                if shift + code_bits > 8:
                    word |= int(raw[offset + byte + 1]) << 8
                if shift + code_bits > 16:
                    word |= int(raw[offset + byte + 2]) << 16
                row.append((word >> shift) & ((1 << code_bits) - 1))
            rows.append(row)
    return torch.tensor(rows, dtype=torch.int32, device=cache.device)


def _unpack_scales(
    cache: torch.Tensor,
    slots: torch.Tensor,
    heads: int,
    offset: int,
) -> torch.Tensor:
    words = []
    page_size = cache.shape[1]
    for slot in slots.cpu().tolist():
        page, position = divmod(int(slot), page_size)
        for head in range(heads):
            raw = cache[page, position, head].cpu()
            words.append(int(raw[offset]) | (int(raw[offset + 1]) << 8))
    return torch.tensor(words, dtype=torch.uint16).view(torch.float16).to(cache.device)


def _reference_materialize(
    codes: torch.Tensor,
    scales: torch.Tensor,
    codebook: torch.Tensor,
    rotation: torch.Tensor,
    tokens: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    rotated = lookup_codewords(codebook, codes.long()).reshape(tokens, heads, head_dim)
    rotated = rotated * scales.reshape(tokens, heads, 1).float()
    return (rotated @ rotation.t()).to(dtype)


@pytest.mark.parametrize("rate", [2, 3, 4])
def test_packed_store_decode_and_attention_match_public_hayatequant(rate):
    torch.manual_seed(100 + rate)
    device = torch.device("cuda")
    dtype = torch.float16
    tokens, kv_heads, query_heads, head_dim, page_size = 17, 2, 4, 128, 16
    recipe, codec, kr, vr, kcb, vcb = _tables(rate, head_dim, device)
    layout = recipe.layout(head_dim)
    pages = (tokens + page_size - 1) // page_size
    cache = torch.full(
        (pages, page_size, kv_heads, layout["slot_size_aligned"]),
        0xA5,
        dtype=torch.uint8,
        device=device,
    )
    slots = torch.arange(tokens, dtype=torch.int32, device=device)
    key = torch.randn(tokens, kv_heads, head_dim, dtype=dtype, device=device)
    value = torch.randn_like(key)

    payload = triton_hayatequant_store(
        key,
        value,
        cache,
        slots,
        kr,
        vr,
        kcb,
        vcb,
        recipe,
        require_fused=True,
    )
    torch.cuda.synchronize()
    public_key = codec._quant_rotor(key, rate, "k")  # pylint: disable=protected-access
    public_value = codec._quant_rotor(value, rate, "v")  # pylint: disable=protected-access
    assert torch.equal(payload["key_codes"], public_key["codes"].reshape_as(payload["key_codes"]))
    assert torch.equal(
        payload["value_codes"], public_value["codes"].reshape_as(payload["value_codes"])
    )
    assert torch.equal(payload["key_scales"], public_key["scale"].reshape(-1))
    assert torch.equal(payload["value_scales"], public_value["scale"].reshape(-1))

    packed_key = _unpack_codes(
        cache,
        slots,
        kv_heads,
        0,
        recipe.key_code_bits_per_block,
        head_dim // recipe.key_block_size,
    )
    packed_value = _unpack_codes(
        cache,
        slots,
        kv_heads,
        layout["value_codes_offset"],
        recipe.value_code_bits_per_block,
        head_dim // recipe.value_block_size,
    )
    packed_key_scale = _unpack_scales(cache, slots, kv_heads, layout["key_scale_offset"])
    packed_value_scale = _unpack_scales(cache, slots, kv_heads, layout["value_scale_offset"])
    assert torch.equal(packed_key, payload["key_codes"])
    assert torch.equal(packed_value, payload["value_codes"])
    assert torch.equal(packed_key_scale, payload["key_scales"])
    assert torch.equal(packed_value_scale, payload["value_scales"])

    block_table = torch.arange(pages, dtype=torch.int32, device=device).view(1, -1)
    allocated = pages * page_size
    key_out = torch.empty(1, kv_heads, allocated, head_dim, dtype=dtype, device=device)
    value_out = torch.empty_like(key_out)
    triton_hayatequant_dequant_kv(
        cache,
        block_table,
        kcb,
        vcb,
        kr,
        vr,
        recipe,
        key_out,
        value_out,
    )
    key_reference = _reference_materialize(
        packed_key,
        packed_key_scale,
        kcb,
        kr,
        tokens,
        kv_heads,
        head_dim,
        dtype,
    )
    value_reference = _reference_materialize(
        packed_value,
        packed_value_scale,
        vcb,
        vr,
        tokens,
        kv_heads,
        head_dim,
        dtype,
    )
    torch.testing.assert_close(
        key_out[0, :, :tokens].transpose(0, 1), key_reference, rtol=0, atol=1e-3
    )
    torch.testing.assert_close(
        value_out[0, :, :tokens].transpose(0, 1), value_reference, rtol=0, atol=1e-3
    )

    query = torch.randn(1, query_heads, head_dim, dtype=dtype, device=device)
    seq_lens = torch.tensor([tokens], dtype=torch.int32, device=device)
    native = triton_hayatequant_decode_attention(
        query,
        cache,
        block_table,
        seq_lens,
        kr,
        vr,
        kcb,
        vcb,
        recipe,
        head_dim**-0.5,
        max_num_kv_splits=1,
    )
    groups = query_heads // kv_heads
    keys = key_reference.repeat_interleave(groups, dim=1)
    values = value_reference.repeat_interleave(groups, dim=1)
    scores = torch.einsum("bhd,thd->bht", query.float(), keys.float()) * head_dim**-0.5
    reference = torch.einsum("bht,thd->bhd", scores.softmax(dim=-1), values.float()).to(dtype)
    torch.testing.assert_close(native, reference, rtol=3e-3, atol=1.5e-2)


@pytest.mark.parametrize("rate", [2, 3, 4])
def test_canonical_lut_kernel_matches_tensor_primitive(rate, monkeypatch):
    device = torch.device("cuda")
    recipe, _codec, _kr, _vr, key_codebook, _value_codebook = _tables(rate, 128, device)
    scoring = key_codebook.half().contiguous()
    csq = scoring.pow(2).sum(dim=1)
    lut = _block_lut_code_bits_on(
        recipe.key_code_bits_per_block,
        256,
        2,
        str(device),
    )
    generator = torch.Generator(device=device).manual_seed(2000 + rate)
    rows = torch.randn(50_003, 2, dtype=torch.float32, device=device, generator=generator)

    monkeypatch.setattr(triton_step, "HAS_TRITON", False)
    expected = quantize_via_lut(
        rows,
        lut,
        256,
        key_codebook,
        scoring_cb=scoring,
        scoring_csq=csq,
    )
    monkeypatch.setattr(triton_step, "HAS_TRITON", True)
    actual = triton_step.quantize_lut_canonical_t2_triton(
        rows,
        lut,
        256,
        scoring,
        csq,
    )

    assert torch.equal(actual, expected)

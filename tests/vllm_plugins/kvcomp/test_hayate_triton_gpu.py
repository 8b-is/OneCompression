from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("vllm")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

from onecomp.kvcomp.codec.hayate import triton_step  # noqa: E402
from onecomp.kvcomp.codec.hayate.codec import _block_lut_code_bits_on  # noqa: E402
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
from vllm_plugins.kvcomp.hayate_native import (  # noqa: E402
    _fold_value_rotation_into_projection,
    _fold_value_rotation_into_qkv_projection,
    _to_folded_value_basis,
)
from vllm_plugins.kvcomp.hayate_triton import (  # noqa: E402
    _rotate_dense_small_rows,
    triton_hayatequant_decode_attention,
    triton_hayatequant_dequant_kv,
    triton_hayatequant_store,
)


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


@pytest.mark.parametrize("rows", [8, 16, 32])
def test_small_dense_rotation_matches_ieee_matmul(rows):
    torch.manual_seed(900 + rows)
    device = torch.device("cuda")
    x = torch.randn(rows, 128, dtype=torch.float16, device=device)
    rotation = torch.linalg.qr(torch.randn(128, 128, device=device)).Q.contiguous()
    output = torch.empty_like(x, dtype=torch.float32)

    actual = _rotate_dense_small_rows(x, rotation, output)
    reference = x.float() @ rotation

    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, reference, rtol=0, atol=3e-6)


@pytest.mark.parametrize("rate", [2, 3, 4])
def test_one_token_small_rotation_preserves_public_codes_and_scales(rate):
    torch.manual_seed(950 + rate)
    device = torch.device("cuda")
    tokens, kv_heads, head_dim = 1, 8, 128
    recipe, codec, kr, vr, kcb, vcb = _tables(rate, head_dim, device)
    layout = recipe.layout(head_dim)
    cache = torch.zeros(
        (1, 16, kv_heads, layout["slot_size_aligned"]),
        dtype=torch.uint8,
        device=device,
    )
    slots = torch.zeros(tokens, dtype=torch.int32, device=device)
    key = torch.randn(tokens, kv_heads, head_dim, dtype=torch.float16, device=device)
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
    public_key = codec._quant_rotor(key, rate, "k")  # pylint: disable=protected-access
    public_value = codec._quant_rotor(value, rate, "v")  # pylint: disable=protected-access

    assert torch.equal(payload["key_codes"], public_key["codes"].reshape_as(payload["key_codes"]))
    assert torch.equal(
        payload["value_codes"], public_value["codes"].reshape_as(payload["value_codes"])
    )
    assert torch.equal(payload["key_scales"], public_key["scale"].reshape(-1))
    assert torch.equal(payload["value_scales"], public_value["scale"].reshape(-1))


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
    direct_cache = torch.full_like(cache, 0xA5)
    assert (
        triton_hayatequant_store(
            key,
            value,
            direct_cache,
            slots,
            kr,
            vr,
            kcb,
            vcb,
            recipe,
            return_payload=False,
            require_fused=True,
        )
        is None
    )
    folded_value_cache = torch.full_like(cache, 0xA5)
    folded_value_payload = triton_hayatequant_store(
        key,
        value.float() @ vr,
        folded_value_cache,
        slots,
        kr,
        vr,
        kcb,
        vcb,
        recipe,
        value_in_rotated_basis=True,
        require_fused=True,
    )
    direct_folded_value_cache = torch.full_like(cache, 0xA5)
    assert (
        triton_hayatequant_store(
            key,
            value.float() @ vr,
            direct_folded_value_cache,
            slots,
            kr,
            vr,
            kcb,
            vcb,
            recipe,
            value_in_rotated_basis=True,
            return_payload=False,
            require_fused=True,
        )
        is None
    )
    torch.cuda.synchronize()
    assert torch.equal(direct_cache, cache)
    assert torch.equal(direct_folded_value_cache, folded_value_cache)
    public_key = codec._quant_rotor(key, rate, "k")  # pylint: disable=protected-access
    public_value = codec._quant_rotor(value, rate, "v")  # pylint: disable=protected-access
    assert torch.equal(payload["key_codes"], public_key["codes"].reshape_as(payload["key_codes"]))
    assert torch.equal(
        payload["value_codes"], public_value["codes"].reshape_as(payload["value_codes"])
    )
    assert torch.equal(payload["key_scales"], public_key["scale"].reshape(-1))
    assert torch.equal(payload["value_scales"], public_value["scale"].reshape(-1))
    assert torch.equal(folded_value_payload["value_codes"], payload["value_codes"])
    assert torch.equal(folded_value_payload["value_scales"], payload["value_scales"])

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
    folded_key_out = torch.empty_like(key_out)
    folded_value_out = torch.empty_like(value_out)
    triton_hayatequant_dequant_kv(
        cache,
        block_table,
        kcb,
        vcb,
        kr,
        vr,
        recipe,
        folded_key_out,
        folded_value_out,
        value_in_rotated_basis=True,
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
    value_rotated_reference = lookup_codewords(vcb, packed_value.long()).reshape(
        tokens, kv_heads, head_dim
    )
    value_rotated_reference *= packed_value_scale.reshape(tokens, kv_heads, 1).float()
    torch.testing.assert_close(
        folded_key_out[0, :, :tokens].transpose(0, 1),
        key_reference,
        rtol=0,
        atol=1e-3,
    )
    torch.testing.assert_close(
        folded_value_out[0, :, :tokens].transpose(0, 1),
        value_rotated_reference.to(dtype),
        rtol=0,
        atol=1e-3,
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
        max_num_kv_splits=32,
    )
    rotated_native = triton_hayatequant_decode_attention(
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
        max_num_kv_splits=32,
        output_in_rotated_basis=True,
    )
    assert rotated_native.dtype == dtype
    explicit_inverse = (rotated_native.float() @ vr.t()).to(dtype)
    torch.testing.assert_close(explicit_inverse, native, rtol=2e-3, atol=8e-3)
    groups = query_heads // kv_heads
    keys = key_reference.repeat_interleave(groups, dim=1)
    values = value_reference.repeat_interleave(groups, dim=1)
    scores = torch.einsum("bhd,thd->bht", query.float(), keys.float()) * head_dim**-0.5
    reference = torch.einsum("bht,thd->bhd", scores.softmax(dim=-1), values.float()).to(dtype)
    torch.testing.assert_close(native, reference, rtol=3e-3, atol=1.5e-2)


class UnquantizedLinearMethod:
    """Minimal stand-in matching vLLM's safe projection-method contract."""


@pytest.mark.parametrize("rate", [2, 3, 4])
def test_value_rotation_folds_exactly_into_unquantized_projection(rate):
    torch.manual_seed(500 + rate)
    device = torch.device("cuda")
    dtype = torch.float16
    tokens, heads, head_dim, output_dim = 7, 4, 128, 96
    _recipe, _codec, _kr, value_rotation, _kcb, _vcb = _tables(rate, head_dim, device)
    original_weight = torch.randn(
        output_dim,
        heads * head_dim,
        dtype=dtype,
        device=device,
    )
    projection = SimpleNamespace(
        weight=original_weight.clone(),
        quant_method=UnquantizedLinearMethod(),
    )
    layer = SimpleNamespace(_hayate_output_projection=projection)

    assert _fold_value_rotation_into_projection(layer, value_rotation, head_dim)
    layer._hayate_projection_folded = True
    layer._hayate_value_rotation = value_rotation

    rotated_output = torch.randn(
        tokens,
        heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    explicit_output = (rotated_output.float() @ value_rotation.t()).to(dtype)
    expected_decode = F.linear(explicit_output.flatten(1), original_weight)
    fused_decode = F.linear(rotated_output.flatten(1), projection.weight)
    torch.testing.assert_close(fused_decode, expected_decode, rtol=3e-3, atol=3e-2)

    ordinary_prefill = torch.randn_like(rotated_output)
    expected_prefill = F.linear(ordinary_prefill.flatten(1), original_weight)
    folded_prefill = _to_folded_value_basis(ordinary_prefill, layer)
    fused_prefill = F.linear(folded_prefill.flatten(1), projection.weight)
    torch.testing.assert_close(fused_prefill, expected_prefill, rtol=3e-3, atol=3e-2)


@pytest.mark.parametrize("rate", [2, 3, 4])
def test_value_rotation_folds_into_qkv_value_projection_only(rate):
    torch.manual_seed(800 + rate)
    device = torch.device("cuda")
    dtype = torch.float16
    tokens, query_heads, kv_heads, head_dim, hidden_dim = 5, 4, 2, 128, 192
    _recipe, _codec, _kr, value_rotation, _kcb, _vcb = _tables(rate, head_dim, device)
    q_size = query_heads * head_dim
    kv_size = kv_heads * head_dim
    output_size = q_size + 2 * kv_size
    original_weight = torch.randn(
        output_size,
        hidden_dim,
        dtype=dtype,
        device=device,
    )
    original_bias = torch.randn(output_size, dtype=dtype, device=device)
    projection = SimpleNamespace(
        weight=original_weight.clone(),
        bias=original_bias.clone(),
        quant_method=UnquantizedLinearMethod(),
    )
    layer = SimpleNamespace(
        _hayate_qkv_projection=projection,
        _hayate_q_size=q_size,
        _hayate_kv_size=kv_size,
    )

    assert _fold_value_rotation_into_qkv_projection(layer, value_rotation, head_dim)
    hidden = torch.randn(tokens, hidden_dim, dtype=dtype, device=device)
    ordinary = F.linear(hidden, original_weight, original_bias)
    folded = F.linear(hidden, projection.weight, projection.bias)
    value_start = q_size + kv_size

    torch.testing.assert_close(folded[:, :value_start], ordinary[:, :value_start], rtol=0, atol=0)
    expected_value = (
        ordinary[:, value_start:].reshape(tokens, kv_heads, head_dim).float() @ value_rotation
    ).to(dtype)
    folded_value = folded[:, value_start:].reshape(tokens, kv_heads, head_dim)
    torch.testing.assert_close(folded_value, expected_value, rtol=3e-3, atol=3e-2)

    layer._hayate_projection_folded = True
    layer._hayate_value_input_folded = True
    layer._hayate_value_rotation = value_rotation
    assert _to_folded_value_basis(folded_value, layer) is folded_value


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

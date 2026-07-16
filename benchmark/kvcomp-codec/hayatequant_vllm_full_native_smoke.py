#!/usr/bin/env python3
"""GPU correctness/parity smoke for the faithful packed HQ native backend.

This is not a model-quality benchmark and does not compare methods.  It proves
the native data-plane contract against the public HayateQuant encoder:

* public-encoder-identical code indices and signed-fp16 scales for both K/V;
* formal LUT T=2 distortion against the all-codeword exact reference;
* shape-independent one-token-store versus prefill payload bytes;
* bit packing/unpacking and untouched slot padding;
* full dequantization after inverse role rotations;
* packed decode-attention against a materialized reference;
* the linear rotated-value accumulation identity used by the native kernel.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import torch

from onecomp.kvcomp.codec.hayate.codec import (
    HayateQuantCodec,
    _block_codebook_by_code_bits,
)
from onecomp.kvcomp.codec.hayate.vq import lookup_codewords
from vllm_plugins.kvcomp.hayate_full_config import (
    FullHQNativeRecipe,
    fixed_hq_native_recipe,
    paper_hq_native_recipe,
)
from vllm_plugins.kvcomp.hayate_full_triton import (
    _assignment_lut,
    triton_full_hayatequant_decode_attention,
    triton_full_hayatequant_dequant_kv,
    triton_full_hayatequant_store,
)

REPO = Path(__file__).resolve().parents[2]


def _cmd(argv: list[str]) -> str | None:
    try:
        return subprocess.run(
            argv, cwd=REPO, check=True, text=True, capture_output=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recipe(args: argparse.Namespace) -> FullHQNativeRecipe:
    common = {
        "dist": args.dist,
        "codebook_init": args.codebook_init,
        "rotation_mode": args.rotation_mode,
        "seed": args.seed,
        "allow_canonical_k10v6_b4": args.allow_canonical_k10v6_b4,
    }
    if args.assignment_mode is not None:
        common["assignment_mode"] = args.assignment_mode
    if args.lut_candidates is not None:
        common["lut_candidates"] = args.lut_candidates
    if args.lut_bins is not None:
        common["lut_bins"] = args.lut_bins
    explicit = (
        args.key_code_bits,
        args.key_block_size,
        args.value_code_bits,
        args.value_block_size,
    )
    if any(value is not None for value in explicit):
        if not all(value is not None for value in explicit):
            raise ValueError("all four explicit role fields are required")
        return FullHQNativeRecipe(
            key_code_bits_per_block=args.key_code_bits,
            key_block_size=args.key_block_size,
            value_code_bits_per_block=args.value_code_bits,
            value_block_size=args.value_block_size,
            **common,
        )
    if args.recipe == "hq":
        return fixed_hq_native_recipe(args.avg_bits, **common)
    return paper_hq_native_recipe(args.avg_bits, **common)


def _tables(recipe: FullHQNativeRecipe, d: int, device: torch.device):
    resolver = HayateQuantCodec(
        bits=max(1, min(8, round(recipe.key_rate))),
        value_bits=max(1, min(8, round(recipe.value_rate))),
        dist=recipe.dist,
        max_codebook_bits=recipe.max_codebook_bits,
        codebook_init=recipe.codebook_init,
        rotation_mode=recipe.rotation_mode,
        encoder=recipe.assignment_mode,
        key_scale_mode="nc",
        value_scale_mode="ls_iter",
        seed=recipe.seed,
    )
    kr = resolver._rt_rotation(  # pylint: disable=protected-access
        d, recipe.key_block_size, recipe.key_rate, device, role="k"
    ).contiguous()
    vr = resolver._rt_rotation(  # pylint: disable=protected-access
        d, recipe.value_block_size, recipe.value_rate, device, role="v"
    ).contiguous()
    kcb = (
        _block_codebook_by_code_bits(
            recipe.key_code_bits_per_block,
            recipe.key_block_size,
            recipe.dist,
            d,
            recipe.codebook_init,
        )
        .to(device=device, dtype=torch.float32)
        .contiguous()
    )
    vcb = (
        _block_codebook_by_code_bits(
            recipe.value_code_bits_per_block,
            recipe.value_block_size,
            recipe.dist,
            d,
            recipe.codebook_init,
        )
        .to(device=device, dtype=torch.float32)
        .contiguous()
    )
    return resolver, kr, vr, kcb, vcb


def _public_payload(
    x: torch.Tensor,
    recipe: FullHQNativeRecipe,
    role: str,
    assignment_mode: str | None = None,
) -> dict:
    code_bits = recipe.key_code_bits_per_block if role == "k" else recipe.value_code_bits_per_block
    block = recipe.key_block_size if role == "k" else recipe.value_block_size
    nominal = max(1, min(8, int(round(code_bits / block))))
    selected_assignment = assignment_mode or recipe.assignment_mode
    codec_kwargs = {}
    if selected_assignment == "lut":
        codec_kwargs.update(
            lut_bins=recipe.lut_bins,
            lut_candidates=recipe.lut_candidates,
        )
    codec = HayateQuantCodec(
        bits=nominal,
        value_bits=nominal,
        dist=recipe.dist,
        block_size=block,
        max_codebook_bits=max(recipe.max_codebook_bits, code_bits),
        codebook_init=recipe.codebook_init,
        rotation_mode=recipe.rotation_mode,
        encoder=selected_assignment,
        key_scale_mode="nc",
        value_scale_mode="ls_iter",
        seed=recipe.seed,
        **codec_kwargs,
    )
    return codec._quant_rotor(  # pylint: disable=protected-access
        x,
        nominal,
        role,
        code_bits_override=code_bits,
    )


def _luts(recipe: FullHQNativeRecipe, d: int, device: torch.device):
    """Return the same cached public assignment tables used by native store."""

    return (
        _assignment_lut(recipe, "k", d, device),
        _assignment_lut(recipe, "v", d, device),
    )


def _scoring_tables(
    recipe: FullHQNativeRecipe,
    key_codebook: torch.Tensor,
    value_codebook: torch.Tensor,
):
    """Hoist exact-search codebook casts and squared norms out of store."""

    key_score = key_codebook.half() if key_codebook.is_cuda else key_codebook
    value_score = value_codebook.half() if value_codebook.is_cuda else value_codebook
    return (
        key_score,
        key_score.pow(2).sum(dim=1),
        value_score,
        value_score.pow(2).sum(dim=1),
    )


def _unpack_stream(
    cache: torch.Tensor,
    slots: torch.Tensor,
    heads: int,
    offset: int,
    code_bits: int,
    nblocks: int,
) -> torch.Tensor:
    """Auditable tensor unpacker independent of the native decode kernel."""

    rows = []
    page_size = cache.shape[1]
    for slot in slots.cpu().tolist():
        page, pos = divmod(int(slot), page_size)
        for head in range(heads):
            raw = cache[page, pos, head].cpu()
            codes = []
            for block in range(nblocks):
                bit = block * code_bits
                byte, shift = divmod(bit, 8)
                word = int(raw[offset + byte])
                if shift + code_bits > 8:
                    word |= int(raw[offset + byte + 1]) << 8
                if shift + code_bits > 16:
                    word |= int(raw[offset + byte + 2]) << 16
                codes.append((word >> shift) & ((1 << code_bits) - 1))
            rows.append(codes)
    return torch.tensor(rows, dtype=torch.int32, device=cache.device)


def _unpack_scale(cache, slots, heads, offset):
    values = []
    page_size = cache.shape[1]
    for slot in slots.cpu().tolist():
        page, pos = divmod(int(slot), page_size)
        for head in range(heads):
            raw = cache[page, pos, head].cpu()
            word = int(raw[offset]) | (int(raw[offset + 1]) << 8)
            values.append(word)
    words = torch.tensor(values, dtype=torch.uint16)
    return words.view(torch.float16).to(cache.device)


def _materialize(codes, scales, cb, block, rotation, shape, dtype):
    rotated = lookup_codewords(cb, codes.long()).reshape(*shape[:-1], shape[-1])
    rotated = rotated * scales.reshape(*shape[:-1], 1).float()
    return (rotated @ rotation.t()).to(dtype)


def _materialize_paged_dequant_reference(
    codes, scales, cb, rotation, *, tokens, heads, alloc_len, head_dim, dtype
):
    """Match the native bulk-dequant GEMM batch/row shape before comparison.

    CUDA fp16/bf16 output rounding can differ by one ulp when the same fp32
    operands are sent through different GEMM row/batch shapes.  The old smoke
    compared native ``[1,H,alloc,D]`` rotation against a compact ``[N,H,D]``
    rotation, so ``tokens=1`` failed even on the unmodified baseline.  This
    reference pads in rotated space and uses the native batch shape; it still
    compares the requested token payload at zero tolerance.
    """

    rotated = lookup_codewords(cb, codes.long()).reshape(tokens, heads, head_dim)
    rotated = rotated * scales.reshape(tokens, heads, 1).float()
    padded = torch.zeros(1, heads, alloc_len, head_dim, device=codes.device)
    padded[0, :, :tokens].copy_(rotated.transpose(0, 1))
    materialized = (padded @ rotation.t()).to(dtype)
    return materialized[0, :, :tokens].transpose(0, 1)


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    recipe = _recipe(args)
    recipe.validate_head_dim(args.head_dim)
    layout = recipe.layout(args.head_dim)
    torch.manual_seed(args.seed)
    n, hk, hq, d = args.tokens, args.kv_heads, args.query_heads, args.head_dim
    if hq % hk:
        raise ValueError("query_heads must be divisible by kv_heads")
    page_size = 16
    pages = (n + page_size - 1) // page_size
    cache = torch.full(
        (pages, page_size, hk, layout["slot_size_aligned"]),
        0xA5,
        device=device,
        dtype=torch.uint8,
    )
    slots = torch.arange(n, device=device, dtype=torch.int32)
    key = torch.randn(n, hk, d, device=device, dtype=dtype)
    value = torch.randn_like(key)
    query = torch.randn(1, hq, d, device=device, dtype=dtype)
    _, kr, vr, kcb, vcb = _tables(recipe, d, device)
    key_lut, value_lut = _luts(recipe, d, device)
    key_score, key_csq, value_score, value_csq = _scoring_tables(recipe, kcb, vcb)
    store_aux = {
        "key_lut": key_lut,
        "value_lut": value_lut,
        "key_scoring_codebook": key_score,
        "key_scoring_csq": key_csq,
        "value_scoring_codebook": value_score,
        "value_scoring_csq": value_csq,
    }

    public_k = _public_payload(key, recipe, "k")
    public_v = _public_payload(value, recipe, "v")
    payload = triton_full_hayatequant_store(
        key,
        value,
        cache,
        slots,
        kr,
        vr,
        kcb,
        vcb,
        recipe,
        **store_aux,
    )
    torch.cuda.synchronize()
    assert torch.equal(
        payload["key_codes"], public_k["codes"].reshape_as(payload["key_codes"]).int()
    )
    assert torch.equal(
        payload["value_codes"], public_v["codes"].reshape_as(payload["value_codes"]).int()
    )
    assert torch.equal(payload["key_scales"], public_k["scale"].reshape(-1))
    assert torch.equal(payload["value_scales"], public_v["scale"].reshape(-1))

    packed_k = _unpack_stream(
        cache, slots, hk, 0, recipe.key_code_bits_per_block, d // recipe.key_block_size
    )
    packed_v = _unpack_stream(
        cache,
        slots,
        hk,
        layout["value_codes_offset"],
        recipe.value_code_bits_per_block,
        d // recipe.value_block_size,
    )
    packed_ks = _unpack_scale(cache, slots, hk, layout["key_scale_offset"])
    packed_vs = _unpack_scale(cache, slots, hk, layout["value_scale_offset"])
    assert torch.equal(packed_k, payload["key_codes"])
    assert torch.equal(packed_v, payload["value_codes"])
    assert torch.equal(packed_ks, payload["key_scales"])
    assert torch.equal(packed_vs, payload["value_scales"])
    if layout["slot_padding_bytes"]:
        assert torch.equal(
            cache[..., layout["slot_size"] :],
            torch.full_like(cache[..., layout["slot_size"] :], 0xA5),
        )

    # The formal table assignment is not allowed to change with workload
    # shape.  Store the last vector again as a one-token update and require
    # identical returned payload fields and packed bytes to the prefill row.
    step_cache = torch.full(
        (1, page_size, hk, layout["slot_size_aligned"]),
        0xA5,
        device=device,
        dtype=torch.uint8,
    )
    step_slot = torch.zeros(1, device=device, dtype=torch.int32)
    step_payload = triton_full_hayatequant_store(
        key[-1:],
        value[-1:],
        step_cache,
        step_slot,
        kr,
        vr,
        kcb,
        vcb,
        recipe,
        **store_aux,
    )
    last_rows = slice((n - 1) * hk, n * hk)
    for name in ("key_codes", "key_scales", "value_codes", "value_scales"):
        assert torch.equal(step_payload[name], payload[name][last_rows]), name
    last_page, last_pos = divmod(n - 1, page_size)
    assert torch.equal(step_cache[0, 0], cache[last_page, last_pos])

    bt = torch.arange(pages, device=device, dtype=torch.int32).view(1, -1)
    k_out = torch.empty(1, hk, pages * page_size, d, device=device, dtype=dtype)
    v_out = torch.empty_like(k_out)
    triton_full_hayatequant_dequant_kv(cache, bt, kcb, vcb, kr, vr, recipe, k_out, v_out)
    torch.cuda.synchronize()
    alloc_len = pages * page_size
    k_ref = _materialize_paged_dequant_reference(
        packed_k,
        packed_ks,
        kcb,
        kr,
        tokens=n,
        heads=hk,
        alloc_len=alloc_len,
        head_dim=d,
        dtype=dtype,
    )
    v_ref = _materialize_paged_dequant_reference(
        packed_v,
        packed_vs,
        vcb,
        vr,
        tokens=n,
        heads=hk,
        alloc_len=alloc_len,
        head_dim=d,
        dtype=dtype,
    )
    torch.testing.assert_close(k_out[0, :, :n].transpose(0, 1), k_ref, rtol=0, atol=0)
    torch.testing.assert_close(v_out[0, :, :n].transpose(0, 1), v_ref, rtol=0, atol=0)

    distortion = {}
    if recipe.assignment_mode == "lut":
        exact_k = _public_payload(key, recipe, "k", assignment_mode="exact")
        exact_v = _public_payload(value, recipe, "v", assignment_mode="exact")
        exact_k_codes = exact_k["codes"].reshape_as(packed_k)
        exact_v_codes = exact_v["codes"].reshape_as(packed_v)
        exact_k_scale = exact_k["scale"].reshape(-1)
        exact_v_scale = exact_v["scale"].reshape(-1)
        lut_recon = {
            "key": _materialize(
                packed_k, packed_ks, kcb, recipe.key_block_size, kr, key.shape, torch.float32
            ),
            "value": _materialize(
                packed_v, packed_vs, vcb, recipe.value_block_size, vr, value.shape, torch.float32
            ),
        }
        exact_recon = {
            "key": _materialize(
                exact_k_codes,
                exact_k_scale,
                kcb,
                recipe.key_block_size,
                kr,
                key.shape,
                torch.float32,
            ),
            "value": _materialize(
                exact_v_codes,
                exact_v_scale,
                vcb,
                recipe.value_block_size,
                vr,
                value.shape,
                torch.float32,
            ),
        }
        for role, source, lut_codes, exact_codes in (
            ("key", key, packed_k, exact_k_codes),
            ("value", value, packed_v, exact_v_codes),
        ):
            lut_mse = (lut_recon[role] - source.float()).square().mean().item()
            exact_mse = (exact_recon[role] - source.float()).square().mean().item()
            excess_pct = (lut_mse / exact_mse - 1.0) * 100.0
            distortion[role] = {
                "lut_mse": lut_mse,
                "exact_all_codewords_mse": exact_mse,
                "lut_excess_mse_pct": excess_pct,
                "final_code_mismatch_fraction": ((lut_codes != exact_codes).float().mean().item()),
                "passes_max_excess_mse_pct": excess_pct <= args.max_excess_distortion_pct,
            }
            if excess_pct > args.max_excess_distortion_pct:
                raise AssertionError(
                    f"{role} LUT excess distortion {excess_pct:.6f}% exceeds "
                    f"gate {args.max_excess_distortion_pct:.6f}%"
                )
    else:
        distortion = {
            "reference_only": True,
            "reason": "paper-original/exact development assignment",
        }

    seq_lens = torch.tensor([n], device=device, dtype=torch.int32)
    native = triton_full_hayatequant_decode_attention(
        query,
        cache,
        bt,
        seq_lens,
        kr,
        vr,
        kcb,
        vcb,
        recipe,
        d**-0.5,
        max_num_kv_splits=1,
    )
    groups = hq // hk
    k_gqa = k_ref.repeat_interleave(groups, dim=1)
    v_gqa = v_ref.repeat_interleave(groups, dim=1)
    scores = torch.einsum("bhd,thd->bht", query.float(), k_gqa.float()) * d**-0.5
    weights = scores.softmax(dim=-1)
    reference = torch.einsum("bht,thd->bhd", weights, v_gqa.float()).to(dtype)
    atol = 1.5e-2 if dtype == torch.bfloat16 else 4e-3
    torch.testing.assert_close(native, reference, rtol=2e-3, atol=atol)

    # Explicitly prove the value-linearity optimization independently of the
    # packed decode kernel and before low-precision output rounding.
    vrot = lookup_codewords(vcb, packed_v.long()).reshape(n, hk, d)
    vrot = vrot * packed_vs.reshape(n, hk, 1).float()
    vrot_gqa = vrot.repeat_interleave(groups, dim=1)
    folded = torch.einsum("bht,thd->bhd", weights, vrot_gqa) @ vr.t()
    # Compare before the final model-dtype cast.  The two expressions use a
    # different fp32 GEMM/reduction order, so exact associativity is not
    # expected; the bound covers that accumulation-order effect, not an extra
    # cache quantization approximation (the stored scales above are already
    # the exact public-codec fp16 values).
    vraw = vrot @ vr.t()
    direct = torch.einsum("bht,thd->bhd", weights, vraw.repeat_interleave(groups, dim=1))
    torch.testing.assert_close(folded, direct, rtol=2e-4, atol=3e-4)

    # Small synchronized timing is diagnostic only; the separate
    # counterbalanced fresh-process harness owns cross-method speed claims.
    for _ in range(args.warmups):
        triton_full_hayatequant_store(
            key,
            value,
            cache,
            slots,
            kr,
            vr,
            kcb,
            vcb,
            recipe,
            **store_aux,
        )
        triton_full_hayatequant_decode_attention(
            query,
            cache,
            bt,
            seq_lens,
            kr,
            vr,
            kcb,
            vcb,
            recipe,
            d**-0.5,
            max_num_kv_splits=1,
        )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        triton_full_hayatequant_store(
            key,
            value,
            cache,
            slots,
            kr,
            vr,
            kcb,
            vcb,
            recipe,
            **store_aux,
        )
    torch.cuda.synchronize()
    store_ms = (time.perf_counter() - t0) * 1000 / args.iters
    t0 = time.perf_counter()
    for _ in range(args.iters):
        triton_full_hayatequant_decode_attention(
            query,
            cache,
            bt,
            seq_lens,
            kr,
            vr,
            kcb,
            vcb,
            recipe,
            d**-0.5,
            max_num_kv_splits=1,
        )
    torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - t0) * 1000 / args.iters

    sources = [
        Path("vllm_plugins/kvcomp/hayate_full_config.py"),
        Path("vllm_plugins/kvcomp/hayate_full_triton.py"),
        Path("vllm_plugins/kvcomp/hayate_full_native.py"),
        Path("benchmark/kvcomp-codec/hayatequant_vllm_full_native_smoke.py"),
    ]
    return {
        "schema": {"name": "onecomp.hq_full_native_correctness_smoke", "version": 2},
        "ok": True,
        "formal_quality_benchmark": False,
        "cross_method_speed_claim_allowed": False,
        "implementation_contract": {
            "proposal_name": "HQ",
            "main_story": "paper HayateQuant fixed role-aware recipe",
            "key_path": (
                f"data-free rotation + fixed grid-LUT T={recipe.lut_candidates} "
                "block-VQ assignment + NC"
                if recipe.assignment_mode == "lut"
                else "data-free rotation + exact short block-VQ + NC"
            ),
            "value_path": (
                f"data-free rotation + fixed grid-LUT T={recipe.lut_candidates} "
                "block-VQ + LS-reassign-LS"
                if recipe.assignment_mode == "lut"
                else "data-free rotation + exact block-VQ + LS-reassign-LS"
            ),
            "public_encoder": recipe.assignment_mode,
            "exact_assignment_implementation": (
                "all_codewords_tensorcore_score_fused_argmax"
                if recipe.assignment_mode == "exact"
                and recipe.key_block_size == 2
                and recipe.value_block_size == 2
                else "public_tensor_primitive"
            ),
            "approximate_lut": recipe.assignment_mode == "lut",
            "persistent_encoder_acceleration_bytes": 0,
            "scale_storage": "one signed fp16 scale per role vector",
            "value_inverse_rotation": "one fp32 GEMM after rotated-space attention accumulation",
            "payload_parity_with_public_codec": True,
        },
        "recipe": recipe.provenance(d),
        "parity": {
            "key_codes_exact": True,
            "value_codes_exact": True,
            "key_fp16_scales_exact": True,
            "value_fp16_scales_exact": True,
            "packed_stream_exact": True,
            "store1_matches_prefill_payload_and_bytes": True,
            "full_dequant_exact_at_output_dtype": True,
            "attention_tolerance": {"rtol": 2e-3, "atol": atol},
            "rotated_value_linearity_tolerance": {"rtol": 2e-4, "atol": 3e-4},
        },
        "quality_gate_against_exact_assignment": {
            "max_role_excess_mse_pct": args.max_excess_distortion_pct,
            "distortion": distortion,
            "passed": all(
                row.get("passes_max_excess_mse_pct", True)
                for row in distortion.values()
                if isinstance(row, dict)
            ),
            "ppl_required_separately": recipe.assignment_mode == "lut",
        },
        "diagnostic_timing": {"store_ms": store_ms, "decode_ms": decode_ms},
        "provenance": {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "git_commit": _cmd(["git", "rev-parse", "HEAD"]),
            "git_status_porcelain": _cmd(["git", "status", "--porcelain"]),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "source_sha256": {str(path): _sha(REPO / path) for path in sources},
            "arguments": vars(args),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--avg-bits", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--recipe", choices=("hq", "paper-original"), default="hq")
    parser.add_argument("--key-code-bits", type=int)
    parser.add_argument("--key-block-size", type=int)
    parser.add_argument("--value-code-bits", type=int)
    parser.add_argument("--value-block-size", type=int)
    parser.add_argument("--allow-canonical-k10v6-b4", action="store_true")
    parser.add_argument("--assignment-mode", choices=("exact", "lut"))
    parser.add_argument("--lut-bins", type=int)
    parser.add_argument("--lut-candidates", type=int, choices=(2, 4, 8, 16))
    parser.add_argument("--dist", choices=("gaussian", "beta"), default="gaussian")
    parser.add_argument("--codebook-init", choices=("auto", "fibquant", "forgy"), default="auto")
    parser.add_argument(
        "--rotation-mode",
        choices=("auto", "dense", "hybrid", "hadamard_gemm"),
        default="auto",
    )
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--query-heads", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--max-excess-distortion-pct", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True, default=str)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

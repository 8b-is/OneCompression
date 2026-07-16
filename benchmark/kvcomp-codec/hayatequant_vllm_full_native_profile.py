#!/usr/bin/env python3
"""Stage-level GPU profile for canonical exact packed HQ.

This is a development diagnostic, not a cross-method benchmark.  It isolates
the faithful store's rotations, full-codebook assignments, codeword gathers,
scale/refit work, and final bit packing, and records the dominant CUDA
operators for both store and direct packed decode.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess

import torch
from torch.profiler import ProfilerActivity, profile

from hayatequant_vllm_full_native_smoke import _recipe, _scoring_tables, _tables
from onecomp.kvcomp.codec.hayate.vq import (
    lookup_codewords,
    quantize_exact_pre,
)
from vllm_plugins.kvcomp.hayate_full_triton import (
    _encode_role,
    _pack_encoded_hq,
    _quantize_exact_b2_tensorcore,
    triton_full_hayatequant_decode_attention,
    triton_full_hayatequant_store,
)

REPO = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, check=True, text=True, capture_output=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _event_summary(fn, *, warmups: int, iters: int) -> dict[str, float | int]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    values = [float(start.elapsed_time(end)) for start, end in zip(starts, ends, strict=True)]
    return {
        "n": len(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _operator_profile(fn, limit: int = 30) -> list[dict[str, float | int | str]]:
    fn()
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        fn()
        torch.cuda.synchronize()
    rows = []
    for event in prof.key_averages():
        device_us = float(event.self_device_time_total)
        if device_us <= 0.0:
            continue
        rows.append(
            {
                "operator": str(event.key),
                "self_device_ms": device_us / 1000.0,
                "device_total_ms": float(event.device_time_total) / 1000.0,
                "calls": int(event.count),
            }
        )
    return sorted(rows, key=lambda row: float(row["self_device_ms"]), reverse=True)[:limit]


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.assignment_mode != "exact":
        raise ValueError("the canonical profiling contract is exact assignment only")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    n, hk, hq, d = args.tokens, args.kv_heads, args.query_heads, args.head_dim
    recipe = _recipe(args)
    if recipe.assignment_mode != "exact":
        raise AssertionError("fixed HQ unexpectedly resolved away from exact")
    layout = recipe.layout(d)
    pages = (n + args.page_size - 1) // args.page_size
    key = torch.randn(n, hk, d, device=device, dtype=dtype)
    value = torch.randn_like(key)
    query = torch.randn(1, hq, d, device=device, dtype=dtype)
    slots = torch.arange(n, device=device, dtype=torch.int32)
    cache = torch.empty(
        pages,
        args.page_size,
        hk,
        layout["slot_size_aligned"],
        device=device,
        dtype=torch.uint8,
    )
    block_table = torch.arange(pages, device=device, dtype=torch.int32).view(1, -1)
    seq_lens = torch.tensor([n], device=device, dtype=torch.int32)
    _, kr, vr, kcb, vcb = _tables(recipe, d, device)
    kscore, kcsq, vscore, vcsq = _scoring_tables(recipe, kcb, vcb)
    store_aux = {
        "key_scoring_codebook": kscore,
        "key_scoring_csq": kcsq,
        "value_scoring_codebook": vscore,
        "value_scoring_csq": vcsq,
    }
    mid = torch.empty(1, hq, args.max_splits, d + 1, device=device, dtype=torch.float32)
    rotated = torch.empty(1, hq, d, device=device, dtype=torch.float32)
    lse = torch.empty(1, hq, device=device, dtype=torch.float32)

    # Fixed inputs for isolated stages.  Their setup is intentionally outside
    # each stage's event interval.
    kxr = key.float() @ kr
    vxr = value.float() @ vr
    krms = kxr.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1.0e-8)
    vrms = vxr.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1.0e-8)
    kblocks = (kxr / krms).reshape(-1, recipe.key_block_size).half()
    vblocks = (vxr / vrms).reshape(-1, recipe.value_block_size).half()
    kcodes = quantize_exact_pre(kblocks, kscore, kcsq)
    vcodes = quantize_exact_pre(vblocks, vscore, vcsq)

    def key_encode():
        return _encode_role(
            key,
            kr,
            kcb,
            recipe.key_block_size,
            "nc",
            lut=None,
            lut_bins=recipe.lut_bins,
            lut_limit=recipe.lut_limit,
            scoring_codebook=kscore,
            scoring_csq=kcsq,
        )

    def value_encode():
        return _encode_role(
            value,
            vr,
            vcb,
            recipe.value_block_size,
            "ls_iter",
            lut=None,
            lut_bins=recipe.lut_bins,
            lut_limit=recipe.lut_limit,
            scoring_codebook=vscore,
            scoring_csq=vcsq,
        )

    encoded_key_codes, encoded_key_scales = key_encode()
    encoded_value_codes, encoded_value_scales = value_encode()

    def pack_only():
        return _pack_encoded_hq(
            encoded_key_codes,
            encoded_key_scales,
            encoded_value_codes,
            encoded_value_scales,
            cache,
            slots,
            recipe,
            d,
        )

    def store_all():
        return triton_full_hayatequant_store(
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

    def decode():
        return triton_full_hayatequant_decode_attention(
            query,
            cache,
            block_table,
            seq_lens,
            kr,
            vr,
            kcb,
            vcb,
            recipe,
            d**-0.5,
            mid_o_buf=mid,
            rotated_output_buf=rotated,
            lse_buf=lse,
            max_num_kv_splits=args.max_splits,
        )

    stage_fns = {
        "store_total": store_all,
        "store_key_rotation": lambda: key.float() @ kr,
        "store_value_rotation": lambda: value.float() @ vr,
        "store_key_public_exact_assignment_reference": lambda: quantize_exact_pre(
            kblocks, kscore, kcsq
        ),
        "store_value_public_exact_assignment_one_pass_reference": lambda: quantize_exact_pre(
            vblocks, vscore, vcsq
        ),
        "store_key_native_fused_exact_assignment": lambda: _quantize_exact_b2_tensorcore(
            kblocks, kscore, kcsq
        ),
        "store_value_native_fused_exact_assignment_one_pass": lambda: (
            _quantize_exact_b2_tensorcore(vblocks, vscore, vcsq)
        ),
        "store_key_codeword_lookup": lambda: lookup_codewords(kcb, kcodes),
        "store_value_codeword_lookup_one_pass": lambda: lookup_codewords(vcb, vcodes),
        "store_key_full_encode": key_encode,
        "store_value_full_encode_two_assignments": value_encode,
        "store_pack_only": pack_only,
        "decode_total": decode,
        "decode_query_rotation": lambda: query.float() @ kr,
        "decode_value_inverse_rotation": lambda: rotated @ vr.t(),
    }
    stages = {
        name: _event_summary(fn, warmups=args.warmups, iters=args.iters)
        for name, fn in stage_fns.items()
    }
    total_store = float(stages["store_total"]["median_ms"])
    estimated_search = float(
        stages["store_key_native_fused_exact_assignment"]["median_ms"]
    ) + 2.0 * float(stages["store_value_native_fused_exact_assignment_one_pass"]["median_ms"])
    attribution = {
        "estimated_exact_search_ms_key_plus_two_value_passes": estimated_search,
        "estimated_exact_search_fraction_of_store": estimated_search / total_store,
        "assignment_implementation": "all_codewords_tensorcore_score_fused_argmax",
        "approximate_lut": False,
        "persistent_acceleration_table_bytes": 0,
        "sum_full_role_encode_plus_pack_ms": (
            float(stages["store_key_full_encode"]["median_ms"])
            + float(stages["store_value_full_encode_two_assignments"]["median_ms"])
            + float(stages["store_pack_only"]["median_ms"])
        ),
        "key_codebook_size": int(kcb.shape[0]),
        "value_codebook_size": int(vcb.shape[0]),
    }
    sources = [
        Path("benchmark/kvcomp-codec/hayatequant_vllm_full_native_profile.py"),
        Path("vllm_plugins/kvcomp/hayate_full_config.py"),
        Path("vllm_plugins/kvcomp/hayate_full_triton.py"),
        Path("onecomp/kvcomp/codec/hayate/vq.py"),
    ]
    return {
        "schema": {"name": "onecomp.hq_full_native_stage_profile", "version": 1},
        "development_only": True,
        "cross_method_speed_claim_allowed": False,
        "recipe": recipe.provenance(d),
        "shape": {
            "tokens": n,
            "kv_heads": hk,
            "query_heads": hq,
            "head_dim": d,
            "dtype": args.dtype,
            "page_size": args.page_size,
            "max_splits": args.max_splits,
        },
        "stages": stages,
        "attribution": attribution,
        "operator_profiles": {
            "store_top_cuda_operators": _operator_profile(store_all),
            "decode_top_cuda_operators": _operator_profile(decode),
        },
        "provenance": {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_status_porcelain": _git("status", "--porcelain"),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "source_sha256": {str(path): _sha(REPO / path) for path in sources},
            "arguments": vars(args),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--avg-bits", type=int, choices=(3, 4), default=3)
    parser.add_argument("--recipe", choices=("hq",), default="hq")
    parser.add_argument("--assignment-mode", choices=("exact",), default="exact")
    parser.add_argument("--dist", choices=("gaussian",), default="gaussian")
    parser.add_argument("--codebook-init", choices=("auto",), default="auto")
    parser.add_argument("--rotation-mode", choices=("auto",), default="auto")
    parser.add_argument("--allow-canonical-k10v6-b4", action="store_true")
    parser.add_argument("--key-code-bits", type=int)
    parser.add_argument("--key-block-size", type=int)
    parser.add_argument("--value-code-bits", type=int)
    parser.add_argument("--value-block-size", type=int)
    parser.add_argument("--lut-bins", type=int)
    parser.add_argument("--lut-candidates", type=int)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--max-splits", type=int, default=32)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
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

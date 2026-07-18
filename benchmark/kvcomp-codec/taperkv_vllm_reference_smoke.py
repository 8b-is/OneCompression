#!/usr/bin/env python3
"""Run a small, provenance-bearing TapeKV vLLM semantic smoke test.

This launcher intentionally exercises ``TaperKVNativeBackend`` rather than
passing ``kv_cache_dtype``.  The backend uses an FP16/BF16 paged cache and
fake-quantizes logical KV values before FlashAttention, so its JSON output is
labelled ``fp16_reference_fake_quant`` and cannot be consumed by packed-memory
or throughput tables.

Example (from any checkout)::

    python benchmark/kvcomp-codec/taperkv_vllm_reference_smoke.py \
      --model /models/Qwen3-0.6B --output results/taperkv_vllm_smoke.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_PROMPTS = [
    "The capital of France is",
    "Explain one correctness risk of mutating a paged KV cache during batched decoding.",
    (
        "KV cache compression must preserve request isolation, page-table order, layer "
        "selection, prefix-cache safety, and the distinction between a logical fake-quant "
        "reference path and a physically packed serving backend. Summarize this requirement."
    ),
]


def _runtime_evidence_path(output: Path) -> Path:
    return output.with_suffix(f"{output.suffix}.runtime.jsonl")


def _load_runtime_evidence(path: Path, *, combined: bool) -> tuple[list[dict], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"TapeKV backend produced no runtime evidence: {path}") from exc
    events = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid runtime evidence JSON at {path}:{line_number}") from exc
        if not isinstance(event, dict):
            raise RuntimeError(f"runtime evidence line {line_number} is not an object")
        events.append(event)
    expected_schema = {"name": "onecomp.kvcomp_vllm_runtime_evidence", "version": 1}
    matching = [
        event
        for event in events
        if event.get("schema") == expected_schema
        and event.get("backend") == "vllm_plugins.kvcomp.taper_native.TaperKVNativeBackend"
        and event.get("event") == "logical_paged_cache_regraded"
        and event.get("hayate_then_taper") is combined
    ]
    if not matching:
        raise RuntimeError(
            "runtime evidence does not prove the selected TapeKV regrade path executed"
        )
    if not any(
        isinstance(event.get("maximum_pages_per_request"), int)
        and event["maximum_pages_per_request"] > 1
        for event in matching
    ):
        raise RuntimeError("runtime evidence does not prove a multi-page cache regrade")
    return events, hashlib.sha256(raw).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local/HF model path accepted by vLLM")
    parser.add_argument("--output", required=True, type=Path, help="JSON receipt to write")
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to generate; repeat for a batch (defaults to a three-prompt page-boundary suite)",
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--policy", choices=("taperkv", "block_taperkv"), default="taperkv")
    parser.add_argument("--key-bits", type=int, default=8)
    parser.add_argument("--value-bits", type=int, default=4)
    parser.add_argument("--key-floor", type=int, default=2)
    parser.add_argument("--value-floor", type=int, default=2)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--recent-tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--requant-stride", type=int, default=None)
    parser.add_argument(
        "--hayate-bits",
        type=int,
        choices=(2, 3, 4),
        default=None,
        help="Apply HayateQuant once to arriving KV before TapeKV (disabled by default)",
    )
    parser.add_argument("--hayate-value-bits", type=int, choices=(2, 3, 4), default=None)
    parser.add_argument("--hayate-block-size", type=int, default=2)
    parser.add_argument("--hayate-encoder", choices=("exact", "lut"), default="exact")
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.output = args.output.expanduser().resolve()
    evidence_path = _runtime_evidence_path(args.output)
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if evidence_path.exists():
        raise FileExistsError(f"runtime evidence already exists: {evidence_path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["KVCOMP_VLLM_RUNTIME_EVIDENCE"] = str(evidence_path)
    prompts = args.prompts or DEFAULT_PROMPTS
    if not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts):
        raise ValueError("all prompts must be non-empty strings")
    venv_bin = Path(sys.executable).parent
    os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"
    os.environ.setdefault("VLLM_PLUGINS", "kvcomp")
    # This BF16 attention correctness smoke does not exercise optional weight
    # DeepGEMM.  Avoid an unrelated vLLM 0.20 B200 warmup failure when the
    # bundled DeepGEMM package is present but lacks a newer required symbol.
    os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
    # Do not import vLLM before the explicit registry installation: EngineCore
    # children receive this environment payload and the general plugin repeats
    # registration in their fresh interpreter.
    from vllm_plugins.kvcomp.taper_config import TaperVLLMRecipe
    from vllm_plugins.kvcomp import vllm_plugin

    policy = {
        "name": args.policy,
        "key_bits": args.key_bits,
        "value_bits": args.value_bits,
        "b_min": args.key_floor,
        "b_min_value": args.value_floor,
        "sink_tokens": args.sink_tokens,
        "recent_tokens": args.recent_tokens,
        "requant_stride": args.requant_stride,
    }
    if args.policy == "block_taperkv":
        policy.update({"block_size": args.block_size, "residual_fp16": True})
    hayate_config = None
    if args.hayate_bits is not None:
        from onecomp.kvcomp.codec.hayate import HayateQuantCodec

        hayate_config = HayateQuantCodec(
            bits=args.hayate_bits,
            value_bits=args.hayate_value_bits or args.hayate_bits,
            block_size=args.hayate_block_size,
            encoder=args.hayate_encoder,
        ).get_kv_config()
    recipe = TaperVLLMRecipe.from_policy(policy, hayate_config=hayate_config)
    plan = vllm_plugin.KVCompVLLMPlan(
        native_supported=True,
        taper_native=True,
        taper_config=recipe.to_mapping(),
    )
    vllm_plugin._install_taper_native_backend(plan, vllm_plugin._get_logger())
    # The model itself is intentionally unmodified.  Preserve the explicit
    # recipe in spawned EngineCore workers, where the automatic model-config
    # hook would otherwise (correctly for ordinary models) clear stale state.
    os.environ["KVCOMP_VLLM_TAPER_CONFIG_EXPLICIT"] = "1"

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=Path(args.model).is_dir(),
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    prompt_token_lengths = [len(tokenizer(prompt).input_ids) for prompt in prompts]
    required_tokens = max(prompt_token_lengths) + args.max_tokens
    if required_tokens > args.max_model_len:
        raise ValueError(
            "longest prompt plus max tokens exceeds max model length: "
            f"{max(prompt_token_lengths)}+{args.max_tokens}>{args.max_model_len}"
        )

    llm = LLM(
        model=args.model,
        kv_cache_dtype="auto",
        enable_prefix_caching=False,
        enforce_eager=True,
        attention_backend="FLASH_ATTN",
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        gpu_memory_utilization=0.5,
    )
    cache_block_size = int(llm.llm_engine.vllm_config.cache_config.block_size)
    if max(prompt_token_lengths) <= cache_block_size:
        raise RuntimeError(
            "TapeKV smoke must include a prompt spanning more than one physical KV page; "
            f"longest prompt={max(prompt_token_lengths)}, block_size={cache_block_size}"
        )
    results = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True),
    )
    generations = []
    total_generated_tokens = 0
    if len(results) != len(prompts):
        raise RuntimeError("vLLM returned a different number of requests than submitted")
    for prompt, result in zip(prompts, results):
        if not result.outputs:
            raise RuntimeError("vLLM returned a request without a generation candidate")
        output = result.outputs[0]
        token_count = len(output.token_ids)
        if token_count != args.max_tokens:
            raise RuntimeError(
                f"expected {args.max_tokens} generated tokens with ignore_eos, got {token_count}"
            )
        total_generated_tokens += token_count
        generations.append(
            {
                "prompt": prompt,
                "text": output.text,
                "tokens": token_count,
                "finish_reason": output.finish_reason,
            }
        )
    runtime_evidence, runtime_evidence_sha256 = _load_runtime_evidence(
        evidence_path, combined=hayate_config is not None
    )
    receipt = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model": args.model,
        "backend": "vllm_plugins.kvcomp.taper_native.TaperKVNativeBackend",
        "implementation": recipe.implementation,
        "physical_kv_cache": "vllm model-dtype paged cache (not packed)",
        "eligible_for_packed_memory_or_speed_table": False,
        "prefix_caching": False,
        "enforce_eager": True,
        "policy": recipe.to_mapping(),
        "num_prompts": len(prompts),
        "prompt_token_lengths": prompt_token_lengths,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "cache_block_size": cache_block_size,
        "exercises_batched_requests": len(prompts) > 1,
        "exercises_multi_page_prompt": max(prompt_token_lengths) > cache_block_size,
        "generated_tokens": total_generated_tokens,
        "prompt": prompts[0],
        "generated_text": generations[0]["text"],
        "generations": generations,
        "runtime_evidence": {
            "path": str(evidence_path),
            "sha256": runtime_evidence_sha256,
            "events": runtime_evidence,
            "custom_backend_execution_proved": True,
        },
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    args.output.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

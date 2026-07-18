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
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local/HF model path accepted by vLLM")
    parser.add_argument("--output", required=True, type=Path, help="JSON receipt to write")
    parser.add_argument("--prompt", default="The capital of France is")
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
    venv_bin = Path(sys.executable).parent
    os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"
    os.environ.setdefault("VLLM_PLUGINS", "kvcomp")
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
    result = llm.generate(
        [args.prompt], SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    )
    text = result[0].outputs[0].text
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
        "prompt": args.prompt,
        "generated_text": text,
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

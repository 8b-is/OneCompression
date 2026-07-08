"""Generate text with native vLLM TurboQuant vs HayateQuant.

This is a paper-supporting sanity/quality harness rather than a kernel smoke:
it loads an actual HF model through vLLM, generates from a small multilingual
prompt suite, records generated text, token throughput, and GPU memory stats.

Run each variant in a fresh process.  HayateQuant is implemented by setting a
``KVCOMP_VLLM_HAYATE_NATIVE_CONFIG`` environment payload and registering an
override of vLLM's ``TURBOQUANT`` attention backend; keeping variants in fresh
processes avoids registry ambiguity.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch


DEFAULT_PROMPTS = [
    "Explain why the sky appears blue in three concise sentences.",
    "Write a Python function that returns the Fibonacci numbers up to n.",
    "次の文章を自然な日本語で一段落にまとめてください: 大規模言語モデルの推論ではKVキャッシュが長文処理のボトルネックになります。",
    "A researcher compresses the KV cache of a transformer. Describe one risk and one validation check.",
    "Translate to English: 量子化はメモリ使用量を削減するが、精度劣化を慎重に測定する必要がある。",
]

_HAYATE_NATIVE_CONFIG_ENV = "KVCOMP_VLLM_HAYATE_NATIVE_CONFIG"


def _dtype_for(bits: int) -> str:
    if bits == 3:
        return "turboquant_3bit_nc"
    if bits == 4:
        return "turboquant_4bit_nc"
    raise ValueError("native vLLM generation eval supports bits in {3, 4}")


def _configure_hayate(bits: int, key_scale_mode: str, codebook_init: str) -> None:
    from vllm.v1.attention.backends.registry import (  # pylint: disable=import-outside-toplevel
        AttentionBackendEnum,
        register_backend,
    )

    cfg = {
        "bits": bits,
        "value_bits": bits,
        "dist": "gaussian",
        "block_size": None,
        "max_codebook_bits": 9,
        "codebook_init": codebook_init,
        "rotation_mode": "auto",
        "key_scale_mode": key_scale_mode,
        "seed": 0,
    }
    os.environ[_HAYATE_NATIVE_CONFIG_ENV] = json.dumps(cfg, sort_keys=True)
    register_backend(
        AttentionBackendEnum.TURBOQUANT,
        "vllm_plugins.kvcomp.hayate_native.HayateQuantNativeBackend",
    )


def _language_metrics(text: str) -> dict:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return {"chars": 0, "printable_ratio": 0.0, "unique_char_ratio": 0.0}
    printable = sum(c.isprintable() for c in chars) / len(chars)
    unique = len(set(chars)) / len(chars)
    return {
        "chars": len(chars),
        "printable_ratio": round(printable, 4),
        "unique_char_ratio": round(unique, 4),
    }


def _build_llm(args: argparse.Namespace):
    from vllm import LLM  # pylint: disable=import-outside-toplevel

    kv_cache_dtype = "auto"
    if args.variant == "turboquant":
        kv_cache_dtype = _dtype_for(args.bits)
    elif args.variant == "hayate":
        kv_cache_dtype = _dtype_for(args.bits)
        _configure_hayate(args.bits, args.key_scale_mode, args.codebook_init)
    elif args.variant != "auto":
        raise ValueError(f"unknown variant: {args.variant}")

    return LLM(
        model=str(args.model_path),
        trust_remote_code=True,
        dtype=args.dtype,
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )


def run(args: argparse.Namespace) -> dict:
    from vllm import SamplingParams  # pylint: disable=import-outside-toplevel

    prompts = DEFAULT_PROMPTS if args.prompt_file is None else [
        line.strip() for line in args.prompt_file.read_text().splitlines() if line.strip()
    ]
    torch.cuda.reset_peak_memory_stats()
    mem_free_before, mem_total = torch.cuda.mem_get_info()
    t_load0 = time.perf_counter()
    llm = _build_llm(args)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - t_load0
    mem_free_after_load, _ = torch.cuda.mem_get_info()

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
    )
    # Warmup one short prompt so the measured pass is not dominated by first-use
    # Triton/CUDA graph setup.
    llm.generate([prompts[0]], SamplingParams(temperature=0.0, max_tokens=8))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    mem_free_after, _ = torch.cuda.mem_get_info()
    gen_rows = []
    gen_tokens = 0
    for prompt, out in zip(prompts, outputs):
        text = out.outputs[0].text
        token_ids = out.outputs[0].token_ids
        gen_tokens += len(token_ids)
        gen_rows.append({
            "prompt": prompt,
            "text": text,
            "tokens": len(token_ids),
            "finish_reason": out.outputs[0].finish_reason,
            "language_metrics": _language_metrics(text),
        })

    chars = [row["language_metrics"]["chars"] for row in gen_rows]
    printable = [row["language_metrics"]["printable_ratio"] for row in gen_rows]
    unique = [row["language_metrics"]["unique_char_ratio"] for row in gen_rows]
    hayate_cfg = None
    if os.environ.get(_HAYATE_NATIVE_CONFIG_ENV):
        hayate_cfg = json.loads(os.environ[_HAYATE_NATIVE_CONFIG_ENV])
    return {
        "variant": args.variant,
        "bits": None if args.variant == "auto" else args.bits,
        "model_path": str(args.model_path),
        "vllm_plugins": os.environ.get("VLLM_PLUGINS"),
        "hayate_native_config": hayate_cfg,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "num_prompts": len(prompts),
        "generated_tokens": gen_tokens,
        "elapsed_seconds": round(elapsed, 4),
        "tokens_per_second": round(gen_tokens / max(elapsed, 1e-9), 3),
        "load_seconds": round(load_seconds, 4),
        "cuda_memory_total_gib": round(mem_total / 2**30, 3),
        "cuda_memory_used_after_load_gib": round((mem_total - mem_free_after_load) / 2**30, 3),
        "cuda_memory_used_after_generate_gib": round((mem_total - mem_free_after) / 2**30, 3),
        "torch_peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        "quality_summary": {
            "min_nonspace_chars": min(chars) if chars else 0,
            "mean_nonspace_chars": round(statistics.mean(chars), 2) if chars else 0.0,
            "mean_printable_ratio": round(statistics.mean(printable), 4) if printable else 0.0,
            "mean_unique_char_ratio": round(statistics.mean(unique), 4) if unique else 0.0,
        },
        "generations": gen_rows,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--variant", choices=["auto", "turboquant", "hayate"], required=True)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--key-scale-mode", choices=["nc", "ub", "ip"], default="nc")
    ap.add_argument("--codebook-init", choices=["auto", "fibquant"], default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--prompt-file", type=Path)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_PLUGINS", "kvcomp")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    venv_bin = Path(sys.executable).resolve().parent
    os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

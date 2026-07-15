"""Zero-shot downstream benchmark for HayateQuant KV codecs.

This complements ``codec_benchmark.py`` with lm-eval-style multiple-choice
accuracy under the same ``CompressedDynamicCache`` path used for perplexity.
The paper driver evaluates the same model set as Table 1 and reports the mean
accuracy over ARC-Easy, ARC-Challenge, PIQA, HellaSwag, and WinoGrande.

Example
-------
Run one model in a GPU allocation:

    srun --gres=gpu:1 python benchmark/kvcomp-codec/zeroshot_benchmark.py \
        --model-path /lustre/.../models/llama-3.1-8b-instruct \
        --output-root /lustre/.../ichikawa_works/kvcomp_hayatequant/zeroshot

Use ``--limit`` only for smoke tests or fast ablations.  The paper command
defaults to the full validation splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from statistics import mean
from typing import Iterable

import torch
from filelock import FileLock
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from codec_benchmark import _codecs  # noqa: E402
from onecomp.kvcomp import CompressedDynamicCache, detect_outlier_layers  # noqa: E402
from onecomp.kvcomp.eval import zeroshot_accuracy  # noqa: E402


DEFAULT_TASKS = ("arc_easy", "arc_challenge", "piqa", "hellaswag", "winogrande")
DEFAULT_METHODS = (
    "FullKV",
    "TurboQuant-paper",
    "FibQuant",
    "BlockQuant",
    "HayateQuant-selected",
    "HayateQuant-frontier",
    "HayateQuant-auto",
    "AdaptiveHQ",
)


def _parse_bits(values: Iterable[str]) -> list[int]:
    return [int(v) for v in values]


def _method_list(args: argparse.Namespace, bits: int) -> list[str]:
    return list(args.methods)


def _build_cache_factory(
    model,
    codec,
    sink_tokens: int,
    recent_tokens: int,
    skip_layers: set[int],
):
    return lambda: CompressedDynamicCache(
        codec=codec,
        config=model.config,
        sink_tokens=sink_tokens,
        recent_tokens=recent_tokens,
        skip_layers=skip_layers if skip_layers else None,
    )


def _output_json(args: argparse.Namespace) -> Path:
    model_tag = os.path.basename(str(args.model_path).rstrip("/"))
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    return args.output_root / f"zeroshot_{model_tag}{suffix}.json"


def _benchmark_config(args: argparse.Namespace) -> dict:
    """Scientific run identity; output location and device are not results."""
    return {
        "schema_version": 2,
        "model_path": str(args.model_path.resolve()),
        "dtype": args.dtype,
        "bits": list(args.bits),
        "tasks": list(args.tasks),
        "methods": list(args.methods),
        "include_fullkv": bool(args.include_fullkv),
        "limit": args.limit,
        "sample_seed": args.sample_seed,
        "sink_tokens": args.sink_tokens,
        "recent_tokens": args.recent_tokens,
        "skip_outliers": bool(args.skip_outliers),
        "outlier_ratio": args.outlier_ratio,
        "ablation": bool(args.ablation),
    }


def _fingerprint(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _validate_existing(out_json: Path, expected_fingerprint: str) -> None:
    try:
        payload = json.loads(out_json.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Existing result is unreadable: {out_json}") from exc
    actual = payload.get("run_fingerprint")
    if payload.get("schema_version") != 2 or actual is None:
        raise RuntimeError(
            f"Refusing to reuse legacy result without a run fingerprint: {out_json}"
        )
    if actual != expected_fingerprint:
        raise RuntimeError(
            f"Existing result was produced by a different benchmark config: {out_json}"
        )


def _atomic_write_json(path: Path, payload: dict) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


@torch.no_grad()
def _detect_skips(model, tok, args: argparse.Namespace) -> set[int]:
    if not args.skip_outliers:
        return set()
    probe_texts = (
        "Solve: what is 17 times 23? Reason step by step.",
        "Repeat the following key exactly: QH-4821-ZX.",
        "Write a Python function that returns the n-th Fibonacci number.",
        "Explain briefly why the daytime sky appears blue.",
    )

    def _ids(text):
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True,
                return_tensors="pt", return_dict=True)["input_ids"].to(model.device)
        return tok(text, return_tensors="pt", truncation=True,
                   max_length=512).input_ids.to(model.device)

    return set(detect_outlier_layers(
        model, [_ids(text) for text in probe_texts],
        ratio_thresh=args.outlier_ratio, agg="mean"))


def _run(args: argparse.Namespace) -> Path:
    args.output_root.mkdir(parents=True, exist_ok=True)
    out_json = _output_json(args)
    run_config = _benchmark_config(args)
    run_fingerprint = _fingerprint(run_config)
    output_lock = FileLock(str(out_json) + ".lock")
    output_lock.acquire()
    if out_json.exists() and out_json.stat().st_size > 0:
        try:
            _validate_existing(out_json, run_fingerprint)
            print(f"[skip] matching existing result: {out_json}", flush=True)
            return out_json
        finally:
            output_lock.release()

    dtype = getattr(torch, args.dtype)
    try:
        tok = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=dtype,
            device_map=args.device,
        ).eval()
        skip_layers = _detect_skips(model, tok, args)
        if skip_layers:
            print(f"outlier layers (kept FP16): {sorted(skip_layers)}", flush=True)

        results: dict[str, dict] = {}
        fullkv_factory = lambda: DynamicCache(config=model.config)
        if args.include_fullkv:
            results["full"] = {
                "FullKV": _eval_method(model, tok, args, cache_factory=fullkv_factory)
            }

        for bits in args.bits:
            codecs = _codecs(bits, ablation=args.ablation)
            bit_key = str(bits)
            results[bit_key] = {}
            for method in _method_list(args, bits):
                if method == "FullKV":
                    if args.include_fullkv and "FullKV" in results.get("full", {}):
                        results[bit_key][method] = results["full"]["FullKV"]
                    else:
                        results[bit_key][method] = _eval_method(
                            model, tok, args, cache_factory=fullkv_factory
                        )
                    continue
                if method not in codecs:
                    print(f"[skip] {bits}bit {method}: not defined", flush=True)
                    continue
                cache_factory = _build_cache_factory(
                    model, codecs[method], args.sink_tokens,
                    args.recent_tokens, skip_layers
                )
                results[bit_key][method] = _eval_method(
                    model, tok, args, cache_factory=cache_factory
                )
                print(
                    f"[{bits}bit] {method}: {results[bit_key][method]}",
                    flush=True,
                )

        payload = {
            "schema_version": 2,
            "run_fingerprint": run_fingerprint,
            "model": os.path.basename(str(args.model_path).rstrip("/")),
            "tasks": list(args.tasks),
            "limit": args.limit,
            "sample_seed": args.sample_seed,
            "config": run_config,
            "results": results,
        }
        _atomic_write_json(out_json, payload)
        print(f"\nWrote {out_json}")
        print(_markdown(payload))
        return out_json
    finally:
        if output_lock.is_locked:
            output_lock.release()


def _eval_method(model, tok, args: argparse.Namespace, cache_factory):
    tasks = {}
    for task in args.tasks:
        res = zeroshot_accuracy(
            model,
            tok,
            task=task,
            cache_factory=cache_factory,
            limit=args.limit,
            sample_seed=args.sample_seed,
        ).as_dict()
        tasks[task] = res
        print(f"  {task}: {res}", flush=True)
    acc = [v["acc"] for v in tasks.values()]
    acc_norm = [v["acc_norm"] for v in tasks.values()]
    primary = [v["primary_score"] for v in tasks.values()]
    return {
        "tasks": tasks,
        "avg_acc": round(mean(acc), 4),
        "avg_acc_norm": round(mean(acc_norm), 4),
        "avg_primary_score": round(mean(primary), 4),
        "primary_metric_definition": (
            "custom unweighted task macro: lm-eval acc_norm for ARC/PIQA/"
            "HellaSwag and lm-eval acc for WinoGrande"
        ),
        "n_total": sum(v["n"] for v in tasks.values()),
    }


def _best_hayate(methods: dict) -> tuple[str, dict] | tuple[None, None]:
    candidates = [
        (name, data)
        for name, data in methods.items()
        if name.startswith("HayateQuant")
    ]
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[1].get(
        "avg_primary_score", item[1].get("avg_acc_norm", -1.0)))


def _markdown(payload: dict) -> str:
    lines = [f"### {payload['model']}", ""]
    lines += [
        "| bits | method | protocol-aligned macro | avg acc | avg acc_norm | n |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for bits, methods in payload["results"].items():
        for name, data in methods.items():
            lines.append(
                f"| {bits} | {name} | {data['avg_primary_score']:.4f} "
                f"| {data['avg_acc']:.4f} | {data['avg_acc_norm']:.4f} "
                f"| {data['n_total']} |"
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--output-suffix", default="",
                    help="Optional filename suffix for task/bit shards.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--bits", nargs="+", default=["2", "3"], type=str)
    ap.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    ap.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    ap.add_argument("--include-fullkv", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sample-seed", type=int, default=2027)
    ap.add_argument("--sink-tokens", type=int, default=4)
    ap.add_argument("--recent-tokens", type=int, default=0)
    ap.add_argument("--skip-outliers", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--outlier-ratio", type=float, default=50.0)
    ap.add_argument("--ablation", action="store_true",
                    help="Expose HayateQuant FibQuant/BlockQuant ablation rows.")
    args = ap.parse_args()
    args.bits = _parse_bits(args.bits)
    return args


def main() -> None:
    _run(parse_args())


if __name__ == "__main__":
    main()

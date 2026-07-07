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
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from codec_benchmark import _codecs  # noqa: E402
from onecomp.kvcomp import CompressedDynamicCache, detect_outlier_layers  # noqa: E402
from onecomp.kvcomp.eval import zeroshot_accuracy  # noqa: E402


DEFAULT_TASKS = ("arc_easy", "arc_challenge", "piqa", "hellaswag", "winogrande")
DEFAULT_METHODS = (
    "FullKV",
    "TurboQuant-hadamard",
    "HayateQuant-auto",
    "HayateQuant-K12",
    "HayateQuant-dense-LS",
)


def _parse_bits(values: Iterable[str]) -> list[int]:
    return [int(v) for v in values]


def _method_list(args: argparse.Namespace, bits: int) -> list[str]:
    methods = list(args.methods)
    if bits == 3 and "HayateQuant-K4V2" not in methods:
        methods.append("HayateQuant-K4V2")
    return methods


def _build_cache_factory(model, codec, sink_tokens: int, skip_layers: set[int]):
    return lambda: CompressedDynamicCache(
        codec=codec,
        config=model.config,
        sink_tokens=sink_tokens,
        skip_layers=skip_layers if skip_layers else None,
    )


def _output_json(args: argparse.Namespace) -> Path:
    model_tag = os.path.basename(str(args.model_path).rstrip("/"))
    return args.output_root / f"zeroshot_{model_tag}.json"


def _acquire_output_lock(out_json: Path) -> Path | None:
    lock_dir = out_json.with_suffix(out_json.suffix + ".lock")
    while True:
        if out_json.exists() and out_json.stat().st_size > 0:
            print(f"[skip] existing result: {out_json}", flush=True)
            return None
        try:
            lock_dir.mkdir()
        except FileExistsError:
            print(f"[wait] another process is writing {out_json}", flush=True)
            time.sleep(30)
            continue
        if out_json.exists() and out_json.stat().st_size > 0:
            try:
                lock_dir.rmdir()
            finally:
                print(f"[skip] existing result: {out_json}", flush=True)
            return None
        return lock_dir


@torch.no_grad()
def _detect_skips(model, tok, args: argparse.Namespace) -> set[int]:
    if not args.skip_outliers:
        return set()
    calib_text = "The history of cartography spans many centuries. " * 8
    ids = tok(
        calib_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).input_ids.to(model.device)
    return set(detect_outlier_layers(model, ids, ratio_thresh=args.outlier_ratio))


def _run(args: argparse.Namespace) -> Path:
    args.output_root.mkdir(parents=True, exist_ok=True)
    out_json = _output_json(args)
    lock_dir = _acquire_output_lock(out_json)
    if lock_dir is None:
        return out_json

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
        if args.include_fullkv:
            results["full"] = {
                "FullKV": _eval_method(model, tok, args, cache_factory=None)
            }

        for bits in args.bits:
            codecs = _codecs(bits, ablation=False)
            bit_key = str(bits)
            results[bit_key] = {}
            for method in _method_list(args, bits):
                if method == "FullKV":
                    if args.include_fullkv and "FullKV" in results.get("full", {}):
                        results[bit_key][method] = results["full"]["FullKV"]
                    else:
                        results[bit_key][method] = _eval_method(
                            model, tok, args, cache_factory=None
                        )
                    continue
                if method not in codecs:
                    print(f"[skip] {bits}bit {method}: not defined", flush=True)
                    continue
                cache_factory = _build_cache_factory(
                    model, codecs[method], args.sink_tokens, skip_layers
                )
                results[bit_key][method] = _eval_method(
                    model, tok, args, cache_factory=cache_factory
                )
                print(
                    f"[{bits}bit] {method}: {results[bit_key][method]}",
                    flush=True,
                )

        payload = {
            "model": os.path.basename(str(args.model_path).rstrip("/")),
            "tasks": list(args.tasks),
            "limit": args.limit,
            "sample_seed": args.sample_seed,
            "config": {
                "model_path": str(args.model_path),
                "dtype": args.dtype,
                "device": args.device,
                "bits": args.bits,
                "sink_tokens": args.sink_tokens,
                "skip_outliers": args.skip_outliers,
                "outlier_ratio": args.outlier_ratio,
            },
            "results": results,
        }
        tmp_json = out_json.with_suffix(out_json.suffix + f".tmp.{os.getpid()}")
        tmp_json.write_text(json.dumps(payload, indent=2) + "\n")
        tmp_json.replace(out_json)
        print(f"\nWrote {out_json}")
        print(_markdown(payload))
        return out_json
    finally:
        try:
            lock_dir.rmdir()
        except OSError:
            pass


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
    return {
        "tasks": tasks,
        "avg_acc": round(mean(acc), 4),
        "avg_acc_norm": round(mean(acc_norm), 4),
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
    return max(candidates, key=lambda item: item[1].get("avg_acc_norm", -1.0))


def _markdown(payload: dict) -> str:
    lines = [f"### {payload['model']}", ""]
    lines += [
        "| bits | method | avg acc | avg acc_norm | n |",
        "|---:|---|---:|---:|---:|",
    ]
    for bits, methods in payload["results"].items():
        for name, data in methods.items():
            lines.append(
                f"| {bits} | {name} | {data['avg_acc']:.4f} "
                f"| {data['avg_acc_norm']:.4f} | {data['n_total']} |"
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--bits", nargs="+", default=["2", "3"], type=str)
    ap.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    ap.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    ap.add_argument("--include-fullkv", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sample-seed", type=int, default=2027)
    ap.add_argument("--sink-tokens", type=int, default=4)
    ap.add_argument("--skip-outliers", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--outlier-ratio", type=float, default=50.0)
    args = ap.parse_args()
    args.bits = _parse_bits(args.bits)
    return args


def main() -> None:
    _run(parse_args())


if __name__ == "__main__":
    main()

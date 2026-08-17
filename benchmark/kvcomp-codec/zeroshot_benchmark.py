"""Zero-shot downstream benchmark for HayateQuant KV codecs.

This complements ``codec_benchmark.py`` with lm-eval-style multiple-choice
accuracy under the same ``CompressedDynamicCache`` path used for perplexity.
The paper driver evaluates the same model set as Table 1 and reports the mean
accuracy over ARC-Easy, ARC-Challenge, PIQA, HellaSwag, and WinoGrande.

Example
-------
Run one model in a GPU allocation:

    srun --gres=gpu:1 "$PYTHON" benchmark/kvcomp-codec/zeroshot_benchmark.py \
        --model-path "$MODELS_ROOT/llama-3.1-8b-instruct" \
        --output-root "$OUTPUT_ROOT/zeroshot"

Use ``--start``/``--limit`` for disjoint recovery shards or smoke tests.  A
formal aggregate must cover each pinned split exactly once (ARC uses test;
PIQA, HellaSwag, and WinoGrande use validation).
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

from codec_benchmark import _codecs, hayatequant_provenance  # noqa: E402
from onecomp.kvcomp import CompressedDynamicCache  # noqa: E402
from onecomp.kvcomp.eval import zeroshot_accuracy  # noqa: E402
from onecomp.kvcomp.eval.provenance import (  # noqa: E402
    load_run_provenance,
    run_code_commit,
    run_code_revision,
    validate_model_provenance,
)
from onecomp.kvcomp.codec.hayate.recipes import HAYATEQUANT_NAME  # noqa: E402

DEFAULT_TASKS = ("arc_easy", "arc_challenge", "piqa", "hellaswag", "winogrande")
DEFAULT_METHODS = (
    "FullKV",
    "turboquant",
    HAYATEQUANT_NAME,
)


def _parse_bits(values: Iterable[str]) -> list[int]:
    return [int(v) for v in values]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_methods(methods: Iterable[str]) -> list[str]:
    """Validate and de-duplicate the two canonical method labels."""

    methods = list(dict.fromkeys(str(name) for name in methods))
    supported = {"FullKV", "turboquant", HAYATEQUANT_NAME}
    unknown = sorted(set(methods) - supported)
    if unknown:
        raise ValueError(f"unknown methods {unknown}; use FullKV, turboquant, and hayatequant")
    return methods


def _method_list(args: argparse.Namespace, bits: int) -> list[str]:
    _ = bits
    return _canonical_methods(args.methods)


def _build_cache_factory(model, codec):
    return lambda: CompressedDynamicCache(
        codec=codec,
        config=model.config,
        sink_tokens=0,
        recent_tokens=0,
    )


def _output_json(args: argparse.Namespace) -> Path:
    model_tag = os.path.basename(str(args.model_path).rstrip("/"))
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    return args.output_root / f"zeroshot_{model_tag}{suffix}.json"


def _benchmark_config(args: argparse.Namespace, model_attestation: dict | None) -> dict:
    """Scientific run identity; only the output location is excluded."""
    return {
        "schema_version": 2,
        "model_path": str(args.model_path.resolve()),
        "device": args.device,
        "dtype": args.dtype,
        "bits": list(args.bits),
        "tasks": list(args.tasks),
        "methods": list(args.methods),
        "include_fullkv": bool(args.include_fullkv),
        "start": args.start,
        "limit": args.limit,
        "sample_seed": args.sample_seed,
        "cache_contract": {
            "sink_tokens": 0,
            "recent_tokens": 0,
            "skip_layers": [],
        },
        "code_commit": run_code_commit(),
        "code_revision": run_code_revision(),
        "model_config_sha256": _sha256_file(args.model_path / "config.json"),
        "model_checkpoint_fingerprint": (
            model_attestation.get("checkpoint", {}).get("fingerprint_sha256")
            if model_attestation is not None
            else None
        ),
    }


def _model_head_dim(config) -> int:
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    value = getattr(text, "head_dim", None)
    if isinstance(value, int) and value > 0:
        return value
    hidden = getattr(text, "hidden_size", None)
    heads = getattr(text, "num_attention_heads", None)
    if (
        isinstance(hidden, int)
        and isinstance(heads, int)
        and hidden > 0
        and heads > 0
        and hidden % heads == 0
    ):
        return hidden // heads
    raise ValueError("Model config has no unambiguous attention head dimension")


def _fingerprint(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _validate_existing(out_json: Path, expected_fingerprint: str) -> dict:
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
    return payload


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


def _run(args: argparse.Namespace) -> Path:
    args.output_root.mkdir(parents=True, exist_ok=True)
    out_json = _output_json(args)
    model_attestation = validate_model_provenance(args.model_path)
    run_config = _benchmark_config(args, model_attestation)
    run_fingerprint = _fingerprint(run_config)
    output_lock = FileLock(str(out_json) + ".lock")
    output_lock.acquire()
    existing_payload = None
    if out_json.exists() and out_json.stat().st_size > 0:
        existing_payload = _validate_existing(out_json, run_fingerprint)
        # Documents written before resumable checkpoints had no ``complete``
        # marker and were only ever emitted after every task completed.
        if existing_payload.get("complete", True):
            print(f"[skip] matching existing result: {out_json}", flush=True)
            output_lock.release()
            return out_json

    dtype = getattr(torch, args.dtype)
    try:
        provenance = load_run_provenance()
        payload = existing_payload or {
            "schema_version": 2,
            "run_fingerprint": run_fingerprint,
            "model": os.path.basename(str(args.model_path).rstrip("/")),
            "tasks": list(args.tasks),
            "start": args.start,
            "limit": args.limit,
            "sample_seed": args.sample_seed,
            "run_provenance": [provenance] if provenance is not None else [],
            "config": run_config,
            "results": {},
        }
        payload["complete"] = False
        _atomic_write_json(out_json, payload)

        tok = AutoTokenizer.from_pretrained(args.model_path, fix_mistral_regex=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=dtype,
            device_map=args.device,
        ).eval()
        head_dim = _model_head_dim(model.config)

        results: dict[str, dict] = payload.setdefault("results", {})

        def fullkv_factory():
            return DynamicCache(config=model.config)

        def evaluate_into(container: dict, method: str, cache_factory) -> dict:
            def checkpoint(partial: dict) -> None:
                container[method] = partial
                _atomic_write_json(out_json, payload)

            result = _eval_method(
                model,
                tok,
                args,
                cache_factory=cache_factory,
                existing=container.get(method),
                on_task=checkpoint,
            )
            container[method] = result
            _atomic_write_json(out_json, payload)
            return result

        if args.include_fullkv:
            full_results = results.setdefault("full", {})
            evaluate_into(full_results, "FullKV", fullkv_factory)

        for bits in args.bits:
            codecs = _codecs(bits)
            bit_key = str(bits)
            results.setdefault(bit_key, {})
            for method in _method_list(args, bits):
                if method == "FullKV":
                    if args.include_fullkv and "FullKV" in results.get("full", {}):
                        results[bit_key][method] = results["full"]["FullKV"]
                    else:
                        evaluate_into(results[bit_key], method, fullkv_factory)
                    results[bit_key][method]["cache_policy"] = {
                        "sink_tokens": 0,
                        "recent_tokens": 0,
                        "skip_layers": [],
                    }
                    continue
                if method not in codecs:
                    print(f"[skip] {bits}bit {method}: not defined", flush=True)
                    continue
                cache_factory = _build_cache_factory(model, codecs[method])
                codec = codecs[method]
                evaluate_into(results[bit_key], method, cache_factory)
                results[bit_key][method].update(
                    {
                        "codec_config": codec.get_kv_config(),
                        "resolved_codec_config": codec.get_resolved_kv_config(head_dim),
                        "cache_policy": {
                            "sink_tokens": 0,
                            "recent_tokens": 0,
                            "skip_layers": [],
                        },
                    }
                )
                if method == HAYATEQUANT_NAME:
                    results[bit_key][method].update(hayatequant_provenance())
                print(
                    f"[{bits}bit] {method}: "
                    f"primary={results[bit_key][method]['avg_primary_score']:.4f}, "
                    f"n={results[bit_key][method]['n_total']}",
                    flush=True,
                )

        payload["complete"] = True
        _atomic_write_json(out_json, payload)
        print(f"\nWrote {out_json}")
        print(_markdown(payload))
        return out_json
    finally:
        if output_lock.is_locked:
            output_lock.release()


def _method_summary(tasks: dict[str, dict], expected_tasks: Iterable[str]) -> dict:
    if not tasks:
        return {"tasks": {}}
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
        "complete": all(task in tasks for task in expected_tasks),
    }


def _eval_method(
    model,
    tok,
    args: argparse.Namespace,
    cache_factory,
    *,
    existing: dict | None = None,
    on_task=None,
):
    tasks = dict((existing or {}).get("tasks", {}))
    for task in args.tasks:
        if task in tasks:
            print(f"  [resume] {task}: checkpoint already present", flush=True)
            continue
        res = zeroshot_accuracy(
            model,
            tok,
            task=task,
            cache_factory=cache_factory,
            start=args.start,
            limit=args.limit,
            sample_seed=args.sample_seed,
        ).as_dict()
        tasks[task] = res
        print(
            f"  {task}: primary={res['primary_score']:.4f}, "
            f"acc={res['acc']:.4f}, acc_norm={res['acc_norm']:.4f}, n={res['n']}",
            flush=True,
        )
        if on_task is not None:
            on_task(_method_summary(tasks, args.tasks))
    return _method_summary(tasks, args.tasks)


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
    ap.add_argument(
        "--output-suffix", default="", help="Optional filename suffix for task/bit shards."
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--bits", nargs="+", default=["2", "3"], type=str)
    ap.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    ap.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    ap.add_argument("--include-fullkv", action="store_true")
    ap.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index in the valid-example order after the optional fixed shuffle.",
    )
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sample-seed", type=int, default=2027)
    args = ap.parse_args()
    if args.start < 0:
        ap.error("--start must be non-negative")
    if args.limit is not None and args.limit <= 0:
        ap.error("--limit must be positive")
    args.bits = _parse_bits(args.bits)
    # Preserve the canonical requested method set here.
    args.methods = _canonical_methods(args.methods)
    return args


def main() -> None:
    _run(parse_args())


if __name__ == "__main__":
    main()

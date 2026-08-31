"""Standalone TaperKV long-context correctness benchmark.

This intentionally compares only FullKV and the two TaperKV granularities.
HayateQuant/TurboQuant paper reproduction lives in ``benchmark/kvcomp-codec``.
All result paths are caller supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from onecomp.kvcomp import (
    BlockTaperKVPolicy,
    KVCompressionConfig,
    TaperKVPolicy,
    build_compressed_cache,
)
from onecomp.kvcomp.eval import needle_in_haystack, wikitext2_perplexity
from onecomp.kvcomp.eval.provenance import (
    load_run_provenance,
    run_code_commit,
    run_code_revision,
    validate_model_provenance,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy(method: str):
    if method == "taperkv":
        return TaperKVPolicy()
    if method == "block_taperkv":
        return BlockTaperKVPolicy()
    if method == "fullkv":
        return None
    raise ValueError(f"unknown method {method!r}")


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to replace existing TaperKV result: {args.output}")
    model_path = args.model.resolve()
    model_config = model_path / "config.json"
    if not model_path.is_dir() or not model_config.is_file():
        raise FileNotFoundError(
            f"TaperKV benchmark requires a local model with config.json: {model_path}"
        )
    model_attestation = validate_model_provenance(model_path)
    checkpoint_fingerprint = (
        model_attestation.get("checkpoint", {}).get("fingerprint_sha256")
        if model_attestation is not None
        else None
    )
    run_provenance = load_run_provenance()
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=args.device,
        trust_remote_code=args.trust_remote_code,
    ).eval()
    results = {}
    for method in args.methods:
        policy = _policy(method)
        kv_config = (
            None if policy is None else KVCompressionConfig(policy=policy, residual_window=0)
        )

        def cache_factory(config=kv_config):
            return build_compressed_cache(model, config) if config is not None else None

        factory = None if kv_config is None else cache_factory
        entry = {
            "kv_compression_config": None if kv_config is None else kv_config.to_dict(),
            "needle": needle_in_haystack(
                model,
                tokenizer,
                cache_factory=factory,
                context_lens=args.context_lengths,
                depths=args.depths,
                seeds=args.seeds,
                max_new_tokens=args.max_new_tokens,
            ).as_dict(),
        }
        if args.ppl_windows != 0:
            entry["wikitext2"] = wikitext2_perplexity(
                model,
                tokenizer,
                cache_factory=factory,
                max_length=args.ppl_max_length,
                stride=args.ppl_stride,
                max_windows=args.ppl_windows,
            )
        results[method] = entry
    accelerator = None
    if torch.cuda.is_available():
        accelerator = torch.cuda.get_device_name(torch.cuda.current_device())
    payload = {
        "schema_version": 2,
        "model": str(model_path),
        "model_config_sha256": _sha256(model_config),
        "model_checkpoint_fingerprint": checkpoint_fingerprint,
        "code_commit": run_code_commit(),
        "code_revision": run_code_revision(),
        "run_provenance": [run_provenance] if run_provenance is not None else [],
        "dtype": args.dtype,
        "device": args.device,
        "accelerator": accelerator,
        "software": {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
        },
        "methods": list(args.methods),
        "context_lengths": args.context_lengths,
        "depths": args.depths,
        "seeds": args.seeds,
        "max_new_tokens": args.max_new_tokens,
        "ppl_windows": args.ppl_windows,
        "ppl_max_length": args.ppl_max_length,
        "ppl_stride": args.ppl_stride,
        "trust_remote_code": args.trust_remote_code,
        "results": results,
    }
    _atomic_json(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("fullkv", "taperkv", "block_taperkv"),
        default=["fullkv", "taperkv", "block_taperkv"],
    )
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[4096, 8192])
    parser.add_argument("--depths", nargs="+", type=float, default=[0.1, 0.5, 0.9])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--ppl-windows", type=int, default=0)
    parser.add_argument("--ppl-max-length", type=int, default=2048)
    parser.add_argument("--ppl-stride", type=int, default=1024)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.seeds < 1 or args.max_new_tokens < 1:
        parser.error("--seeds and --max-new-tokens must be positive")
    if args.ppl_windows is not None and args.ppl_windows < 0:
        parser.error("--ppl-windows must be non-negative")
    if any(length < 1 for length in args.context_lengths):
        parser.error("--context-lengths must contain positive integers")
    if any(not 0.0 <= depth <= 1.0 for depth in args.depths):
        parser.error("--depths entries must lie in [0, 1]")
    if args.ppl_max_length < 2 or not 0 < args.ppl_stride <= args.ppl_max_length:
        parser.error("PPL requires 0 < --ppl-stride <= --ppl-max-length and length >= 2")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))

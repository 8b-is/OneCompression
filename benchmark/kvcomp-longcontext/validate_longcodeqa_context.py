#!/usr/bin/env python3
"""Tokenize official LongCodeQA prompts and prove selected brackets fit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from onecomp.kvcomp.eval.longcodeqa import (
    LONGCODEQA_DATASET_REVISION,
    LONGCODEQA_DISCUSSION_URL,
    LONGCODEQA_PAIRED_MAX_NEW_TOKENS,
    LONGCODEQA_PROTOCOL_COMMIT,
    _effective_eos_token_ids,
    _encode_prompt,
    load_longcodeqa,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _model_revision(model: Path) -> dict:
    revisions = set()
    metadata_root = model / ".cache" / "huggingface" / "download"
    for path in metadata_root.glob("*.metadata"):
        try:
            first = path.read_text(encoding="utf-8").splitlines()[0]
        except (OSError, UnicodeDecodeError, IndexError):
            continue
        if re.fullmatch(r"[0-9a-f]{40}", first):
            revisions.add(first)
    return {
        "revision": next(iter(revisions)) if len(revisions) == 1 else None,
        "snapshot_revisions": sorted(revisions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--brackets", nargs="+", required=True)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=LONGCODEQA_PAIRED_MAX_NEW_TOKENS,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, fix_mistral_regex=True
    )
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    # This lightweight stand-in is enough for the shared resolver and prevents
    # loading any model weights during the context-fit preflight.
    generation_path = model_path / "generation_config.json"
    if not generation_path.is_file():
        raise FileNotFoundError(
            "LongCodeQA paired decoding requires pinned generation_config.json: "
            f"{generation_path}"
        )
    with generation_path.open(encoding="utf-8") as stream:
        generation_document = json.load(stream)
    if not isinstance(generation_document, dict):
        raise ValueError(f"Expected an object in {generation_path}")
    generation_config = type(
        "PinnedGenerationConfig",
        (),
        {"eos_token_id": generation_document.get("eos_token_id")},
    )()
    model_stub = type("PinnedModel", (), {"generation_config": generation_config})()
    eos_token_ids, eos_source = _effective_eos_token_ids(model_stub, tokenizer)
    if eos_source != "model.generation_config.eos_token_id" or not eos_token_ids:
        raise ValueError(
            "LongCodeQA formal checkpoints must pin at least one EOS token in "
            f"{generation_path}"
        )
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    context_limit = getattr(text_config, "max_position_embeddings", None)
    if not isinstance(context_limit, int) or context_limit <= 0:
        raise ValueError("Model config has no positive max_position_embeddings")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    results = {}
    all_overflows = []
    for bracket in args.brackets:
        rows, path, digest = load_longcodeqa(args.data_root, bracket)
        lengths = []
        overflows = []
        per_row = []
        for index, row in enumerate(rows):
            tokens = int(_encode_prompt(tokenizer, row["prompt"]).shape[1])
            margin = context_limit - tokens - args.max_new_tokens
            lengths.append(tokens)
            per_row.append({"index": index, "input_tokens": tokens, "margin": margin})
            if margin < 0:
                overflows.append(index)
                all_overflows.append((bracket, index, tokens))
        results[bracket] = {
            "rows": len(rows),
            "dataset_path": str(path),
            "dataset_sha256": digest,
            "input_tokens": {
                "min": min(lengths),
                "median": statistics.median(lengths),
                "max": max(lengths),
            },
            "minimum_margin_tokens": min(item["margin"] for item in per_row),
            "overflow_rows": overflows,
            "per_row": per_row,
        }
        print(
            f"{bracket}: n={len(rows)} tokens={min(lengths)}..{max(lengths)} "
            f"min_margin={results[bracket]['minimum_margin_tokens']}",
            flush=True,
        )

    chat_template = getattr(tokenizer, "chat_template", None)
    model_meta = _model_revision(model_path)
    for filename in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        path = model_path / filename
        if path.is_file():
            model_meta[filename] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    document = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "LongCodeBench/LongCodeQA context-fit validation",
        "model": {
            "path": str(model_path),
            **model_meta,
            "context_limit": context_limit,
            "generation_eos_token_ids": eos_token_ids,
            "generation_eos_token_ids_source": eos_source,
            "chat_template_sha256": (
                hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
                if isinstance(chat_template, str)
                else None
            ),
        },
        "protocol": {
            "scope": (
                "release exact dataset, prompt, parser, and per-bracket accuracy; "
                "preregistered paired greedy decoding adaptation"
            ),
            "dataset_revision": LONGCODEQA_DATASET_REVISION,
            "prompt": "released prompt verbatim as raw causal text",
            "prompt_release_clarification": LONGCODEQA_DISCUSSION_URL,
            "chat_template": "none",
            "truncation": "none",
            "answer_parser_reference_commit": LONGCODEQA_PROTOCOL_COMMIT,
            "answer_parser": (
                "first /Final Answer:\\s*([ABCD])/i; otherwise trailing " "/\\b([ABCD])\\s*$/i"
            ),
            "reference_evaluator_executable_verbatim": False,
            "reference_evaluator_known_issue": (
                "pinned evaluator stores response but reads prediction text"
            ),
            "generation": {
                "protocol": "hayatequant deterministic greedy paired comparison",
                "do_sample": False,
                "max_new_tokens": args.max_new_tokens,
                "seed": None,
                "eos_token_ids_source": eos_source,
                "stop_conditions": ["any_eos_token_id", "max_new_tokens"],
                "official_reference_max_output_length": None,
                "official_reference_uniform_decoding_config": False,
                "official_reference_decoding_exact": False,
            },
        },
        "brackets": results,
        "all_rows_fit": not all_overflows,
    }
    _atomic_json(args.output.expanduser().resolve(), document)
    if all_overflows:
        raise RuntimeError(
            f"{len(all_overflows)} LongCodeQA prompts exceed the model context; "
            f"see {args.output}"
        )


if __name__ == "__main__":
    main()

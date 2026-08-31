"""LongCodeBench LongCodeQA evaluation under KV compression.

LongCodeQA is the multiple-choice code-comprehension half of LongCodeBench
(Rando et al., COLM 2025).  The released dataset contains 443 questions split
into six context-length brackets.  Each row already carries the official prompt
and a single ``correct_letter`` in ``A``--``D``; this evaluator deliberately
uses those fields verbatim as raw causal-LM text instead of reconstructing,
chat-wrapping, or truncating prompts.  This matches LongCodeBench's released
self-hosted Hugging Face input path.

The released reference does *not* define one portable decoding protocol:
``eval.py`` passes ``max_output_length=None``, while the provider wrappers and
model generation configs impose different defaults.  For a paired KV-cache
comparison we therefore use an explicitly labelled deterministic greedy
adaptation with a fixed token budget.  The final filtered dataset, prompt,
answer parser, and per-bracket accuracy are release-faithful; the decoding
policy is not claimed to reproduce the heterogeneous decoding used for the
paper's original model table.

Long inputs that do not fit the model are rejected.  Silently head/tail
truncating a 128K example would change the benchmark while still looking like a
128K score, which is especially misleading in a KV-compression paper.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import torch

from .compression_accounting import (
    PER_EXAMPLE_CACHE_LIFETIME,
    PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION,
    aggregate_compression_sufficient_statistics,
    aggregate_mem_fraction,
    cache_compression_sufficient_statistics,
    cache_mem_fraction,
)
from .longcontext import greedy_answer, proportion_stats

LONGCODEQA_BRACKET_COUNTS = {
    "32K": 113,
    "64K": 76,
    "128K": 92,
    "256K": 65,
    "512K": 47,
    "1M": 50,
}
LONGCODEQA_TOTAL = sum(LONGCODEQA_BRACKET_COUNTS.values())
LONGCODEQA_DATASET_REVISION = "989d5eff750d65a72c522e47f8f745ef2e22906b"
LONGCODEQA_DATASET_SHA256 = {
    "32K": "a4de950aa887c97ed74d64d50a1202b28e566105b9b94e62b7c1a50a48b46d98",
    "64K": "c4c77b52582187753f94d4e3e78d68b6d2549791496b7305fc93b558166bdc74",
    "128K": "c4b44c40f4a2efbeea6f69c210b80ca9a90d3156f032335fcb1182eb2992e1d8",
    "256K": "3c28f1aa8bd6502c70b298e7f4390d5c9b7a0263b94c06141779ef1cb353aad9",
    "512K": "1f4bd0339d68d11cae27af80abf6bb618ac5847ab41cb0a1b68c3d55a21c9fc2",
    "1M": "eeaf9b1ac950d43341153e47a806c37b104572c9c1be5591700abecf94275680",
}
# Commit which introduced the released LongCodeQA evaluator.  Its parser is
# unchanged on upstream's current default branch at the time of this audit.
LONGCODEQA_PROTOCOL_COMMIT = "91db30f5144363bb809534ec7e4a84a56ea43987"
LONGCODEQA_PAIRED_MAX_NEW_TOKENS = 16
LONGCODEQA_DISCUSSION_URL = "https://huggingface.co/datasets/Steefano/LCB/discussions/1"

_FINAL_ANSWER = re.compile(r"Final Answer:\s*([ABCD])", re.IGNORECASE)
_TRAILING_LETTER = re.compile(r"\b([ABCD])\s*$", re.IGNORECASE)


def extract_longcodeqa_letter(text: str) -> Optional[str]:
    """Apply LongCodeBench's released A--D parser exactly.

    Upstream first accepts the first ``Final Answer: X`` occurrence, then falls
    back to a letter at the end of the response.  Broader ``answer is X`` or
    ``option X`` patterns are deliberately not accepted because they change the
    benchmark's accuracy definition.
    """
    match = _FINAL_ANSWER.search(text)
    if match:
        return match.group(1).upper()
    match = _TRAILING_LETTER.search(text.strip())
    return match.group(1).upper() if match else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_file(data_root: str, bracket: str) -> Path:
    if bracket not in LONGCODEQA_BRACKET_COUNTS:
        raise ValueError(
            f"Unknown LongCodeQA bracket {bracket!r}; expected "
            f"{list(LONGCODEQA_BRACKET_COUNTS)}"
        )
    root = Path(data_root).expanduser().resolve()
    candidates = (root / f"{bracket}.json", root / "LQA" / f"{bracket}.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"LongCodeQA {bracket} data not found below {root}; expected "
        f"{candidates[0]} or {candidates[1]}"
    )


def load_longcodeqa(data_root: str, bracket: str) -> tuple[List[dict], Path, str]:
    """Load and byte-validate one final-filtered LongCodeQA bracket."""
    path = _data_file(data_root, bracket)
    with path.open(encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}")
    expected = LONGCODEQA_BRACKET_COUNTS[bracket]
    if len(rows) != expected:
        raise RuntimeError(
            f"Expected {expected} official LongCodeQA {bracket} rows, got {len(rows)}"
        )
    required = {"prompt", "correct_letter", "repo", "question"}
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"LongCodeQA {bracket} row {index} is missing {sorted(missing)}")
        letter = str(row["correct_letter"]).strip().upper()
        if letter not in {"A", "B", "C", "D"}:
            raise ValueError(
                f"LongCodeQA {bracket} row {index} has invalid correct_letter={letter!r}"
            )
        if not isinstance(row["prompt"], str) or not row["prompt"].strip():
            raise ValueError(f"LongCodeQA {bracket} row {index} has an empty prompt")
        if not isinstance(row["repo"], str) or not row["repo"].strip():
            raise ValueError(f"LongCodeQA {bracket} row {index} has an empty repo")
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError(f"LongCodeQA {bracket} row {index} has an empty question")
    digest = _sha256(path)
    expected_digest = LONGCODEQA_DATASET_SHA256[bracket]
    if digest != expected_digest:
        raise RuntimeError(
            f"LongCodeQA {bracket} SHA-256 mismatch: expected final-filtered "
            f"Steefano/LCB@{LONGCODEQA_DATASET_REVISION} file {expected_digest}, "
            f"got {digest} for {path}"
        )
    return rows, path, digest


def _encode_prompt(tokenizer, prompt: str) -> torch.Tensor:
    """Tokenize the released prompt as raw text, matching self-hosted upstream."""
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1:
        raise TypeError("LongCodeQA tokenizer must return input_ids with shape (1, sequence)")
    if ids.shape[1] == 0:
        raise ValueError("LongCodeQA tokenizer returned an empty prompt")
    return ids


def _model_context_limit(model) -> int:
    config = model.config
    if hasattr(config, "get_text_config"):
        config = config.get_text_config()
    limit = getattr(config, "max_position_embeddings", None)
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("Model config does not expose a positive max_position_embeddings")
    return limit


def _token_id_list(value, *, label: str) -> List[int]:
    """Normalize a generation token-id field without accepting bools/strings."""
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    normalized: List[int] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{label} must contain non-negative integer token IDs")
        if item not in normalized:
            normalized.append(item)
    return normalized


def _effective_eos_token_ids(model, tokenizer) -> tuple[List[int], str]:
    """Resolve EOS exactly as HF ``generate`` would, then record the source.

    A model's loaded ``generation_config`` takes precedence over the tokenizer.
    This matters for both formal checkpoints: their generation configs expose
    more than one EOS token whereas their tokenizers expose only one.
    """
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        configured = getattr(generation_config, "eos_token_id", None)
        if configured is not None:
            values = _token_id_list(configured, label="model.generation_config.eos_token_id")
            if not values:
                raise ValueError("model.generation_config.eos_token_id must not be empty")
            return values, "model.generation_config.eos_token_id"
    values = _token_id_list(
        getattr(tokenizer, "eos_token_id", None),
        label="tokenizer.eos_token_id",
    )
    if not values:
        raise ValueError("LongCodeQA requires at least one tokenizer/model EOS token ID")
    return values, "tokenizer.eos_token_id"


@dataclass
class LongCodeQAResult:
    bracket: str
    n: int
    correct: int
    accuracy: float
    accuracy_stderr: float
    accuracy_ci95: tuple
    input_tokens: List[int]
    dataset_path: str
    dataset_sha256: str
    dataset_revision: str
    max_context_tokens: int
    max_new_tokens: int
    protocol_commit: str = LONGCODEQA_PROTOCOL_COMMIT
    start: int = 0
    limit: Optional[int] = None
    mem_fraction: float = 1.0
    mem_fraction_aggregation: str = "fullkv_no_cache"
    eos_token_ids: List[int] = field(default_factory=list)
    eos_token_ids_source: str = "tokenizer.eos_token_id"
    per_example: List[dict] = field(default_factory=list)
    compression_sufficient_statistics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        expected = LONGCODEQA_BRACKET_COUNTS[self.bracket]
        complete = (
            self.start == 0
            and self.limit is None
            and self.n == expected
            and self.dataset_revision == LONGCODEQA_DATASET_REVISION
            and self.dataset_sha256 == LONGCODEQA_DATASET_SHA256[self.bracket]
            and self.protocol_commit == LONGCODEQA_PROTOCOL_COMMIT
            and self.max_new_tokens == LONGCODEQA_PAIRED_MAX_NEW_TOKENS
        )
        lengths = self.input_tokens or [0]
        return {
            "benchmark": "LongCodeBench/LongCodeQA",
            "task": "multiple_choice_code_comprehension",
            "metric": "exact_match_accuracy",
            "bracket": self.bracket,
            "n": self.n,
            "expected_bracket_n": expected,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 6),
            "accuracy_pct": round(100.0 * self.accuracy, 2),
            "accuracy_stderr": round(self.accuracy_stderr, 6),
            "accuracy_ci95": [round(self.accuracy_ci95[0], 6), round(self.accuracy_ci95[1], 6)],
            "input_tokens": {
                "min": min(lengths),
                "median": round(float(statistics.median(lengths)), 1),
                "max": max(lengths),
            },
            "mem_fraction": round(self.mem_fraction, 6),
            "mem_fraction_aggregation": self.mem_fraction_aggregation,
            "example_cache_lifetime": PER_EXAMPLE_CACHE_LIFETIME,
            "cross_iteration_tensor_retention": (PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION),
            "max_context_tokens": self.max_context_tokens,
            "max_new_tokens": self.max_new_tokens,
            "start": self.start,
            "limit": self.limit,
            "released_data_complete_bracket": complete,
            "released_data_complete_dataset": False,
            "dataset": {
                "repo_id": "Steefano/LCB",
                "revision": self.dataset_revision,
                "path": self.dataset_path,
                "sha256": self.dataset_sha256,
                "final_filtered_release": True,
                "release_clarification": LONGCODEQA_DISCUSSION_URL,
            },
            "protocol_commit": self.protocol_commit,
            "reference_evaluator_executable_verbatim": False,
            "reference_evaluator_known_issue": (
                "pinned evaluator stores response but reads prediction text"
            ),
            "protocol_scope": (
                "release_exact_dataset_prompt_parser_and_per_bracket_accuracy; "
                "preregistered_paired_greedy_decoding_adaptation"
            ),
            "prompt_protocol": "released_prompt_verbatim_raw_causal_text",
            "chat_template": "none",
            "truncation": "none",
            "generation": {
                "protocol": "hayatequant_deterministic_greedy_paired_comparison",
                "do_sample": False,
                "max_new_tokens": self.max_new_tokens,
                "seed": None,
                "eos_token_ids": self.eos_token_ids,
                "eos_token_ids_source": self.eos_token_ids_source,
                "stop_conditions": ["any_eos_token_id", "max_new_tokens"],
                "official_reference_max_output_length": None,
                "official_reference_uniform_decoding_config": False,
                "official_reference_decoding_exact": False,
            },
            "answer_parser": (
                "first /Final Answer:\\s*([ABCD])/i; otherwise trailing " "/\\b([ABCD])\\s*$/i"
            ),
            "per_example_source_fields": [
                "index",
                "repo",
                "question",
                "correct_letter_as_gold",
            ],
            # The released prompts can be hundreds of thousands of tokens, so
            # duplicating them in every result shard would be wasteful.  The
            # formal aggregator instead reopens the content-hashed source file
            # and the tokenizer-specific context-fit manifest.  Keep this
            # limitation machine-readable so no later report can pretend to
            # have compared a prompt copy that the result does not contain.
            "per_example_prompt_retained": False,
            "per_example": self.per_example,
            "compression_sufficient_statistics": self.compression_sufficient_statistics,
        }


@torch.no_grad()
def longcodeqa(
    model,
    tokenizer,
    data_root: str,
    bracket: str,
    cache_factory: Optional[Callable[[], object]] = None,
    max_context_tokens: Optional[int] = None,
    max_new_tokens: int = LONGCODEQA_PAIRED_MAX_NEW_TOKENS,
    start: int = 0,
    limit: Optional[int] = None,
    evict: bool = False,
    attn_capture: bool = False,
    dataset_revision: str = LONGCODEQA_DATASET_REVISION,
) -> LongCodeQAResult:
    """Evaluate one final-filtered LongCodeQA bracket with exact-match accuracy.

    ``limit``/``start`` are smoke and recovery controls.  A subset is always
    labelled incomplete in :meth:`LongCodeQAResult.as_dict`.  Even a complete
    bracket records that its deterministic decoding is an adaptation rather
    than the heterogeneous upstream model/provider decoding.
    """
    from transformers import DynamicCache

    if not isinstance(start, int) or isinstance(start, bool) or start < 0:
        raise ValueError("start must be non-negative")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
        raise ValueError("limit must be positive or None")
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens <= 0
    ):
        raise ValueError("max_new_tokens must be positive")
    if dataset_revision != LONGCODEQA_DATASET_REVISION:
        raise ValueError(
            "LongCodeQA evaluation is pinned to the final-filtered dataset revision "
            f"{LONGCODEQA_DATASET_REVISION}; got {dataset_revision!r}"
        )
    if not isinstance(evict, bool) or not isinstance(attn_capture, bool):
        raise ValueError("evict and attn_capture must be bools")
    rows, path, digest = load_longcodeqa(data_root, bracket)
    selected = rows[start:] if limit is None else rows[start : start + limit]
    if not selected:
        raise ValueError(f"LongCodeQA {bracket} selection is empty (start={start}, limit={limit})")
    advertised_limit = _model_context_limit(model)
    context_limit = advertised_limit if max_context_tokens is None else max_context_tokens
    if not isinstance(context_limit, int) or isinstance(context_limit, bool) or context_limit <= 0:
        raise ValueError("max_context_tokens must be a positive integer or None")
    if context_limit > advertised_limit:
        raise ValueError(
            f"Declared LongCodeQA context {context_limit} exceeds the model's "
            f"max_position_embeddings={advertised_limit}"
        )
    eos_token_ids, eos_source = _effective_eos_token_ids(model, tokenizer)
    hits = 0
    input_lengths: List[int] = []
    mem: List[float] = []
    records: List[dict] = []
    for offset, row in enumerate(selected):
        row_index = start + offset
        ids = _encode_prompt(tokenizer, row["prompt"])
        input_len = int(ids.shape[1])
        if input_len + max_new_tokens > context_limit:
            raise RuntimeError(
                f"LongCodeQA {bracket} row {row_index} needs {input_len} input + "
                f"{max_new_tokens} generation tokens, exceeding the declared model "
                f"limit {context_limit}. Refusing benchmark-changing truncation."
            )
        cache = cache_factory() if cache_factory is not None else DynamicCache(config=model.config)
        post_step = (lambda value: value.evict_now()) if evict else None
        generated = greedy_answer(
            model,
            ids,
            cache,
            max_new_tokens=max_new_tokens,
            post_prefill=post_step,
            eos=eos_token_ids or None,
            attn_capture=attn_capture,
        )
        compression = cache_compression_sufficient_statistics(cache)
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        prediction = extract_longcodeqa_letter(text)
        gold = str(row["correct_letter"]).strip().upper()
        correct = prediction == gold
        hits += int(correct)
        input_lengths.append(input_len)
        fraction = cache_mem_fraction(cache)
        if fraction is not None:
            mem.append(fraction)
        records.append(
            {
                "index": row_index,
                "repo": row["repo"],
                "question": row["question"],
                "input_tokens": input_len,
                "generated_text": text,
                "generated_tokens": int(generated.shape[1]),
                "prediction": prediction,
                "gold": gold,
                "correct": correct,
                **compression,
            }
        )
        # ``records`` contains only detached host values.  Releasing here (not
        # on the next assignment) prevents a transient two-cache GPU peak.
        del cache, generated, ids
    n = len(selected)
    stderr, low, high = proportion_stats(hits, n)
    compression_statistics = aggregate_compression_sufficient_statistics(records)
    aggregate_fraction, fraction_aggregation = aggregate_mem_fraction(compression_statistics, mem)
    return LongCodeQAResult(
        bracket=bracket,
        n=n,
        correct=hits,
        accuracy=hits / n,
        accuracy_stderr=stderr,
        accuracy_ci95=(low, high),
        input_tokens=input_lengths,
        dataset_path=str(path),
        dataset_sha256=digest,
        dataset_revision=dataset_revision,
        protocol_commit=LONGCODEQA_PROTOCOL_COMMIT,
        max_context_tokens=context_limit,
        max_new_tokens=max_new_tokens,
        start=start,
        limit=limit,
        mem_fraction=aggregate_fraction,
        mem_fraction_aggregation=fraction_aggregation,
        eos_token_ids=eos_token_ids,
        eos_token_ids_source=eos_source,
        per_example=records,
        compression_sufficient_statistics=compression_statistics,
    )


def aggregate_longcodeqa(results: List[LongCodeQAResult]) -> dict:
    """Pool complete, disjoint bracket results without macro-average bias."""
    if not results:
        raise ValueError("At least one LongCodeQA result is required")
    brackets = [result.bracket for result in results]
    if len(set(brackets)) != len(brackets):
        raise ValueError(f"Duplicate LongCodeQA brackets: {brackets}")
    docs = {result.bracket: result.as_dict() for result in results}
    n = sum(result.n for result in results)
    correct = sum(result.correct for result in results)
    stderr, low, high = proportion_stats(correct, n)
    complete = (
        set(brackets) == set(LONGCODEQA_BRACKET_COUNTS)
        and all(doc["released_data_complete_bracket"] for doc in docs.values())
        and n == LONGCODEQA_TOTAL
    )
    all_generation_records = [record for result in results for record in result.per_example]
    return {
        "benchmark": "LongCodeBench/LongCodeQA",
        "metric": "micro_exact_match_accuracy",
        "official_primary_reporting": "per_bracket_exact_match_accuracy",
        "derived_cross_bracket_summary": True,
        "protocol_scope": (
            "release_exact_dataset_prompt_parser_and_per_bracket_accuracy; "
            "preregistered_paired_greedy_decoding_adaptation"
        ),
        "official_reference_decoding_exact": False,
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 6),
        "accuracy_pct": round(100.0 * correct / n, 2),
        "accuracy_stderr": round(stderr, 6),
        "accuracy_ci95": [round(low, 6), round(high, 6)],
        "example_cache_lifetime": PER_EXAMPLE_CACHE_LIFETIME,
        "cross_iteration_tensor_retention": (PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION),
        "brackets": docs,
        "compression_sufficient_statistics": (
            aggregate_compression_sufficient_statistics(all_generation_records)
            if all_generation_records
            else {}
        ),
        "released_data_complete_dataset": complete,
        "expected_dataset_n": LONGCODEQA_TOTAL,
    }

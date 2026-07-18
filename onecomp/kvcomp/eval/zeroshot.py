"""Zero-shot accuracy via length-normalised log-likelihood.

Implements the lm-eval-harness ``acc``/``acc_norm`` multiple-choice protocol:
for each example we score every candidate continuation by its summed token
log-probability (and a length-normalised variant) under the model, then pick
the arg-max.  Each example is scored with a fresh cache from ``cache_factory``
so KV compression affects the prediction exactly as at inference time.
For a codec whose update fits statistics across tokens, the context alone is
prefilled and continuation targets are forwarded incrementally; a candidate
can never influence its own earlier conditional probability.
Compression is aggregated from exact cache-bit counters when available;
otherwise it uses full-input-token-weighted harmonic aggregation.  It is never
the arithmetic mean of per-choice ratios (choices have unequal lengths).

Supported tasks (all loaded via ``datasets``):
  * ``arc_easy``      -- AI2 ARC-Easy (test).
  * ``arc_challenge`` -- AI2 ARC-Challenge (test).
  * ``piqa``          -- physical commonsense QA (validation).
  * ``hellaswag``     -- grounded commonsense (validation, acc_norm headline).
  * ``winogrande``    -- pronoun/coreference commonsense (validation).

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from logging import getLogger
from typing import Callable, List, Optional

import torch

from ._teacher_forcing import (
    SINGLE_FORWARD_PROTOCOL,
    cache_temporal_dependency,
    causal_teacher_forced_logits,
    teacher_forced_protocol,
)
from .compression_accounting import (
    PER_EXAMPLE_CACHE_LIFETIME,
    PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION,
)

logger = getLogger(__name__)

# Exact upstream reference used to derive the prompts, target delimiter,
# context/continuation boundary handling, and acc_norm denominator.  Recording
# the Git commit (not just a package version) makes a formal result auditable
# even if a release tag or task file is later moved.
LM_EVAL_VERSION = "0.4.11"
LM_EVAL_PROTOCOL_COMMIT = "27988a293647d5853e48edea291640b4af54740c"
LM_EVAL_TASK_METADATA_VERSION = "1.0"
PROTOCOL_INPUT_HASH_SCHEMA = "lm_eval_multiple_choice_prompt_v1"
ZEROSHOT_PROTOCOL_PROBE_CACHE_LIFETIME = (
    "released_after_protocol_dependency_snapshot_before_first_choice_factory"
)


@dataclass
class ZeroShotResult:
    task: str
    n: int
    acc: float
    acc_norm: float
    compression_ratio: float = 1.0
    compression_aggregation: str = "fullkv_no_cache"
    compression_fp16_bits: Optional[int] = None
    compression_stored_bits: Optional[int] = None
    dataset_size: Optional[int] = None
    split: Optional[str] = None
    dataset_revision: Optional[str] = None
    correct: Optional[int] = None
    correct_norm: Optional[int] = None
    start: int = 0
    limit: Optional[int] = None
    sample_seed: Optional[int] = None
    per_example: Optional[List[dict]] = None
    protocol_input_sha256: Optional[str] = None
    teacher_forced_scoring_protocol: str = SINGLE_FORWARD_PROTOCOL
    teacher_forced_temporal_dependency: Optional[str] = None

    def as_dict(self) -> dict:
        # lm-eval exposes acc_norm for ARC/PIQA/HellaSwag, but WinoGrande's
        # official task exposes only plain acc.  A cross-task aggregate must not
        # silently substitute the non-standard WinoGrande acc_norm value.
        primary_metric = "acc" if self.task == "winogrande" else "acc_norm"
        primary_score = self.acc if primary_metric == "acc" else self.acc_norm
        out = {
            "task": self.task,
            "n": self.n,
            "acc": round(self.acc, 4),
            "acc_norm": round(self.acc_norm, 4),
            "primary_metric": primary_metric,
            "primary_score": round(primary_score, 4),
            "compression_ratio": round(self.compression_ratio, 3),
            "compression_aggregation": self.compression_aggregation,
            "compression_fp16_bits": self.compression_fp16_bits,
            "compression_stored_bits": self.compression_stored_bits,
            "evaluator": f"lm-eval-compatible@{LM_EVAL_VERSION}",
            "lm_eval_protocol_commit": LM_EVAL_PROTOCOL_COMMIT,
            "lm_eval_task_metadata_version": LM_EVAL_TASK_METADATA_VERSION,
            "num_fewshot": 0,
            "target_delimiter": " ",
            "apply_chat_template": False,
            "protocol_input_hash_schema": PROTOCOL_INPUT_HASH_SCHEMA,
            "protocol_input_sha256": self.protocol_input_sha256,
            "teacher_forced_scoring_protocol": self.teacher_forced_scoring_protocol,
            "teacher_forced_temporal_dependency": self.teacher_forced_temporal_dependency,
            "protocol_probe_cache_lifetime": ZEROSHOT_PROTOCOL_PROBE_CACHE_LIFETIME,
            "choice_cache_lifetime": PER_EXAMPLE_CACHE_LIFETIME,
            "cross_iteration_tensor_retention": (
                PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION
            ),
            "correct": self.correct,
            "correct_norm": self.correct_norm,
            "selection": {
                "start_valid_example": self.start,
                "stop_valid_example_exclusive": self.start + self.n,
                "limit": self.limit,
                "sample_seed": self.sample_seed,
                "order": (
                    f"datasets.shuffle(seed={self.sample_seed})"
                    if self.sample_seed is not None and (self.start > 0 or self.limit is not None)
                    else "released split order"
                ),
            },
        }
        if self.dataset_size is not None:
            out["dataset_size"] = int(self.dataset_size)
            out["complete_dataset_eval"] = (
                self.start == 0 and self.limit is None and self.n == self.dataset_size
            )
        if self.split is not None:
            out["split"] = self.split
        if self.dataset_revision is not None:
            out["dataset_revision"] = self.dataset_revision
        if self.per_example is not None:
            out["per_example"] = self.per_example
        return out


def _arc_easy(ds_split):
    for ex in ds_split:
        choices = ex["choices"]["text"]
        labels = ex["choices"]["label"]
        try:
            gold = labels.index(ex["answerKey"])
        except ValueError:
            continue
        ctx = f"Question: {ex['question']}\nAnswer:"
        conts = [f" {c}" for c in choices]
        yield ctx, conts, gold


def _arc_challenge(ds_split):
    yield from _arc_easy(ds_split)


def _piqa(ds_split):
    for ex in ds_split:
        ctx = f"Question: {ex['goal']}\nAnswer:"
        conts = [f" {ex['sol1']}", f" {ex['sol2']}"]
        yield ctx, conts, int(ex["label"])


def _hellaswag(ds_split):
    def _clean(text: str) -> str:
        # Keep this byte-for-byte aligned with lm-eval 0.4.11's
        # ``hellaswag.utils.preprocess``.  Collapsing arbitrary whitespace with
        # ``\\s+`` looks harmless, but it changes both the token boundary and
        # the character-count denominator used by ``acc_norm``.
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub(r"\[.*?\]", "", text)
        return text.replace("  ", " ")

    for ex in ds_split:
        # Match lm-eval's canonical HellaSwag preprocessing.  The raw ``ctx``
        # field omits the activity label and keeps bracketed artefacts.
        body = f"{ex['ctx_a']} {str(ex['ctx_b']).capitalize()}"
        ctx = _clean(f"{ex['activity_label']}: {body}")
        conts = [" " + _clean(e) for e in ex["endings"]]
        try:
            gold = int(ex["label"])
        except (ValueError, TypeError):
            continue
        yield ctx, conts, gold


def _winogrande(ds_split):
    for ex in ds_split:
        # Match lm-eval's multiple-input protocol exactly: each option completes
        # the prefix *in the context*, and only the common suffix is scored as
        # the continuation. Scoring option+suffix from a shared prefix is a
        # different task and can change predictions through option frequency.
        split = ex["sentence"].index("_")
        prefix = ex["sentence"][:split]
        suffix = ex["sentence"][split + 1 :].strip()
        ctx = [prefix + ex["option1"], prefix + ex["option2"]]
        conts = [f" {suffix}", f" {suffix}"]
        try:
            gold = int(ex["answer"]) - 1
        except (ValueError, TypeError):
            continue
        yield ctx, conts, gold


_TASKS = {
    # lm-eval evaluates ARC on its test split, not validation.
    "arc_easy": (
        "allenai/ai2_arc",
        "ARC-Easy",
        "test",
        _arc_easy,
        "210d026faf9955653af8916fad021475a3f00453",
    ),
    "arc_challenge": (
        "allenai/ai2_arc",
        "ARC-Challenge",
        "test",
        _arc_challenge,
        "210d026faf9955653af8916fad021475a3f00453",
    ),
    "piqa": ("baber/piqa", None, "validation", _piqa, "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1"),
    "hellaswag": (
        "Rowan/hellaswag",
        None,
        "validation",
        _hellaswag,
        "218ec52e09a7e7462a5400043bb9a69a41d06b76",
    ),
    "winogrande": (
        "allenai/winogrande",
        "winogrande_xl",
        "validation",
        _winogrande,
        "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
    ),
}


def _load_task(task: str):
    from datasets import load_dataset  # pylint: disable=import-outside-toplevel

    path, name, split, fn, revision = _TASKS[task]
    kwargs = {"split": split, "revision": revision}
    ds = load_dataset(path, name, **kwargs) if name else load_dataset(path, **kwargs)
    return ds, fn


def _encode_pair(tokenizer, context: str, continuation: str):
    """Match lm-eval's causal ``HFLM._encode_pair`` boundary handling.

    Tokenizing context and continuation independently can change BPE/SentencePiece
    merges at their boundary.  lm-eval moves trailing context whitespace into
    the continuation, tokenizes the joined text once, and splits that sequence
    at the separately-tokenized context length.
    """
    if not context:
        raise ValueError("multiple-choice context must not be empty")
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole = tokenizer(context + continuation, return_tensors="pt").input_ids
    ctx = tokenizer(context, return_tensors="pt").input_ids
    cont = whole[:, ctx.shape[1] :]
    if cont.shape[1] == 0:
        raise ValueError("Continuation tokenized to zero tokens under lm-eval boundary handling")
    return ctx, cont


def _update_protocol_input_digest(
    digest,
    *,
    task: str,
    index: int,
    context: str | list[str],
    continuations: list[str],
    gold: int,
    choice_char_lengths: list[int],
) -> None:
    """Hash the exact lm-eval request strings and normalization denominators.

    The formal shard aggregator independently reconstructs this canonical JSON
    line from the pinned Arrow split.  This makes prompt, delimiter, gold-label,
    WinoGrande multiple-input, and character-normalization drift detectable
    without storing the (sometimes long) prompt text in every result row.
    """
    record = {
        "schema": PROTOCOL_INPUT_HASH_SCHEMA,
        "task": task,
        "index": int(index),
        "context": list(context) if isinstance(context, (list, tuple)) else context,
        "continuations": list(continuations),
        "gold": int(gold),
        "choice_char_lengths": [int(value) for value in choice_char_lengths],
    }
    digest.update(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")


def _cache_compression_accounting(cache, reference_tokens: int) -> dict:
    """Return an exactly aggregatable accounting record for one choice."""
    if reference_tokens <= 0:
        raise ValueError("compression reference_tokens must be positive")
    if cache is None:
        return {
            "mode": "fullkv_no_cache",
            "numerator": float(reference_tokens),
            "denominator": float(reference_tokens),
        }

    fp16_bits = getattr(cache, "fp16_bits", None)
    stored_bits = getattr(cache, "stored_bits", None)
    exact = (
        not isinstance(fp16_bits, bool)
        and isinstance(fp16_bits, int)
        and fp16_bits >= 0
        and not isinstance(stored_bits, bool)
        and isinstance(stored_bits, int)
        and stored_bits >= 0
    )
    if exact:
        if fp16_bits == 0:
            raise RuntimeError(
                "Compressed cache recorded zero fp16_bits for a non-empty zero-shot choice"
            )
        return {
            "mode": "exact_cache_bit_counters",
            "numerator": int(fp16_bits),
            "denominator": int(stored_bits),
        }
    if fp16_bits is not None or stored_bits is not None:
        raise ValueError(
            "Cache exposes incomplete/non-integer fp16_bits or stored_bits; "
            "zero-shot compression accounting requires exact non-negative integers"
        )

    ratio_fn = getattr(cache, "compression_ratio", None)
    if callable(ratio_fn):
        ratio = float(ratio_fn())
        if math.isnan(ratio) or ratio <= 0.0:
            raise ValueError(f"Cache returned invalid compression ratio {ratio!r}")
        weight = float(reference_tokens)
        return {
            "mode": "token_weighted_cache_ratio",
            "numerator": weight,
            "denominator": 0.0 if math.isinf(ratio) else weight / ratio,
        }
    return {
        "mode": "fullkv_no_cache",
        "numerator": float(reference_tokens),
        "denominator": float(reference_tokens),
    }


def _aggregate_compression_accounting(records) -> dict:
    """Aggregate choices/rows as total bits, never a mean of ratios."""
    if not records:
        return {
            "mode": "fullkv_no_cache",
            "numerator": 0.0,
            "denominator": 0.0,
            "compression_ratio": 1.0,
            "compression_aggregation": "fullkv_no_cache",
            "compression_fp16_bits": None,
            "compression_stored_bits": None,
        }
    modes = {record.get("mode") for record in records}
    if len(modes) != 1:
        raise RuntimeError(
            "Cache compression accounting changed representation across zero-shot choices: "
            f"{sorted(str(mode) for mode in modes)}"
        )
    mode = next(iter(modes))
    if mode not in (
        "exact_cache_bit_counters",
        "token_weighted_cache_ratio",
        "fullkv_no_cache",
    ):
        raise ValueError(f"Unknown zero-shot compression accounting mode {mode!r}")
    numerator = sum(record["numerator"] for record in records)
    denominator = sum(record["denominator"] for record in records)
    if numerator < 0 or denominator < 0:
        raise ValueError("Compression accounting totals must be non-negative")
    ratio = float("inf") if denominator == 0 else float(numerator / denominator)
    return {
        "mode": mode,
        "numerator": numerator,
        "denominator": denominator,
        "compression_ratio": ratio,
        "compression_aggregation": mode,
        "compression_fp16_bits": (int(numerator) if mode == "exact_cache_bit_counters" else None),
        "compression_stored_bits": (
            int(denominator) if mode == "exact_cache_bit_counters" else None
        ),
    }


@torch.no_grad()
def _score_continuation(
    model,
    tokenizer,
    ctx_ids,
    cont_ids,
    cache_factory,
    *,
    expected_protocol: Optional[str] = None,
    expected_dependency: Optional[str] = None,
):
    """Summed and per-token mean log-prob of ``cont_ids`` given ``ctx_ids``."""
    del tokenizer
    device = model.device
    full = torch.cat([ctx_ids, cont_ids], dim=1).to(device)
    past = cache_factory() if cache_factory is not None else None
    cont_len = cont_ids.shape[1]
    first_target = int(full.shape[1]) - int(cont_len)
    logits, protocol, dependency = causal_teacher_forced_logits(model, full, first_target, past)
    if expected_protocol is not None and protocol != expected_protocol:
        raise RuntimeError(
            "Cache factory changed teacher-forced protocol between the probe and "
            f"a scored choice: {expected_protocol!r} -> {protocol!r}"
        )
    if expected_protocol is not None and dependency != expected_dependency:
        raise RuntimeError(
            "Cache factory changed temporal dependency between the probe and "
            f"a scored choice: {expected_dependency!r} -> {dependency!r}"
        )
    logits = logits.float()
    if not bool(torch.isfinite(logits).all().item()):
        raise FloatingPointError(
            "Zero-shot continuation scoring produced non-finite logits; "
            "refusing to turn NaN/inf into a multiple-choice prediction"
        )
    targets = full[:, first_target:]
    logprobs = torch.log_softmax(logits, dim=-1)
    if not bool(torch.isfinite(logprobs).all().item()):
        raise FloatingPointError(
            "Zero-shot continuation scoring produced non-finite log-probabilities"
        )
    tok_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
    if not bool(torch.isfinite(tok_lp).all().item()):
        raise FloatingPointError(
            "Zero-shot continuation scoring produced a non-finite target log-probability"
        )
    score = tok_lp.sum()
    if not bool(torch.isfinite(score).item()):
        raise FloatingPointError(
            "Zero-shot continuation scoring produced a non-finite choice score"
        )
    accounting = _cache_compression_accounting(past, reference_tokens=full.numel())
    return float(score.item()), int(cont_len), accounting


@torch.no_grad()
def _score_choices(
    model,
    tokenizer,
    ctx_ids,
    cont_ids_list,
    cache_factory,
    *,
    expected_protocol: Optional[str] = None,
    expected_dependency: Optional[str] = None,
):
    """Score choices independently so padding never enters a compressed cache.

    Padding is harmless for FullKV with an attention mask, but a per-channel KV
    codec can use padded token activations when fitting group scales. That changes
    valid-token logits and makes the metric depend on the other choices' lengths.
    """
    contexts = ctx_ids if isinstance(ctx_ids, (list, tuple)) else [ctx_ids] * len(cont_ids_list)
    if len(contexts) != len(cont_ids_list):
        raise ValueError("Multiple-input contexts must align one-to-one with choices")
    sums, lengths, accounting_records = [], [], []
    for one_ctx, cont in zip(contexts, cont_ids_list):
        score, length, accounting = _score_continuation(
            model,
            tokenizer,
            one_ctx,
            cont,
            cache_factory,
            expected_protocol=expected_protocol,
            expected_dependency=expected_dependency,
        )
        sums.append(score)
        lengths.append(length)
        accounting_records.append(accounting)
    return sums, lengths, _aggregate_compression_accounting(accounting_records)


def _flush_examples(
    model,
    tokenizer,
    examples,
    cache_factory,
    *,
    expected_protocol: Optional[str] = None,
    expected_dependency: Optional[str] = None,
):
    out = []
    for ctx_ids, cont_ids, gold, choice_char_lens in examples:
        sums, token_lens, accounting = _score_choices(
            model,
            tokenizer,
            ctx_ids,
            cont_ids,
            cache_factory,
            expected_protocol=expected_protocol,
            expected_dependency=expected_dependency,
        )
        # ``_score_continuation`` returns Python doubles. Keep that precision
        # through the decision: an implicit float32 conversion can manufacture
        # a tie between close log-likelihoods and select the wrong first option.
        sums_t = torch.tensor(sums, dtype=torch.float64)
        # lm-eval's acc_norm divides by the character length of the documented
        # choice, not by the tokenizer-dependent number of continuation tokens.
        norm_t = sums_t / torch.tensor(choice_char_lens, dtype=torch.float64).clamp_min(1)
        out.append((int(sums_t.argmax()), int(norm_t.argmax()), gold, accounting, len(token_lens)))
    return out


@torch.no_grad()
def zeroshot_accuracy(
    model,
    tokenizer,
    task: str = "arc_easy",
    cache_factory: Optional[Callable[[], object]] = None,
    limit: Optional[int] = None,
    sample_seed: Optional[int] = None,
    start: int = 0,
) -> ZeroShotResult:
    """Zero-shot ``acc`` / ``acc_norm`` for a multiple-choice task."""
    if task not in _TASKS:
        raise ValueError(f"Unknown zero-shot task {task!r}; expected {sorted(_TASKS)}")
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("start must be a non-negative integer")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive integer or None")
    if sample_seed is not None and (
        isinstance(sample_seed, bool) or not isinstance(sample_seed, int)
    ):
        raise ValueError("sample_seed must be an integer or None")
    protocol_probe = cache_factory() if cache_factory is not None else None
    scoring_protocol = teacher_forced_protocol(protocol_probe)
    temporal_dependency = cache_temporal_dependency(protocol_probe)
    # The probe is needed only to snapshot the immutable scoring contract.  In
    # particular, it must not coexist with the first real choice cache: large
    # compressed caches otherwise double the peak allocation before any task
    # example has been scored.
    del protocol_probe
    ds, fn = _load_task(task)
    dataset_size = len(ds)
    if (start > 0 or limit is not None) and sample_seed is not None:
        ds = ds.shuffle(seed=int(sample_seed))
    n_correct = n_correct_norm = n = 0
    compression_records: List[dict] = []
    per_example: List[dict] = []
    pending = []
    pending_indices: List[int] = []
    protocol_input_digest = hashlib.sha256()
    batch_size = 1
    valid_index = 0
    for ctx, conts, gold in fn(ds):
        if valid_index < start:
            valid_index += 1
            continue
        if limit is not None and n >= limit:
            break
        example_index = valid_index
        valid_index += 1
        if not isinstance(conts, (list, tuple)) or not conts:
            raise ValueError(f"Zero-shot {task} example {example_index} has no choices")
        if isinstance(ctx, (list, tuple)) and len(ctx) != len(conts):
            raise ValueError(
                f"Zero-shot {task} example {example_index} has mismatched contexts/choices"
            )
        if isinstance(gold, bool) or not isinstance(gold, int) or not 0 <= gold < len(conts):
            raise ValueError(
                f"Zero-shot {task} example {example_index} has invalid gold index {gold!r}"
            )
        contexts = list(ctx) if isinstance(ctx, (list, tuple)) else [ctx] * len(conts)
        encoded = [
            _encode_pair(tokenizer, one_ctx, cont) for one_ctx, cont in zip(contexts, conts)
        ]
        ctx_ids = (
            [pair[0] for pair in encoded] if isinstance(ctx, (list, tuple)) else encoded[0][0]
        )
        cont_ids_list = [pair[1] for pair in encoded]
        # lm-eval prepends exactly one request-formatting space to each
        # ``doc_to_choice`` string, then normalizes by the *original choice
        # string's* character length.  Remove exactly that one space here; an
        # additional leading space created by task preprocessing (possible in
        # HellaSwag after a bracket artefact is removed) remains part of the
        # official denominator.
        # WinoGrande is lm-eval's multiple-input case: its (unreported)
        # acc_norm denominator is the option-completed context choice. All other
        # tasks use the continuation choice itself.
        choice_text = ctx if isinstance(ctx, (list, tuple)) else conts
        if isinstance(ctx, (list, tuple)):
            choice_char_lens = [max(1, len(choice)) for choice in choice_text]
        else:
            choice_char_lens = [
                max(1, len(choice[1:] if choice.startswith(" ") else choice))
                for choice in choice_text
            ]
        _update_protocol_input_digest(
            protocol_input_digest,
            task=task,
            index=example_index,
            context=ctx,
            continuations=conts,
            gold=gold,
            choice_char_lengths=choice_char_lens,
        )
        pending.append((ctx_ids, cont_ids_list, gold, choice_char_lens))
        pending_indices.append(example_index)
        target_batch = batch_size if limit is None else max(1, min(batch_size, limit - n))
        if len(pending) >= target_batch:
            flushed = _flush_examples(
                model,
                tokenizer,
                pending,
                cache_factory,
                expected_protocol=scoring_protocol,
                expected_dependency=temporal_dependency,
            )
            if len(flushed) != len(pending_indices):
                raise RuntimeError("Zero-shot scorer returned the wrong number of examples")
            for example_index_i, (pred, pred_norm, gold_i, accounting, _n_choices) in zip(
                pending_indices, flushed
            ):
                n_correct += int(pred == gold_i)
                n_correct_norm += int(pred_norm == gold_i)
                compression_records.append(accounting)
                per_example.append(
                    {
                        "index": int(example_index_i),
                        "gold": int(gold_i),
                        "prediction": int(pred),
                        "prediction_norm": int(pred_norm),
                        "correct": bool(pred == gold_i),
                        "correct_norm": bool(pred_norm == gold_i),
                    }
                )
                n += 1
            pending.clear()
            pending_indices.clear()
        if limit is not None and n >= limit:
            break
    if pending and (limit is None or n < limit):
        flushed = _flush_examples(
            model,
            tokenizer,
            pending,
            cache_factory,
            expected_protocol=scoring_protocol,
            expected_dependency=temporal_dependency,
        )
        if len(flushed) != len(pending_indices):
            raise RuntimeError("Zero-shot scorer returned the wrong number of examples")
        for example_index_i, (pred, pred_norm, gold_i, accounting, _n_choices) in zip(
            pending_indices, flushed
        ):
            n_correct += int(pred == gold_i)
            n_correct_norm += int(pred_norm == gold_i)
            compression_records.append(accounting)
            per_example.append(
                {
                    "index": int(example_index_i),
                    "gold": int(gold_i),
                    "prediction": int(pred),
                    "prediction_norm": int(pred_norm),
                    "correct": bool(pred == gold_i),
                    "correct_norm": bool(pred_norm == gold_i),
                }
            )
            n += 1
            if limit is not None and n >= limit:
                break
    if n == 0:
        raise ValueError(
            f"Zero-shot selection is empty for {task}: start={start}, limit={limit}, "
            f"dataset_size={dataset_size}"
        )
    if start == 0 and limit is None and n != dataset_size:
        raise RuntimeError(
            f"Complete {task} evaluation produced {n} valid examples, but the pinned "
            f"split has {dataset_size}; refusing to label an incomplete run as full"
        )
    compression = _aggregate_compression_accounting(compression_records)
    return ZeroShotResult(
        task=task,
        n=n,
        acc=n_correct / max(n, 1),
        acc_norm=n_correct_norm / max(n, 1),
        compression_ratio=compression["compression_ratio"],
        compression_aggregation=compression["compression_aggregation"],
        compression_fp16_bits=compression["compression_fp16_bits"],
        compression_stored_bits=compression["compression_stored_bits"],
        dataset_size=dataset_size,
        split=_TASKS[task][2],
        dataset_revision=_TASKS[task][4],
        correct=n_correct,
        correct_norm=n_correct_norm,
        start=start,
        limit=limit,
        sample_seed=sample_seed,
        per_example=per_example,
        protocol_input_sha256=protocol_input_digest.hexdigest(),
        teacher_forced_scoring_protocol=scoring_protocol,
        teacher_forced_temporal_dependency=temporal_dependency,
    )

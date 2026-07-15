"""Zero-shot accuracy via length-normalised log-likelihood.

Implements the lm-eval-harness ``acc``/``acc_norm`` multiple-choice protocol:
for each example we score every candidate continuation by its summed token
log-probability (and a length-normalised variant) under the model, then pick
the arg-max.  Each example is scored with a fresh cache from ``cache_factory``
so KV compression affects the prediction exactly as at inference time.

Supported tasks (all loaded via ``datasets``):
  * ``arc_easy``      -- AI2 ARC-Easy (test).
  * ``arc_challenge`` -- AI2 ARC-Challenge (test).
  * ``piqa``          -- physical commonsense QA (validation).
  * ``hellaswag``     -- grounded commonsense (validation, acc_norm headline).
  * ``winogrande``    -- pronoun/coreference commonsense (validation).

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from logging import getLogger
from typing import Callable, List, Optional

import torch

logger = getLogger(__name__)


@dataclass
class ZeroShotResult:
    task: str
    n: int
    acc: float
    acc_norm: float
    compression_ratio: float = 1.0
    dataset_size: Optional[int] = None
    split: Optional[str] = None

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
            "evaluator": "lm-eval-compatible@0.4.11",
        }
        if self.dataset_size is not None:
            out["dataset_size"] = int(self.dataset_size)
            out["complete_dataset_eval"] = self.n == self.dataset_size
        if self.split is not None:
            out["split"] = self.split
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
        text = text.strip().replace(" [title]", ". ")
        text = re.sub(r"\[.*?\]", "", text)
        return re.sub(r"\s+", " ", text).strip()

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
        suffix = ex["sentence"][split + 1:].strip()
        ctx = [prefix + ex["option1"], prefix + ex["option2"]]
        conts = [f" {suffix}", f" {suffix}"]
        try:
            gold = int(ex["answer"]) - 1
        except (ValueError, TypeError):
            continue
        yield ctx, conts, gold


_TASKS = {
    # lm-eval evaluates ARC on its test split, not validation.
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test", _arc_easy),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test", _arc_challenge),
    "piqa": ("baber/piqa", None, "validation", _piqa),
    "hellaswag": ("Rowan/hellaswag", None, "validation", _hellaswag),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation", _winogrande),
}


def _load_task(task: str):
    from datasets import load_dataset  # pylint: disable=import-outside-toplevel

    path, name, split, fn = _TASKS[task]
    ds = load_dataset(path, name, split=split) if name else load_dataset(path, split=split)
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
    cont = whole[:, ctx.shape[1]:]
    if cont.shape[1] == 0:
        raise ValueError(
            "Continuation tokenized to zero tokens under lm-eval boundary handling"
        )
    return ctx, cont


@torch.no_grad()
def _score_continuation(model, tokenizer, ctx_ids, cont_ids, cache_factory):
    """Summed and per-token mean log-prob of ``cont_ids`` given ``ctx_ids``."""
    device = model.device
    full = torch.cat([ctx_ids, cont_ids], dim=1).to(device)
    past = cache_factory() if cache_factory is not None else None
    out = model(full, past_key_values=past, use_cache=past is not None)
    logits = out.logits[:, :-1].float()
    targets = full[:, 1:]
    logprobs = torch.log_softmax(logits, dim=-1)
    tok_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
    cont_len = cont_ids.shape[1]
    cont_lp = tok_lp[-cont_len:]
    ratio = past.compression_ratio() if (past is not None and hasattr(past, "compression_ratio")) else 1.0
    return float(cont_lp.sum()), int(cont_len), ratio


@torch.no_grad()
def _score_choices(model, tokenizer, ctx_ids, cont_ids_list, cache_factory):
    """Score choices independently so padding never enters a compressed cache.

    Padding is harmless for FullKV with an attention mask, but a per-channel KV
    codec can use padded token activations when fitting group scales. That changes
    valid-token logits and makes the metric depend on the other choices' lengths.
    """
    contexts = ctx_ids if isinstance(ctx_ids, (list, tuple)) else [ctx_ids] * len(cont_ids_list)
    if len(contexts) != len(cont_ids_list):
        raise ValueError("Multiple-input contexts must align one-to-one with choices")
    sums, lengths, ratios = [], [], []
    for one_ctx, cont in zip(contexts, cont_ids_list):
        score, length, ratio = _score_continuation(
            model, tokenizer, one_ctx, cont, cache_factory)
        sums.append(score)
        lengths.append(length)
        ratios.append(ratio)
    return sums, lengths, (sum(ratios) / len(ratios) if ratios else 1.0)


@torch.no_grad()
def _score_choice_batch(model, tokenizer, examples, cache_factory):
    """Score candidate continuations for several examples in one forward pass."""
    device = model.device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    rows = []
    lengths = []
    cont_lens = []
    group_sizes = []
    for example in examples:
        ctx_ids, cont_ids_list = example[:2]
        group_sizes.append(len(cont_ids_list))
        ctx_rows = ctx_ids if isinstance(ctx_ids, (list, tuple)) else [ctx_ids] * len(cont_ids_list)
        if len(ctx_rows) != len(cont_ids_list):
            raise ValueError("Multiple-input contexts must align one-to-one with choices")
        for one_ctx_ids, cont_ids in zip(ctx_rows, cont_ids_list):
            full = torch.cat([one_ctx_ids, cont_ids], dim=1)[0]
            rows.append(full)
            lengths.append(int(full.numel()))
            cont_lens.append(int(cont_ids.shape[1]))

    max_len = max(lengths)
    batch = rows[0].new_full((len(rows), max_len), int(pad_id))
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    for i, row in enumerate(rows):
        batch[i, : row.numel()] = row
        attention_mask[i, : row.numel()] = 1

    batch = batch.to(device)
    attention_mask = attention_mask.to(device)
    past = cache_factory() if cache_factory is not None else None
    out = model(
        batch,
        attention_mask=attention_mask,
        past_key_values=past,
        use_cache=past is not None,
    )
    logits = out.logits[:, :-1].float()
    targets = batch[:, 1:]
    logprobs = torch.log_softmax(logits, dim=-1)
    tok_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    sums = []
    for i, (seq_len, cont_len) in enumerate(zip(lengths, cont_lens)):
        start = seq_len - cont_len - 1
        end = seq_len - 1
        sums.append(float(tok_lp[i, start:end].sum()))
    ratio = past.compression_ratio() if (past is not None and hasattr(past, "compression_ratio")) else 1.0
    grouped = []
    offset = 0
    for size in group_sizes:
        grouped.append((sums[offset : offset + size], cont_lens[offset : offset + size]))
        offset += size
    return grouped, ratio


def _flush_examples(model, tokenizer, examples, cache_factory):
    out = []
    for ctx_ids, cont_ids, gold, choice_char_lens in examples:
        sums, token_lens, ratio = _score_choices(
            model, tokenizer, ctx_ids, cont_ids, cache_factory)
        sums_t = torch.tensor(sums)
        # lm-eval's acc_norm divides by the character length of the documented
        # choice, not by the tokenizer-dependent number of continuation tokens.
        norm_t = sums_t / torch.tensor(choice_char_lens).clamp_min(1)
        out.append((int(sums_t.argmax()), int(norm_t.argmax()), gold, ratio, len(token_lens)))
    return out


@torch.no_grad()
def zeroshot_accuracy(
    model,
    tokenizer,
    task: str = "arc_easy",
    cache_factory: Optional[Callable[[], object]] = None,
    limit: Optional[int] = None,
    sample_seed: Optional[int] = None,
) -> ZeroShotResult:
    """Zero-shot ``acc`` / ``acc_norm`` for a multiple-choice task."""
    ds, fn = _load_task(task)
    dataset_size = len(ds)
    if limit is not None and sample_seed is not None:
        ds = ds.shuffle(seed=int(sample_seed))
    n_correct = n_correct_norm = n = 0
    ratios: List[float] = []
    pending = []
    batch_size = 1
    for ctx, conts, gold in fn(ds):
        contexts = list(ctx) if isinstance(ctx, (list, tuple)) else [ctx] * len(conts)
        encoded = [_encode_pair(tokenizer, one_ctx, cont)
                   for one_ctx, cont in zip(contexts, conts)]
        ctx_ids = [pair[0] for pair in encoded] if isinstance(ctx, (list, tuple)) else encoded[0][0]
        cont_ids_list = [pair[1] for pair in encoded]
        # Leading continuation whitespace is an evaluation formatting detail and
        # is not part of lm-eval's ``doc_to_choice`` string length.
        # WinoGrande is lm-eval's multiple-input case: its (unreported)
        # acc_norm denominator is the option-completed context choice. All other
        # tasks use the continuation choice itself.
        choice_text = ctx if isinstance(ctx, (list, tuple)) else conts
        choice_char_lens = [max(1, len(choice.lstrip())) for choice in choice_text]
        pending.append((ctx_ids, cont_ids_list, gold, choice_char_lens))
        target_batch = batch_size if limit is None else max(1, min(batch_size, limit - n))
        if len(pending) >= target_batch:
            for pred, pred_norm, gold_i, ratio, n_choices in _flush_examples(
                model, tokenizer, pending, cache_factory
            ):
                n_correct += int(pred == gold_i)
                n_correct_norm += int(pred_norm == gold_i)
                ratios.extend([ratio] * n_choices)
                n += 1
            pending.clear()
        if limit is not None and n >= limit:
            break
    if pending and (limit is None or n < limit):
        for pred, pred_norm, gold_i, ratio, n_choices in _flush_examples(
            model, tokenizer, pending, cache_factory
        ):
            n_correct += int(pred == gold_i)
            n_correct_norm += int(pred_norm == gold_i)
            ratios.extend([ratio] * n_choices)
            n += 1
            if limit is not None and n >= limit:
                break
    return ZeroShotResult(
        task=task,
        n=n,
        acc=n_correct / max(n, 1),
        acc_norm=n_correct_norm / max(n, 1),
        compression_ratio=float(sum(ratios) / len(ratios)) if ratios else 1.0,
        dataset_size=dataset_size,
        split=_TASKS[task][2],
    )

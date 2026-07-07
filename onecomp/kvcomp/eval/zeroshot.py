"""Zero-shot accuracy via length-normalised log-likelihood.

Implements the lm-eval-harness ``acc``/``acc_norm`` multiple-choice protocol:
for each example we score every candidate continuation by its summed token
log-probability (and a length-normalised variant) under the model, then pick
the arg-max.  Each example is scored with a fresh cache from ``cache_factory``
so KV compression affects the prediction exactly as at inference time.

Supported tasks (all loaded via ``datasets``):
  * ``arc_easy``      -- AI2 ARC-Easy (validation).
  * ``arc_challenge`` -- AI2 ARC-Challenge (validation).
  * ``piqa``          -- physical commonsense QA (validation).
  * ``hellaswag``     -- grounded commonsense (validation, acc_norm headline).
  * ``winogrande``    -- pronoun/coreference commonsense (validation).

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

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

    def as_dict(self) -> dict:
        return {
            "task": self.task,
            "n": self.n,
            "acc": round(self.acc, 4),
            "acc_norm": round(self.acc_norm, 4),
            "compression_ratio": round(self.compression_ratio, 3),
        }


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
    for ex in ds_split:
        ctx = ex["ctx"]
        conts = [" " + e for e in ex["endings"]]
        try:
            gold = int(ex["label"])
        except (ValueError, TypeError):
            continue
        yield ctx, conts, gold


def _winogrande(ds_split):
    for ex in ds_split:
        prefix, suffix = ex["sentence"].split("_", 1)
        ctx = prefix
        conts = [f"{ex['option1']}{suffix}", f"{ex['option2']}{suffix}"]
        try:
            gold = int(ex["answer"]) - 1
        except (ValueError, TypeError):
            continue
        yield ctx, conts, gold


_TASKS = {
    "arc_easy": (" allenai/ai2_arc".strip(), "ARC-Easy", "validation", _arc_easy),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "validation", _arc_challenge),
    "piqa": ("baber/piqa", None, "validation", _piqa),
    "hellaswag": ("Rowan/hellaswag", None, "validation", _hellaswag),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation", _winogrande),
}


def _load_task(task: str):
    from datasets import load_dataset  # pylint: disable=import-outside-toplevel

    path, name, split, fn = _TASKS[task]
    ds = load_dataset(path, name, split=split) if name else load_dataset(path, split=split)
    return ds, fn


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
    """Score all candidate continuations for one example in a single batch."""
    grouped, ratio = _score_choice_batch(
        model, tokenizer, [(ctx_ids, cont_ids_list)], cache_factory
    )
    sums, cont_lens = grouped[0]
    return sums, cont_lens, ratio


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
        for cont_ids in cont_ids_list:
            full = torch.cat([ctx_ids, cont_ids], dim=1)[0]
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
    scored, ratio = _score_choice_batch(model, tokenizer, examples, cache_factory)
    out = []
    for (_, _, gold), (sums, lens) in zip(examples, scored):
        sums_t = torch.tensor(sums)
        norm_t = sums_t / torch.tensor(lens).clamp_min(1)
        out.append((int(sums_t.argmax()), int(norm_t.argmax()), gold, ratio, len(lens)))
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
    if limit is not None and sample_seed is not None:
        ds = ds.shuffle(seed=int(sample_seed))
    n_correct = n_correct_norm = n = 0
    ratios: List[float] = []
    pending = []
    batch_size = 1
    for ctx, conts, gold in fn(ds):
        ctx_ids = tokenizer(ctx, return_tensors="pt").input_ids
        cont_ids_list = [
            tokenizer(cont, return_tensors="pt", add_special_tokens=False).input_ids
            for cont in conts
        ]
        pending.append((ctx_ids, cont_ids_list, gold))
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
    )

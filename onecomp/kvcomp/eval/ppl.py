"""Perplexity evaluation under a (possibly compressed) KV cache.

``wikitext2_perplexity`` runs the standard sliding-window NLL on WikiText-2
(raw, test split).  Each window starts from a *fresh* cache built by
``cache_factory`` so a codec is measured exactly the way it is used at decode
time.  This is the KV analogue of the weight side's perplexity validation.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from logging import getLogger
from typing import Callable, Optional

import torch

logger = getLogger(__name__)


def _load_wikitext2_text() -> str:
    """Return the concatenated WikiText-2 raw test text (requires ``datasets``)."""
    from datasets import load_dataset  # pylint: disable=import-outside-toplevel

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(ds["text"])


@torch.no_grad()
def perplexity_over_text(
    model,
    tokenizer,
    text: str,
    cache_factory: Optional[Callable[[], object]] = None,
    max_length: int = 2048,
    stride: int = 512,
    max_windows: Optional[int] = None,
) -> dict:
    """Sliding-window perplexity over ``text``.

    Args:
        model: HF causal LM (already on device, eval mode).
        tokenizer: Matching tokenizer.
        text: Raw evaluation text.
        cache_factory: ``() -> transformers.Cache`` built fresh per window.
            ``None`` uses the model's default (FullKV) cache.
        max_length: Context window length.
        stride: Sliding stride (overlap = max_length - stride).
        max_windows: Cap the number of windows (for quick smoke runs).

    Returns:
        dict with ``ppl``, ``nll``, ``n_tokens``, ``compression_ratio``.
    """
    device = model.device
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    seq_len = input_ids.shape[1]

    nlls = []
    n_tokens = 0
    ratios = []
    prev_end = 0
    n_win = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end  # number of newly-scored targets
        ids = input_ids[:, begin:end]
        target = ids.clone()
        target[:, :-trg_len] = -100

        past = cache_factory() if cache_factory is not None else None
        out = model(ids, past_key_values=past, use_cache=past is not None)
        logits = out.logits[:, :-1].float()
        tgt = target[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100,
            reduction="sum",
        )
        valid = (tgt != -100).sum().item()
        if valid > 0:
            nlls.append(loss)
            n_tokens += valid
        if past is not None and hasattr(past, "compression_ratio"):
            ratios.append(past.compression_ratio())
        prev_end = end
        n_win += 1
        if end == seq_len or (max_windows is not None and n_win >= max_windows):
            break

    total_nll = torch.stack(nlls).sum() / max(n_tokens, 1)
    return {
        "ppl": float(torch.exp(total_nll)),
        "nll": float(total_nll),
        "n_tokens": n_tokens,
        "n_windows": n_win,
        "compression_ratio": float(sum(ratios) / len(ratios)) if ratios else 1.0,
    }


def wikitext2_perplexity(
    model,
    tokenizer,
    cache_factory: Optional[Callable[[], object]] = None,
    max_length: int = 2048,
    stride: int = 512,
    max_windows: Optional[int] = None,
    fallback_text: Optional[str] = None,
) -> dict:
    """WikiText-2 (raw, test) sliding-window perplexity.

    Falls back to ``fallback_text`` when ``datasets`` / network is unavailable.
    """
    try:
        text = _load_wikitext2_text()
        source = "wikitext-2-raw-v1/test"
    except Exception as exc:  # pragma: no cover - network/datasets dependent
        if fallback_text is None:
            raise
        logger.warning("WikiText-2 load failed (%s); using fallback text", exc)
        text = fallback_text
        source = "fallback"
    out = perplexity_over_text(
        model, tokenizer, text, cache_factory=cache_factory,
        max_length=max_length, stride=stride, max_windows=max_windows,
    )
    out["source"] = source
    return out

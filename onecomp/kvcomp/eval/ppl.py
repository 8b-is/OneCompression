"""Perplexity evaluation under a (possibly compressed) KV cache.

``wikitext2_perplexity`` runs the standard sliding-window NLL on WikiText-2
(raw, test split).  Each window starts from a *fresh* cache built by
``cache_factory`` so a codec is measured exactly the way it is used at decode
time.  This is the KV analogue of the weight side's perplexity validation.

Measurement-validity guards
---------------------------
* **Causal cache state.**  A channel-axis group scale (KIVI/NVFP4/Ternary),
  dynamic outlier-channel set, online bit split, block-global preconditioner,
  trailing full-precision window, age-dependent precision schedule, ThinKV
  group, or eviction decision must
  not be fitted/applied using future target tokens.  Such caches are scored by
  pre-filling only the unscored prefix and forwarding scored targets one token
  at a time.  Token-local HQ/TQ configurations keep the equivalent
  single-forward causal-mask path.
* **Non-finite windows.**  A window whose summed loss is ``NaN``/``inf`` makes
  the complete run invalid. It is never silently dropped from the denominator.
* **Instruct / multimodal models.**  Raw-text WikiText PPL is only meaningful for
  a *base* causal LM.  On instruction-tuned or multimodal (``*ForConditional
  Generation``) checkpoints scored on raw text, the absolute PPL is unreliable
    and KV quantization can occasionally lower the observed test NLL by acting as
  a perturbation -- prefer the base model, and compare codecs only relative to the
  *same* model's FullKV (never across models). A small negative delta is not proof
  that the codec is lossless or generally improves the model.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import hashlib
import math
from logging import getLogger
from typing import Callable, Optional

import torch

from ._teacher_forcing import causal_teacher_forced_logits

logger = getLogger(__name__)

WIKITEXT2_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT2_TEXT_SHA256 = "696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83"


def _load_wikitext2_text() -> str:
    """Return the concatenated WikiText-2 raw test text (requires ``datasets``).

    Formal inputs pin the immutable ``Salesforce/wikitext`` parquet source.
    Do not try the legacy unqualified alias first: whichever cache happens to
    exist on a worker must not choose the scientific input.  The joined-text
    digest also fixes the row separator and detects upstream/cache drift.
    """
    from datasets import load_dataset  # pylint: disable=import-outside-toplevel

    ds = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
        revision=WIKITEXT2_REVISION,
    )
    text = "\n\n".join(ds["text"])
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != WIKITEXT2_TEXT_SHA256:
        raise RuntimeError(
            "Pinned WikiText-2 joined text changed: "
            f"expected {WIKITEXT2_TEXT_SHA256}, got {digest}"
        )
    return text


@torch.no_grad()
def perplexity_over_text(
    model,
    tokenizer,
    text: str,
    cache_factory: Optional[Callable[[], object]] = None,
    max_length: int = 2048,
    stride: int = 512,
    max_windows: Optional[int] = None,
    fail_on_nonfinite: bool = True,
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
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 2:
        raise ValueError("max_length must be an integer >= 2 to score next-token loss")
    if (
        isinstance(stride, bool)
        or not isinstance(stride, int)
        or stride <= 0
        or stride > max_length
    ):
        raise ValueError("stride must be an integer satisfying 0 < stride <= max_length")
    if max_windows is not None and (
        isinstance(max_windows, bool) or not isinstance(max_windows, int) or max_windows <= 0
    ):
        raise ValueError("max_windows must be a positive integer or None")
    if not isinstance(fail_on_nonfinite, bool):
        raise ValueError("fail_on_nonfinite must be a bool")
    device = model.device
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    seq_len = input_ids.shape[1]
    if seq_len < 2:
        raise ValueError("tokenized PPL text must contain at least two tokens")

    nlls = []
    n_tokens = 0
    compression_mode = None
    compression_numerator: int | float = 0
    compression_denominator: int | float = 0
    prev_end = 0
    n_win = 0
    n_skipped = 0
    complete_text = False
    scoring_protocol = None
    temporal_dependency = None
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end  # number of newly-scored targets
        ids = input_ids[:, begin:end]

        past = cache_factory() if cache_factory is not None else None
        # Only a suffix of each overlapping window is newly scored.  A
        # temporally coupled codec may prefill the unscored prefix, but each
        # scored target must be predicted before that target enters the cache.
        # Token-local HQ/TQ configurations retain the fast one-forward path.
        first_target = max(1, int(ids.shape[1]) - int(trg_len))
        logits, window_protocol, window_dependency = causal_teacher_forced_logits(
            model, ids, first_target, past
        )
        if scoring_protocol not in (None, window_protocol):
            raise RuntimeError("Teacher-forced scoring protocol changed across PPL windows")
        if temporal_dependency not in (None, window_dependency) and window_dependency is not None:
            raise RuntimeError("Cache temporal dependency changed across PPL windows")
        scoring_protocol = window_protocol
        if window_dependency is not None:
            temporal_dependency = window_dependency
        logits = logits.float()
        tgt = ids[:, first_target:]
        token_losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            reduction="none",
        )
        valid = tgt.numel()
        # The model probabilities are intentionally evaluated in fp32, but a
        # fp32 reduction over thousands of tokens needlessly rounds the NLL
        # numerator.  Preserve every fp32 token loss and accumulate it in fp64;
        # formal HQ/TQ comparisons are based on the within-model NLL difference.
        loss = token_losses.to(dtype=torch.float64).sum()
        prev_end = end
        n_win += 1
        # Count non-finite windows, then invalidate the complete run below. Using
        # a different denominator for each codec can make an unstable codec look
        # artificially better than FullKV.
        if valid > 0 and torch.isfinite(loss):
            nlls.append(loss)
            n_tokens += valid
        elif valid > 0:
            n_skipped += 1
            logger.warning(
                "Skipping window [%d:%d] with non-finite loss (NaN/inf); "
                "excluded from perplexity.",
                begin,
                end,
            )
        if past is not None:
            fp16_bits = getattr(past, "fp16_bits", None)
            stored_bits = getattr(past, "stored_bits", None)
            exact_counters = (
                not isinstance(fp16_bits, bool)
                and isinstance(fp16_bits, int)
                and fp16_bits >= 0
                and not isinstance(stored_bits, bool)
                and isinstance(stored_bits, int)
                and stored_bits >= 0
            )
            if exact_counters:
                if compression_mode not in (None, "exact_cache_bit_counters"):
                    raise RuntimeError(
                        "Cache compression accounting changed representation across PPL windows"
                    )
                compression_mode = "exact_cache_bit_counters"
                compression_numerator += fp16_bits
                compression_denominator += stored_bits
            elif hasattr(past, "compression_ratio"):
                if compression_mode not in (None, "token_weighted_cache_ratio"):
                    raise RuntimeError(
                        "Cache compression accounting changed representation across PPL windows"
                    )
                ratio = float(past.compression_ratio())
                if math.isnan(ratio) or ratio <= 0.0:
                    raise ValueError(f"Cache returned invalid compression ratio {ratio!r}")
                # For a fixed model, the window's uncompressed KV bits are a
                # constant multiple of its token count.  Weighting by input
                # tokens therefore reconstructs sum(fp16_bits)/sum(stored_bits)
                # even for eviction/graded caches that expose only a ratio.
                reference_weight = float(ids.numel())
                compression_mode = "token_weighted_cache_ratio"
                compression_numerator += reference_weight
                compression_denominator += 0.0 if math.isinf(ratio) else reference_weight / ratio
        if end == seq_len:
            complete_text = True
            break
        if max_windows is not None and n_win >= max_windows:
            break

    if not nlls:
        raise ValueError(
            "No finite-loss windows scored; perplexity is undefined "
            f"({n_skipped} non-finite windows skipped). The model/text combination "
            "is numerically unstable (e.g. raw text on an instruct/multimodal model)."
        )
    if n_skipped:
        if fail_on_nonfinite:
            raise ValueError(
                f"Perplexity run is invalid: {n_skipped}/{n_win} windows had non-finite loss."
            )
        logger.warning(
            "%d/%d windows had non-finite loss and were excluded; treat the PPL "
            "as approximate and check the model/text suitability.",
            n_skipped,
            n_win,
        )
    # Preserve the unrounded numerator as well as the mean.  Formal codec
    # comparisons use within-model NLL differences: rounding PPL before taking
    # a ratio can manufacture or erase the small deltas seen at mild rates.
    nll_sum = torch.stack(nlls).to(dtype=torch.float64).sum()
    total_nll = nll_sum / max(n_tokens, 1)
    if compression_mode is None or compression_numerator == 0.0:
        compression_ratio = 1.0
    elif compression_denominator == 0.0:
        compression_ratio = float("inf")
    else:
        compression_ratio = compression_numerator / compression_denominator
    return {
        "ppl": float(torch.exp(total_nll)),
        "nll": float(total_nll),
        "nll_sum": float(nll_sum),
        "n_tokens": n_tokens,
        "n_windows": n_win,
        "n_skipped_nonfinite": n_skipped,
        "complete_text_eval": complete_text,
        "max_windows": max_windows,
        "compression_ratio": float(compression_ratio),
        "compression_aggregation": compression_mode or "fullkv_no_cache",
        "compression_fp16_bits": (
            int(compression_numerator) if compression_mode == "exact_cache_bit_counters" else None
        ),
        "compression_stored_bits": (
            int(compression_denominator)
            if compression_mode == "exact_cache_bit_counters"
            else None
        ),
        "teacher_forced_scoring_protocol": scoring_protocol,
        "teacher_forced_temporal_dependency": temporal_dependency,
    }


def wikitext2_perplexity(
    model,
    tokenizer,
    cache_factory: Optional[Callable[[], object]] = None,
    max_length: int = 2048,
    stride: int = 512,
    max_windows: Optional[int] = None,
    fallback_text: Optional[str] = None,
    fail_on_nonfinite: bool = True,
) -> dict:
    """WikiText-2 (raw, test) sliding-window perplexity.

    Falls back to ``fallback_text`` when ``datasets`` / network is unavailable.
    """
    try:
        text = _load_wikitext2_text()
        source = "wikitext-2-raw-v1/test"
        dataset_revision = WIKITEXT2_REVISION
    except Exception as exc:  # pragma: no cover - network/datasets dependent
        if fallback_text is None:
            raise
        logger.warning("WikiText-2 load failed (%s); using fallback text", exc)
        text = fallback_text
        source = "fallback"
        dataset_revision = None
    out = perplexity_over_text(
        model,
        tokenizer,
        text,
        cache_factory=cache_factory,
        max_length=max_length,
        stride=stride,
        max_windows=max_windows,
        fail_on_nonfinite=fail_on_nonfinite,
    )
    out["source"] = source
    out["dataset_revision"] = dataset_revision
    out["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return out

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

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from logging import getLogger
from typing import Callable, Optional

import torch

from ._teacher_forcing import causal_teacher_forced_logits

logger = getLogger(__name__)

WIKITEXT2_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT2_TEXT_SHA256 = "696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83"
PPL_NLL_ACCUMULATION = (
    "fp32_token_cross_entropy_per_window_fp64_sum_host_binary64_math_fsum"
)
PPL_WINDOW_CACHE_LIFETIME = "released_before_next_cache_factory"
PPL_CROSS_WINDOW_TENSOR_RETENTION = (
    "tokenized_input_ids_only; window_cache_logits_targets_losses_released; "
    "nll_host_binary64_scalars"
)


def _ppl_window_plan(
    sequence_length: int,
    *,
    max_length: int,
    stride: int,
    max_windows: Optional[int],
) -> list[dict]:
    """Return the exact global target-token indices scored by each window.

    A causal LM cannot score global token zero.  Every later target needs its
    predecessor in the same reset window, which is why a disjoint multi-window
    stride is invalid: the first token of each later block has no predecessor
    and the historical implementation silently omitted it.  ``range`` objects
    keep this plan O(number of windows) even for the full WikiText stream while
    still making the exact target-index set executable in regression tests.
    """

    if (
        isinstance(sequence_length, bool)
        or not isinstance(sequence_length, int)
        or sequence_length < 2
    ):
        raise ValueError("tokenized PPL text must contain at least two tokens")
    if stride == max_length and sequence_length > max_length and max_windows != 1:
        raise ValueError(
            "stride == max_length is invalid for multi-window PPL: a disjoint "
            "reset window silently leaves its first global target unscored; "
            "use stride < max_length (or max_windows == 1 for a one-window control)"
        )

    windows: list[dict] = []
    previous_end = 0
    for begin in range(0, sequence_length, stride):
        end = min(begin + max_length, sequence_length)
        new_suffix_length = end - previous_end
        first_target = max(1, (end - begin) - new_suffix_length)
        target_indices = range(begin + first_target, end)
        windows.append(
            {
                "begin": begin,
                "end": end,
                "first_target": first_target,
                "target_indices": target_indices,
            }
        )
        previous_end = end
        if end == sequence_length:
            break
        if max_windows is not None and len(windows) >= max_windows:
            break

    # This is both a construction invariant and a guard against future changes
    # to window arithmetic: the union must be exactly [1, traversed_end), with
    # neither duplicate nor dropped targets.
    expected_next = 1
    for window in windows:
        targets = window["target_indices"]
        if targets.start != expected_next or targets.step != 1 or targets.stop < targets.start:
            raise RuntimeError(
                "PPL window plan does not cover a contiguous exact target-index set: "
                f"expected next={expected_next}, got range({targets.start}, {targets.stop})"
            )
        expected_next = targets.stop
    if not windows or expected_next != windows[-1]["end"]:
        raise RuntimeError("PPL window plan failed to cover every traversed next-token target")
    return windows


def _adaptive_hq_runtime_snapshot(cache: object) -> dict | None:
    """Return one cache window's compact, validated AdaptiveHQ ledger."""

    callback = getattr(cache, "get_adaptive_hq_provenance", None)
    if callback is None:
        return None
    if not callable(callback):
        raise TypeError("get_adaptive_hq_provenance must be callable")
    provenance = callback()
    runtime = provenance.get("runtime") if isinstance(provenance, Mapping) else None
    if not runtime:
        return None
    aggregate = runtime.get("aggregate") if isinstance(runtime, Mapping) else None
    if runtime.get("kind") != "adaptive_hq" or not isinstance(aggregate, Mapping):
        raise RuntimeError("AdaptiveHQ PPL window provenance is incomplete")
    updates = aggregate.get("encoded_updates")
    tokens = aggregate.get("encoded_tokens")
    selector_bits = aggregate.get("selector_bits_total")
    if (
        runtime.get("schema_version") != 1
        or not isinstance(runtime.get("policy_version"), str)
        or not runtime["policy_version"]
        or isinstance(updates, bool)
        or not isinstance(updates, int)
        or updates < 1
        or isinstance(tokens, bool)
        or not isinstance(tokens, int)
        or tokens < updates
        or isinstance(selector_bits, bool)
        or not isinstance(selector_bits, int)
        or selector_bits not in (0, updates)
        or aggregate.get("selector_incremental_stored_bits_total") != 0
        or not isinstance(aggregate.get("selector_carrier_version"), str)
    ):
        raise RuntimeError("AdaptiveHQ PPL window counters are malformed")
    raw_histogram = aggregate.get("split_histogram")
    if not isinstance(raw_histogram, Mapping) or not raw_histogram:
        raise RuntimeError("AdaptiveHQ PPL window split histogram is missing")
    histogram = {}
    for key, raw in sorted(raw_histogram.items()):
        if (
            not isinstance(key, str)
            or not isinstance(raw, Mapping)
            or not isinstance(raw.get("pair"), list)
            or len(raw["pair"]) != 2
            or any(
                isinstance(bit, bool) or not isinstance(bit, int) or bit < 1 for bit in raw["pair"]
            )
            or key != f"{raw['pair'][0]}:{raw['pair'][1]}"
        ):
            raise RuntimeError("AdaptiveHQ PPL window split histogram is malformed")
        bucket = {"pair": [int(raw["pair"][0]), int(raw["pair"][1])]}
        for field in ("updates", "tokens", "vectors"):
            count = raw.get(field)
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise RuntimeError("AdaptiveHQ PPL window histogram count is malformed")
            bucket[field] = int(count)
        if not bucket["updates"] <= bucket["tokens"] <= bucket["vectors"]:
            raise RuntimeError("AdaptiveHQ PPL window histogram counters are inconsistent")
        histogram[key] = bucket
    if (
        sum(bucket["updates"] for bucket in histogram.values()) != updates
        or sum(bucket["tokens"] for bucket in histogram.values()) != tokens
        or (selector_bits == 0 and len(histogram) != 1)
    ):
        raise RuntimeError("AdaptiveHQ PPL window histogram does not replay")
    return {
        "schema_version": 1,
        "policy_version": runtime["policy_version"],
        "encoded_updates": updates,
        "encoded_tokens": tokens,
        "selector_bits_per_encode": selector_bits // updates,
        "selector_bits_total": selector_bits,
        "selector_incremental_stored_bits_total": 0,
        "selector_carrier_version": aggregate["selector_carrier_version"],
        "split_histogram": histogram,
    }


def _aggregate_adaptive_hq_ppl_windows(windows: Sequence[Mapping]) -> dict | None:
    """Merge per-window selector histograms without losing window boundaries."""

    if not windows:
        return None
    runtimes = [window.get("runtime") for window in windows]
    if any(not isinstance(runtime, Mapping) for runtime in runtimes):
        raise RuntimeError("AdaptiveHQ PPL window runtime is missing")
    policy_versions = {runtime["policy_version"] for runtime in runtimes}
    carrier_versions = {runtime["selector_carrier_version"] for runtime in runtimes}
    selector_widths = {runtime["selector_bits_per_encode"] for runtime in runtimes}
    if len(policy_versions) != 1 or len(carrier_versions) != 1 or len(selector_widths) != 1:
        raise RuntimeError("AdaptiveHQ PPL policy/carrier changed across windows")
    histogram = {}
    for runtime in runtimes:
        for key, bucket in runtime["split_histogram"].items():
            merged = histogram.setdefault(
                key,
                {"pair": list(bucket["pair"]), "updates": 0, "tokens": 0, "vectors": 0},
            )
            if merged["pair"] != bucket["pair"]:
                raise RuntimeError("AdaptiveHQ PPL split identity changed across windows")
            for field in ("updates", "tokens", "vectors"):
                merged[field] += int(bucket[field])
    return {
        "schema_version": 1,
        "policy_version": next(iter(policy_versions)),
        "encoded_updates": sum(int(runtime["encoded_updates"]) for runtime in runtimes),
        "encoded_tokens": sum(int(runtime["encoded_tokens"]) for runtime in runtimes),
        "selector_bits_per_encode": next(iter(selector_widths)),
        "selector_bits_total": sum(int(runtime["selector_bits_total"]) for runtime in runtimes),
        "selector_incremental_stored_bits_total": 0,
        "selector_carrier_version": next(iter(carrier_versions)),
        "split_histogram": histogram,
    }


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
    force_incremental_control: bool = False,
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
        force_incremental_control: Run even a token-local cache through the
            same prefix+token-step scorer as a temporally coupled codec.  This
            is an explicitly labelled matched-protocol control, not a new
            formal method.

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
    if not isinstance(force_incremental_control, bool):
        raise ValueError("force_incremental_control must be a bool")
    device = model.device
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    del enc
    seq_len = input_ids.shape[1]
    window_plan = _ppl_window_plan(
        int(seq_len), max_length=max_length, stride=stride, max_windows=max_windows
    )

    # Apart from the tokenized source IDs needed to form the next slice, keep
    # only host fp64 NLL scalars across windows.  Retaining even scalar CUDA
    # outputs can retain allocator/storage metadata, and (more decisively) the
    # previous cache must be deleted *before* invoking the next cache factory:
    # assignment evaluates the right-hand side first and otherwise creates a
    # transient two-cache peak that can OOM a full WikiText run.
    window_nll_sums: list[float] = []
    finite_target_ranges: list[list[int]] = []
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
    adaptive_runtime_mode = None
    adaptive_window_provenance = []
    for planned_window in window_plan:
        begin = planned_window["begin"]
        end = planned_window["end"]
        first_target = planned_window["first_target"]
        planned_targets = planned_window["target_indices"]
        ids = input_ids[:, begin:end]

        past = cache_factory() if cache_factory is not None else None
        # Only a suffix of each overlapping window is newly scored.  A
        # temporally coupled codec may prefill the unscored prefix, but each
        # scored target must be predicted before that target enters the cache.
        # Token-local HQ/TQ configurations retain the fast one-forward path.
        logits, window_protocol, window_dependency = causal_teacher_forced_logits(
            model,
            ids,
            first_target,
            past,
            force_incremental_control=force_incremental_control,
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
        if valid != len(planned_targets):
            raise RuntimeError(
                "PPL scorer diverged from its exact target-index plan: "
                f"window=[{begin}:{end}], scored={valid}, planned={len(planned_targets)}"
            )
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
            window_nll_sums.append(float(loss.item()))
            finite_target_ranges.append([planned_targets.start, planned_targets.stop])
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
            adaptive_runtime = _adaptive_hq_runtime_snapshot(past)
            present = adaptive_runtime is not None
            if adaptive_runtime_mode is None:
                adaptive_runtime_mode = present
            elif adaptive_runtime_mode != present:
                raise RuntimeError("AdaptiveHQ PPL runtime provenance changed across windows")
            if adaptive_runtime is not None:
                adaptive_window_provenance.append(
                    {
                        "window_index": n_win - 1,
                        "input_begin": int(begin),
                        "input_end_exclusive": int(end),
                        "input_tokens": int(ids.numel()),
                        "first_target_index": int(first_target),
                        "scored_target_tokens": int(valid),
                        "cache_reset_for_window": True,
                        "teacher_forced_scoring_protocol": window_protocol,
                        "teacher_forced_temporal_dependency": window_dependency,
                        "runtime": copy.deepcopy(adaptive_runtime),
                    }
                )
        if end == seq_len:
            complete_text = True
            del ids, past, logits, tgt, token_losses, loss
            break
        # Do not rely on the next loop assignment to release ``past``: Python
        # calls cache_factory() before rebinding the name, briefly keeping two
        # complete KV caches alive.  The explicit release also makes the
        # multi-window lifetime contract testable with weak references.
        del ids, past, logits, tgt, token_losses, loss

    if not window_nll_sums:
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
    # With strict overlap, the first window scores positions [1, end) and every
    # later window scores exactly the newly introduced suffix.  Thus every
    # finite run must have scored each traversed next-token target exactly once.
    # Keep this executable invariant next to the denominator rather than
    # trusting window arithmetic that a future refactor could subtly change.
    if n_skipped == 0:
        expected_n_tokens = sum(len(window["target_indices"]) for window in window_plan)
        if n_tokens != expected_n_tokens:
            raise RuntimeError(
                "PPL target coverage is not contiguous and exact: "
                f"scored={n_tokens}, expected={expected_n_tokens}, traversed={prev_end}"
            )
    # Preserve the unrounded numerator as well as the mean.  Formal codec
    # comparisons use within-model NLL differences: rounding PPL before taking
    # a ratio can manufacture or erase the small deltas seen at mild rates.
    nll_sum = math.fsum(window_nll_sums)
    total_nll = nll_sum / max(n_tokens, 1)
    if compression_mode is None or compression_numerator == 0.0:
        compression_ratio = 1.0
    elif compression_denominator == 0.0:
        compression_ratio = float("inf")
    else:
        compression_ratio = compression_numerator / compression_denominator
    exact_target_coverage = n_skipped == 0
    return {
        "ppl": math.exp(total_nll),
        "nll": float(total_nll),
        "nll_sum": nll_sum,
        "nll_accumulation": PPL_NLL_ACCUMULATION,
        "n_tokens": n_tokens,
        "n_windows": n_win,
        "traversed_tokens": prev_end,
        "scored_target_index_set": f"[1,{prev_end})" if exact_target_coverage else None,
        "scored_target_ranges": finite_target_ranges,
        "scored_target_coverage": (
            "contiguous_exact_once"
            if exact_target_coverage
            else "incomplete_nonfinite_windows_excluded"
        ),
        "window_cache_lifetime": PPL_WINDOW_CACHE_LIFETIME,
        "cross_window_tensor_retention": PPL_CROSS_WINDOW_TENSOR_RETENTION,
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
        "force_incremental_control": force_incremental_control,
        "adaptive_hq_runtime": _aggregate_adaptive_hq_ppl_windows(adaptive_window_provenance),
        "adaptive_hq_window_provenance": adaptive_window_provenance,
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
    force_incremental_control: bool = False,
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
        force_incremental_control=force_incremental_control,
    )
    out["source"] = source
    out["dataset_revision"] = dataset_revision
    out["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return out

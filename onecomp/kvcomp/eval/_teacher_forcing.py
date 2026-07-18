"""Causally valid teacher-forced forwarding for compressed KV caches.

The attention mask alone is insufficient when a codec fits statistics across
the sequence dimension.  A full-window cache update can then use a future
token to alter an earlier key before that earlier query is evaluated.  This
module keeps the fast one-forward protocol for token-local codecs and uses a
prefix prefill followed by token-step updates when the cache declares such a
dependency.
"""

from __future__ import annotations

from typing import Optional

import torch

SINGLE_FORWARD_PROTOCOL = "single_forward_causal_mask_token_local_cache"
CAUSAL_INCREMENTAL_PROTOCOL = "causal_prefix_prefill_then_token_step_targets"
FORCED_INCREMENTAL_CONTROL_DEPENDENCY = "forced_matched_incremental_control"


def cache_temporal_dependency(cache) -> Optional[str]:
    """Return a cache-declared cross-token dependency, validating its type."""
    if cache is None:
        return None
    describe = getattr(cache, "teacher_forced_temporal_dependency", None)
    if describe is None:
        return None
    if not callable(describe):
        raise TypeError("teacher_forced_temporal_dependency must be callable")
    dependency = describe()
    if dependency is not None and not (isinstance(dependency, str) and dependency.strip()):
        raise TypeError(
            "teacher_forced_temporal_dependency() must return a non-empty string or None"
        )
    return dependency


def teacher_forced_protocol(cache) -> str:
    """Return the scoring protocol required by ``cache``."""
    return (
        CAUSAL_INCREMENTAL_PROTOCOL
        if cache_temporal_dependency(cache) is not None
        else SINGLE_FORWARD_PROTOCOL
    )


def _forward_step(model, input_ids, cache, cache_positions=None):
    """Forward one causal step and run optional cache lifecycle hooks."""
    kwargs = {
        "past_key_values": cache,
        "use_cache": cache is not None,
    }
    forward_kwargs = getattr(cache, "teacher_forced_forward_kwargs", None)
    if forward_kwargs is not None:
        if not callable(forward_kwargs):
            raise TypeError("teacher_forced_forward_kwargs must be callable")
        extras = forward_kwargs(cache_positions)
        if not isinstance(extras, dict):
            raise TypeError("teacher_forced_forward_kwargs() must return a dict")
        overlap = set(kwargs) & set(extras)
        if overlap:
            raise ValueError(
                "teacher_forced_forward_kwargs() cannot override " f"{sorted(overlap)}"
            )
        kwargs.update(extras)
    out = model(input_ids, **kwargs)
    logits = getattr(out, "logits", None)
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 3
        or logits.shape[0] != input_ids.shape[0]
        or logits.shape[1] != input_ids.shape[1]
        or logits.shape[2] == 0
        or not torch.is_floating_point(logits)
    ):
        raise RuntimeError(
            "teacher-forced model must return floating logits with shape "
            "(batch, input_sequence, non-empty_vocab)"
        )
    post_forward = getattr(cache, "teacher_forced_post_forward", None)
    if post_forward is not None:
        if not callable(post_forward):
            raise TypeError("teacher_forced_post_forward must be callable")
        post_forward()
    return out


@torch.no_grad()
def causal_teacher_forced_logits(
    model,
    input_ids,
    first_target_index: int,
    cache,
    *,
    force_incremental_control: bool = False,
):
    """Return logits for a contiguous target suffix without future-stat leakage.

    ``first_target_index`` is the index in ``input_ids`` of the first target to
    score.  The returned tensor has ``seq_len - first_target_index`` positions:
    position zero predicts ``input_ids[:, first_target_index]``.

    For a temporally coupled codec, the unscored prefix is safe to prefill as a
    block: its last query has no future positions within that prefix.  Every
    scored target is then predicted before that target enters the cache.  The
    final target is forwarded once after scoring so compression accounting
    covers exactly the same complete input window as the one-forward path.
    """
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("teacher-forced input_ids must have shape (batch, sequence)")
    if input_ids.shape[0] == 0:
        raise ValueError("teacher-forced input_ids batch must be non-empty")
    seq_len = int(input_ids.shape[1])
    if (
        isinstance(first_target_index, bool)
        or not isinstance(first_target_index, int)
        or not (1 <= first_target_index < seq_len)
    ):
        raise ValueError(
            "first_target_index must satisfy 1 <= first_target_index < sequence length"
        )

    if not isinstance(force_incremental_control, bool):
        raise ValueError("force_incremental_control must be a bool")
    dependency = cache_temporal_dependency(cache)
    if dependency is None and force_incremental_control:
        if cache is None:
            raise ValueError("forced incremental control requires a cache instance")
        dependency = FORCED_INCREMENTAL_CONTROL_DEPENDENCY
    if dependency is None:
        out = _forward_step(model, input_ids, cache)
        logits = out.logits[:, first_target_index - 1 : -1]
        expected = seq_len - first_target_index
        if logits.shape[1] != expected:
            raise RuntimeError(
                f"teacher-forced model returned {logits.shape[1]} logits; expected {expected}"
            )
        return logits, SINGLE_FORWARD_PROTOCOL, None

    if cache is None:  # Defensive: a dependency can only come from a cache.
        raise RuntimeError("incremental teacher forcing requires a cache instance")

    prefix = input_ids[:, :first_target_index]
    prefix_positions = torch.arange(first_target_index, device=input_ids.device)
    out = _forward_step(model, prefix, cache, prefix_positions)
    pieces = [out.logits[:, -1:]]
    for previous_index in range(first_target_index, seq_len - 1):
        position = torch.tensor([previous_index], device=input_ids.device)
        out = _forward_step(
            model,
            input_ids[:, previous_index : previous_index + 1],
            cache,
            position,
        )
        pieces.append(out.logits[:, -1:])

    # Include the last token in the cache solely for identical complete-window
    # bit accounting. Its output does not enter any likelihood numerator.
    final_position = torch.tensor([seq_len - 1], device=input_ids.device)
    _forward_step(model, input_ids[:, -1:], cache, final_position)
    logits = torch.cat(pieces, dim=1)
    expected = seq_len - first_target_index
    if logits.shape[1] != expected:
        raise RuntimeError(
            f"incremental scorer produced {logits.shape[1]} logits; expected {expected}"
        )
    return logits, CAUSAL_INCREMENTAL_PROTOCOL, dependency

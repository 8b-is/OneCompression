"""ThinKV: thought-adaptive hybrid KV compression for reasoning models.

ThinKV (Thought-Adaptive KV Cache Compression) compresses the KV cache of
long chain-of-thought generation by:

* classifying each *thought segment* into Reasoning / Execution / Transition
  from attention **sparsity** (TBQ/TBE driver).  Sparsity order is T > R > E,
* **TBQ** (Think-Before-you-Quantize): a full-precision buffer of ``g`` tokens
  is group-quantized per thought importance (R/E -> NVFP4 4-bit, T -> ternary
  2-bit; the formal scheme also allows R -> FP8 8-bit),
* **TBE** (Think-Before-you-Evict): on every transition thought the preceding
  segments are annealed down the retention schedule ``R={64,32,16,8,4}``
  (``keep = R[n]`` where ``n`` is the number of prior evictions of that
  segment), plus oldest/least-important eviction when the budget is exceeded,
  using K-means on post-RoPE keys.

Implementation maps directly to the paper's generation-loop code listing and
Problem Formulations (TBQ/TBE) / Algorithm 1 (KDE calibration).

This module provides:

* :class:`ThoughtClassifier` -- sparsity measurement + R/E/T classification
  with KDE-based threshold calibration (k-means fallback).
* :class:`ThinKVEngine` -- the per-layer decode-time state machine (TBQ buffer
  + group quantization + TBE), unit-testable on synthetic streams.
* :class:`ThinKVPolicy` -- a :class:`~onecomp.kvcomp.KVPolicy` exposing ThinKV
  (incl. L* optimal-layer selection) to the Runner / cache integration.

Reference: Ramachandran et al., "ThinKV: Thought-Adaptive KV Cache Compression
for Efficient Reasoning Models" (ICLR 2026 submission).

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from logging import getLogger
from typing import List, Optional

import torch

from .._base import KVPolicy
from .._registry import register_kv_policy
from ..codec.lowbit_fp import NVFP4Codec, TernaryCodec
from ..codec.fp8 import FP8Codec

# Thought type ids follow the paper's code listing: 0=Transition, 1=Execution, 2=Reasoning.
T_TRANSITION, T_EXECUTION, T_REASONING = 0, 1, 2
_ID_TO_NAME = {T_TRANSITION: "T", T_EXECUTION: "E", T_REASONING: "R"}
_IMPORTANCE = {T_TRANSITION: 0, T_EXECUTION: 1, T_REASONING: 2}  # rho


def attention_sparsity(attn_probs: torch.Tensor, rel_threshold: float = 0.01) -> torch.Tensor:
    """Fraction of attention weights below ``rel_threshold * row_max``.

    Args:
        attn_probs: ``(..., seq_len)`` softmax attention probabilities.
        rel_threshold: Relative cut-off (paper uses 1%).

    Returns:
        Sparsity ratio per row (``attn_probs.shape[:-1]``).
    """
    row_max = attn_probs.amax(dim=-1, keepdim=True)
    inactive = (attn_probs < rel_threshold * row_max).float().sum(dim=-1)
    return inactive / attn_probs.shape[-1]


def kmeans_1d(values: torch.Tensor, k: int, iters: int = 50) -> torch.Tensor:
    """Simple 1-D k-means; returns sorted centroids (length ``k``)."""
    v = values.flatten().float()
    if v.numel() < k:
        return torch.sort(v)[0]
    qs = torch.linspace(0, 1, k + 2)[1:-1]
    centroids = torch.quantile(v, qs)
    for _ in range(iters):
        d = (v[:, None] - centroids[None, :]).abs()
        assign = d.argmin(dim=1)
        new = centroids.clone()
        for j in range(k):
            sel = v[assign == j]
            if sel.numel() > 0:
                new[j] = sel.mean()
        if torch.allclose(new, centroids, atol=1e-6):
            break
        centroids = new
    return torch.sort(centroids)[0]


def kde_estimate(samples: torch.Tensor, grid: int = 512, bandwidth: float = None):
    """Gaussian KDE of ``samples`` on a uniform grid over ``[0, 1]``.

    Sparsity ratios live in ``[0, 1]``.  Bandwidth defaults to Silverman's rule.
    Returns ``(xs, density)``.
    """
    s = samples.flatten().float()
    xs = torch.linspace(0.0, 1.0, grid)
    if bandwidth is None:
        std = float(s.std()) if s.numel() > 1 else 0.1
        bandwidth = 1.06 * max(std, 1e-3) * (max(s.numel(), 1) ** (-1.0 / 5.0))
    bandwidth = max(bandwidth, 1e-3)
    d = (xs[:, None] - s[None, :]) / bandwidth
    dens = torch.exp(-0.5 * d * d).sum(dim=1) / (s.numel() * bandwidth * math.sqrt(2 * math.pi))
    return xs, dens


def kde_modes_and_minima(xs: torch.Tensor, dens: torch.Tensor):
    """Return interior local maxima (modes) and minima indices of a 1-D density."""
    diff = dens[1:] - dens[:-1]
    maxima, minima = [], []
    for i in range(1, len(dens) - 1):
        if diff[i - 1] > 0 and diff[i] <= 0:
            maxima.append(i)
        elif diff[i - 1] < 0 and diff[i] >= 0:
            minima.append(i)
    return maxima, minima


def kde_thresholds(samples: torch.Tensor, num_thoughts: int = 3):
    """Estimate ``num_thoughts - 1`` thresholds as KDE minima between modes.

    Returns ``(thresholds, num_modes)``.  ``thresholds`` is ``None`` when the
    density does not exhibit ``num_thoughts`` modes (caller may fall back).
    """
    xs, dens = kde_estimate(samples)
    maxima, minima = kde_modes_and_minima(xs, dens)
    num_modes = len(maxima)
    if num_modes < num_thoughts or len(maxima) < 2:
        return None, num_modes
    # Keep minima that lie strictly between the first and last retained mode.
    lo, hi = maxima[0], maxima[num_thoughts - 1] if num_thoughts <= len(maxima) else maxima[-1]
    between = [m for m in minima if lo < m < hi]
    if len(between) < num_thoughts - 1:
        return None, num_modes
    sel = between[: num_thoughts - 1]
    return tuple(float(xs[i]) for i in sel), num_modes


def kmeans_keep_indices(keys: torch.Tensor, keep: int, iters: int = 20) -> torch.Tensor:
    """Cluster ``keys`` (n, dim) into ``keep`` clusters; keep nearest-to-centroid.

    Returns ``keep`` token indices (sorted) representative of the segment.
    """
    n = keys.shape[0]
    if keep >= n:
        return torch.arange(n, device=keys.device)
    k = max(keep, 1)
    perm = torch.randperm(n, device=keys.device)[:k]
    centroids = keys[perm].clone().float()
    kf = keys.float()
    assign = torch.zeros(n, dtype=torch.long, device=keys.device)
    for _ in range(iters):
        d = torch.cdist(kf, centroids)
        assign = d.argmin(dim=1)
        for j in range(k):
            sel = kf[assign == j]
            if sel.numel() > 0:
                centroids[j] = sel.mean(dim=0)
    # For each cluster pick the token closest to its centroid.
    d = torch.cdist(kf, centroids)
    chosen = []
    for j in range(k):
        members = (assign == j).nonzero(as_tuple=True)[0]
        if members.numel() == 0:
            continue
        best = members[d[members, j].argmin()]
        chosen.append(int(best))
    return torch.tensor(sorted(set(chosen)), device=keys.device, dtype=torch.long)


@dataclass
class ThoughtClassifier:
    """Classify decode steps into R/E/T from attention sparsity.

    Sparsity order is T > R > E, so thresholds ``(theta1, theta2)`` split:
    ``u <= theta1 -> E``, ``theta1 < u <= theta2 -> R``, ``u > theta2 -> T``.
    """

    thresholds: tuple = (0.5, 0.8)
    rel_threshold: float = 0.01

    def measure(self, attn_probs: torch.Tensor) -> float:
        """Mean sparsity over the provided attention rows."""
        return float(attention_sparsity(attn_probs, self.rel_threshold).mean())

    def classify(self, sparsity: float) -> int:
        t1, t2 = self.thresholds
        if sparsity > t2:
            return T_TRANSITION
        if sparsity > t1:
            return T_REASONING
        return T_EXECUTION

    def calibrate(self, sparsity_samples: torch.Tensor, num_thoughts: int = 3):
        """Fit thresholds from the KDE local-minima between modes (Algorithm 1).

        Falls back to midpoints of ``num_thoughts`` k-means clusters when the
        density does not show ``num_thoughts`` clean modes.
        """
        thr, _ = kde_thresholds(sparsity_samples, num_thoughts)
        if thr is None:
            centroids = kmeans_1d(sparsity_samples, num_thoughts)
            thr = tuple(
                float((centroids[i] + centroids[i + 1]) / 2) for i in range(num_thoughts - 1)
            )
        self.thresholds = tuple(thr)
        return self.thresholds


@dataclass
class _Segment:
    """A contiguous run of same-thought tokens in the (compressed) cache."""

    thought: int
    indices: List[int]  # original token positions retained
    evict_count: int = 0


@dataclass
class ThinKVEngine:
    """Per-layer ThinKV decode-time state machine (TBQ buffer + TBE).

    Mirrors the paper's ``generation_loop`` for a single layer.  Feed
    ``add_prefill`` once (prefill tokens are treated as Reasoning, per the
    paper), then ``add_decode_step`` per generated token.  A full-precision
    buffer of ``group_size`` tokens is accumulated and group-quantized per the
    current thought; ``token_budget`` bounds the *committed* (quantized) cache.

    Tensor layout for keys/values: ``(num_kv_heads, seq, head_dim)`` (single
    batch element; the cache integration loops over the batch).
    """

    classifier: ThoughtClassifier
    refresh_interval: int = 128
    group_size: int = 16
    retention_schedule: tuple = (64, 32, 16, 8, 4)
    token_budget: int = 1024
    precision: dict = field(default_factory=lambda: {"R": 4, "E": 4, "T": 2})

    def __post_init__(self):
        self.logger = getLogger(__name__)
        self._codecs = {2: TernaryCodec(), 4: NVFP4Codec(), 8: FP8Codec()}
        self.segments: List[_Segment] = []
        self.keys: List[torch.Tensor] = []  # committed (quantized) per-token keys
        self.values: List[torch.Tensor] = []
        self.thoughts: List[int] = []
        self.origin: List[int] = []  # original token index of each committed token
        self.kbuf: List[torch.Tensor] = []  # full-precision token buffer (TBQ)
        self.vbuf: List[torch.Tensor] = []
        self.obuf: List[int] = []  # original token index of each buffered token
        self._orig_next: int = 0
        self.cur_thought: int = T_REASONING
        self.prev_thought: int = T_REASONING

    # --- properties --------------------------------------------------------
    @property
    def committed_tokens(self) -> int:
        return len(self.keys)

    @property
    def total_tokens(self) -> int:
        return len(self.keys) + len(self.kbuf)

    def codec_for(self, thought: int):
        """Map thought importance to its quantization codec (B={2,4,8})."""
        bits = self.precision.get(_ID_TO_NAME[thought], 4)
        if bits <= 2:
            return self._codecs[2]
        if bits <= 4:
            return self._codecs[4]
        return self._codecs[8]

    # --- TBQ ---------------------------------------------------------------
    def _quantize_block(self, keys_block, values_block, thought):
        """Group-quantize a ``(heads, n, dim)`` K/V block under ``thought``."""
        codec = self.codec_for(thought)
        kq, vq = codec.decode(codec.encode(keys_block[None], values_block[None]))
        return kq[0], vq[0]

    def _append_block(self, keys_block, values_block, thought, origins):
        """Append a quantized ``(heads, n, dim)`` block as ``thought`` tokens."""
        n = keys_block.shape[1]
        kq, vq = self._quantize_block(keys_block, values_block, thought)
        start = self.committed_tokens
        for t in range(n):
            self.keys.append(kq[:, t])
            self.values.append(vq[:, t])
            self.thoughts.append(thought)
            self.origin.append(int(origins[t]))
        if self.segments and self.segments[-1].thought == thought:
            self.segments[-1].indices.extend(range(start, start + n))
        else:
            self.segments.append(_Segment(thought, list(range(start, start + n))))

    def _commit_buffer(self, count=None):
        """Group-quantize and commit ``count`` (default all) buffered tokens."""
        if not self.kbuf:
            return
        count = len(self.kbuf) if count is None else min(count, len(self.kbuf))
        kblock = torch.stack(self.kbuf[:count], dim=1)  # (heads, count, dim)
        vblock = torch.stack(self.vbuf[:count], dim=1)
        self._append_block(kblock, vblock, self.cur_thought, self.obuf[:count])
        self.kbuf = self.kbuf[count:]
        self.vbuf = self.vbuf[count:]
        self.obuf = self.obuf[count:]

    # --- ingestion ---------------------------------------------------------
    def add_prefill(self, keys, values, thought: int = T_REASONING):
        """Add prompt tokens (treated as a single Reasoning thought block)."""
        self.cur_thought = thought
        seq = keys.shape[1]
        origins = list(range(self._orig_next, self._orig_next + seq))
        self._orig_next += seq
        self._append_block(keys, values, thought, origins)

    def add_decode_step(self, k_tok, v_tok, attn_probs, step: int, thought: int = None):
        """Add one generated token; refresh thought, TBQ buffer, then TBE.

        ``thought`` may be supplied directly (e.g. from the L*-averaged sparsity
        computed by the driver); otherwise it is classified from ``attn_probs``.
        """
        refresh = step % self.refresh_interval == 0 and (attn_probs is not None or thought is not None)
        if refresh:
            self.prev_thought = self.cur_thought
            if thought is not None:
                self.cur_thought = thought
            else:
                self.cur_thought = self.classifier.classify(self.classifier.measure(attn_probs))
            # Flush the partial buffer at a thought boundary so a group is never
            # mixed across thoughts.
            if self.cur_thought != self.prev_thought and self.kbuf:
                # Commit the buffered tokens under the *previous* thought.
                cur = self.cur_thought
                self.cur_thought = self.prev_thought
                self._commit_buffer()
                self.cur_thought = cur

        # TBQ: buffer full-precision tokens, group-quantize when full.
        self.kbuf.append(k_tok)
        self.vbuf.append(v_tok)
        self.obuf.append(self._orig_next)
        self._orig_next += 1
        if len(self.kbuf) >= self.group_size:
            self._commit_buffer(self.group_size)

        # TBE Case 1: anneal preceding segments at a transition refresh.
        if refresh and self.prev_thought == T_TRANSITION:
            self._anneal_preceding_segments()

        # TBE Case 2: budget-constrained eviction on the committed cache.
        if self.committed_tokens > self.token_budget:
            self._evict_for_budget()

    # --- TBE ---------------------------------------------------------------
    def _anneal_size(self, evict_count: int) -> int:
        sched = self.retention_schedule
        return sched[min(evict_count, len(sched) - 1)]

    def _anneal_preceding_segments(self):
        """Case 1: on a transition end, shrink preceding segments one step."""
        for seg in self.segments[:-1]:
            keep = self._anneal_size(seg.evict_count)
            if len(seg.indices) > keep:
                self._shrink_segment(seg, keep)
                seg.evict_count += 1

    def _evict_for_budget(self):
        """Case 2: shrink the oldest, least-important segment until under budget.

        The smart segment-annealing layer reduces older, low-importance
        segments first.  A hard trim then *guarantees* the budget invariant
        (the active segment can otherwise grow unbounded), mirroring how a
        real paged KV cache enforces a strict capacity.
        """
        guard = 0
        while self.committed_tokens > self.token_budget and len(self.segments) > 1 and guard < 512:
            guard += 1
            min_rho = min(_IMPORTANCE[s.thought] for s in self.segments[:-1])
            target = next(s for s in self.segments[:-1] if _IMPORTANCE[s.thought] == min_rho)
            keep = self._anneal_size(target.evict_count)
            if len(target.indices) <= keep:
                keep = max(len(target.indices) // 2, self.retention_schedule[-1])
            if len(target.indices) <= keep:
                self.segments.remove(target)
                continue
            self._shrink_segment(target, keep)
            target.evict_count += 1
        if self.committed_tokens > self.token_budget:
            self._hard_trim_to_budget()

    def _hard_trim_to_budget(self):
        """Strict fallback: keep sink + most-recent tokens up to the budget."""
        n = self.committed_tokens
        sink_keep = min(self.retention_schedule[-1], max(self.token_budget // 8, 1))
        recent_keep = self.token_budget - sink_keep
        keep = set(range(0, sink_keep)) | set(range(max(n - recent_keep, sink_keep), n))
        self._apply_keep(keep)

    def _apply_keep(self, keep_set):
        """Compact the cache to ``keep_set`` positions; rebuild segments."""
        new_keys, new_values, new_thoughts, new_origin = [], [], [], []
        for i in range(self.committed_tokens):
            if i in keep_set:
                new_keys.append(self.keys[i])
                new_values.append(self.values[i])
                new_thoughts.append(self.thoughts[i])
                new_origin.append(self.origin[i])
        self.keys, self.values, self.thoughts = new_keys, new_values, new_thoughts
        self.origin = new_origin
        self._rebuild_segments_from_thoughts()

    def _rebuild_segments_from_thoughts(self):
        """Recompute contiguous same-thought segments from ``self.thoughts``."""
        segments: List[_Segment] = []
        for pos, th in enumerate(self.thoughts):
            if segments and segments[-1].thought == th:
                segments[-1].indices.append(pos)
            else:
                segments.append(_Segment(th, [pos]))
        self.segments = segments

    def _shrink_segment(self, seg: _Segment, keep: int):
        """K-means keep ``keep`` representative tokens of ``seg`` (post-RoPE keys)."""
        if keep >= len(seg.indices):
            return
        seg_keys = torch.stack([self.keys[i].reshape(-1) for i in seg.indices], dim=0)
        local = kmeans_keep_indices(seg_keys, keep)
        kept_positions = sorted(seg.indices[i] for i in local.tolist())
        self._compact(set(kept_positions), seg, kept_positions)

    def _compact(self, keep_set, seg, kept_positions):
        """Drop evicted tokens of ``seg`` and rebuild contiguous index lists."""
        evicted = set(seg.indices) - keep_set
        if not evicted:
            return
        new_keys, new_values, new_thoughts, new_origin = [], [], [], []
        remap = {}
        for old_pos in range(self.committed_tokens):
            if old_pos in evicted:
                continue
            remap[old_pos] = len(new_keys)
            new_keys.append(self.keys[old_pos])
            new_values.append(self.values[old_pos])
            new_thoughts.append(self.thoughts[old_pos])
            new_origin.append(self.origin[old_pos])
        self.keys, self.values, self.thoughts = new_keys, new_values, new_thoughts
        self.origin = new_origin
        for s in self.segments:
            s.indices = [remap[p] for p in s.indices if p in remap]
        self.segments = [s for s in self.segments if s.indices]

    # --- read-out ----------------------------------------------------------
    def materialize(self):
        """Return the cache as ``(keys, values)`` of ``(num_kv_heads, n, dim)``.

        Concatenates the committed (quantized) tokens and the full-precision
        TBQ buffer (most recent tokens), matching what attention would read.
        """
        keys = list(self.keys) + list(self.kbuf)
        values = list(self.values) + list(self.vbuf)
        if not keys:
            return None, None
        return torch.stack(keys, dim=1), torch.stack(values, dim=1)

    def retained_origins(self) -> set:
        """Original token indices currently retained (committed + buffer)."""
        return set(self.origin) | set(self.obuf)

    def stats(self) -> dict:
        counts = {"R": 0, "E": 0, "T": 0}
        for t in self.thoughts:
            counts[_ID_TO_NAME[t]] += 1
        return {
            "committed_tokens": self.committed_tokens,
            "buffer_tokens": len(self.kbuf),
            "total_tokens": self.total_tokens,
            "num_segments": len(self.segments),
            "thought_counts": counts,
            "thresholds": self.classifier.thresholds,
        }


@register_kv_policy("thinkv")
@dataclass
class ThinKVPolicy(KVPolicy):
    """Thought-adaptive policy exposing ThinKV to the Runner / cache layer."""

    num_thoughts: int = 3
    refresh_interval: int = 128
    group_size: int = 16
    retention_schedule: tuple = (64, 32, 16, 8, 4)
    token_budget: int = 1024
    precision: dict = field(default_factory=lambda: {"R": 4, "E": 4, "T": 2})
    thresholds: tuple = (0.5, 0.8)
    calib_layers: Optional[list] = None
    max_calib_layers: int = 4  # |L*| in the paper
    sink_tokens: int = 4  # attention-sink tokens kept FP (StreamingLLM / KIVI)
    skip_layers: Optional[list] = None  # outlier layers left un-quantized

    def make_classifier(self) -> ThoughtClassifier:
        return ThoughtClassifier(thresholds=tuple(self.thresholds))

    def make_engine(self) -> ThinKVEngine:
        return ThinKVEngine(
            classifier=self.make_classifier(),
            refresh_interval=self.refresh_interval,
            group_size=self.group_size,
            retention_schedule=tuple(self.retention_schedule),
            token_budget=self.token_budget,
            precision=dict(self.precision),
        )

    def select_optimal_layers(self, layer_sparsity: dict):
        """Choose L*: layers whose sparsity KDE exhibits ``num_thoughts`` modes.

        ``layer_sparsity`` maps ``layer_idx -> 1-D tensor of sparsity samples``.
        Per Algorithm 1, layers with exactly ``|T|`` modes are eligible; we keep
        up to ``max_calib_layers`` of them and average their KDE-minima
        thresholds.
        """
        eligible = {}
        for layer, samples in layer_sparsity.items():
            thr, num_modes = kde_thresholds(torch.as_tensor(samples), self.num_thoughts)
            if thr is not None and num_modes == self.num_thoughts:
                eligible[layer] = thr
        if not eligible:
            # Fallback: pool all layers' sparsity and use k-means thresholds.
            pooled = torch.cat([torch.as_tensor(v).flatten() for v in layer_sparsity.values()])
            self.thresholds = self.make_classifier().calibrate(pooled, self.num_thoughts)
            self.calib_layers = sorted(layer_sparsity)[: self.max_calib_layers]
            return self.calib_layers, self.thresholds
        chosen = sorted(eligible)[: self.max_calib_layers]
        avg = [sum(eligible[l][j] for l in chosen) / len(chosen) for j in range(self.num_thoughts - 1)]
        self.calib_layers = chosen
        self.thresholds = tuple(avg)
        return self.calib_layers, self.thresholds

    def calibrate(self, sparsity_samples=None, layer_sparsity=None, calib_layers=None):
        """Fit thresholds (and L*) from calibration sparsity.

        Pass ``layer_sparsity`` (dict) for full Algorithm-1 behaviour (L*
        selection), or ``sparsity_samples`` (1-D) for pooled threshold fitting.
        """
        if layer_sparsity is not None:
            return self.select_optimal_layers(layer_sparsity)[1]
        self.thresholds = tuple(self.make_classifier().calibrate(sparsity_samples, self.num_thoughts))
        if calib_layers is not None:
            self.calib_layers = list(calib_layers)
        return self.thresholds

    def get_kv_config(self) -> dict:
        return {
            "name": "thinkv",
            "num_thoughts": self.num_thoughts,
            "refresh_interval": self.refresh_interval,
            "group_size": self.group_size,
            "retention_schedule": list(self.retention_schedule),
            "token_budget": self.token_budget,
            "precision": dict(self.precision),
            "thresholds": list(self.thresholds),
            "calib_layers": self.calib_layers,
            "max_calib_layers": self.max_calib_layers,
            "sink_tokens": self.sink_tokens,
            "skip_layers": self.skip_layers,
        }


# --- one-shot faithful configuration -------------------------------------------

_CALIB_PROMPTS = (
    "Natalia sold clips to 48 friends in April, then sold half as many in May. "
    "How many clips did she sell altogether?",
    "A rectangle has length 3 more than twice its width. Its perimeter is 36. Find its area.",
    "Compute the sum of all positive integers below 50 divisible by 3 or 5.",
)


@torch.no_grad()
def detect_outlier_layers(model, input_ids, ratio_thresh: float = 50.0,
                          agg: str = "mean") -> list:
    """Layers whose per-channel key range >> median (massive activations).

    These are left un-quantized by every KV-quant method (StreamingLLM/KIVI
    robustness); quantizing them with low-bit NVFP4 is what breaks reasoning
    models, so detecting + skipping them is part of a *faithful* ThinKV setup.

    ``input_ids`` may be a single tensor (one probe) **or a list/tuple of probe
    tensors**.  Aggregating the per-layer range-ratio across several diverse
    probes makes the skip set reproducible instead of hostage to one short prompt
    (benchmark review B-6):

    * ``agg="mean"`` (default): skip a layer if its *mean* ratio across probes
      exceeds ``ratio_thresh``.
    * ``agg="vote"``: skip a layer if it is an outlier in a *majority* of probes.

    A single-tensor argument reproduces the original single-probe behaviour.
    """
    probes = list(input_ids) if isinstance(input_ids, (list, tuple)) else [input_ids]
    if not probes:
        return []
    per_probe = []
    for ids in probes:
        out = model(ids.to(model.device), use_cache=True)
        layers = out.past_key_values.layers
        ratios = []
        for layer in layers:
            k = layer.keys.float()[0]
            ratios.append(
                float((k.abs().amax(dim=-1)
                       / k.abs().median(dim=-1).values.clamp_min(1e-6)).mean()))
        per_probe.append(ratios)

    n_layers = len(per_probe[0])
    skip = []
    for li in range(n_layers):
        vals = [p[li] for p in per_probe]
        if agg == "vote":
            is_outlier = sum(v > ratio_thresh for v in vals) >= (len(vals) + 1) // 2
        else:  # mean
            is_outlier = (sum(vals) / len(vals)) > ratio_thresh
        if is_outlier:
            skip.append(li)
    return skip


@torch.no_grad()
def auto_configure_thinkv(model, tokenizer, *, precision=None, sink_tokens: int = 4,
                          token_budget: int = 1 << 30, max_new_tokens: int = 200,
                          calib_prompts=None, skip_layers=None) -> "ThinKVPolicy":
    """Build a fully-calibrated :class:`ThinKVPolicy` (the faithful ``08`` setup).

    Runs outlier-layer detection, then collects attention-sparsity samples over a
    few CoT generations to fit L* and the R/E/T thresholds (Algorithm 1).  This
    is the consolidated entry point so eval harnesses engage real thought
    classification instead of degenerate all-Reasoning quantization.

    ``skip_layers`` may be passed to reuse a model-level outlier detection shared
    with the other quant methods (every KV-quant method must skip the same
    massive-activation layers for a fair comparison).
    """
    precision = precision or {"R": 4, "E": 4, "T": 2}
    calib_prompts = list(calib_prompts or _CALIB_PROMPTS)
    cfg = model.config.get_text_config()
    n_layers = cfg.num_hidden_layers

    def _ids(p):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True,
            return_tensors="pt", return_dict=True)["input_ids"].to(model.device)

    skip = list(skip_layers) if skip_layers is not None else detect_outlier_layers(model, _ids(calib_prompts[0]))

    pol = ThinKVPolicy(token_budget=token_budget, sink_tokens=sink_tokens,
                       skip_layers=skip, precision=dict(precision))
    ls = {i: [] for i in range(n_layers)}
    for p in calib_prompts:
        out = model.generate(_ids(p), max_new_tokens=max_new_tokens, do_sample=False,
                             output_attentions=True, return_dict_in_generate=True,
                             pad_token_id=tokenizer.eos_token_id)
        for step_attn in out.attentions[1:]:
            for li, a in enumerate(step_attn):
                ls[li].append(float(attention_sparsity(a[:, :, -1, :].float()).mean()))
    ls = {k: torch.tensor(v) for k, v in ls.items() if v}
    calib_layers, thr = pol.select_optimal_layers(ls)
    pol.calib_layers, pol.thresholds = calib_layers, thr
    return pol

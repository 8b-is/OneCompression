"""Long-context retrieval evaluation (Needle / synthetic multi-key retrieval).

This is the quality axis that distinguishes *eviction* from *graded
quantization*: a needle hidden mid-context is unrecoverable once evicted, but
survives (lossily) under a non-zero precision floor.  It is the missing piece
of :mod:`onecomp.kvcomp.eval` (which already covers PPL, zero-shot and
reasoning) and the primary metric for the TaperKV vs ThinKV / PM-KVQ comparison.

Like the rest of :mod:`onecomp.kvcomp.eval`, every harness takes a
``cache_factory`` callable ``() -> transformers.Cache`` so the same code scores
FullKV and any compressed cache.  Eviction caches cannot be trimmed mid-forward
by HF ``generate`` (mask/position bookkeeping breaks; see tasks/lessons.md K17),
so generation uses an explicit greedy loop that trims **after prefill** and
drives ``cache_position`` / ``position_ids`` directly.  This loop is also
correct for length-preserving (graded / static-quant) caches, so one path
covers every method.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import random
import re
from dataclasses import dataclass, field
from logging import getLogger
from typing import Callable, List, Optional

import torch

from .compression_accounting import (
    PER_EXAMPLE_CACHE_LIFETIME,
    PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION,
    aggregate_compression_sufficient_statistics,
    aggregate_mem_fraction,
    cache_compression_sufficient_statistics,
)

logger = getLogger(__name__)

_Z95 = 1.959963984540054  # standard normal 97.5th percentile
NEEDLE_PROTOCOL_VERSION = 2
NEEDLE_PROTOCOL_LABEL = "deterministic synthetic needle-in-a-haystack v2"
NEEDLE_PROTOCOL_SCOPE = (
    "common deterministic prompts across methods; rendered prompt lengths, "
    "depths, predictions, and per-trial cache-accounting ledgers retained; "
    "Pareto comparison over realized memory rather than an external benchmark"
)


def _generation_token_id_list(value, *, label: str) -> List[int]:
    """Normalize one HF generation token-ID field without coercion."""
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
    if isinstance(value, (set, frozenset)):
        values = sorted(values)
    normalized: List[int] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{label} must contain non-negative integer token IDs")
        if item not in normalized:
            normalized.append(item)
    return normalized


def effective_generation_eos_token_ids(model, tokenizer) -> tuple[List[int], str]:
    """Resolve EOS the way HF generation does and retain its provenance."""
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        configured = getattr(generation_config, "eos_token_id", None)
        if configured is not None:
            normalized = _generation_token_id_list(
                configured, label="model.generation_config.eos_token_id"
            )
            if not normalized:
                raise ValueError("model.generation_config.eos_token_id must not be empty")
            return normalized, "model.generation_config.eos_token_id"
    return (
        _generation_token_id_list(
            getattr(tokenizer, "eos_token_id", None),
            label="tokenizer.eos_token_id",
        ),
        "tokenizer.eos_token_id",
    )


def proportion_stats(hits: int, n: int):
    """(stderr, ci95_low, ci95_high) for a binomial accuracy ``hits/n``.

    Reporting an error bar is required to claim a method "beats" another: with a
    handful of needles a 1/24 difference is well within noise.  Uses the normal
    standard error for the point estimate and the **Wilson** 95% interval (which
    stays inside [0, 1] and is well-behaved near 0 % / 100 % accuracy, unlike the
    Wald interval).
    """
    if isinstance(hits, bool) or not isinstance(hits, int):
        raise ValueError("hits must be an integer")
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    if hits < 0 or hits > n:
        raise ValueError("hits must satisfy 0 <= hits <= n")
    if n == 0:
        return 0.0, 0.0, 0.0
    p = hits / n
    se = math.sqrt(max(p * (1.0 - p), 0.0) / n)
    z = _Z95
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(max(p * (1.0 - p) / n + z * z / (4.0 * n * n), 0.0)) / denom
    return se, max(0.0, center - half), min(1.0, center + half)


def _number_hit(gold: str, text: str) -> bool:
    """True iff every standalone integer prediction equals ``gold``.

    A raw substring metric gives credit to ``"1234, or perhaps 5678"`` for
    gold ``1234`` even though that response does not answer the retrieval
    question.  The prompt asks for one number, so accept a terse answer or a
    repeated explanation of the same number, but reject a response containing
    any competing integer.  Digit boundaries prevent partial-number matches.
    """
    numbers = re.findall(r"(?<!\d)(\d+)(?!\d)", text)
    return bool(numbers) and all(number == gold for number in numbers)


_CITIES = [
    "Paris",
    "Tokyo",
    "Cairo",
    "Lima",
    "Oslo",
    "Delhi",
    "Quito",
    "Accra",
    "Hanoi",
    "Sofia",
    "Riga",
    "Bern",
    "Doha",
    "Kyiv",
    "Amman",
    "Dakar",
]

# Diverse, high-entropy filler (memo C).  The old filler was the *same* four
# sentences repeated to fill the context, which is so low-entropy that the model
# can recover the needle from almost any surviving token -- unrealistically
# forgiving to quantization noise and eviction.  We instead sample, without
# immediate repetition, from a broad bank of unrelated essay-style sentences
# (history, science, economics, nature, technology, art) so the haystack has the
# token diversity of real long-context text and the retrieval task actually
# stresses the codec.  Seeded, so every (length, depth, seed) cell is
# deterministic and reproducible.
_FILLER_BANK = [
    "The history of cartography spans many centuries of patient observation.",
    "Merchants traded spices along routes that crossed deserts and mountains.",
    "Scholars debated the shape of coastlines while sailors charted the stars.",
    "Libraries collected manuscripts describing distant provinces and rivers.",
    "A startup succeeds when it makes something a small group of people want badly.",
    "Compound interest rewards the investor who is patient through quiet decades.",
    "Photosynthesis converts sunlight into the chemical energy that feeds the forest.",
    "The printing press lowered the cost of copying ideas almost to nothing.",
    "Glaciers carve valleys slowly, grinding granite into pale silt over millennia.",
    "A good essay begins with a question the writer cannot yet answer.",
    "Tides rise and fall as the moon pulls the patient oceans across the globe.",
    "Markets aggregate the scattered knowledge of thousands of strangers into a price.",
    "The cathedral took three generations of masons to lift its vaults skyward.",
    "Bees navigate by polarized light and dance to share the bearing of flowers.",
    "Encryption turns a private message into noise that only a key can undo.",
    "Volcanic ash enriches the soil, and farmers return to the dangerous slopes.",
    "A violin's tone depends on the slow seasoning of spruce and maple wood.",
    "Railways shrank the continent, binding distant towns to a single timetable.",
    "Antibiotics gave physicians a brief, astonishing power over old infections.",
    "Rivers braid and wander, abandoning oxbow lakes as they search for the sea.",
    "The chess endgame rewards calm calculation more than brilliant sacrifice.",
    "Coral reefs build limestone cities that shelter a quarter of marine life.",
    "Inflation quietly transfers wealth from savers to borrowers year by year.",
    "Migratory birds cross oceans guided by stars, smell, and the earth's field.",
    "A compiler translates human intention into instructions a machine obeys.",
    "Monsoon winds reverse with the seasons, watering half the world's harvests.",
    "The telescope turned wandering lights into worlds with moons of their own.",
    "Sourdough rises because wild yeast and bacteria quietly digest the flour.",
    "Supply chains thread raw ore through a dozen ports before it becomes a phone.",
    "Desert foxes hunt at night, their great ears shedding the day's stored heat.",
]


def _content_token_count(tokenizer, text: str) -> int:
    """Count message-content tokens without injecting BOS/EOS tokens.

    The old filler builder called ``tokenizer(sentence)`` for every sentence.
    Decoder tokenizers commonly add a BOS there, so it systematically believed
    the haystack was longer than the text actually passed through the chat
    template.  This helper makes the requested *filler* budget meaningful;
    the final rendered prompt length is recorded separately below.
    """

    try:
        encoded = tokenizer(text, add_special_tokens=False)
    except TypeError as exc:
        raise RuntimeError(
            "synthetic retrieval tokenizer must support add_special_tokens=False "
            "for reproducible content-token accounting"
        ) from exc
    input_ids = getattr(encoded, "input_ids", None)
    if input_ids is None and isinstance(encoded, dict):
        input_ids = encoded.get("input_ids")
    if isinstance(input_ids, torch.Tensor):
        if input_ids.ndim == 2:
            if input_ids.shape[0] != 1:
                raise RuntimeError(
                    "synthetic retrieval tokenizer returned a batched content encoding"
                )
            return int(input_ids.shape[1])
        if input_ids.ndim == 1:
            return int(input_ids.shape[0])
    if isinstance(input_ids, (list, tuple)):
        if input_ids and isinstance(input_ids[0], (list, tuple)):
            if len(input_ids) != 1:
                raise RuntimeError(
                    "synthetic retrieval tokenizer returned a batched content encoding"
                )
            input_ids = input_ids[0]
        if all(isinstance(token, int) for token in input_ids):
            return len(input_ids)
    raise RuntimeError("synthetic retrieval tokenizer returned invalid content input_ids")


def _build_filler(tokenizer, target_tokens: int, rng) -> str:
    """Sample diverse sentences (no immediate repeat) until ~``target_tokens``."""
    if target_tokens <= 0:
        return ""
    out, ntok, prev = [], 0, -1
    while ntok < target_tokens:
        i = rng.randrange(len(_FILLER_BANK))
        if i == prev and len(_FILLER_BANK) > 1:
            i = (i + 1) % len(_FILLER_BANK)
        prev = i
        s = _FILLER_BANK[i] + " "
        out.append(s)
        ntok += _content_token_count(tokenizer, s)
    return "".join(out)


@dataclass
class NeedleResult:
    """Needle-in-a-haystack retrieval result over a depth x length grid."""

    n: int
    hits: int
    accuracy: float
    mem_fraction: float
    by_length: dict = field(default_factory=dict)  # ctx_len -> accuracy
    by_depth: dict = field(default_factory=dict)  # requested depth -> accuracy
    grid: List[dict] = field(default_factory=list)  # per (len, depth, seed)
    accuracy_stderr: float = 0.0  # binomial SE of accuracy
    accuracy_ci95: tuple = (0.0, 0.0)  # Wilson 95% interval
    mem_fraction_aggregation: str = "fullkv_no_cache"
    compression_sufficient_statistics: dict = field(default_factory=dict)
    benchmark: str = NEEDLE_PROTOCOL_LABEL
    protocol_scope: str = NEEDLE_PROTOCOL_SCOPE
    # Kept in memory so callers that split the context grid can pool the raw
    # per-generation ledgers. Each persisted grid row also carries its ledger,
    # while the aggregate totals above make the reported memory fraction easy
    # to independently recompute.
    compression_records: List[dict] = field(default_factory=list, repr=False)

    def as_dict(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "protocol_version": NEEDLE_PROTOCOL_VERSION,
            "official_reference": False,
            "protocol_scope": self.protocol_scope,
            "metric": "all standalone generated integers equal the hidden gold integer",
            "n": self.n,
            "hits": self.hits,
            "accuracy": round(self.accuracy, 4),
            "accuracy_stderr": round(self.accuracy_stderr, 4),
            "accuracy_ci95": [round(self.accuracy_ci95[0], 4), round(self.accuracy_ci95[1], 4)],
            "mem_fraction": round(self.mem_fraction, 4),
            "mem_fraction_aggregation": self.mem_fraction_aggregation,
            "example_cache_lifetime": PER_EXAMPLE_CACHE_LIFETIME,
            "cross_iteration_tensor_retention": (PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION),
            "compression_sufficient_statistics": self.compression_sufficient_statistics,
            "by_length": {k: round(v, 4) for k, v in self.by_length.items()},
            "by_depth": {str(k): round(v, 4) for k, v in self.by_depth.items()},
            "grid": self.grid,
        }


def aggregate_cache_compression_records(records: List[dict]) -> tuple[dict, float, str]:
    """Pool per-generation cache ledgers without averaging exact ratios.

    Exact codecs retain integer ``sum(stored_bits) / sum(fp16_bits)``.  Caches
    that expose only a legacy ratio keep the explicitly named arithmetic mean
    of per-generation memory fractions.  The shared accounting helper rejects
    mixed exact/legacy/FullKV records so a context-dependent cache factory
    cannot silently change the metric halfway through a grid.
    """

    statistics = aggregate_compression_sufficient_statistics(records)
    legacy_fractions = []
    for record in records:
        if record.get("compression_aggregation") != "legacy_cache_ratio_only":
            continue
        ratio = float(record["compression_ratio"])
        legacy_fractions.append(0.0 if math.isinf(ratio) else 1.0 / ratio)
    fraction, aggregation = aggregate_mem_fraction(statistics, legacy_fractions)
    return statistics, fraction, aggregation


def make_needle_prompt(
    tokenizer,
    context_tokens: int,
    depth: float = 0.5,
    seed: int = 0,
    *,
    return_metadata: bool = False,
):
    """Build a needle-in-haystack chat prompt and its gold answer.

    A secret ``magic number`` is hidden at fractional ``depth`` inside a
    requested number of *content filler* tokens.  The chat-template prompt is
    intentionally not called a ``context_tokens``-exact grid: template,
    question, and needle tokens differ by model.  Their rendered lengths are
    retained by :func:`needle_in_haystack` for every trial.

    The public default remains ``(messages, gold_str)`` for compatibility.
    ``return_metadata=True`` adds deterministic prompt-construction facts used
    by the audit ledger.
    """
    if (
        isinstance(context_tokens, bool)
        or not isinstance(context_tokens, int)
        or context_tokens <= 0
    ):
        raise ValueError("context_tokens must be a positive integer")
    if (
        isinstance(depth, bool)
        or not isinstance(depth, (int, float))
        or not math.isfinite(float(depth))
        or not 0.0 <= float(depth) <= 1.0
    ):
        raise ValueError("depth must be finite and in [0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    rng = random.Random(seed)
    magic = rng.randint(1000, 9999)
    city = rng.choice(_CITIES)
    needle = f" The magic number for {city} is {magic}. "
    pre = _build_filler(tokenizer, int(context_tokens * depth), rng)
    post = _build_filler(tokenizer, int(context_tokens * (1.0 - depth)), rng)
    haystack = pre + needle + post
    question = (
        f"\n\nQuestion: What is the magic number for {city}? "
        f"Answer with only the number.\nAnswer:"
    )
    messages = [{"role": "user", "content": haystack + question}]
    if not return_metadata:
        return messages, str(magic)
    pre_tokens = _content_token_count(tokenizer, pre)
    post_tokens = _content_token_count(tokenizer, post)
    filler_tokens = pre_tokens + post_tokens
    return (
        messages,
        str(magic),
        {
            "requested_filler_tokens": context_tokens,
            "requested_depth": float(depth),
            "pre_filler_tokens": pre_tokens,
            "post_filler_tokens": post_tokens,
            "actual_filler_tokens": filler_tokens,
            "realized_filler_depth": (pre_tokens / filler_tokens if filler_tokens else 0.0),
            "prompt_seed": seed,
            "city": city,
        },
    )


def _encode(tokenizer, messages) -> torch.Tensor:
    # Retrieval answers are a single short number, so disable chain-of-thought for
    # thinking models (Qwen3/QwQ): with reasoning on, a short max_new budget is
    # consumed entirely inside the <think> block and the model never emits the
    # answer (FullKV would score ~0). enable_thinking is ignored by non-thinking
    # templates (Llama/Mistral), so this is safe across model families.
    try:
        enc = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        )
    except TypeError as exc:
        # Older tokenizer implementations may expose a fixed Python signature
        # without template kwargs.  Retry only for that precise compatibility
        # case; a malformed template or prompt must never silently change the
        # benchmark into raw causal text.
        message = str(exc)
        if "enable_thinking" not in message or not any(
            marker in message
            for marker in ("unexpected keyword", "unexpected argument", "got an unexpected")
        ):
            raise RuntimeError("synthetic retrieval chat-template rendering failed") from exc
        try:
            enc = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            )
        except Exception as retry_exc:
            raise RuntimeError("synthetic retrieval chat-template rendering failed") from retry_exc
    except Exception as exc:
        raise RuntimeError("synthetic retrieval chat-template rendering failed") from exc

    try:
        input_ids = enc["input_ids"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("synthetic retrieval chat template did not return input_ids") from exc
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise RuntimeError("synthetic retrieval chat template returned invalid input_ids")
    return input_ids


def _model_context_capacity(model) -> int:
    """Return an explicitly advertised model context capacity.

    Synthetic context labels are approximate because the needle, question and
    chat wrapper add tokens.  Capacity therefore has to be checked against the
    *rendered* prompt instead of trusting the requested filler length.
    """
    config = getattr(model, "config", None)
    candidates = [config, getattr(config, "text_config", None)]
    values = []
    for candidate in candidates:
        if candidate is None:
            continue
        value = getattr(candidate, "max_position_embeddings", None)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            continue
        values.append(value)
    if not values:
        raise ValueError(
            "synthetic retrieval requires config.max_position_embeddings for "
            "a fail-closed prompt+generation capacity check"
        )
    if len(set(values)) != 1:
        raise ValueError("ambiguous model context capacity between config and text_config")
    return values[0]


def _validate_prompt_capacity(model, input_ids: torch.Tensor, max_new_tokens: int) -> int:
    capacity = _model_context_capacity(model)
    prompt_tokens = int(input_ids.shape[1])
    required = prompt_tokens + int(max_new_tokens)
    if required > capacity:
        raise ValueError(
            "synthetic retrieval prompt+generation exceeds model context: "
            f"{prompt_tokens} + {max_new_tokens} > {capacity} tokens"
        )
    return capacity


def _validate_grid(
    context_lens: List[int],
    *,
    seeds: int,
    max_new_tokens: int,
    depths: Optional[List[float]] = None,
) -> None:
    if not context_lens:
        raise ValueError("context_lens must be non-empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in context_lens
    ):
        raise ValueError("context_lens must contain positive integers")
    if len(set(context_lens)) != len(context_lens):
        raise ValueError("context_lens must not contain duplicates")
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds <= 0:
        raise ValueError("seeds must be a positive integer")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    if depths is not None:
        if not depths:
            raise ValueError("depths must be non-empty")
        if any(
            isinstance(depth, bool)
            or not isinstance(depth, (int, float))
            or not math.isfinite(float(depth))
            or not 0.0 <= float(depth) <= 1.0
            for depth in depths
        ):
            raise ValueError("depths must contain finite values in [0, 1]")
        if len(set(float(depth) for depth in depths)) != len(depths):
            raise ValueError("depths must not contain duplicates")


@torch.no_grad()
def greedy_answer(
    model, input_ids, cache, max_new_tokens=32, post_prefill=None, eos=None, attn_capture=False
):
    """Greedy decode with explicit ``cache_position`` (robust to eviction).

    ``post_prefill(cache)`` is called after prefill and after every decode cache
    update.  This maintains a bounded StreamingLLM window throughout generation;
    calling it only once lets generated tokens grow without bound.

    ``attn_capture=True`` materializes ``attn_weights`` (eager + output_attentions)
    and feeds attention sparsity into a ThinKV cache so its thought
    classification engages during the explicit decode loop.
    """
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or input_ids.shape[1] == 0
    ):
        raise ValueError("greedy_answer input_ids must have shape (1, non-empty sequence)")
    if post_prefill is not None and not callable(post_prefill):
        raise TypeError("post_prefill must be callable or None")
    if not isinstance(attn_capture, bool):
        raise ValueError("attn_capture must be a bool")
    eos_ids = set(_generation_token_id_list(eos, label="eos"))

    def finite_argmax(logits, *, stage: str):
        if not isinstance(logits, torch.Tensor) or logits.ndim < 2 or logits.shape[-1] == 0:
            raise TypeError(f"{stage} returned invalid next-token logits")
        next_token_logits = logits[:, -1]
        if not torch.is_floating_point(next_token_logits):
            raise TypeError(f"{stage} next-token logits must have a floating-point dtype")
        if not bool(torch.isfinite(next_token_logits).all().item()):
            raise FloatingPointError(
                f"{stage} produced non-finite next-token logits; "
                "refusing to turn NaN/inf into a greedy token"
            )
        return next_token_logits.argmax(-1, keepdim=True)

    def reached_eos(token) -> bool:
        token_id = int(token.item())
        return token_id in eos_ids

    dev = model.device
    n = input_ids.shape[1]
    pos = torch.arange(n, device=dev)
    if attn_capture:
        from ..cache import thinkv_attention_capture  # local import

        ctx = thinkv_attention_capture(model, cache)
        oa = True
    else:
        ctx = contextlib.nullcontext()
        oa = False
    with ctx:
        out = model(
            input_ids.to(dev),
            past_key_values=cache,
            use_cache=True,
            cache_position=pos,
            output_attentions=oa,
        )
        if post_prefill is not None:
            post_prefill(cache)
        nxt = finite_argmax(out.logits, stage="Greedy prefill")
        toks, cur = [nxt], n
        if reached_eos(nxt):
            return torch.cat(toks, dim=1)
        for _ in range(max_new_tokens - 1):
            cp = torch.tensor([cur], device=dev)
            out = model(
                nxt,
                past_key_values=cache,
                use_cache=True,
                cache_position=cp,
                position_ids=cp.unsqueeze(0),
                output_attentions=oa,
            )
            if post_prefill is not None:
                post_prefill(cache)
            nxt = finite_argmax(out.logits, stage="Greedy decode")
            toks.append(nxt)
            cur += 1
            if reached_eos(nxt):
                break
    return torch.cat(toks, dim=1)


@torch.no_grad()
def needle_in_haystack(
    model,
    tokenizer,
    cache_factory: Optional[Callable[[], object]] = None,
    context_lens: Optional[List[int]] = None,
    depths: Optional[List[float]] = None,
    seeds: int = 3,
    max_new_tokens: int = 32,
    evict: bool = False,
    attn_capture: bool = False,
) -> NeedleResult:
    """Needle retrieval accuracy over a deterministic depth x filler-budget grid.

    Args:
        cache_factory: ``() -> Cache`` (``None`` = FullKV ``DynamicCache``).
        context_lens: requested filler-token budgets (e.g. [2048, 4096]).
            The full rendered prompt length is model-specific and retained for
            every trial; it is never inferred from this label.
        depths: fractional needle positions (e.g. [0.1, .., 0.9]).
        seeds: independent needles per (length, depth) cell.
        max_new_tokens: decode budget (use ~256 for reasoning models that think
            before answering -- the recalled fact appears within the thinking).
        evict: if True, call ``cache.evict_now()`` after prefill (eviction caches).
        attn_capture: if True, engage ThinKV thought classification (eager attn).
    """
    from transformers import DynamicCache  # local import; keeps module import light

    context_lens = [2048, 4096] if context_lens is None else context_lens
    depths = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9] if depths is None else depths
    _validate_grid(context_lens, seeds=seeds, max_new_tokens=max_new_tokens, depths=depths)
    eos, _ = effective_generation_eos_token_ids(model, tokenizer)

    hit, n = 0, 0
    compression_records: List[dict] = []
    by_len_hit: dict = {c: [0, 0] for c in context_lens}
    by_depth_hit: dict = {float(depth): [0, 0] for depth in depths}
    grid: List[dict] = []
    for ctx in context_lens:
        for di, depth in enumerate(depths):
            for s in range(seeds):
                prompt_seed = di * 97 + s * 1009 + 1
                prompt = make_needle_prompt(
                    tokenizer,
                    ctx,
                    depth=depth,
                    seed=prompt_seed,
                    return_metadata=True,
                )
                if not isinstance(prompt, tuple) or len(prompt) not in (2, 3):
                    raise RuntimeError("make_needle_prompt returned an invalid prompt contract")
                messages, gold = prompt[:2]
                metadata = (
                    dict(prompt[2])
                    if len(prompt) == 3 and isinstance(prompt[2], dict)
                    else {
                        "requested_filler_tokens": ctx,
                        "requested_depth": float(depth),
                        "prompt_seed": prompt_seed,
                    }
                )
                ids = _encode(tokenizer, messages)
                capacity = _validate_prompt_capacity(model, ids, max_new_tokens)
                cache = (
                    cache_factory()
                    if cache_factory is not None
                    else DynamicCache(config=model.config)
                )
                pp = (lambda c: c.evict_now()) if evict else None
                gen = greedy_answer(
                    model,
                    ids,
                    cache,
                    max_new_tokens,
                    post_prefill=pp,
                    eos=eos,
                    attn_capture=attn_capture,
                )
                txt = tokenizer.decode(gen[0], skip_special_tokens=True)
                ok = _number_hit(gold, txt)
                hit += int(ok)
                n += 1
                by_len_hit[ctx][0] += int(ok)
                by_len_hit[ctx][1] += 1
                by_depth_hit[float(depth)][0] += int(ok)
                by_depth_hit[float(depth)][1] += 1
                cache_accounting = cache_compression_sufficient_statistics(cache)
                compression_records.append(cache_accounting)
                content = messages[0].get("content") if messages else None
                if not isinstance(content, str):
                    raise RuntimeError(
                        "needle prompt must be one user message with string content"
                    )
                grid.append(
                    {
                        "ctx": ctx,
                        "depth": float(depth),
                        "seed": s,
                        "requested_filler_tokens": ctx,
                        "requested_depth": float(depth),
                        "seed_index": s,
                        "prompt_seed": prompt_seed,
                        "rendered_prompt_tokens": int(ids.shape[1]),
                        "reserved_prompt_plus_generation_tokens": int(ids.shape[1])
                        + max_new_tokens,
                        "model_context_capacity": capacity,
                        "prompt_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                        "gold": gold,
                        "prediction": txt,
                        "hit": ok,
                        "cache_accounting": cache_accounting,
                        **metadata,
                    }
                )
                del cache, gen, ids
    se, lo, hi = proportion_stats(hit, n)
    compression, mem_fraction, mem_fraction_aggregation = aggregate_cache_compression_records(
        compression_records
    )
    return NeedleResult(
        n=n,
        hits=hit,
        accuracy=hit / max(1, n),
        mem_fraction=mem_fraction,
        by_length={c: by_len_hit[c][0] / max(1, by_len_hit[c][1]) for c in context_lens},
        by_depth={
            depth: by_depth_hit[depth][0] / max(1, by_depth_hit[depth][1])
            for depth in by_depth_hit
        },
        grid=grid,
        accuracy_stderr=se,
        accuracy_ci95=(lo, hi),
        mem_fraction_aggregation=mem_fraction_aggregation,
        compression_sufficient_statistics=compression,
        compression_records=compression_records,
    )


def make_multikey_prompt(
    tokenizer, context_tokens: int, num_keys: int = 4, depth: float = 0.5, seed: int = 0
):
    """Build a synthetic multi-key retrieval prompt and its gold answer.

    ``num_keys`` distinct ``magic number for <city>`` facts are scattered through
    ~``context_tokens`` of filler; the query asks for one of them.  Distractor
    keys (same surface form, different cities) make this strictly harder than the
    single-needle task -- the model must disambiguate, not just locate, which is
    where lossy precision floors and eviction diverge most.
    """
    rng = random.Random(seed)
    cities = rng.sample(_CITIES, k=min(num_keys, len(_CITIES)))
    facts = {c: rng.randint(1000, 9999) for c in cities}
    target = rng.choice(cities)

    # Place each needle at a fractional position in [0, 1): the *target* sits at
    # ``depth`` (so retrieval can be probed at a controlled context depth) and the
    # distractors are spread evenly across the remaining context, forcing the
    # model to disambiguate the queried key from look-alikes at known separations.
    pos = {target: float(min(max(depth, 0.0), 0.999))}
    others = [c for c in cities if c != target]
    for i, c in enumerate(others):
        pos[c] = (i + 0.5) / max(1, len(others))
    ordered = sorted(cities, key=lambda c: pos[c])

    body = ""
    prev = 0.0
    for c in ordered:
        f = pos[c]
        body += _build_filler(tokenizer, int(round((f - prev) * context_tokens)), rng)
        body += f" The magic number for {c} is {facts[c]}. "
        prev = f
    body += _build_filler(tokenizer, int(round((1.0 - prev) * context_tokens)), rng)

    question = (
        f"\n\nQuestion: What is the magic number for {target}? "
        f"Answer with only the number.\nAnswer:"
    )
    return [{"role": "user", "content": body + question}], str(facts[target])


@torch.no_grad()
def ruler_multikey(
    model,
    tokenizer,
    cache_factory: Optional[Callable[[], object]] = None,
    context_lens: Optional[List[int]] = None,
    num_keys: int = 4,
    seeds: int = 5,
    max_new_tokens: int = 32,
    evict: bool = False,
    attn_capture: bool = False,
) -> NeedleResult:
    """Synthetic multi-key retrieval accuracy over context lengths.

    This is not NVIDIA's multi-task RULER benchmark.  The historical function
    name is kept for API compatibility; result tables must label it synthetic.

    Like :func:`needle_in_haystack` but with ``num_keys`` competing facts so the
    model must disambiguate among distractors -- the regime where eviction loses
    the queried key and graded quantization keeps it (lossily) recoverable.
    """
    from transformers import DynamicCache  # local import

    context_lens = [2048, 4096] if context_lens is None else context_lens
    _validate_grid(context_lens, seeds=seeds, max_new_tokens=max_new_tokens)
    if (
        isinstance(num_keys, bool)
        or not isinstance(num_keys, int)
        or not 1 <= num_keys <= len(_CITIES)
    ):
        raise ValueError(f"num_keys must be an integer in [1, {len(_CITIES)}]")
    eos, _ = effective_generation_eos_token_ids(model, tokenizer)

    hit, n = 0, 0
    compression_records: List[dict] = []
    by_len_hit: dict = {c: [0, 0] for c in context_lens}
    grid: List[dict] = []
    for ctx in context_lens:
        for s in range(seeds):
            messages, gold = make_multikey_prompt(
                tokenizer, ctx, num_keys=num_keys, seed=s * 1009 + 7
            )
            ids = _encode(tokenizer, messages)
            capacity = _validate_prompt_capacity(model, ids, max_new_tokens)
            cache = (
                cache_factory() if cache_factory is not None else DynamicCache(config=model.config)
            )
            pp = (lambda c: c.evict_now()) if evict else None
            gen = greedy_answer(
                model,
                ids,
                cache,
                max_new_tokens,
                post_prefill=pp,
                eos=eos,
                attn_capture=attn_capture,
            )
            txt = tokenizer.decode(gen[0], skip_special_tokens=True)
            ok = _number_hit(gold, txt)
            hit += int(ok)
            n += 1
            by_len_hit[ctx][0] += int(ok)
            by_len_hit[ctx][1] += 1
            cache_accounting = cache_compression_sufficient_statistics(cache)
            compression_records.append(cache_accounting)
            content = messages[0].get("content") if messages else None
            if not isinstance(content, str):
                raise RuntimeError(
                    "synthetic multi-key prompt must be one user message with string content"
                )
            grid.append(
                {
                    "ctx": ctx,
                    "seed": s,
                    "requested_filler_tokens": ctx,
                    "requested_depth": 0.5,
                    "num_keys": num_keys,
                    "seed_index": s,
                    "prompt_seed": s * 1009 + 7,
                    "rendered_prompt_tokens": int(ids.shape[1]),
                    "reserved_prompt_plus_generation_tokens": int(ids.shape[1]) + max_new_tokens,
                    "model_context_capacity": capacity,
                    "prompt_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "gold": gold,
                    "prediction": txt,
                    "hit": ok,
                    "cache_accounting": cache_accounting,
                }
            )
            del cache, gen, ids
    se, lo, hi = proportion_stats(hit, n)
    compression, mem_fraction, mem_fraction_aggregation = aggregate_cache_compression_records(
        compression_records
    )
    return NeedleResult(
        n=n,
        hits=hit,
        accuracy=hit / max(1, n),
        mem_fraction=mem_fraction,
        by_length={c: by_len_hit[c][0] / max(1, by_len_hit[c][1]) for c in context_lens},
        grid=grid,
        accuracy_stderr=se,
        accuracy_ci95=(lo, hi),
        mem_fraction_aggregation=mem_fraction_aggregation,
        compression_sufficient_statistics=compression,
        compression_records=compression_records,
        benchmark="deterministic synthetic multi-key retrieval v2",
        protocol_scope=(
            "common deterministic prompts across methods; rendered prompt lengths, "
            "predictions, and per-trial cache-accounting ledgers retained; local diagnostic rather "
            "than NVIDIA RULER"
        ),
    )

"""Long-context retrieval evaluation (Needle-in-a-Haystack / simple RULER).

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
import math
import random
import re
from dataclasses import dataclass, field
from logging import getLogger
from typing import Callable, List, Optional

import torch

logger = getLogger(__name__)

_Z95 = 1.959963984540054  # standard normal 97.5th percentile


def proportion_stats(hits: int, n: int):
    """(stderr, ci95_low, ci95_high) for a binomial accuracy ``hits/n``.

    Reporting an error bar is required to claim a method "beats" another: with a
    handful of needles a 1/24 difference is well within noise.  Uses the normal
    standard error for the point estimate and the **Wilson** 95% interval (which
    stays inside [0, 1] and is well-behaved near 0 % / 100 % accuracy, unlike the
    Wald interval).
    """
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = hits / n
    se = math.sqrt(max(p * (1.0 - p), 0.0) / n)
    z = _Z95
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(max(p * (1.0 - p) / n + z * z / (4.0 * n * n), 0.0)) / denom
    return se, max(0.0, center - half), min(1.0, center + half)


def _number_hit(gold: str, text: str) -> bool:
    """True iff ``gold`` appears in ``text`` as a standalone integer.

    Uses digit boundaries so e.g. gold ``1234`` is not falsely matched inside
    ``91234`` / ``12345`` -- a substring check would inflate accuracy.
    """
    return re.search(rf"(?<!\d){re.escape(gold)}(?!\d)", text) is not None

_CITIES = ["Paris", "Tokyo", "Cairo", "Lima", "Oslo", "Delhi", "Quito", "Accra",
           "Hanoi", "Sofia", "Riga", "Bern", "Doha", "Kyiv", "Amman", "Dakar"]

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
        ntok += len(tokenizer(s).input_ids)
    return "".join(out)


@dataclass
class NeedleResult:
    """Needle-in-a-haystack retrieval result over a depth x length grid."""
    n: int
    accuracy: float
    mem_fraction: float
    by_length: dict = field(default_factory=dict)   # ctx_len -> accuracy
    grid: List[dict] = field(default_factory=list)   # per (len, depth, seed)
    accuracy_stderr: float = 0.0                     # binomial SE of accuracy
    accuracy_ci95: tuple = (0.0, 0.0)                # Wilson 95% interval

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "accuracy_stderr": round(self.accuracy_stderr, 4),
            "accuracy_ci95": [round(self.accuracy_ci95[0], 4),
                              round(self.accuracy_ci95[1], 4)],
            "mem_fraction": round(self.mem_fraction, 4),
            "by_length": {k: round(v, 4) for k, v in self.by_length.items()},
        }


def make_needle_prompt(tokenizer, context_tokens: int, depth: float = 0.5, seed: int = 0):
    """Build a needle-in-haystack chat prompt and its gold answer.

    A secret ``magic number`` is hidden at fractional ``depth`` inside ~``context_tokens``
    tokens of filler.  Returns ``(messages, gold_str)``.
    """
    rng = random.Random(seed)
    magic = rng.randint(1000, 9999)
    city = rng.choice(_CITIES)
    needle = f" The magic number for {city} is {magic}. "
    pre = _build_filler(tokenizer, int(context_tokens * depth), rng)
    post = _build_filler(tokenizer, int(context_tokens * (1.0 - depth)), rng)
    haystack = pre + needle + post
    question = (f"\n\nQuestion: What is the magic number for {city}? "
                f"Answer with only the number.\nAnswer:")
    return [{"role": "user", "content": haystack + question}], str(magic)


def _encode(tokenizer, messages) -> torch.Tensor:
    # Retrieval answers are a single short number, so disable chain-of-thought for
    # thinking models (Qwen3/QwQ): with reasoning on, a short max_new budget is
    # consumed entirely inside the <think> block and the model never emits the
    # answer (FullKV would score ~0). enable_thinking is ignored by non-thinking
    # templates (Llama/Mistral), so this is safe across model families.
    try:
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, enable_thinking=False)
        return enc["input_ids"]
    except Exception:  # template rejects the kwarg -> retry without it
        pass
    try:
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        return enc["input_ids"]
    except Exception:  # tokenizer without chat template
        return tokenizer(messages[0]["content"], return_tensors="pt").input_ids


@torch.no_grad()
def greedy_answer(model, input_ids, cache, max_new_tokens=32, post_prefill=None,
                  eos=None, attn_capture=False):
    """Greedy decode with explicit ``cache_position`` (robust to eviction).

    ``post_prefill(cache)`` is called once after the prefill forward so an
    eviction cache can trim itself; survivors keep their true positions while
    new tokens decode at their true position index.

    ``attn_capture=True`` materializes ``attn_weights`` (eager + output_attentions)
    and feeds attention sparsity into a ThinKV cache so its thought
    classification engages during the explicit decode loop.
    """
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
        out = model(input_ids.to(dev), past_key_values=cache, use_cache=True,
                    cache_position=pos, output_attentions=oa)
        if post_prefill is not None:
            post_prefill(cache)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        toks, cur = [nxt], n
        for _ in range(max_new_tokens - 1):
            cp = torch.tensor([cur], device=dev)
            out = model(nxt, past_key_values=cache, use_cache=True,
                        cache_position=cp, position_ids=cp.unsqueeze(0),
                        output_attentions=oa)
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            toks.append(nxt)
            cur += 1
            if eos is not None and nxt.item() == eos:
                break
    return torch.cat(toks, dim=1)


@torch.no_grad()
def needle_in_haystack(
    model, tokenizer,
    cache_factory: Optional[Callable[[], object]] = None,
    context_lens: Optional[List[int]] = None,
    depths: Optional[List[float]] = None,
    seeds: int = 3,
    max_new_tokens: int = 32,
    evict: bool = False,
    attn_capture: bool = False,
) -> NeedleResult:
    """Needle retrieval accuracy over a depth x context-length grid.

    Args:
        cache_factory: ``() -> Cache`` (``None`` = FullKV ``DynamicCache``).
        context_lens: approximate prompt lengths in tokens (e.g. [2048, 4096]).
        depths: fractional needle positions (e.g. [0.1, .., 0.9]).
        seeds: independent needles per (length, depth) cell.
        max_new_tokens: decode budget (use ~256 for reasoning models that think
            before answering -- the recalled fact appears within the thinking).
        evict: if True, call ``cache.evict_now()`` after prefill (eviction caches).
        attn_capture: if True, engage ThinKV thought classification (eager attn).
    """
    from transformers import DynamicCache  # local import; keeps module import light

    context_lens = context_lens or [2048, 4096]
    depths = depths or [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]
    eos = tokenizer.eos_token_id

    hit, n = 0, 0
    mem: List[float] = []
    by_len_hit: dict = {c: [0, 0] for c in context_lens}
    grid: List[dict] = []
    for ctx in context_lens:
        for di, depth in enumerate(depths):
            for s in range(seeds):
                messages, gold = make_needle_prompt(
                    tokenizer, ctx, depth=depth, seed=di * 97 + s * 1009 + 1)
                ids = _encode(tokenizer, messages)
                cache = cache_factory() if cache_factory is not None else DynamicCache()
                pp = (lambda c: c.evict_now()) if evict else None
                gen = greedy_answer(model, ids, cache, max_new_tokens, post_prefill=pp,
                                    eos=eos, attn_capture=attn_capture)
                txt = tokenizer.decode(gen[0], skip_special_tokens=True)
                ok = _number_hit(gold, txt)
                hit += int(ok); n += 1
                by_len_hit[ctx][0] += int(ok); by_len_hit[ctx][1] += 1
                if hasattr(cache, "mem_fraction"):
                    mem.append(cache.mem_fraction())
                elif hasattr(cache, "compression_ratio"):
                    mem.append(1.0 / max(1e-9, cache.compression_ratio()))
                grid.append({"ctx": ctx, "depth": depth, "seed": s, "hit": ok})
    se, lo, hi = proportion_stats(hit, n)
    return NeedleResult(
        n=n,
        accuracy=hit / max(1, n),
        mem_fraction=(sum(mem) / len(mem) if mem else 1.0),
        by_length={c: by_len_hit[c][0] / max(1, by_len_hit[c][1]) for c in context_lens},
        grid=grid,
        accuracy_stderr=se,
        accuracy_ci95=(lo, hi),
    )


def make_multikey_prompt(tokenizer, context_tokens: int, num_keys: int = 4,
                         depth: float = 0.5, seed: int = 0):
    """Build a RULER-style multi-key retrieval prompt and its gold answer.

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

    question = (f"\n\nQuestion: What is the magic number for {target}? "
                f"Answer with only the number.\nAnswer:")
    return [{"role": "user", "content": body + question}], str(facts[target])


@torch.no_grad()
def ruler_multikey(
    model, tokenizer,
    cache_factory: Optional[Callable[[], object]] = None,
    context_lens: Optional[List[int]] = None,
    num_keys: int = 4,
    seeds: int = 5,
    max_new_tokens: int = 32,
    evict: bool = False,
    attn_capture: bool = False,
) -> NeedleResult:
    """RULER-style multi-key retrieval accuracy over context lengths.

    Like :func:`needle_in_haystack` but with ``num_keys`` competing facts so the
    model must disambiguate among distractors -- the regime where eviction loses
    the queried key and graded quantization keeps it (lossily) recoverable.
    """
    from transformers import DynamicCache  # local import

    context_lens = context_lens or [2048, 4096]
    eos = tokenizer.eos_token_id

    hit, n = 0, 0
    mem: List[float] = []
    by_len_hit: dict = {c: [0, 0] for c in context_lens}
    grid: List[dict] = []
    for ctx in context_lens:
        for s in range(seeds):
            messages, gold = make_multikey_prompt(
                tokenizer, ctx, num_keys=num_keys, seed=s * 1009 + 7)
            ids = _encode(tokenizer, messages)
            cache = cache_factory() if cache_factory is not None else DynamicCache()
            pp = (lambda c: c.evict_now()) if evict else None
            gen = greedy_answer(model, ids, cache, max_new_tokens, post_prefill=pp,
                                eos=eos, attn_capture=attn_capture)
            txt = tokenizer.decode(gen[0], skip_special_tokens=True)
            ok = _number_hit(gold, txt)
            hit += int(ok); n += 1
            by_len_hit[ctx][0] += int(ok); by_len_hit[ctx][1] += 1
            if hasattr(cache, "mem_fraction"):
                mem.append(cache.mem_fraction())
            elif hasattr(cache, "compression_ratio"):
                mem.append(1.0 / max(1e-9, cache.compression_ratio()))
            grid.append({"ctx": ctx, "num_keys": num_keys, "seed": s, "hit": ok})
    se, lo, hi = proportion_stats(hit, n)
    return NeedleResult(
        n=n,
        accuracy=hit / max(1, n),
        mem_fraction=(sum(mem) / len(mem) if mem else 1.0),
        by_length={c: by_len_hit[c][0] / max(1, by_len_hit[c][1]) for c in context_lens},
        grid=grid,
        accuracy_stderr=se,
        accuracy_ci95=(lo, hi),
    )

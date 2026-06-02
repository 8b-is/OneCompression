"""Compute TaperKV avg-bits / mem_fraction for candidate operating points.

Replicates ``onecomp.kvcomp.cache.taper_cache.taper_bits`` *exactly* (verified
against the source) so it runs on a CPU login node without importing the heavy
onecomp package (which pulls in gemlite/CUDA at import time).

Goal: pick operating points where the *floor* bits (b_min / b_min_value) keep
the needle recoverable -- TaperKV uses plain affine per-token quant, and the
benchmark already shows affine Uniform-INT2 -> needle 0.0 while INT3/INT4 ->
needle 1.0.  So the oldest tokens (which hold a mid-context needle) need a
>= 3-bit floor to survive.  This script reports the average bit-width and the
fp16 memory fraction (avg_bits/16) at each benchmark context length so we can
choose floors that are retrieval-safe at the target memory budget.

Run:
    python tests/taper_schedule.py
"""

from __future__ import annotations

import torch


def taper_bits(age, total, sink_tokens, recent_tokens, hi_bits, b_min, ladder=None):
    """Verbatim copy of onecomp.kvcomp.cache.taper_cache.taper_bits."""
    bits = torch.full_like(age, hi_bits, dtype=torch.int64)
    is_sink = age >= (total - sink_tokens)
    is_recent = age < recent_tokens
    mid = ~(is_sink | is_recent)
    if mid.any():
        if ladder is None:
            ladder = list(range(hi_bits, b_min - 1, -1)) or [b_min]
        nb = len(ladder)
        a = age[mid].float()
        lo, hi = recent_tokens, max(recent_tokens + 1, total - sink_tokens)
        frac = ((a - lo) / max(1.0, (hi - lo))).clamp(0, 1 - 1e-6)
        band = (frac * nb).long().clamp(0, nb - 1)
        ladder_t = torch.tensor(ladder, device=age.device, dtype=torch.int64)
        bits[mid] = ladder_t[band]
    return bits.clamp(min=b_min, max=hi_bits)


def avg_bits_and_mem(key_bits, value_bits, b_min, b_min_value,
                     sink_tokens, recent_tokens, total):
    age = torch.arange(total - 1, -1, -1)
    kb = taper_bits(age, total, sink_tokens, recent_tokens, key_bits, b_min).float()
    vb = taper_bits(age, total, sink_tokens, recent_tokens, value_bits, b_min_value).float()
    kavg = float(kb.mean())
    vavg = float(vb.mean())
    combined = (kavg + vavg) / 2.0
    return kavg, vavg, combined, combined / 16.0


# Candidate operating points: (label, key_bits, value_bits, b_min, b_min_value)
CANDIDATES = [
    ("current TaperKV (8,4,2,2)",      8, 4, 2, 2),   # b_min=2: needle dies (floor 2-bit affine)
    ("current TaperKV-3bit (4,3,2,2)", 4, 3, 2, 2),
    ("retr-safe A (8,4,4,3)",          8, 4, 4, 3),   # key floor 4, value floor 3
    ("retr-safe B (8,4,4,4)",          8, 4, 4, 4),   # both floor 4 (Uniform-INT4 floor)
    ("retr-safe C (8,4,3,3)",          8, 4, 3, 3),   # both floor 3 (Uniform-INT3 floor)
    ("retr-safe 3bit (4,4,3,3)",       4, 4, 3, 3),   # all >=3, lower hi -> ~3.x avg
    ("retr-safe 3bit (4,3,3,3)",       4, 3, 3, 3),
    ("retr-safe 3bit (3,3,3,3)=Uni3",  3, 3, 3, 3),   # degenerates to uniform 3-bit
]

SINK = 4
RECENT = 128  # paper_taperkv.yaml recent_tokens
CONTEXTS = [4096, 8192, 16384]

print(f"sink={SINK} recent={RECENT}")
print(f"{'operating point':<34} | " + " | ".join(f"ctx={c}" for c in CONTEXTS))
print(f"{'':34} | " + " | ".join("kavg vavg avg  mem " for _ in CONTEXTS))
print("-" * 110)
for label, kb, vb, bmn, bmnv in CANDIDATES:
    cells = []
    for ctx in CONTEXTS:
        ka, va, avg, mem = avg_bits_and_mem(kb, vb, bmn, bmnv, SINK, RECENT, ctx)
        cells.append(f"{ka:4.1f} {va:4.1f} {avg:4.2f} {mem:.3f}")
    print(f"{label:<34} | " + " | ".join(cells))

print()
print("Reference iso-memory targets: 3-bit=0.1875, 3.5-bit=0.219, 4-bit=0.250")
print("Needle survival (affine floor, from completed Llama run):")
print("  Uniform-INT4=1.0  INT3=1.0  INT2=0.0  -> floor must be >=3 for needle")

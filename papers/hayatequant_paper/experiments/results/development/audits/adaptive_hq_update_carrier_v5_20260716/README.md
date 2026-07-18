# AdaptiveHQ update-carrier v5 audit (2026-07-16)

This repo-local development bundle records the source-contract correction from
AdaptiveHQ policy v4 to
`adaptive_hq_cache_causal_safe_two_pair_rd_v5`, with distortion table
`source_qmc128_exact_codebook_fp16scale_safe_two_pair_update_carrier_v4`.
The correction was decided during the pre-commit implementation audit, before
and without inspecting any formal benchmark or model outcome.  This bundle
contains no PPL, accuracy, long-context, generation-quality, or speed result.

## Correction

AdaptiveHQ stores exactly one candidate-pair selector in the low fraction bit
of the first existing fp16 value scale for each encode update.  The previous
proxy cleared only one scale in a fixed 128-vector QMC panel and therefore
implicitly amortized the carrier over 128 vectors, even when an actual update
contained a different number of vectors.

The v5 contract separates the carrier-free value distortion, `D_V^0`, from a
signed per-vector carrier delta, `Delta_V`.  The frozen QMC calculation clears
the selector bit in every one of its 128 value scales and computes
`Delta_V = D_all_clear - D_V^0`.  An update with `M` actual vectors uses

```text
D_V(update) = D_V^0 + Delta_V / M
M = batch * heads * sequence_length
```

The delta remains signed; only the final effective distortion is required to
be finite and positive.  This models one carrier-bearing value scale per
update without changing stored bits, the two-pair candidate set, the causal
power statistic, or the original HQ method story.

## Frozen formal recipe mapping

| Average rate | Candidate K/V code bits per two-coordinate block | Value-table rows |
|---:|---|---|
| 2 | `(4,4)`, `(5,3)` | 4, 3 |
| 3 | `(6,6)`, `(7,5)` | 6, 5 |
| 4 | `(8,8)`, `(9,7)` | 8, 7 |

`audit.json` freezes the complete value-side QMC constants for code bits 3
through 8 and the effective values for update sizes 1, 2, 8, and 128.  These
are deterministic data-free source/codebook checks, not model measurements.

## Verification result

The focused CPU suite passed: **10 passed, 302 deselected, 14 warnings** in
12.15 seconds.  It covers the signed amortization formula and malformed input,
all six frozen QMC deltas, an update-size-dependent synthetic RD boundary, a
real-QMC boundary at 1 versus 128 vectors, provenance, and required carrier
decode behavior.

All source hashes in `source_manifest.json` are hashes of the shared dirty
pre-commit working tree, not of the recorded Git HEAD.  The manifest labels
this explicitly so the commit identifier cannot be mistaken for a clean
source snapshot.  `SHA256SUMS` covers every file in this directory except
itself.

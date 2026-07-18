# HQ/AHQ quality refinement development evidence

Status: complete. Slurm quality array `69327_[0-13]` and packed-native array
`69341_[0-4]` all completed with exit code 0. The fail-closed analyzer passed.
This is a one-window WikiText-2 development panel, not formal paper evidence.

The frozen protocol, candidate set, exact-rate accounting, and ranking rule are
in `PREREGISTRATION.md`; the independent native protocol is in
`NATIVE_PREREGISTRATION.md`. All comparisons below use strict unrounded NLL.

## Fixed HQ result

Rates 3 and 4 are the unchanged current b=2 HQ. Only rate 2 changes for the
b=4 candidates.

| Development recipe | TQ | FibQuant | BlockQuant | All external |
|---|---:|---:|---:|---:|
| Current b=2 HQ | 18/21 | 16/21 | 4/7 | 38/49 (77.55%) |
| b=4 K10/V6 Gaussian | 18/21 | 18/21 | 4/7 | 40/49 (81.63%) |
| b=4 K10/V6 finite Beta | 18/21 | 17/21 | 4/7 | 39/49 (79.59%) |

Gaussian `auto` and forced `fibquant` initialization are one equivalence
class: every unrounded NLL and every key/value codebook SHA-256 is identical.
There is no shipped K1024 artifact, so both paths converge to the same current
online-fit fixed point. The Gaussian candidate improves current HQ in 6/7
rate-2 models and has summed NLL delta -1.514556, but regresses Qwen3-8B by
+0.179663 NLL. Its 40/49 result is below the requested 90% target.

## Safe AdaptiveHQ result

Every rate uses only `(2r,2r)` and fixed-HQ `(2r+1,2r-1)`. The selected global
weight is `key_error_weight:value_error_weight = 4:1` under the frozen rank.

| Comparison | Rate 2 | Rate 3 | Rate 4 | Total |
|---|---:|---:|---:|---:|
| Safe AdaptiveHQ vs all external baselines | 13/14 | 16/21 | 11/14 | 40/49 (81.63%) |
| Previous AdaptiveHQ vs all external baselines | 0/14 | 12/21 | 10/14 | 22/49 (44.90%) |
| Safe AdaptiveHQ vs current fixed HQ | 4/7 | 4/7 | 3/7 | 11/21 |

All three predeclared global weights score the same 40/49 external cells;
`4:1` wins the lower-order frozen criteria. It beats TQ in 20/21, FibQuant in
17/21, and BlockQuant in 3/7 cells. Compared directly with the previous
AdaptiveHQ output it wins 13/21 cells (7/7 at rate 2), with summed NLL delta
-14.907778. It is an opt-in AdaptiveHQ policy, not a replacement for fixed HQ:
its summed NLL delta versus current fixed HQ is +1.092892 despite 11 pointwise
wins.

Across 7 models, 3 rates, and 3 weights, all 63 adaptive cells use exactly one
selector bit per encoded update and zero additional stored bits. There are
1,032,192 encoded updates in total. Every observed split belongs to the frozen
two-pair set; no K7/V1 or other extreme split occurs. TQ, FibQuant, fixed HQ,
and safe AdaptiveHQ have equal exact integer fp16/stored-bit counters in every
applicable cell.

## Native parity and speed eligibility

All four packed-native smoke tests pass exact public/native key and value codes,
fp16 scales, packed streams, output-dtype dequantization, and store-one/prefill
identity. Every candidate uses the same 68-byte physical slot (2,228,224 bytes
for the ABBA cache). The b=4 recipe requires the explicit K10 safety-cap
exception, has a 54.4x larger fp32 codebook, and uses the generic rather than
the specialized b=2 decode path.

Fresh-process B200 ABBA medians for the quality-leading b=4 Gaussian candidate:

| Native operation | Current b=2 HQ (ms) | b=4 candidate (ms) | Candidate/current latency |
|---|---:|---:|---:|
| Prefill store | 2.670784 | 7.015200 | 2.627x |
| Decode attention | 0.103712 | 0.135392 | 1.305x |
| Store-one + decode | 0.763536 | 0.808608 | 1.059x |
| CUDA-Graph store-one + decode | 0.189688 | 0.236968 | 1.249x |

Thus b=4 passes correctness and exact-rate gates but fails the native
speed-Pareto gate on all four operations. The unique current-round decision is
to retain b=2 K5/V3 current HQ and not promote b=4. Safe AdaptiveHQ `4:1` is the
quality-selected adaptive development recipe, pending the separately
preregistered fixed-artifact rerun.

`analysis.json` is the machine-readable source of all summaries, cell deltas,
payload audits, codebook identities, and native ABBA validation. `cells.csv`,
`adaptive_split_traces.json`, and `codebook_hashes.json` are flattened views.
Raw JSON, stdout/stderr, submission scripts, source state, commands, Slurm
accounting, secret scan, and SHA-256 manifest are retained in this directory.

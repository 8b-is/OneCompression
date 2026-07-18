# HQ variant pruning (2026-07-17)

This note records the three-model screening used before changing the public
HayateQuant recipe.  The run uses the same codec benchmark, model weights,
WikiText-2 revision, 128-token KV capture, 256-token/one-window PPL smoke
protocol, and `skip_outliers=false` for every row.  It is a screening gate,
not a replacement for the full paper matrix: each PPL is based on 255 scored
targets.

## Candidates

The run compares `HQ` (current fixed recipe), the historical pre-v2
`HQ-frontier-candidate` (UB key scale and integer K(r+1)/V(r-1) split),
`AdaptiveHQ`, and the external TQ/Fib/Block quality rows.  The candidate is
explicitly diagnostic and is not silently substituted for the public `HQ`.

| model | candidate wins vs HQ | HQ wins vs candidate |
|---|---:|---:|
| Qwen3-0.6B | 3/3 rates | 0/3 |
| DeepSeek-R1-Distill-Qwen-1.5B | 3/3 | 0/3 |
| Llama-3.1-8B-Instruct | 0/3 | 3/3 |

The complete JSON payloads are in
`frontier_candidate_20260717/`; the legacy-branch frontier payloads are
preserved under `frontier_legacy_labkvcomp_20260717/`.

## Speed gate

The measured quantity is codec encode latency (`ms_per_encode`), not
end-to-end decode/inference.  Relative to TurboQuant, the frontier candidate
is slower in every measured cell (about 1.04--3.13x); current HQ is also
usually slower (about 0.76--1.29x).  Therefore no claim that HQ is faster than
TurboQuant is justified by this run.  FibQuant and BlockQuant are quality-only
rows because their public implementations do not provide a serving kernel.

## Pruning decision

For production/default runs keep only the small named family:

1. `HQ` (current fixed proposal),
2. one fixed frontier candidate under evaluation,
3. `AdaptiveHQ`, and
4. `RotationHQ` only when its independent calibration artifact is complete.

`auto`, `exact`, `K12`, `dense-LS`, `bs1/bs2`, scale/init proxies, and other
development controls remain available only for explicit regression/ablation
commands and are excluded from the default paper method set.  The frontier
candidate is not promoted to the public `HQ` name yet: its advantage is
model-dependent and its latency does not meet the TurboQuant-speed gate.

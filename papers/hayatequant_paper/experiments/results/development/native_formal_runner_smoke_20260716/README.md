# Packed-native formal-runner development smoke (2026-07-16)

This directory retains the development validation of the new fail-closed
packed-native HQ/TurboQuant speed runner. It is **not** a formal paper result:
the runner and tests were intentionally uncommitted while being exercised, and
the short runs use four fresh processes, one warmup, two timed iterations, and
32 or 64 tokens. Every raw JSON correctly records these ineligibility reasons.

The purpose was to verify the measurement contract before any new formal run:

- both HQ and unmodified upstream vLLM 0.20 TurboQuant first store their
  physical packed cache;
- each cache is independently dequantized and used in materialized fp32 GQA;
- eager and CUDA-graph packed-attention outputs must pass the elementwise
  tolerance (`rtol=0.002`, bf16 `atol=0.015`) before timing;
- non-finite values, modified slot padding, competing compute processes, and
  fresh-process changes in logical packed bytes or output hashes fail closed;
- worker commands, stdout, stderr, exit status, GPU UUID/driver/clocks/power,
  source hashes, upstream vLLM hashes, and failure JSON paths are retained.

All three controller runs passed their correctness and determinism gates. The
largest eager packed-output absolute error was `0.00390625`; the largest
relative L2 error was `0.0011601375348690005`. The rate-3 32-token and rate-4
32-token reruns also confirmed that every selected-GPU snapshot contained only
the worker PID, with B200 clocks at 1965/3996 MHz and driver 580.126.20.

The latency samples are deliberately retained even though HQ loses every
short-smoke point estimate:

| Rate | Tokens | Prefill store | Packed attention | Eager store1+attention | Graph store1+attention |
|---:|---:|---:|---:|---:|---:|
| 3 | 64 | 0.162445x | 0.562853x | 0.180801x | 0.324804x |
| 3 | 32 | 0.235003x | 0.821199x | 0.230099x | 0.330194x |
| 4 | 32 | 0.171466x | 0.430574x | 0.169611x | 0.304248x |

Ratios are TurboQuant latency divided by HQ latency, so values greater than one
would favor HQ. These tiny workloads and repeat counts are runner diagnostics,
not evidence for a speed claim.

These smokes exercised the draft contract whose SHA-256, recorded in every raw
result, is
`fe655abd53948b3db840a4747ce5b58b5ebaea30fb32d6466a79af321001d16c`.
Before any formal result was run, the contract was tightened to pin synthetic
input seed zero plus the Gaussian generator and all HQ source-law/rotation
arguments. The final contract at
`benchmark/kvcomp-codec/hayatequant_native_speed_formal_contract.json` has
SHA-256 `0e35a5df5fb99f06163165f1155c0916522f324c8758c5d55c46f9f28620bb9a`.
It freezes 48 cells (two common native rates, two model-derived GQA shapes,
three contexts, four endpoints), twenty fresh processes per cell, strict ties
as losses, at least 44/48 overall wins, at least 11/12 wins per endpoint, and
at least 22/24 wins per rate. A single result can never authorize a formal
claim. The contract explicitly states that earlier negative development data
was known before preregistration; no blind preregistration is claimed.

`SHA256SUMS` inventories every retained file except itself.

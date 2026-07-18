# Clean-source packed-native HQ/TurboQuant diagnostic (2026-07-16)

This directory is an immutable **development diagnostic**, not a formal paper
result.  It records the first clean-source rerun of the faithful fixed-HQ packed
vLLM backend against the unmodified upstream vLLM 0.20 TurboQuant kernels on an
NVIDIA B200 (`kagura-gpu11`).  The source was detached at
`8aee55bebaa6974d5ca3362697982d6892359498` with an empty porcelain status.

The negative result is retained deliberately.  At 4,096 cached tokens, current
exact HQ loses every reported point estimate:

| Rate | HQ/TQ prefill-store speedup | HQ/TQ packed-attention speedup | HQ/TQ eager store-one+attention | HQ/TQ graph store-one+attention |
|---:|---:|---:|---:|---:|
| 3 | 0.027673x | 0.747481x | 0.206008x | 0.431184x |
| 4 | 0.013183x | 0.732309x | 0.193503x | 0.359012x |

The ratios are `TurboQuant latency / HQ latency`; values above one favor HQ.
No speed-superiority claim is supported by these measurements.  In particular,
the much faster fixed-HQ research fake-quant path is a different implementation
contract and is not combined with this packed-native comparison.

## Scope and limitations

- Two fresh processes per method were produced by the four-process ABBA order;
  this is a diagnostic sample, not the planned five-process formal matrix.
- The runner checks its supported layout and method contracts, but this schema
  does not persist per-run decoded-output/checksum error fields.  Formal native
  runs must add and pass packed-output correctness gates before timing.
- Upstream vLLM 0.20 has native TurboQuant presets for rates 3 and 4 only in this
  harness.  Rate 2 was not silently synthesized or excluded from a formal gate;
  it remains unmeasured here.
- `formal_eligible=true` inside the raw files means the source/provenance checks
  passed.  It does not promote this development-only context/trial matrix into a
  formal paper result.
- AdaptiveHQ is not implemented by this native backend and was not measured.

Each raw JSON contains exact arguments, GPU UUID, source hashes, upstream vLLM
hashes, per-process samples, physical slot widths, and the clean git state.
`SHA256SUMS` covers every archived file except itself.

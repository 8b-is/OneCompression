# HQ native speed development record (2026-07-16)

This directory is an immutable development record for the faithful packed HQ
vLLM backend and its rejected LUT acceleration candidates.  It is not a
formal paper-result directory.  Raw JSON includes source/model/dataset hashes
and exact arguments; `logs/` includes successful, negative, harness-bug, and
environment-failure Slurm output.  No failed attempt was removed.

## Decision

Canonical **HQ** and **Adaptive HQ** remain on the shape-independent
all-codeword `encoder="exact"` assignment.  The public method remains the
paper HayateQuant story (data-free role-aware rotation, short block VQ,
fixed K/V allocation, key NC, and value LS-reassign-LS).  Exact and LUT are
implementation references, not method variants.

The fixed 256-bin LUT candidates T=2/4/8/16 were rejected.  The final fused
T=4 attempt proved payload correctness on its fixtures and made prefill much
faster than the generic candidate implementation, but it failed both required
adoption gates:

1. Qwen3-0.6B WikiText rate 2 was +1.65855% PPL relative to exact (limit
   +0.25%).  Rates 3 and 4 were -6.37770% and -0.54015%, respectively, but a
   single fixed all-rate candidate must pass every rate.
2. Against unmodified upstream vLLM 0.20 TurboQuant, fused T=4 still lost all
   serving speed point estimates.  At rate 3 its HQ/TQ speedups were 0.228x
   prefill, 0.870x decode, 0.257x eager serving step, and 0.487x CUDA-graph
   serving step.  At rate 4 they were 0.215x, 0.793x, 0.248x, and 0.405x.

A rate-resolved rule (rate 2 exact, rates 3/4 T=4) was also rejected because
the speed gate fails.  Therefore the preregistered seven-model x three-rate
development panel was correctly not started; that panel is required only
after the single-model quality and speed gates both pass.

## Correctness evidence

`full_native_exact_precomputed_smoke_rate3.json` passed public/native code,
signed-fp16 scale, packed-byte, store-one/prefill, dequantization, direct
packed-attention, and rotated-value linearity checks.

The final fused T=4 CUDA kernel combines cell arithmetic, candidate gather,
canonical half-GEMM score-minus-norm, and smallest-index tie handling.  A
1,048,576-row complete-candidate CUDA fixture matched the canonical full
half-GEMM codes bit-for-bit (job 69311).  Final rate-2/3/4 native smokes
(jobs 69314/69313/69312) passed all payload and attention contracts, with
zero code mismatch and zero excess MSE on their pinned random fixtures.  This
local parity success is recorded separately from the failed model-level
quality gate; it does not justify adopting the table.

The T=16 assignment audit at rate 2 found all exact winners in the candidate
sets over 7,340,032 K blocks and 14,680,064 V blocks.  The generic batched
candidate scorer differed once in the V path despite a complete K=8 set;
that single cache assignment changed the short-window PPL by +1.65855%.
This is why random-fixture distortion alone was not accepted as a quality
gate.

## Exact-path root cause

Stage profiles at 4096 tokens show that full-codebook search consumes an
estimated 90.99% of the rate-3 store and 95.49% of the rate-4 store.  Exact
rate-3 store was 4.732 ms: key search 1.860 ms and each value search 1.223 ms
(value is assigned twice).  Exact rate-4 store was 9.436 ms: key search
5.305 ms and each value search 1.853 ms.  Packing was only 0.036 ms.

Decode is already direct over packed codes.  Its dominant stage-1 kernel was
about 0.067--0.070 ms; query and inverse-value rotations were each small.
Total HQ decode remained slower than TQ, so eliminating prefill search alone
cannot satisfy the requested end-to-end speed dominance.

Canonical exact fresh-process results (dirty worktree is the sole formal
eligibility blocker recorded in JSON):

| Rate | Metric | HQ ms | TQ ms | HQ/TQ speedup |
|---:|---|---:|---:|---:|
| 3 | prefill store | 4.7315 | 0.1309 | 0.0277x |
| 3 | decode | 0.1055 | 0.0825 | 0.7814x |
| 3 | eager store1+decode | 0.7863 | 0.1742 | 0.2216x |
| 3 | graph store1+decode | 0.2101 | 0.0907 | 0.4315x |
| 4 | prefill store | 9.4508 | 0.1247 | 0.0132x |
| 4 | decode | 0.1080 | 0.0858 | 0.7944x |
| 4 | eager store1+decode | 0.8224 | 0.1772 | 0.2155x |
| 4 | graph store1+decode | 0.2155 | 0.0762 | 0.3534x |

## Failed attempts retained

- Jobs 69279, 69286, 69288, and 69289 exited 2 intentionally because their
  pinned PPL quality gates failed; these were not execution crashes.
- Jobs 69283--69285 exposed a smoke-reference propagation bug for T=4.  The
  harness was corrected and all subsequent candidate metadata/table widths
  were audited.
- Jobs 69304--69310 preserve the first fused-kernel compile failure
  (`zi[:,0]` unsupported in Triton).  The one-line indexing fix was followed
  by the passing kernel test and final jobs 69311--69317.
- Real-model vLLM integration reached model load/cache allocation/prefill.
  Jobs 69273/69275 fixed HQ cache-spec integration issues.  Later failures
  were external vLLM environment issues: DeepGEMM availability, missing
  `ninja`, then a stale FlashInfer JIT cache referencing a removed Python
  environment (job 69318).  These logs are retained and no generation claim
  is made.

## Reproducibility and integrity

- `COMMANDS.md` maps every job to its command family/config and marks which
  older commands were reconstructed.  JSON `provenance.arguments` is the
  authoritative per-run configuration where a result exists.
- `SACCT.txt` is the scheduler status snapshot.
- `SOURCE_SHA256SUMS` hashes the implementation/benchmark sources used at
  archive finalization.  Each JSON also stores the source hashes at run time.
- `SHA256SUMS` hashes every archived file except itself.
- A filename-only secret scan over all source result/log files found no token,
  authorization-header, API-key, GitHub-token, or HF-token patterns before
  archival.


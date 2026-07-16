# Development baseline-quality matrix (2026-07-16)

This directory is the immutable analysis-input snapshot used by the refinement
run: it intentionally contains only this README, `analysis.json`, and
`ppl_cells.csv`.  The complete raw/log/provenance bundle remains alongside this
run at `../../baseline_quality_20260716/`.  The local `SHA256SUMS` covers exactly
the three snapshot payload files and does not pretend that omitted raw files
are present here.

This bundle is a **development-only**, capped WikiText-2 diagnostic.  It is not
a formal-paper result and must not be rendered into a headline table.  Slurm
array `69320` scored one 512-token window (511 next-token targets) on seven
checkpoints with no FP16 sink/recent window and no skipped layers.

The compared rows were `TurboQuant-paper`, independent `FibQuant-paper`, fixed
`HQ`, and `AdaptiveHQ` at rate slots 2/3/4.  The independent, fixed-seed
`BlockQuant-paper` quality row was evaluated only in its defined rate-3 slot.
HQ and AdaptiveHQ used the unchanged canonical exact encoder; this run did not
tune or select a recipe.

## Point-estimate result

A win means strictly lower unrounded JSON-double NLL on the same 511 targets.
Ties are not wins.  The validator fails on missing/non-finite cells, dataset or
denominator drift, non-exact HQ, incorrect causal scoring protocol, or unequal
TQ/Fib/HQ/AdaptiveHQ stored-bit counters.

| Proposal | Baseline | Wins / cells | Win rate |
|---|---|---:|---:|
| HQ | TurboQuant-paper | 18 / 21 | 85.71% |
| HQ | FibQuant-paper | 16 / 21 | 76.19% |
| HQ | BlockQuant-paper (rate 3) | 4 / 7 | 57.14% |
| **HQ** | **all external quality cells** | **38 / 49** | **77.55%** |
| AdaptiveHQ | TurboQuant-paper | 9 / 21 | 42.86% |
| AdaptiveHQ | FibQuant-paper | 9 / 21 | 42.86% |
| AdaptiveHQ | BlockQuant-paper (rate 3) | 4 / 7 | 57.14% |
| **AdaptiveHQ** | **all external quality cells** | **22 / 49** | **44.90%** |

HQ's aggregate rate-stratified counts are 12/14 at rate 2, 15/21 at rate 3,
and 11/14 at rate 4.  AdaptiveHQ's are 0/14, 12/21, and 10/14 respectively.
Thus this short diagnostic does **not** support a 90% quality-win claim.  In
particular, AdaptiveHQ's rate-2 policy is a decisive regression in all seven
checkpoints and should remain outside any formal claim until independently
fixed and re-evaluated.

Fixed-HQ loss cells are:

- vs TQ: Llama-2 rate 4; DeepSeek-R1-Distill-Llama-8B rates 3 and 4.
- vs FibQuant: Llama-3.1 rate 2; Llama-2 rate 2; Qwen3-4B rate 4;
  DeepSeek-R1-Distill-Llama-8B rate 3; DeepSeek-R1-Distill-Qwen-1.5B rate 3.
- vs BlockQuant: Qwen3-4B, DeepSeek-R1-Distill-Llama-8B, and
  DeepSeek-R1-Distill-Qwen-1.5B (all rate 3).

AdaptiveHQ loses every rate-2 TQ/Fib cell.  At rate 3 it loses all three
baseline comparisons on Llama-3.1, Llama-2, and DeepSeek-R1-Distill-Llama-8B.
At rate 4 it loses to TQ on Llama-2 and DeepSeek-R1-Distill-Llama-8B, and to
FibQuant on Qwen3-4B and DeepSeek-R1-Distill-Llama-8B.

## PPL point estimates

These displayed values are rounded for readability; `analysis.json`,
`ppl_cells.csv`, and each raw JSON preserve the unrounded PPL/NLL numerators.

| Model | Rate | TQ | FibQuant | BlockQuant | HQ | AdaptiveHQ |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 2 | 9.49466 | 8.39114 | -- | 8.42755 | 110.882 |
| Llama-3.1-8B-Instruct | 3 | 6.97961 | 6.88901 | 6.98844 | 6.85119 | 7.34929 |
| Llama-3.1-8B-Instruct | 4 | 6.60037 | 6.60891 | -- | 6.58468 | 6.48192 |
| Llama-2-7b | 2 | 6.58648 | 6.13676 | -- | 6.37962 | 71.0744 |
| Llama-2-7b | 3 | 5.51951 | 5.57118 | 5.60794 | 5.47100 | 5.64830 |
| Llama-2-7b | 4 | 5.38009 | 5.41142 | -- | 5.38050 | 5.39098 |
| Qwen3-4B | 2 | 143.136 | 156.891 | -- | 67.2315 | 5038.24 |
| Qwen3-4B | 3 | 17.7708 | 17.6823 | 14.5050 | 14.8274 | 14.0331 |
| Qwen3-4B | 4 | 15.0949 | 13.1089 | -- | 13.7072 | 13.2716 |
| Qwen3-8B | 2 | 20.0493 | 22.6384 | -- | 14.7240 | 193.898 |
| Qwen3-8B | 3 | 10.7266 | 11.2139 | 10.1136 | 9.40310 | 9.94213 |
| Qwen3-8B | 4 | 9.93269 | 9.84255 | -- | 9.14414 | 9.39054 |
| R1-Distill-Llama-8B | 2 | 45.2437 | 37.4496 | -- | 34.1243 | 586.528 |
| R1-Distill-Llama-8B | 3 | 27.2128 | 26.7936 | 25.9151 | 27.5751 | 27.5568 |
| R1-Distill-Llama-8B | 4 | 25.4255 | 26.5200 | -- | 26.3030 | 26.5403 |
| R1-Distill-Qwen-1.5B | 2 | 54211.5 | 64426.2 | -- | 46147.0 | 172203 |
| R1-Distill-Qwen-1.5B | 3 | 54921.0 | 40038.0 | 39049.4 | 47071.7 | 30113.1 |
| R1-Distill-Qwen-1.5B | 4 | 29788.7 | 41261.2 | -- | 13244.4 | 13790.2 |
| R1-Distill-Qwen-7B | 2 | 87747.6 | 107982 | -- | 51370.4 | 363972 |
| R1-Distill-Qwen-7B | 3 | 58340.9 | 82857.1 | 30838.0 | 12787.3 | 1911.19 |
| R1-Distill-Qwen-7B | 4 | 24133.6 | 15825.0 | -- | 851.171 | 1055.30 |

Instruction/distill checkpoints can have extreme raw-text absolute PPL.  Only
within-checkpoint, same-target codec contrasts are interpreted here.  A single
window is also high-variance; full WikiText evaluation is required for formal
claims.

## Accounting and provenance

TQ, FibQuant, HQ, and AdaptiveHQ have exactly equal integer fp16 and stored-bit
counters within every model/rate cell.  BlockQuant is not exactly iso-stored-
rate because the versioned dynamic top-32 completion counts its combinadic
identity field; its fixed-seed row is not the paper's five-seed reproduction.
No FibQuant/BlockQuant speed comparison was made.

- `raw/`: seven benchmark JSON documents, including exact codec configs,
  dataset revision/text digest, unrounded NLL sums, token counts, and bit counts.
- `logs/`: complete Slurm stdout/stderr for all seven tasks.
- `manifests/`: per-task job/node/GPU/source/exit records.
- `analysis.json` and `ppl_cells.csv`: fail-closed aggregate and all 98
  proposal-vs-baseline comparison records.
- `SOURCE_STATE.txt`: launch commit, dirty-tree digests, and relevant source hashes.
- `SLURM_ACCOUNTING.txt`: job IDs, nodes, elapsed time, and exit codes.
- `COMMAND.txt` and `runtime/`: exact launch/analysis/finalization commands.
- `MANIFEST.json` and `SHA256SUMS`: bundle inventory and integrity hashes.

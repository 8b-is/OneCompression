# KV codec: HayateQuant vs TurboQuant / HALO (speed-quality)

Main experiment for the data-free rotated KV codecs, the analogue of the
weight-side `benchmark/<model>-<recipe>/` runs.

**HayateQuant** (`onecomp/kvcomp/codec/hayate/codec.py`) combines block-VQ,
folded rotation, and role-aware scales.  The paper's fixed **HQ** recipe uses
`block_size=2`, Gaussian source codebooks, exact full-codebook assignment,
key NC scaling, value LS-reassign/refit scaling, and integer block-code pairs
K5/V3, K7/V5, and K9/V7 at average rates 2, 3, and 4.  The old
`HayateQuant-frontier` K=`r+1`/V=`r-1` UB recipe is a legacy appendix ablation,
not the formal method.  **AdaptiveHQ** is the only opt-in proposal row.
The `HayateQuant-scalar-matched` row is the causal scalar control: source law,
rotation, scale equation, encoder, and rate match `TurboQuant-paper`, with only
the L2-versus-RMS parameterization of the signed-fp16 scale allowed to produce
low-precision rounding differences.  No compatibility relation alone implies
downstream superiority; all claims come from the coverage-checked formal runs.

| Codec | Quantizer | Rotation | Scale | Role |
|---|---|---|---|---|
| TurboQuant (hadamard) | scalar Lloyd-Max | WHT butterfly | RMS | deployment/rotation ablation |
| TurboQuant-QJL | scalar Lloyd-Max + 1-bit QJL residual | WHT butterfly + Gaussian sketch | RMS + residual norm | TurboQuant-prod appendix baseline |
| **TurboQuant-paper** | scalar finite-Beta Lloyd-Max | Gaussian-QR dense rotation | L2 | formal MSE baseline |
| **FibQuant-paper** | exact spherical-Beta block-VQ | shared Gaussian-QR dense rotation | L2 | independent formal quality-only baseline |
| **BlockQuant-paper** | exact finite-sphere block-VQ, dynamic top-32 key split | per-part Gaussian-QR dense rotation | UB / BSM | fixed-seed external quality-only baseline at the paper KV point |
| HALO (auto) | scalar Lloyd-Max | cheapest GEMM | NC / ls_iter | prior best (superseded) |
| **HQ** | **b=2 K5/V3, K7/V5, K9/V7 block-VQ** | **rate-resolved folded rotation** | **NC / LS-reassign/refit** | **fixed formal proposal** |
| **AdaptiveHQ** | **two-pair causal allocation over the same HQ codec** | **pair-resolved HQ rotation** | **NC / LS-reassign/refit** | **only opt-in formal proposal** |
| HayateQuant scalar/block matched | scalar or block VQ | dense | RMS-equivalent | mechanism controls; exact encoder |
| HQ-Fib-init-proxy | HQ block-VQ with Fib-inspired warm-start | HQ auto | HQ default | internal ablation, not FibQuant |
| HQ-Block-scale-proxy | HQ block-VQ | HQ auto | UB/IP / BSM/LS | internal ablation, not BlockQuant |

`TurboQuant-paper` is a stable internal result key, not a claim that the row
reproduces every TurboQuant paper table.  Formal output displays it as
**TurboQuant Algorithm 1 (MSE)**: finite-Beta Lloyd-Max scalar quantization,
dense Gaussian-QR/Haar rotation, and a per-vector L2 norm.  The paper's
LongBench mixed-precision description does not specify enough of its outlier
selection and K/V recipe for an exact Table 1 reproduction, and its printed
`(32*3 + 96*2)/128 = 2.5` arithmetic is 2.25 payload bits.  The formal
aggregate therefore records `paper_table1_reproduction=false`; the renderer
rejects a missing or stronger scope claim.  The signed-fp16 L2 norm is rounded
in the payload path and included as a fixed-width data-plane field.

The Python `encode`/`decode` payload used by codec microbenchmarks is a
materialized **logical** payload: code indices are unpacked `int64` tensors.
Its `stored_bits` field counts only fixed-width encoded-KV data-plane growth:
logical packed code widths, signed-fp16 scales, and any enabled mask, QJL, or
outlier fields.  Self-describing Python control tags and container bytes are
uncharged out-of-band/cache-level configuration; global codebooks, LUTs, and
rotations are also outside this per-token count.  Consequently `stored_bits`
is neither the payload tensors' allocator-resident footprint nor physical
serving memory, and the public-path timing must not be reported as packed-cache
serving throughput.  The formal output records and revalidates the committed
HQ/AdaptiveHQ codebook manifest and every reachable artifact hash.  Persistent
codebook/LUT/rotation resident bytes remain outside the per-token data-plane
count and are not reported as packed serving memory.

The fair-speed runner records the exact storage bytes of the public payload's
code-index and scale tensors separately from `stored_bits`.  Those bytes still
exclude Python containers, control tags, allocator fragmentation, and
persistent codec/codebook/LUT/rotation state.  Its E2E peak-memory counter is a
third, distinct quantity: `torch.cuda.max_memory_allocated` for a resident model
and the dequantized research fake-quant cache (reserved-but-unallocated bytes
excluded).  It must not be relabelled as a packed KV-cache measurement.

Formal speed is a fixed 60-process matrix: two checkpoints, three rates, two
compile modes, and five fresh Slurm processes.  Every timing process is pinned
to one selected healthy B200 node, while the aggregator independently rejects
any GPU capacity, SM-count, driver, software, or timing-protocol mismatch.  The
renderer exposes every public prefill/decode `encode`, prebuilt-`decode`, and
explicit-roundtrip endpoint; research fused prefill/decode roundtrip; E2E TTFT
and steady decode; and paired allocated-memory ratio/difference.  Per-endpoint
intervals remain descriptive.  A complete speed/measured-memory superiority
statement is possible only if all 132 predeclared primary-mode stochastic
endpoints are present and their one-sided 95% Bonferroni lower bounds favor HQ,
the public unpacked tensor footprints are no larger, and logical `stored_bits`
remain equal.  This gate does not establish downstream-quality superiority.

Independent FibQuant and BlockQuant codecs now live under
`onecomp/kvcomp/codec/`; neither is an HQ warm-start/scale proxy.  The formal
quality matrix registers the exact FibQuant v1 integer recipes as
`FibQuant-paper` for PPL, corrected zero-shot, the explicitly named NeMo-Skills
RULER-v1 100-sample/task protocol (not canonical NVIDIA 500/task), LongBench-E, MATH-500,
and LongCodeQA.  Its logical rate includes one signed-fp16 norm per vector.
`BlockQuant-paper` is registered only at its Llama-3.1 `head_dim=128` paper
operating point (formal rate slot 3).  The paper does not publish the top-32
lifecycle/identity code, the experimental LUT, the p=3 tail rule, or trained
centroids.  The row therefore uses a versioned per-batch/head/update top-L2
completion, fixed-width combinadic identities (`ceil(log2 C(128,32)) = 101`
bits), lower-dimensional tails, data-free fixed-seed codebooks, and exact
assignment.  It is a same-runtime fixed-seed quality baseline, not the paper's
five-random-seed reproduction.  Both exact-search Python implementations have
no packed serving kernel, so FibQuant and BlockQuant are intentionally absent
from every codec/decode/inference speed row.
The `HQ-*-proxy` labels remain internal HQ ablations and are never read aliases
for either independent codec.

`TurboQuant-QJL` follows the TurboQuant-prod construction for keys: for a target
payload of `b` bits per coordinate it uses a `b-1` bit scalar MSE key code plus
a 1-bit Gaussian QJL sketch of the key residual, while values use the regular
`b` bit scalar MSE code.  The residual norm is counted as an enabled data-plane field in
`stored_bits_per_coord`, so this row should be read as an appendix diagnostic
for the inner-product estimator rather than as an equal-stored-bits replacement
for the main TurboQuant-MSE baseline.

Ablation rows included in every run: `HayateQuant-bs1` (scalar compatibility control),
`HayateQuant-exact` (LUT off), `HayateQuant-K12` (large-codebook quality
ceiling), `HayateQuant-dense-LS` (Qwen-family PPL operating point).

**Lever-isolation ablation** (`--config-name paper_hayatequant_ablation`, sets
`ablation: true`) adds four more rows, each disabling exactly one of
HayateQuant-auto's levers: `HayateQuant-noscale` (RMS both roles),
`HayateQuant-hadamard-bvq` (no per-block dense rotation factor),
`HayateQuant-beta` (exact finite-d source law), `HayateQuant-bs2` (explicit
block size). Results must be regenerated with the current fingerprinted
harness before any quantitative lever claim is reported.

Optional SpinQuant-style calibration and K/V-asymmetric operating points remain
available as opt-in ablations. They are not folded into the fixed data-free HQ
or AdaptiveHQ formal rows.

## Portable layout and paper pipeline

The public HQ surface is deliberately limited to `HQ`, `AdaptiveHQ`, and
`RotationHQ`; development candidates and historical `frontier` spellings are
read-only compatibility paths.  Recipe definitions live in
`onecomp/kvcomp/codec/hayate/recipes.py`, codec backends in the adjacent
`codec.py`/`vq.py` modules, and benchmark orchestration in the drivers below.
`benchmark/kvcomp_common/paths.py` is the only shared path resolver: models
are never copied into onecomp and may be local paths, an external
`--models-root` entry, or a Hugging Face id.

For a reproducible multi-axis run, provide an external JSON manifest and keep
all generated evidence in a chosen output root:

```bash
.venv/bin/python benchmark/kvcomp-codec/run_paper_pipeline.py \
  --manifest benchmark/configs/paper_models.example.json \
  --models-root "$MODELS_ROOT" --output-root "$RESULTS_ROOT" \
  --run-local
```

Use `--dry-run` first (or omit `--run-local` to submit the child jobs to
Slurm).  The `codec`, `zeroshot`, and `longcontext` stages cover the paper
quality/retrieval/reasoning axes; `--stages render --aggregate <json>` emits
the validated LaTeX figure/table includes without embedding machine paths.
The example manifest contains references only, not model weights.

## Run

**Standard run:**
```bash
srun --gres=gpu:1 .venv/bin/python -u codec_benchmark.py \
    model_path=/lustre/.../models/llama-3.1-8b-instruct
```

**HayateQuant paper preset** (`conf/paper_hayatequant.yaml`, headline 4/3/2-bit,
multi-model):
```bash
# one model at a time (override model_path); ~30 min/model on one GPU
srun --gres=gpu:1 .venv/bin/python codec_benchmark.py --config-name paper_hayatequant \
    model_path=/.../models/llama-3.1-8b-instruct
srun --gres=gpu:1 .venv/bin/python codec_benchmark.py --config-name paper_hayatequant \
    model_path=/.../models/DeepSeek-R1-Distill-Qwen-7B
```

**Cross-model launcher / summarizer** (`hayatequant_cross_model.py`):
```bash
# submit the default paper grid (Llama2/3.1, Qwen3-4B/8B, Ministral,
# R1-Llama-8B, R1-Qwen-7B/1.5B)
.venv/bin/python benchmark/kvcomp-codec/hayatequant_cross_model.py --submit

# add extra coverage or run inside an interactive GPU allocation
.venv/bin/python benchmark/kvcomp-codec/hayatequant_cross_model.py \
    --models qwen3-14b r1-qwen-14b --submit

# summarize finished results_*.json files as a compact Markdown table
.venv/bin/python benchmark/kvcomp-codec/hayatequant_cross_model.py \
    --summarize --output-root /lustre/.../kvcomp_hayatequant/results_cross_model_v1
```

**Zero-shot downstream grid** (`hayatequant_zeroshot.py`):
```bash
# same eight models as the main PPL table; ARC-Easy, ARC-Challenge, PIQA,
# HellaSwag, and WinoGrande; reports mean acc/acc_norm for TQ and HQ variants
.venv/bin/python benchmark/kvcomp-codec/hayatequant_zeroshot.py --submit \
    --output-root /lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/kvcomp_hayatequant/zeroshot_v1

# quick smoke test only; do not use --limit for final paper numbers
.venv/bin/python benchmark/kvcomp-codec/hayatequant_zeroshot.py --run-local \
    --models qwen3-0.6b --tasks piqa --limit 4

# summarize completed zeroshot_*.json files
.venv/bin/python benchmark/kvcomp-codec/hayatequant_zeroshot.py --summarize \
    --output-root /lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/kvcomp_hayatequant/zeroshot_v1
```

**Native vLLM kernel smoke** (`hayatequant_vllm_native_smoke.py`):
```bash
# compiles and runs the byte-compatible TurboQuant-layout HayateQuant key
# block-VQ store/decode kernels for 3-bit and 4-bit presets; no model load
.venv-vllm/bin/python benchmark/kvcomp-codec/hayatequant_vllm_native_smoke.py \
    --output /lustre/.../kvcomp_hayatequant/native_vllm/smoke.json
```

**Full HayateQuant paper battery** (`submit_hayatequant_paper.sh`):
```bash
# Submit the fixed eight-job smoke plan first. Formal submission remains
# disabled until every exact output finishes and is explicitly approved.
RUN_ROOT=/lustre/.../kvcomp_hayatequant/paper_final_YYYYMMDD \
PARTITIONS=batch-1gpu \
  bash benchmark/kvcomp-codec/submit_hayatequant_paper.sh smoke

# Inspect results/smoke and the Slurm logs, then create the validated marker.
FORMAL_SMOKE_APPROVAL=I_APPROVE_THE_VALIDATED_HAYATEQUANT_SMOKE_PLAN \
  bash benchmark/kvcomp-codec/submit_hayatequant_paper.sh approve-smoke

# Every formal command re-hashes the approved smoke results and fails closed
# if the commit, input manifest, plan, result bytes, or marker has changed.
bash benchmark/kvcomp-codec/submit_hayatequant_paper.sh formal

# Individual formal slices use the same mandatory smoke approval gate.
bash benchmark/kvcomp-codec/submit_hayatequant_paper.sh ppl
bash benchmark/kvcomp-codec/submit_hayatequant_paper.sh zeroshot
bash benchmark/kvcomp-codec/submit_hayatequant_paper.sh longbench
bash benchmark/kvcomp-codec/submit_hayatequant_paper.sh speed
```

The formal long-code slice is specifically the released **LongCodeQA code
comprehension subtask** (all model-feasible brackets).  It is not a result for
the whole LongCodeBench suite and does not include LongSWE-Bench.

(`conf/paper_halo.yaml` is kept only to reproduce the historical HALO runs.)

Config: `conf/benchmark_codec.yaml`. Metrics per bit-width {2,3,4}: distortion
(relMSE on captured real KV), WikiText-2 PPL under `CompressedDynamicCache`,
and, for methods with an actual timing contract, overhead (ms/encode, Melem/s)
at the model's true head dim. `FibQuant-paper` and `BlockQuant-paper` emit an
explicit speed-omission marker instead of timing their unpacked exact-search
research paths. Results →
`output_dir` (JSON + Markdown).

Teacher-forced PPL and zero-shot likelihood are causal at the *codec-statistic*
level, not merely at the attention-mask level.  A channel-axis sequence group
(KIVI/NVFP4/Ternary), dynamic outlier-channel selection, online HQ bit split,
enabled HALO block preconditioner, or nonzero trailing FP window is evaluated
by pre-filling only the unscored prefix and then advancing scored targets one token at a time.  This
prevents a future target from changing an earlier scale, mask, or allocation.
The formal `TurboQuant-paper`, `FibQuant-paper`, and fixed `HQ` recipes use token
axes, no dynamic outlier split, no online bit allocation, and no block-global
preconditioner, so they retain the faster single-forward quality-evaluation
path. `AdaptiveHQ` makes causal allocation decisions, and `BlockQuant-paper`
selects top-32 key channels over each cache update; both therefore use causal
prefix prefill plus token-step targets. Regression tests assert both protocols.

## Result status

The previous unversioned headline table was removed because it predates the
current result schema, dataset fingerprints, fair speed controls, and fixed HQ
contract. Publishable values must come from a fresh run whose JSON
records the code commit, source/dataset fingerprint, model configuration,
sample denominator, and hardware provenance.

## Historical HALO results (not current evidence)

The old unversioned HALO measurements were also removed from this current-run
README. Historical bundles remain in their archived result directories, but
must not be cited as results from the fingerprinted benchmark contract.

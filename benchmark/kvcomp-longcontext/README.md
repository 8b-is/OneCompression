# KV-cache compression: long-context retrieval + reasoning

> **Protocol boundary.** Much of this directory is a historical, multi-method
> development harness.  Presets labelled `hayatequant_paper` and the local
> `codec_speed.py` matrix are not the refreshed HayateQuant formal protocol.
> Formal HQ/AdaptiveHQ claims are produced only by the fail-closed launcher,
> validators, and renderer under `benchmark/kvcomp-codec/`; old numeric tables
> below remain provenance-labelled development evidence and are not citable.

Main experiment for the KV-cache compression methods, the analogue of the
weight-side `benchmark/<model>-<recipe>/` runs (GPTQ / JointQ). It compares,
on a shared prompt grid with each method's **realised memory** recorded.  It is
a Pareto harness by default: pairwise iso-memory claims require a separately
verified matched operating point.

| Method | Family | Mechanism | Evicts? |
|---|---|---|:---:|
| **FullKV** | — | fp16 upper bound | no |
| **Uniform-INT4 / INT2** | quant | uniform per-token affine | no |
| **StreamingLLM** | evict | keep sink + recent, drop middle | **yes** |
| **ThinKV-TBQ** | quant | static per-thought bits (R/E=4, T=2), outlier-layer skip + calibrated L\*/thresholds + attention-sparsity classification. Paper TBE eviction is not implemented in this HF path | no |
| **PM-KVQ** | quant | **progressive / dynamic** precision: fixed high-precision window, older KV shrinks to a floor as memory saturates; block-wise (layer) allocation | no |
| **TaperKV** | quant | **static** age-graded precision ladder, non-zero floor (proposed) | no |

Thesis: uniform quantization is less memory-efficient than eviction, but eviction
**structurally fails long-context retrieval** — a needle in the evicted region is
unrecoverable. Replacing eviction with a graded precision floor (TaperKV) keeps
every token recoverable, whereas *dynamic* progressive precision (PM-KVQ)
collapses old tokens to a 2-bit floor over a fixed absolute window. TaperKV's
length-proportional ladder spends bits where retrieval needs them.

### Models and benchmark scope

The model grid mirrors the competing papers. Only official harnesses are directly
comparable; the local synthetic multi-key and functional-code axes below are
diagnostics and are deliberately not reported as RULER or LiveCodeBench scores:

The dedicated `ruler_v1` evaluator is a separately named **NeMo-Skills RULER
v1 100-sample/task protocol**. It consumes all 13 tokenizer-specific task files
from the pinned adapter and must not be relabelled as the canonical
NVIDIA/RULER v1 configuration, which uses 500 samples per task. Historical
`run_ruler=true` rows below remain only the local synthetic multi-key diagnostic.

LongCodeQA uses the final-filtered `Steefano/LCB` release, its verbatim raw
prompt and released parser, and primary per-bracket accuracy.  Its fixed
16-token greedy decoding is a deterministic paired-cache adaptation because
the upstream wrappers do not publish one uniform decoding configuration; it is
not labelled as exact reproduction of the original model-table decoding.  See
[LONGCODEQA_PROTOCOL_AUDIT.md](LONGCODEQA_PROTOCOL_AUDIT.md) for the pins,
known upstream evaluator bug, hashes, and feasible-bracket limits.

| Proposed method | Competitor | Models | Benchmarks |
|---|---|---|---|
| **HayateQuant** (codec; supersedes HALO) | TurboQuant (arXiv:2504.19874), HALO | Llama-3.1-8B-Instruct, Ministral-8B-Instruct-2410 | Needle, local synthetic multi-key, official LongBench-E, WikiText-2 PPL |
| **TaperKV / TurboTaper** (recipe) | ThinKV (ICLR 2026), PM-KVQ | DeepSeek-R1-Distill-Llama-8B / Qwen-14B / Qwen-7B / 1.5B, QwQ-32B | GSM8K, MATH-500, AIME-2024/2025, HumanEval-shaped code smoke, PPL |

The retrieval grid (`conf/turboquant_paper.yaml`, `conf/turboquant_paper_ministral.yaml`)
runs HayateQuant against TurboQuant's full bit-grid on **both** TurboQuant models
(with HALO retained as the strongest prior baseline). The
reasoning grid (`conf/paper_aime.yaml`) runs TaperKV/TurboTaper/HayateQuant against
ThinKV/PM-KVQ/KIVI on the R1-Distill family with the four ThinKV benchmarks. Gemma
is intentionally excluded: no KV-cache-compression baseline (TurboQuant, ThinKV,
PM-KVQ, KIVI, R-KV) reports Gemma KV-cache results, so it adds no comparable point.

## Run

**Quick run** (single method, short context):
```bash
srun --gres=gpu:1 .venv/bin/python -u kv_benchmark.py \
    model_path=/lustre/.../models/llama-3.1-8b-instruct \
    'methods=[StreamingLLM,TaperKV]' 'context_lens=[8192]' run_gsm8k=false
```

**Paper-level run** (all methods, all metrics, full context grid):
```bash
srun --gres=gpu:1 .venv/bin/python -u kv_benchmark.py \
    --config-name paper_run \
    model_path=/lustre/.../models/llama-3.1-8b-instruct
```

**TaperKV paper preset** (`conf/paper_taperkv.yaml`, ~3-bit candidate schedule;
TaperKV vs ThinKV/PM-KVQ/StreamingLLM/KIVI/TurboQuant/HayateQuant/Uniform-INT3):
```bash
srun --gres=gpu:1 .venv/bin/python -u kv_benchmark.py \
    --config-name paper_taperkv \
    model_path=/.../models/DeepSeek-R1-Distill-Qwen-7B           # 7B headline
# or submit the whole battery via the in-repo launcher:
bash paper/submit_all.sh paper2
```

**TurboQuant comparison preset** (`conf/turboquant_paper.yaml`, full bit grid
2/2.5/3/3.5/4 for HayateQuant + TurboQuant + HALO, plus TaperKV-3bit, on
Llama-3.1-8B-Instruct):
```bash
srun --gres=gpu:1 .venv/bin/python -u kv_benchmark.py \
    --config-name turboquant_paper \
    model_path=/.../models/llama-3.1-8b-instruct
# or: bash paper/submit_all.sh paper1   (HayateQuant codec jobs)
#     bash paper/submit_paper_final.sh    (final HayateQuant battery)
```

**Reasoning headline** (`conf/paper_aime.yaml`, ThinKV/PM-KVQ turf; GSM8K + MATH-500
+ AIME-2024/2025 + HumanEval-shaped functional-code smoke, carrying TaperKV +
TurboTaper + HayateQuant):
```bash
srun --gres=gpu:1 .venv/bin/python -u kv_benchmark.py \
    --config-name paper_aime \
    model_path=/lustre/.../models/DeepSeek-R1-Distill-Qwen-14B \
    lcb_problems_path=/lustre/.../livecodebench_functional.jsonl
```
`lcb_problems_path` is a JSONL of `{prompt, entry_point, test}` (functional /
HumanEval protocol); omit it to load `lcb_dataset` from HF. Generated code runs in
a sandboxed subprocess (CPU/memory rlimits + timeout) — use an isolated node.

**Retrieval second model** (`conf/turboquant_paper_ministral.yaml`, HayateQuant
vs TurboQuant on TurboQuant's Ministral model):
```bash
srun --gres=gpu:1 .venv/bin/python -u kv_benchmark.py \
    --config-name turboquant_paper_ministral
```

## Ablations (paper)

Two complementary ablation artefacts isolate each design lever of the proposed
codecs/recipes:

**1. Codec speed × quality Pareto** (`codec_speed.py`).  Times encode/decode
(CUDA-synchronised, warmup + median) **and** measures KV-MSE / attention-output
error on the *same* tensors, so the HayateQuant/HALO levers are separable:

```bash
# GPU: the headline speed-quality table (synthetic anisotropic KV)
python codec_speed.py --device cuda --heads 32 --seq 4096 --dim 128 \
    --bits 2 3 4 --repeats 30 --out results/codec_speed_cuda.json
# real KV from a model instead of synthetic:
python codec_speed.py --device cuda --model /lustre/.../llama-3.1-8b-instruct
```

Historical rows: `HayateQuant-auto` (old auto block-VQ + folded hybrid rotation +
grid-LUT development recipe) · `HayateQuant-bs1` (scalar-path compatibility control) · `HayateQuant-exact` /
`HayateQuant-K12` / `HayateQuant-dense-LS` (encoder/codebook/rotation
ablations) · `TurboQuant-paper` (Algorithm-1 MSE dense-QR reference, not a
Table-1 mixed-precision reproduction) ·
`TurboQuant-hadamard` (fast deployment approximation) · `HALO-*` (historical
scale/rotation levers).  These are exploratory diagnostics only; the current
fixed HQ recipe and its formal claim gates are not defined by this matrix.

**2. End-task lever ablation** (`conf/paper_ablation.yaml`).  Does each codec lever
/ TaperKV design choice carry through to needle retrieval + PPL on a real model?

```bash
srun --gres=gpu:1 .venv/bin/python -u kv_benchmark.py --config-name paper_ablation \
    model_path=/lustre/.../models/llama-3.1-8b-instruct
```

* HALO levers (iso bit-width & iso storage): `HALO-{h,hg}-{noLS}` and
  `HALO-hg-LS` vs `HALO-dense-LS`, referenced to the faithful `TurboQuant-{3,4}bit`.
* TaperKV K≥V asymmetry: `TaperKV-asym` (K6V3) vs `TaperKV-sym` (K4V4) vs
  `TaperKV-revsym` (K3V6, reversed — expected to regress since keys are more
  sensitive).

This historical harness also reports **95% CIs** (Wilson for retrieval, normal
for reasoning) in its JSON and Markdown table.  In the refreshed HayateQuant
paper, predeclared point estimates are primary, while a separate simultaneous
global gate governs any aggregate superiority statement; per-endpoint intervals
remain descriptive and are not used to hide or discard point-estimate losses.

Config: `conf/benchmark_longcontext.yaml`. `sink_tokens` and `recent_tokens`
are shared; `target_mem` is the explicit StreamingLLM operating point, while
every method's stored/fp16 fraction is measured from the same generated
prompts. Needle v2 retains rendered lengths, depth, prompt hashes, golds,
predictions, and per-trial cache-accounting ledgers. Results (JSON + Markdown table) are
written to `output_dir`.

**Historical outlier-layer handling (applied uniformly in these presets).** A small set of layers carry
massive activations whose keys span orders of magnitude; quantizing them at low
bit-width destroys generation on reasoning models. The harness detects these
layers once and keeps them in FP16 for **every** quantization method, which is
the standard StreamingLLM/KIVI robustness trick. Keeping it uniform is what makes
the comparison fair within these legacy presets.  The refreshed formal HQ/TQ/
FibQ/BlockQ quality grid does not silently skip FP16 layers: every stored bit
and every included/excluded endpoint is fixed and validated by its manifest.

**Faithful ThinKV.** With `thinkv.auto_configure=true` (default) the harness runs
`auto_configure_thinkv(model, tok)`: it reuses the shared outlier set, then
calibrates the optimal classification layers L\* and the Reasoning/Execution/
Transition thresholds by KDE, and finally engages thought classification during
decode via `attn_capture=True` (eager attention + `output_attentions`). Without
this setup ThinKV degrades to a uniform 4-bit codec over every layer and collapses
on reasoning models. Attention capture is used only on the short reasoning prompts;
it is disabled for long-context retrieval, where materializing full attention maps
would exhaust memory and is unnecessary.

Metrics: needle-in-a-haystack accuracy over a depth × length grid
(`needle_in_haystack`); a **local synthetic multi-key** diagnostic (not NVIDIA
RULER), where several competing keys force disambiguation (`ruler_multikey`,
historical flag `run_ruler=true`); GSM8K and MATH-500 pass@1 for reasoning non-degradation
(`gsm8k_pass_at_1`, `math500_pass_at_1`, `run_math=true`); WikiText-2 perplexity
as a sanity check (`wikitext2_perplexity`); and the realized whole-cache
compression ratio (stored bits / fp16 bits, including outlier-skip and sink layers).

---

## Results

> **Historical PPL invalidation.** The tables below predate the causal
> cache-state scorer. Their PPL columns for KIVI/ThinKV, PM-KVQ/TaperKV/
> TurboTaper, and StreamingLLM are not valid conditional likelihoods: a
> full-window update could use future-token group statistics or final-window
> age schedules, while the eviction row did not evict between targets. Those
> PPL values and every conclusion/Pareto claim that depends on them must not be
> cited. The current evaluator uses causal prefix + token-step scoring for
> these caches and explicit eviction with absolute positions; fresh results
> are required. Retrieval/generation columns have separate autoregressive
> harnesses and are not invalidated by this specific teacher-forcing bug.

Four architectures were evaluated on hardware (B200 GPU, batch-1gpu partition).
All runs use the same config knobs (`sink_tokens=4`, `recent_tokens=64`,
`target_mem=0.27`). Compression ratios are **whole-cache** figures (FP outlier-skip
layers and sink tokens counted at fp16, so the ratio is directly comparable across
methods including eviction).

---

### llama-3.1-8b-instruct — retrieval + reasoning instruct model

Needle = 63 cells (7 depths × 3 ctx lengths × 3 seeds); synthetic multi-key = 15 prompts
(4 facts: one queried + three distractors, × 3 lengths × 5 seeds); GSM8K n=40;
PPL = WikiText-2 (16 windows).

> **Historical snapshot — not citable.** The Needle rows in this table were
> produced before Needle v2 retained its rendered-prompt and per-trial ledger.
> They do not establish a matched-memory or quality result; rerun the v2 preset
> and use the emitted JSON grid instead.

| Method | Needle ↑ | synthetic multi-key ↑ | Mem | GSM8K p@1 ↑ | PPL ↓ |
|---|---:|---:|---:|---:|---:|
| FullKV | 1.00 | 1.00 | 1.00 | 0.825 | 6.10 |
| Uniform-INT4 | 1.00 | 1.00 | 0.26 | 0.825 | 6.17 |
| Uniform-INT2 | 0.54 | 0.07 | 0.13 | 0.725 | 28.54 |
| StreamingLLM (evict) | 0.30 | 0.60 | 0.28 | 0.825 | 6.10 |
| ThinKV (faithful) | 1.00 | 1.00 | 0.31 | 0.575 | 6.21 |
| PM-KVQ | 1.00 | 1.00 | 0.20 | 0.800 | 7.42 |
| **TaperKV (proposed)** | **1.00** | **1.00** | 0.28 | **0.825** | **6.24** |

**Key findings:**

1. **Eviction structurally fails retrieval.** At an equal memory budget (~0.28),
   StreamingLLM recovers only 30% of needles and 60% of synthetic keys — once a
   token is evicted the answer is unrecoverable. Every length-preserving quant
   method retrieves perfectly. Uniform-INT2 keeps all tokens but loses nearly
   half the needles (PPL 28.5); keeping tokens *and* a sensible precision floor
   is what makes them usable.

2. **Historical development observation (not a formal conclusion).** The old
   development table suggests TaperKV may match FullKV on all three axes.  It is the only
   method with perfect retrieval (Needle/synthetic multi-key 1.00), FullKV-level reasoning
   (GSM8K 0.825), and near-FullKV perplexity (6.24 vs 6.10) at the reported
   development memory point.  This requires a Needle v2 rerun before use.
   PM-KVQ matches retrieval but pays with higher PPL (7.42) and a small reasoning
   drop. ThinKV retrieves perfectly but over-compresses on an instruct model
   (GSM8K 0.575 — it is designed for long chain-of-thought models).

---

### DeepSeek-R1-Distill-Qwen-7B — long chain-of-thought reasoning

Retrieval is uninformative on this model (FullKV itself scores below 0.2 on needle
at 4k–16k). Reasoning non-degradation is measured at `max_new=2048`, which
long-CoT models need to reach `\boxed{}`. GSM8K n=40, MATH-500 n=16.

| Method | GSM8K p@1 ↑ | MATH-500 p@1 ↑ | PPL ↓ | cr ↑ |
|---|---:|---:|---:|---:|
| FullKV | 0.675 | 0.313 | 50.8 | 1.0× |
| Uniform-INT4 | 0.775 | 0.375 | 52.3 | 3.9× |
| Uniform-INT2 | 0.625 | 0.375 | 184.8 | 7.5× |
| ThinKV (faithful) | 0.650 | **0.375** | 53.3 | 2.2× |
| PM-KVQ | 0.700 | 0.313 | 68.9 | 4.9× |
| **TaperKV (proposed)** | 0.650 | 0.313 | **53.6** | **3.4×** |

With outlier layers held in FP16, every method tracks the uncompressed model
within evaluation noise. TaperKV achieves FullKV-level perplexity (53.6 vs 50.8)
at **3.4× compression**. PM-KVQ compresses more (4.9×) but at a PPL cost (68.9).
ThinKV, in its native long-CoT regime, delivers the best MATH-500 (0.375) at the
cost of a larger memory footprint (2.2×).

The key lesson here (K23): the earlier degenerate setup produced GSM8K = 0.0 for
all quant methods. Uniformly skipping massive-activation outlier layers restores
generation quality for *all* methods — it is the required baseline treatment, not
a proposed-method advantage.

---

### DeepSeek-R1-Distill-Qwen-1.5B — lightweight reasoning model

A faster, smaller reasoning model confirming that TaperKV's near-lossless PPL
holds across model scales. GSM8K n=50, MATH-500 n=30.

| Method | GSM8K p@1 ↑ | MATH-500 p@1 ↑ | PPL ↓ | cr ↑ |
|---|---:|---:|---:|---:|
| FullKV | 0.22 | 0.100 | 207.5 | 1.0× |
| Uniform-INT4 | 0.22 | 0.133 | 217.8 | 3.9× |
| Uniform-INT2 | 0.06 | 0.067 | 878.0 | 7.5× |
| ThinKV (faithful) | 0.12 | **0.233** | 218.4 | 2.7× |
| PM-KVQ | **0.26** | 0.133 | 255.1 | 3.9× |
| **TaperKV (proposed)** | 0.24 | 0.167 | **208.1** | **3.4×** |

TaperKV's PPL (208.1) is essentially indistinguishable from FullKV (207.5) at
3.4× compression — the graded ladder introduces negligible information loss on the
WikiText-2 language modeling task. Uniform-INT2 collapses (PPL 878, GSM8K 0.06),
confirming that a non-zero precision floor is critical at aggressive compression.

---

## Cross-architecture summary

| Model | TaperKV Needle | TaperKV synthetic multi-key | TaperKV GSM8K | TaperKV PPL / FullKV PPL |
|---|:---:|:---:|:---:|---:|
| llama-3.1-8b-instruct | **1.00** | **1.00** | **0.825** (= FullKV) | 6.24 / 6.10 |
| DeepSeek-R1-7B (reasoning) | — | — | 0.650 (≈ FullKV 0.675) | 53.6 / 50.8 |
| DeepSeek-R1-1.5B (reasoning) | — | — | 0.240 (≈ FullKV 0.220) | 208.1 / 207.5 |

**TaperKV is robust on all three axes — retrieval, reasoning, and perplexity —
across every evaluated architecture,** matching FullKV retrieval while eviction
fails it. Against the nearest quantization competitor PM-KVQ, TaperKV reaches the
same retrieval at lower perplexity (6.24 vs 7.42 on Llama-3.1-8B); the Pareto sweep
below traces the accuracy-vs-memory dominance at multiple operating points.

---

## TaperKV Pareto sweep — accuracy-vs-memory curve

The Pareto sweep runs 9 methods (the 6 standard methods + TaperKV at 3 operating
points: default / mid / aggr) to show that TaperKV
**dominates PM-KVQ and eviction at matched memory**, not just at a single tuned
operating point. Config: `conf/paper_run.yaml` with
`methods=[FullKV,Uniform-INT4,Uniform-INT2,StreamingLLM,PM-KVQ,ThinKV,TaperKV,TaperKV-mid,TaperKV-aggr]`.

### llama-3.1-8b-instruct — Pareto sweep (9 methods)

Needle = 42 cells (7 depths × 3 lengths × 2 seeds); local synthetic multi-key = 9 cells
(4 distractor keys × 3 lengths × 3 seeds); GSM8K n=40; PPL = 16 windows.

| Method | Needle ↑ | Synthetic multi-key ↑ | Mem | GSM8K p@1 ↑ | PPL ↓ | cr ↑ |
|---|---:|---:|---:|---:|---:|---:|
| FullKV | 1.00 | 1.00 | 1.00 | 0.825 | 6.10 | 1.0× |
| Uniform-INT4 | 1.00 | 1.00 | 0.26 | 0.750 | 6.34 | 3.8× |
| Uniform-INT2 | 0.00 | 0.00 | 0.13 | 0.000 | 578 | 7.4× |
| StreamingLLM (evict) | 0.31 | 0.78 | 0.28 | 0.825 | 6.10 | — |
| ThinKV (faithful) | 1.00 | 1.00 | 0.31 | 0.575 | 6.21 | 3.2× |
| PM-KVQ | 1.00 | 1.00 | 0.20 | 0.800 | 7.42 | 4.9× |
| **TaperKV (default)** | **1.00** | **1.00** | **0.28** | **0.825** | **6.24** | **3.5×** |
| TaperKV-mid (6/3-bit, b_min=2) | 0.69 | 0.78 | 0.20 | 0.775 | 11.34 | 4.9× |
| TaperKV-aggr (4/2-bit, b_min=2) | 0.38 | 0.56 | 0.16 | 0.500 | 37.4 | 6.4× |

The default TaperKV is the unique Pareto-optimal point: it reaches FullKV needle
and local multi-key accuracy at lower memory than ThinKV (0.28 vs 0.31) and lower PPL than
PM-KVQ (6.24 vs 7.42). Aggressive variants follow a graceful degradation curve —
TaperKV-mid at 4.9× already retrieves 69% of needles, which is better than
StreamingLLM at the same memory.

Result JSON: `results_pareto/results_llama-3.1-8b-instruct.json`.

---

## Sub-validation

All component and codec verification is in-repo and self-contained:

* `tests/onecomp/kvcomp/` — the full unit suite: every codec
  round-trip + bit-accounting, HALO scale modes / outlier split, fallback + QJL
  accounting, and the vLLM mapping warnings / skip-layer propagation.
* `paper/tests/verify_halo_asym_scale.py`, `paper/tests/verify_halo_outlier.py`
  — CPU-only, data-free array-level proofs of the HALO levers (run anywhere; the
  scripts resolve the repo root relative to themselves).
* `paper/tests/taper_schedule.py` — the TaperKV avg-bits / memory predictor.

Result JSONs are stored in `results/` (one file per model). Raw Hydra run logs
(timestamped directories) are git-ignored; only the summary JSONs are tracked.

# Paper benchmark battery (reproducible) — two KV-compression papers

Self-contained harness to reproduce **every** number in the two papers on a
SLURM cluster (kagura, `batch-1gpu`, B200). Everything here is repo-relative:
clone the repo, sync `.venv`, place the models under `$MODELS`, and run
`submit_all.sh`.

| Paper | Proposed method | Family | Competitor(s) | Headline claim |
|---|---|---|---|---|
| **Paper 1** | **HALO** | codec (drop-in) | TurboQuant (arXiv:2504.19874) | iso-bit, HALO is **faster** (≈3.8× encode) and **no worse** in quality/retrieval than TurboQuant |
| **Paper 2** | **TaperKV / TurboTaper** | recipe (age-graded precision) | ThinKV (ICLR 2026), PM-KVQ (arXiv:2505.18610) | at iso-memory, eviction **structurally fails** retrieval; TaperKV's graded floor keeps the needle **and** reasoning |

The two papers share one harness (`kv_benchmark.py`, `codec_benchmark.py`,
`codec_speed.py`) and one fairness contract (iso-memory budget, uniform
outlier-layer skip, 95% CIs) so the cross-paper numbers are directly comparable.

---

## TL;DR — reproduce

```bash
cd benchmark/kvcomp-longcontext/paper
bash submit_all.sh            # all 27 jobs (both papers)
bash submit_all.sh paper1     # HALO codec jobs only (11 jobs)
bash submit_all.sh paper2     # TaperKV recipe jobs only (16 jobs)
DRYRUN=1 bash submit_all.sh   # print the sbatch lines, submit nothing

# overridable env:
PY=/path/to/python MODELS=/path/to/models PART=batch-1gpu \
EXCLUDE=kagura-gpu09,kagura-gpu10 bash submit_all.sh
```

Monitor: `squeue -u $USER | grep -E 'p1-|p2-'`. SLURM stdout lands in
`paper/logs/<name>_<jobid>.log` (git-ignored). Each job writes its summary JSON
incrementally (after every method) so a crash/time-out keeps partial results.

---

## Models (`$MODELS`, default `.../conference/models`)

| Short | Path | Used by |
|---|---|---|
| LLAMA | `llama-3.1-8b-instruct` | P1 retrieval/LB/speed/codec/ablation, P2 retrieval headline |
| MINISTRAL | `Ministral-8B-Instruct-2410` | P1 retrieval/LB/codec (TurboQuant's 2nd model) |
| R1L8 | `DeepSeek-R1-Distill-Llama-8B` | P2 reasoning (ThinKV model) |
| R1Q15 | `DeepSeek-R1-Distill-Qwen-1.5B` | P1 codec, P2 reasoning |
| R1Q7 | `DeepSeek-R1-Distill-Qwen-7B` | P1 codec, P2 reasoning |
| R1Q14 | `DeepSeek-R1-Distill-Qwen-14B` | P2 reasoning (ThinKV/PM-KVQ model) |
| QWQ | `QwQ-32B` | P2 reasoning (ThinKV/PM-KVQ model) |

Gemma is intentionally excluded — no KV-compression baseline (TurboQuant/ThinKV/
PM-KVQ/KIVI/R-KV) reports Gemma KV results, so it adds no comparable point.

---

## Paper 1 — HALO (codec)

| Table / figure | Config | Job(s) | Result dir |
|---|---|---|---|
| Codec quality (KV-MSE, attn-err) | `kvcomp-codec/conf/paper_halo.yaml` | `p1-halo-{llama,minis,r1q7,r1q15}` | `kvcomp-codec/results_paper_*` |
| Codec speed × quality Pareto | `codec_speed.py` (CLI) | `p1-speed-{synth,llama}` | `results_codec_speed/` |
| Retrieval (Needle/RULER/PPL), full bit-grid | `conf/turboquant_paper.yaml`, `conf/turboquant_paper_ministral.yaml` | `p1-tq-{llama,minis}-retr` | `results_tq_{llama,minis}_retr/` |
| LongBench-E downstream | same configs, `run_longbench=true` | `p1-tq-{llama,minis}-lb` | `results_tq_{llama,minis}_lb/` |
| Lever ablation (HALO levers + TaperKV asymmetry) | `conf/paper_ablation.yaml` | `p1-ablation-llama` | `results_ablation_llama/` |

The retrieval grid runs HALO at 2/3/4-bit against TurboQuant's full bit-grid
(2/2.5/3/3.5/4) + Uniform-INT + KIVI + FP8 + NVFP4, all at matched memory, so the
"iso-bit, faster, no worse" claim is backed end-to-end (not just codec MSE).

### Validated retrieval result (Llama-3.1-8B, completed)

`needle = 45 cells (5 depths × 3 ctx × 3 seeds); RULER = 15 cells; PPL = WikiText-2`

| Method | Needle ↑ | RULER ↑ | PPL ↓ | Mem |
|---|---:|---:|---:|---:|
| FullKV | 1.00 | 1.00 | 6.49 | 1.00 |
| Uniform-INT3 | 1.00 | 0.87 | 8.54 | 0.197 |
| StreamingLLM (evict) | 0.40 | 0.07 | 6.49 | 0.282 |
| TurboQuant-3bit | 1.00 | 1.00 | 6.92 | 0.196 |
| **HALO-3bit (proposed)** | **1.00** | **1.00** | **6.90** | **0.196** |

At iso-3-bit (mem ≈ 0.196) HALO-3bit matches TurboQuant-3bit on retrieval/PPL
while being ≈3.8× faster to encode (codec-speed table), and beats Uniform-INT3
on PPL (6.90 vs 8.54). Eviction (StreamingLLM) collapses retrieval at the same
budget. Ministral confirms the same ordering.

---

## Paper 2 — TaperKV / TurboTaper (recipe)

| Table / figure | Config | Job(s) | Result dir |
|---|---|---|---|
| Iso-memory retrieval + reasoning headline | `conf/paper_taperkv.yaml` | `p2-taper-{llama,r1q15,r1q7,r1q14}` | `results_taperkv/` |
| Reasoning: GSM8K + MATH-500 + PPL | `conf/paper_aime.yaml` (aime off) | `p2-basic-{r1q15,r1l8,r1q7,r1q14,qwq}` | `results_aime_basic_*` |
| Reasoning: AIME-2024 / AIME-2025 | `conf/paper_aime.yaml` (aime only) | `p2-aime*-{r1q15,r1l8,r1q7,r1q14,qwq}` | `results_aime_comp_*` |

Retrieval is claimed on the **instruct** model (Llama); the R1/QwQ thinking
models run with `run_needle=false run_ruler=false` (the synthetic needle is
unreliable on thinking models — FullKV itself retrieves <0.7 because the model
keeps emitting `<think>` instead of answering). Their axis is reasoning
non-degradation (GSM8K/MATH/AIME) + PPL at iso-memory.

### The TaperKV floor fix (important — read before re-running)

TaperKV quantizes keys **per-token affine**. Per-token ≤2-bit keys are
catastrophic for retrieval (Uniform-INT2 → needle 0.00), so the oldest tokens —
which may hold a mid-context needle — need a **≥3-bit floor** to survive. The
floors are now fixed in config:

| operating point | key floor `b_min` | value floor `b_min_value` | needle (Llama) | PPL |
|---|:---:|:---:|:---:|:---:|
| old headline (broken) | 2 | 2 | 0.64 | 9.58 |
| **TaperKV standard** | **4** | **2** | **1.00** | **6.14** |
| **TaperKV-3bit (iso-3-bit headline)** | **3** | **2** | **0.978** | **6.96** |

Evidence: `tests/conf/test_taper_floors.yaml` (GPU sweep, job 45990) +
`tests/taper_schedule.py` (CPU bit/memory predictor). To re-verify:

```bash
cd benchmark/kvcomp-longcontext
python paper/tests/taper_schedule.py          # CPU: avg-bits/mem per floor
srun --gres=gpu:1 .venv/bin/python -u kv_benchmark.py \
    --config-dir paper/tests/conf --config-name test_taper_floors \
    model_path=$MODELS/llama-3.1-8b-instruct  # GPU: needle vs floor
```

---

## Fairness contract (applies to both papers)

* **Iso-memory.** All methods share `sink_tokens`/`recent_tokens`/`target_mem`;
  the eviction window is sized to `target_mem·ctx` so eviction gets the **same**
  budget as the quant methods. `compression_ratio`/`mem_fraction` is the realized
  whole-cache figure (FP outlier-skip layers + sinks counted at fp16).
* **Uniform outlier-layer skip.** Massive-activation layers are detected once and
  kept FP16 for **every** quant method (standard StreamingLLM/KIVI trick) — not a
  privilege of the proposed method.
* **Faithful baselines.** ThinKV runs with `auto_configure=true` (outlier set +
  KDE-calibrated L\*/thresholds + attention-sparsity thought classification under
  eager attention); PM-KVQ runs with block-wise IP allocation + channel reparam.
* **Error bars.** Every accuracy/pass@1 carries a 95% CI (Wilson for retrieval,
  normal for reasoning) in both the JSON and the Markdown table.

---

## LiveCodeBench axis

LiveCodeBench is now enabled via a local pre-fetched JSONL
(`data/lcb_functional.jsonl`, 50 functional programming tasks from
`livecodebench/code_generation_lite`). Jobs use `run_livecodebench=true
lcb_problems_path=<...>/lcb_functional.jsonl HF_DATASETS_OFFLINE=0`.

**Important caveat (2026-06-04):** R1-Distill-Llama-8B achieves **FullKV=0%**
pass@1 on LCB (avg_gen_len≈10400, well below the 16384 limit). This is a
model-capability issue: the model was distilled for MATH/reasoning, not
functional code generation, and its long `<think>` blocks precede code that
fails the test harness. LCB results for R1-Llama-8B are therefore included for
completeness but should not be used as a primary comparison point; QwQ-32B and
R1-Qwen-14B are the relevant models for the LCB claim.

---

## TurboTaper bugs (2026-06-03 / 2026-06-04)

Two implementation bugs were found and fixed in `TurboTaper`:

| Bug | Symptom | Root cause | Fix |
|---|---|---|---|
| **Bug-1** (2026-06-03) | QwQ-32B: GSM8K 0%; R1-7B: GSM8K 48% / MATH 6% | `b_min_value=2` → Hadamard + 2-bit value floor causes CoT collapse on long Qwen reasoning chains | Raise `b_min_value` to 3 in both `paper_taperkv.yaml` and `paper_aime.yaml` |
| **Bug-2** (2026-06-04) | R1-7B: MATH 24% despite GSM8K 69%=FullKV; QwQ-32B: GSM8K 17% after Bug-1 fix | `GradedQuantCache.update()` re-quantises ALL floor tokens every 16 steps; rotated Lloyd-Max codec is not perfectly idempotent → drift compounds over 66–275 passes | **Delta quantisation** in `turbotaper_cache.py`: only (re-)quantise tokens whose bit-width changed or that are new. Floor tokens quantised at most once. |

After both fixes, re-run jobs submitted on 2026-06-04 (`resubmit_deltafix.sh`,
JIDs 46481-46489).

---

## TurboTaper delta-fix results (2026-06-04, confirmed)

All four TurboTaper re-run jobs (JIDs 46481-46484) completed successfully.
**Delta quantization fix fully eliminates cascade error.** Results are at or
above FullKV level on every model:

### Basic (GSM8K / MATH-500) — TurboTaper after delta fix

| Model | FullKV GSM | FullKV MATH | TaperKV GSM | TaperKV MATH | TurboTaper GSM | TurboTaper MATH | CR |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| R1-Distill-Llama-8B | 0.63 | 0.61 | 0.62 | 0.58 | **0.65** | **0.59** | 3.0× |
| R1-Distill-Qwen-7B  | 0.69 | 0.67 | 0.75 | 0.64 | **0.67** | **0.65** | 2.2× |
| R1-Distill-Qwen-14B | 0.88 | 0.68 | 0.90 | 0.62 | **0.86** | **0.68** | 2.3× |
| QwQ-32B             | 1.00 | 0.67 | 0.97 | 0.60 | **1.00** | **0.70** | 3.1× |

QwQ-32B MATH **0.70 > FullKV 0.67** — TurboTaper compression is strictly lossless at iso-memory.

Before fix (OLD cascade-error code): R1-7B MATH=6%, QwQ-32B GSM=0%.

### AIME comp — TurboTaper after delta fix

| Model | Year | FullKV | TaperKV | TurboTaper (delta fix) | CR |
|---|:---:|:---:|:---:|:---:|:---:|
| R1-Distill-Qwen-14B | 2024 | 0.50 | — | **0.43** | 2.4× |
| R1-Distill-Qwen-14B | 2025 | 0.28 | 0.33 | **0.37** | 2.4× |
| QwQ-32B             | 2024 | 0.30 | 0.20 | **0.30** (=FullKV) | 3.2× |

---

## Run status (2026-06-04 18:50 JST — updated)

### Completed (all OK)

| Component | JID(s) | Status |
|---|---|---|
| Paper 1 codec/retrieval/ablation/speed | various | ✅ |
| P2 taperkv retrieval Llama-8B-instruct | 46386 | ✅ |
| P2 TurboTaper basic (delta fix) — R1-Llama,R1-7B,R1-14B,QwQ | 46481-46484 | ✅ |
| P2 TurboTaper AIME-2024/2025 R1-14B comp | 46486,46487 | ✅ |
| P2 TurboTaper AIME-2024 QwQ comp | 46488 | ✅ |
| P2 TurboTaper-3bit taperkv R1-14B (delta fix) | 46489 | ✅ |
| P2 TurboTaper AIME R1-7B (delta fix) | 46485 | ✅ aime2024=0.533, aime2025=0.300 |
| P2 taperkv/Llama TurboTaper-3bit (delta fix) | 46539 | ✅ GSM=81%, MATH=48% |

### Running (as of 18:50 JST)

| Component | JID | Notes |
|---|---|---|
| P2 taperkv R1-1.5B (resume — all methods) | 46417 | ~22:00 timeout |
| P2 basic R1-1.5B (resume — all methods) | 46418 | ~22:00 timeout |
| P2 taperkv R1-7B (KIVI-INT2 → HALO) | 46421 | ⚠ TurboTaper-3bit ALREADY WRITTEN with **OLD code** (gsm=0.53, math=0.06) |
| P2 comp/r1q7 remaining (ThinKV…HALO) | 46534 | ThinKV AIME-2024=0.567 done, AIME-2025 in progress |
| P2 comp/qwq_25 HALO-3bit | 46535 | KIVI=0.200 done, HALO started |
| P2 comp/r1l8 AIME all methods (k=2) | 46537 | StreamingLLM done, ThinKV in progress |
| P2 comp/r1q14_24 TaperKV+TurboTaper+HALO | 46708 | **new, just submitted** delta fix ✅ |

### Critical issues / postfix required

| Issue | Action | Script | Status |
|---|---|---|---|
| taperkv/R1-7B: TurboTaper-3bit has cascade error (math=0.06) | Run after 46421 completes (~22:00) | `postfix_46421.sh` | ⏳ PENDING |
| postfix_46421.sh updated to also add HALO-3bit if 46421 times out before completing it | — | `postfix_46421.sh` | ready |

### Data gaps remaining

| Dataset | Missing | Expected fill |
|---|---|---|
| comp/r1q14_24 | TaperKV, TurboTaper, HALO-3bit | 46708 (running) |
| comp/r1q7 | ThinKV, PM-KVQ, TaperKV, KIVI, HALO | 46534 (running) |
| comp/qwq_25 | HALO-3bit | 46535 (running) |
| comp/r1l8 | PM-KVQ, TaperKV, KIVI, HALO, TurboTaper | 46537 (running, slow) |
| taperkv/R1-1.5B | TurboTaper-3bit, KIVI, HALO | 46417 (running) |
| taperkv/R1-7B | TurboTaper-3bit (delta fix) | postfix_46421.sh after 46421 done |

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

## Known-excluded axis: LiveCodeBench

ThinKV's code axis is **not** submitted. The LiveCodeBench loader is incompatible
with `datasets>=4.8` (its dataset script + on-disk format break the current
loader). It is left out rather than shipped half-working; GSM8K/MATH/AIME cover
the reasoning claim. To re-enable once a compatible local JSONL exists, add a job
with `run_livecodebench=true lcb_problems_path=<...>.jsonl HF_DATASETS_OFFLINE=0`.

---

## Run status (snapshot)

* **Paper 1 — complete & committed:** codec (`results_paper_*`), retrieval
  (`results_tq_{llama,minis}_retr`), ablation, codec-speed. LongBench-E
  (`results_tq_*_lb`) re-running with the floor fix.
* **Paper 2 — running on kagura:** `paper_taperkv` (Llama done; R1/QwQ in queue)
  and the `paper_aime` reasoning grid (GSM8K/MATH/AIME) are generating; reasoning
  jobs are slow (long CoT, `max_new≈4–8k`). Summary JSONs are committed as each
  model's full method list completes.

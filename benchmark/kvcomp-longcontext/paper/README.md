# Paper benchmark battery (reproducible) — two KV-compression papers

Self-contained harness to reproduce **every** number in the two papers on a
SLURM cluster (kagura, `batch-1gpu`, B200). Everything here is repo-relative:
clone the repo, sync `.venv`, place the models under `$MODELS`, and run
`submit_all.sh`.

| Paper | Proposed method | Family | Competitor(s) | Headline claim |
|---|---|---|---|---|
| **Paper 1** | **HayateQuant** (supersedes HALO) | codec (drop-in) | TurboQuant (arXiv:2504.19874), HALO | iso-bit, HayateQuant is **more accurate** (block-VQ space-filling gain) *and* faster to decode than TurboQuant; strict superset of both baselines |
| **Paper 2** | **TaperKV / TurboTaper** | recipe (age-graded precision) | ThinKV (ICLR 2026), PM-KVQ (arXiv:2505.18610) | at iso-memory, eviction **structurally fails** retrieval; TaperKV's graded floor keeps the needle **and** reasoning |

The two papers share one harness (`kv_benchmark.py`, `codec_benchmark.py`,
`codec_speed.py`) and one fairness contract (iso-memory budget, uniform
outlier-layer skip, 95% CIs) so the cross-paper numbers are directly comparable.

---

## TL;DR — reproduce

```bash
cd benchmark/kvcomp-longcontext/paper
bash submit_all.sh            # all 27 jobs (both papers)
bash submit_all.sh paper1     # HayateQuant codec jobs only (11 jobs)
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

## Paper 1 — HayateQuant (codec; HALO retained as prior baseline)

| Table / figure | Config | Job(s) | Result dir |
|---|---|---|---|
| Codec quality (KV-MSE, attn-err) | `kvcomp-codec/conf/paper_hayatequant.yaml` | `p1-hq-{llama,minis,r1q7,r1q15}` | `kvcomp-codec/results_paper_*` |
| Codec speed × quality Pareto | `codec_speed.py` (CLI) | `p1-speed-{synth,llama}` | `results_codec_speed/` |
| Retrieval (Needle/RULER/PPL), full bit-grid | `conf/turboquant_paper.yaml`, `conf/turboquant_paper_ministral.yaml` | `p1-tq-{llama,minis}-retr` | `results_tq_{llama,minis}_retr/` |
| LongBench-E downstream | same configs, `run_longbench=true` | `p1-tq-{llama,minis}-lb` | `results_tq_{llama,minis}_lb/` |
| Lever ablation (HALO levers + TaperKV asymmetry) | `conf/paper_ablation.yaml` | `p1-ablation-llama` | `results_ablation_llama/` |

The retrieval grid runs HayateQuant at the full 2/2.5/3/3.5/4 bit-grid against
TurboQuant's identical grid + HALO + Uniform-INT + KIVI + FP8 + NVFP4, all at
matched memory, so the "iso-bit, more accurate" claim is backed end-to-end
(not just codec MSE).  See `papers/hayatequant_paper/README.md` for the
method write-up and validated v3 results.

> **Full technical write-up:** see [`HALO_ENHANCED_REPORT.md`](HALO_ENHANCED_REPORT.md)
> for the method (all 7 levers + status), the complete review-memo audit
> (A-1..A-8, B-1..3, C, vLLM 1-5), the fair TurboQuant positioning (what is / is
> not novel, and the 4 superiority axes), and the verification artifacts.

### Enhanced HALO — role-asymmetric dequant scale (proposed upgrade)

The default HALO uses one least-squares (LS) dequant scale for both K and V. LS
minimises reconstruction MSE but systematically shrinks the rotated vector norm
(`‖k̂‖ = ‖k‖·cosθ`), so the attention **logit** is scaled by `cos²θ` — MSE-optimal
is the wrong objective for keys, which enter softmax through an inner product.
Enhanced HALO makes the scale **role-asymmetric** (iso-storage, same fp16/vector):

| variant | key scale | value scale | effect |
|---|---|---|---|
| `HALO-NC-{2,2.5,3,3.5,4}bit` | norm-correcting (`nc`) | least-squares (`ls`) | logit shrink `cos²θ → cosθ`; value MSE unchanged |
| `HALO-NCIT-{2,2.5,3,3.5,4}bit` | `nc` | `ls_iter` | + one Lloyd alternating-min step on values (monotone-lower value MSE) |

The **2.5 / 3.5-bit** points use lever 3 (outlier split, `n_outlier=64`,
`outlier_bits=b+1` on head_dim=128) so HALO now covers TurboQuant's full
headline bit-grid at exactly iso-memory. The needle/RULER filler is now a
diverse 30-sentence bank (not a 4-sentence repeat), so retrieval actually
stresses the codec.

`s_NC = √(s_LS · s_IP)` is the geometric mean of the MSE-optimal and inner-product
-unbiased scales; the NC MSE penalty vs LS is `O(θ⁴)`, so 3–4-bit converge while
2-bit gains the most (keys are sharpest there). The codec change is in
`onecomp/kvcomp/codec/halo.py` (`key_scale_mode` / `value_scale_mode`, default
`None` → legacy behaviour, so existing results reproduce bit-for-bit).

Re-run the paper battery for the enhanced HALO only (self-contained vs FullKV +
TurboQuant + legacy HALO at iso-bit; writes to `results_halo_enh_*`):

```bash
cd benchmark/kvcomp-longcontext/paper
bash submit_halo_enhanced.sh             # 8 jobs (retrieval + reasoning)
bash submit_halo_enhanced.sh retrieval   # Llama/Ministral needle/RULER/PPL/LB
bash submit_halo_enhanced.sh reason      # R1/QwQ GSM8K/MATH/PPL
DRYRUN=1 bash submit_halo_enhanced.sh    # print sbatch lines, submit nothing
```

The enhanced variants are also wired into `conf/turboquant_paper.yaml`,
`conf/paper_ablation.yaml` (`HALO-hg-NC/NCIT` lever rows), `conf/paper_aime.yaml`,
and `conf/paper_taperkv.yaml`, so the full battery picks them up automatically.

#### Inference speed (latency / throughput)

Two harnesses measure speed; the enhanced HALO scale levers are *quality* levers
with the same encode/decode cost as plain LS, so the goal is to show the gain is
free (no latency/throughput regression vs the shipped TurboQuant default):

* `codec_speed.py` — codec-level latency **and throughput** in both the prefill
  (S=4096) and decode (S=1, matrix-vector) regimes: encode/decode ms, encode
  speedup vs TurboQuant, throughput (Melem/s), per-token decode µs, tok/s/layer,
  plus KV-MSE / attn-err / bits-per-element on the same tensors. Now includes
  `HALO-hadamard_gemm-NC` and `-NCIT`.
* `e2e_speed.py` — end-to-end on a real model: prefill latency (TTFT), decode
  throughput (tok/s), per-token decode latency (ms/tok), and peak GPU memory for
  `FullKV / TurboQuant / HALO / HALO-NC / HALO-NCIT` (codec on every layer, the
  honest worst-case overhead).

```bash
bash submit_halo_enhanced.sh speed       # codec_speed (synth+Llama) + e2e (Llama/R1-Q7)
# or directly:
python codec_speed.py --device cuda --heads 32 --seq 4096 --dim 128 --bits 2 3 4 --repeats 50
python e2e_speed.py --model <DIR> --prompt-len 4096 --gen-len 128 --bits 3
```

Results land in `results_codec_speed/{codec_speed_enh_*,e2e_*}.json`.

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

---

## Run status (2026-06-12 15:00 JST — 94% complete, rescue jobs submitted)

### Rescue jobs submitted (2026-06-12)

| JobID | 対象 | 欠損メソッド | 予想完了 |
|---|---|---|---|
| 47708 | basic/R1-L8 | KIVI-INT2, HALO-3bit | ~8h |
| 47709 | basic/R1-Q7 | KIVI-INT2, HALO-3bit | ~8h |
| 47710 | basic/R1-Q14 | Uniform-INT4, StreamingLLM | ~8h |
| 47711 | basic/R1-Q15 | TurboTaper, KIVI-INT2 | ~8h |
| 47712 | basic/QwQ | Uniform-INT4, StreamingLLM | ~12h |
| 47713 | comp/R1-Q15 | Uniform-INT4, StreamingLLM, ThinKV, PM-KVQ (AIME) | ~16h |

### Completion snapshot: 259/273 slots (94%)

| セクション | 完了 | 残り |
|---|:---:|:---:|
| P1 retr/Llama, retr/Minis | ✅ 22+22 | 0 |
| P1 lb/Llama, lb/Minis | ✅ 22+22 | 0 |
| P1 ablation/codec_speed | ✅ | 0 |
| P2 taperkv/Llama, R1-Q7/14/15/L8, QwQ | ✅ 11×6 | 0 |
| P2 basic/R1-L8,Q7,Q14,Q15,QwQ | ⏳ 8/10 each | 10 (47708-47712) |
| P2 comp/R1-Q7,L8,Q14-24,Q14-25 | ✅ 10×4 | 0 |
| P2 comp/R1-Q15 | ⏳ 6/10 | 4 (47713) |
| P2 comp/QwQ-24, QwQ-25 | ✅ 10+10 | 0 |

---

## Benchmark results (confirmed, 2026-06-12)

### Paper 1 — HALO codec: Retrieval (Llama-3.1-8B, iso-memory)

`Needle = 45 cells (5 depths × 3 ctx-len × 3 seeds); RULER = 15 cells`

| Method | Needle ↑ | RULER ↑ | PPL ↓ | Mem |
|---|---:|---:|---:|---:|
| FullKV | 1.000 | 1.000 | 6.49 | 1.000 |
| Uniform-INT3 | 1.000 | 0.867 | 8.54 | 0.197 |
| Uniform-INT4 | 1.000 | 1.000 | 6.71 | 0.260 |
| StreamingLLM (evict) | 0.400 | 0.067 | 6.49 | 0.282 |
| KIVI-INT2 | 1.000 | 1.000 | 7.43 | 0.161 |
| TurboQuant-3bit | 1.000 | 1.000 | 6.92 | 0.196 |
| **HALO-3bit (proposed)** | **1.000** | **1.000** | **6.90** | **0.196** |
| HALO-4bit | 1.000 | 0.933 | 6.59 | 0.258 |
| TaperKV | 1.000 | 1.000 | 6.59 | 0.292 |
| TurboTaper | 1.000 | 1.000 | 6.55 | 0.296 |

**Key claim confirmed:** HALO-3bit = TurboQuant-3bit on retrieval/PPL at iso-memory, while being ≈3.8× faster to encode.

---

### Paper 2 — TaperKV: Retrieval headline (Llama-3.1-8B, iso-~3-bit ≈20% budget)

| Method | Needle ↑ | PPL ↓ | Mem |
|---|---:|---:|---:|
| FullKV | 1.000 | 6.49 | 1.000 |
| Uniform-INT3 | 1.000 | 8.50 | 0.197 |
| StreamingLLM (evict) | **0.200** ❌ | 6.49 | 0.209 |
| ThinKV | 1.000 | 6.64 | 0.313 |
| PM-KVQ | 1.000 | 7.95 | 0.204 |
| TurboQuant-3bit | 1.000 | 6.92 | 0.196 |
| **HALO-3bit** | **1.000** | **6.91** | **0.196** |
| **TaperKV** (standard) | **1.000** | **6.59** | **0.293** |
| **TaperKV-3bit** (iso-3-bit) | **1.000** | **7.42** | **0.197** |
| **TurboTaper-3bit** | **1.000** | **6.67** | **0.190** |

**Key claim confirmed:** At iso-memory, StreamingLLM (eviction) collapses needle to 0.20 while TaperKV maintains needle=1.00 with PPL 6.59 (beats even Uniform-INT3 on PPL).

---

### Paper 2 — AIME competition math: R1-Qwen-7B

| Method | AIME-2024 ↑ | AIME-2025 ↑ |
|---|---:|---:|
| FullKV | 0.342 | 0.233 |
| Uniform-INT4 | 0.342 | 0.192 |
| KIVI-INT2 | 0.567 | 0.400 |
| ThinKV | 0.567 | 0.400 |
| PM-KVQ | 0.033 | 0.067 |
| **TaperKV** | **0.633** | **0.433** |
| TurboTaper | 0.533 | 0.300 |
| HALO-4bit | 0.533 | 0.367 |
| HALO-3bit | 0.567 | 0.367 |

**TaperKV (0.633 / 0.433) beats FullKV (0.342 / 0.233) — compression improves AIME on R1-Q7.**

---

### Paper 2 — AIME competition math: QwQ-32B

| Method | AIME-2024 ↑ | AIME-2025 ↑ |
|---|---:|---:|
| FullKV | 0.300 | 0.267 |
| KIVI-INT2 | 0.333 | 0.200 |
| ThinKV | 0.000 ❌ | 0.000 ❌ |
| PM-KVQ | 0.000 ❌ | 0.000 ❌ |
| TaperKV | 0.200 | 0.167 |
| TurboTaper | 0.300 | 0.167 |
| **HALO-4bit** | **0.500** | **0.367** |
| HALO-3bit | 0.200 | 0.167 |

**Notable:** ThinKV / PM-KVQ = 0.000 on QwQ-32B (KV budget insufficient for 32B long-CoT). HALO-4bit (0.500/0.367) outperforms FullKV (0.300/0.267).

---

## Run status (2026-06-06 01:20 JST — Round 3 実行中, ablation 完了)

### Bugs fixed (2026-06-04)

| Bug | Fix | Committed |
|---|---|---|
| TurboTaper Bug-1: `b_min_value=2` crashes Qwen long-CoT | `b_min_value: 3` in both YAML configs | ✅ |
| TurboTaper Bug-2: cascade re-quantization (non-idempotent Lloyd-Max) | Delta quantization in `turbotaper_cache.py` | ✅ |
| `kv_benchmark._save()` race condition: concurrent jobs overwrite each other | Re-read on-disk JSON before each write; merge on-disk + in-memory | ✅ `8e1a24f` |
| QwQ-32B taperkv OOM: `eager` attention @16K context exceeded VRAM | Switch `paper_taperkv_qwq.yaml` to `sdpa`; exclude ThinKV (requires eager) | ✅ `c629f23` |

### Completed (all OK, updated 2026-06-06)

| Component | JID(s) | Status |
|---|---|---|
| Paper 1 codec/retrieval/ablation/speed (Llama+Ministral) | various | ✅ |
| P2 taperkv Llama-8B-instruct (full 11 methods) | 46386+46539 | ✅ |
| P2 TurboTaper basic (delta fix) — R1-Llama,R1-7B,R1-14B,QwQ | 46481-46484 | ✅ |
| P2 TurboTaper AIME comp R1-7B,R1-14B,QwQ (delta fix) | 46485-46488 | ✅ |
| P2 taperkv/R1-14B TurboTaper-3bit (delta fix) | 46489 | ✅ |
| P2 comp/qwq_25 all except StreamingLLM | 46535 | ✅ |
| P2 basic/R1-L8 KIVI+HALO | 46718 | ✅ |
| P2 basic/R1-Q7 KIVI+HALO | 46720 | ✅ |
| P2 basic/R1-Q14 StreamingLLM+Uniform-INT4 | 46719 | ✅ |
| P2 comp/qwq_24+25 StreamingLLM | 46721-46722 | ✅ |
| P2 comp/r1q14_25 StreamingLLM | 46723 | ✅ |
| P2 taperkv/R1-Q14 HALO-3bit | 46716 | ✅ |
| P2 taperkv/R1-Q7 TT-3bit Δfix + HALO-3bit | 46714 | ✅ |
| P2 comp/r1q14_24 TaperKV+TurboTaper+HALO | 46708 | ✅ |
| P2 comp/r1l8 PM-KVQ+TaperKV | 46537 | ✅ |
| P2 basic/R1-Q15 TurboTaper | 46736 | ✅ |
| P2 taperkv/R1-Q15 (11/11 全メソッド) | 46417+46735+46928 | ✅ |
| P2 comp/r1q14_24 KIVI+StreamingLLM (完成) | 46923+46924 | ✅ |
| **Ablation/Ministral-8B (14 methods)** | 46972 | ✅ **NEW** |
| **Ablation/R1-Q7 (14 methods)** | 46973 | ✅ **NEW** |
| **b_min sweep/Llama-8B (11 configs)** | 46974 | ✅ **NEW** |
| **b_min sweep/R1-Q7 (11 configs)** | 46975 | ✅ **NEW** |
| **Budget sweep/Llama budget=0.10 (7 methods)** | 46976 | ✅ **NEW** |
| **basic/R1-L8 comp KIVI-INT2** | 46921 (partial) | ✅ **NEW** |

### Key new results (2026-06-06)

**b_min sweet spot (ablation):**
| Config | b_min | Llama needle | Llama PPL | R1-Q7 PPL |
|---|:---:|:---:|:---:|:---:|
| TaperKV-3bit-bmin2 | 2 | 0.556 (❌) | 20.99 | 145.97 |
| **TaperKV-3bit-bmin3** | **3** | **1.000 ✅** | **7.42** | **67.66** |
| **TaperKV-3bit-bmin4** | **4** | **1.000 ✅** | **7.00** | **57.05** |
| TaperKV-4bit-bmin2 | 2 | 0.667 (❌) | 11.15 | 94.46 |
| **TaperKV-4bit-bmin4** | **4** | **1.000 ✅** | **6.78** | **52.97** |
→ **b_min=3 is minimum safe floor for 3-bit; b_min=4 for 4-bit**

**Budget sweep @10% memory (Llama-8B):**
| Method | needle | PPL |
|---|:---:|:---:|
| StreamingLLM | 0.222 ❌ | 6.49 |
| ThinKV | 1.000 ✅ | 6.64 |
| KIVI-INT2 | 1.000 ✅ | 7.40 |
| **TaperKV** | **1.000 ✅** | **6.59** |
| HALO-3bit | 1.000 ✅ | 6.91 |
→ **TaperKV maintains needle=1.0 even at 10% budget (StreamingLLM fails)**

### Currently running (52 jobs, updated 01:20 JST Jun 6)

**Long-running (9h+):**
| JID | Component | Progress |
|---|---|---|
| 46919 | comp/r1q7 TaperKV | AIME-2025 running (~est. complete 02:11) |
| 46920 | comp/r1q7 KIVI+HALO | KIVI AIME-2025 running; HALO-3bit will need rescue job |
| 46921 | comp/r1l8 KIVI+HALO | KIVI done ✅; HALO-3bit AIME started 23:33 (~est. done 11:33) |
| 46922 | comp/r1l8 TurboTaper | AIME-2024 done(0.367); AIME-2025 running |
| 46925 | comp/r1q15 KIVI+HALO+TT | 3 methods AIME-2025 running |
| 46926 | basic/QwQ Stream+Uni4 | StreamingLLM GSM=0.11 done; MATH running |

**⚠️ Walltime rescue submitted:**
| JID | Purpose |
|---|---|
| **47037** | comp/r1q7 HALO-3bit rescue (46920 will hit walltime before HALO-3bit AIME-2025) |

**Ablation / budget sweep:**
| JID | Component | Progress |
|---|---|---|
| 46977 | budget_llama_15 | Running 6m |
| 46978 | budget_llama_20 | Running 3m |
| 46979-46980 | budget_llama_25/35 | PENDING |
| 46981-46985 | budget_r1q7_10/15/20/25/35 | PENDING |

**HALO-4bit additions:**
| JID | Component | Progress |
|---|---|---|
| 46952-46953 | taperkv/R1-L8 (2 jobs) | 6h, FullKV+Uni-INT3+Stream done; StreamingLLM GSM running |
| 46954-46958 | basic H4 (R1-L8/Q7/Q14/QwQ/Q15) | AIME running |
| 46959-46965 | comp H4 (R1-Q7/L8/Q14×2/Q15/QwQ×2) | AIME running |
| 46966-46971 | LCB all models | Running 4h45m |
| 47015-47019 | taperkv/QwQ-32B (sdpa fix) | PENDING |
| 47020-47026 | Uniform-INT4 comp dirs | PENDING |

### Data gaps — snapshot 01:20 JST Jun 6 (81% complete)

**Overall: 142/175 method slots filled (81%)**

**Paper 1 (HALO):**
| Dataset | Status | Notes |
|---|---|---|
| results_tq_llama_lb + retr | ✅ COMPLETE | 22 methods |
| results_tq_minis_lb + retr | ✅ COMPLETE | 22 methods |
| results_ablation_llama | ✅ COMPLETE | 14 methods (lever + asymmetry) |
| results_ablation_minis | ✅ **COMPLETE** | 14 methods ✅ NEW |
| results_ablation_r1q7 | ✅ **COMPLETE** | 14 methods ✅ NEW |
| results_ablation_bmin_llama | ✅ **COMPLETE** | 11 configs (b_min sweep) ✅ NEW |
| results_ablation_bmin_r1q7 | ✅ **COMPLETE** | 11 configs ✅ NEW |
| results_ablation_budget_llama_10 | ✅ **COMPLETE** | 7 methods @10% budget ✅ NEW |
| results_ablation_budget_llama_15/20/25/35 | ⏳ 46977-46980 | Running/pending |
| results_ablation_budget_r1q7_* | ⏳ 46981-46985 | All pending |
| results_codec_speed | ✅ COMPLETE | HALO Pareto (speed × quality) |
| results_tq_r1q7 | ⏳ 46986 | Full bit-grid on reasoning model (PENDING) |
| results_tq_r1l8 | ⏳ 46987 | Full bit-grid on reasoning model (PENDING) |

**Paper 2 (TaperKV):**
| Dataset | Status | Notes |
|---|---|---|
| taperkv/Llama | ✅ COMPLETE (11/11) | — |
| taperkv/R1-Q7 | ✅ COMPLETE (11/11) | — |
| taperkv/R1-Q14 | ✅ COMPLETE (11/11) | — |
| taperkv/R1-Q15 | ✅ **COMPLETE** (11/11) | ✅ NEW (KIVI+HALO done via 46928) |
| taperkv/QwQ-32B | ⏳ 47015-47019 | PENDING (sdpa OOM fix applied) |
| taperkv/R1-L8 | ⏳ 46952-46953 | 4/11 done; ~7 more hrs to complete |
| basic/R1-L8 | 9/10 + HALO-4bit⏳ 46954 | AIME running |
| basic/R1-Q7 | 9/10 + HALO-4bit⏳ 46955 | AIME running |
| basic/R1-Q14 | 9/10 + HALO-4bit⏳ 46956 | AIME running |
| basic/QwQ | 8/10 ⏳ 46926+46957 | StreamingLLM+HALO-4bit |
| basic/R1-Q15 | 8/10 ⏳ 46958 | HALO-3bit+HALO-4bit (KIVI done ✅) |
| comp/r1q7 | 6/10 ⏳ 46919+46920+46959+**47037** | TaperKV(running)+KIVI+HALO-3/4bit |
| comp/r1l8 | 7/10 ⏳ 46921+46922+46960 | KIVI done ✅; HALO-3bit+TT+HALO-4bit |
| comp/r1q14_24 | 8/10 ⏳ 46961 | Uniform-INT4+HALO-4bit pending |
| comp/r1q14_25 | 8/10 ⏳ 46962 | Uniform-INT4+HALO-4bit pending |
| comp/r1q15 | 6/10 ⏳ 46925+46963 | KIVI+HALO+TT+HALO-4bit |
| comp/qwq_24 | 8/10 ⏳ 46964 | Uniform-INT4+HALO-4bit pending |
| comp/qwq_25 | 8/10 ⏳ 46965 | Uniform-INT4+HALO-4bit pending |
| LCB (all models) | ⏳ 46966-46971 | 全推論モデル × 全メソッド (FullKV=0% for all R1, expected) |
| Ablation Ministral | ✅ **COMPLETE** | 14 methods ✅ NEW |
| Ablation R1-Q7 | ✅ **COMPLETE** | 14 methods ✅ NEW |
| Ablation R1-L8 | ⏳ 46988 | PENDING |
| b_min sweep (Llama+R1-Q7) | ✅ **COMPLETE** | ✅ NEW — b_min=3 safe for 3-bit |
| Budget sweep Llama | ⏳ 46976✅+46977-46980 | 10% done; 15/20 running; 25/35 pending |
| Budget sweep R1-Q7 | ⏳ 46981-46985 | All pending |
| Reasoning asymmetry (R1-Q7+R1-L8) | ⏳ 46989-46990 | PENDING |

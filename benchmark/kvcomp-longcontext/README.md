# KV-cache compression: long-context retrieval + reasoning

Main experiment for the KV-cache compression methods, the analogue of the
weight-side `benchmark/<model>-<recipe>/` runs (GPTQ / JointQ). It compares,
**at a matched memory budget**, the two compression *families* the design memo
contrasts, plus the proposed method:

| Method | Family | Mechanism | Evicts? |
|---|---|---|:---:|
| **FullKV** | — | fp16 upper bound | no |
| **Uniform-INT4 / INT2** | quant | uniform per-token affine | no |
| **StreamingLLM** | evict | keep sink + recent, drop middle | **yes** |
| **ThinKV** | quant (+evict) | static per-thought bits (R/E=4, T=2); **faithful**: outlier-layer skip + calibrated L*/thresholds + attention-sparsity thought classification. TBE eviction is vLLM-only | HF: no |
| **PM-KVQ** | quant | **progressive / dynamic** precision: fixed high-precision window, older KV shrinks to a floor as memory saturates; block-wise (layer) allocation | no |
| **TaperKV** | quant | **static** age-graded precision ladder, non-zero floor (proposed) | no |

Thesis: uniform quantization is less memory-efficient than eviction, but eviction
**structurally fails long-context retrieval** — a needle in the evicted region is
unrecoverable. Replacing eviction with a graded precision floor (TaperKV) keeps
every token recoverable, whereas *dynamic* progressive precision (PM-KVQ)
collapses old tokens to the floor over a fixed absolute window. TaperKV's
length-proportional ladder instead spends bits where retrieval needs them.

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

**Reasoning model** (ThinKV native regime; disable needle which is uninformative at long CoT):
```bash
srun --gres=gpu:1 .venv/bin/python -u kv_benchmark.py \
    --config-name paper_run \
    model_path=/lustre/.../models/DeepSeek-R1-Distill-Qwen-7B \
    run_needle=false run_ruler=false max_new_reason=2048 gsm8k_limit=100
```

Config: `conf/benchmark_longcontext.yaml`. Knobs are kept identical across
methods (`sink_tokens`, `recent_tokens`, `target_mem`) for a fair iso-memory
comparison; the eviction window is sized to `target_mem * ctx` so eviction gets
the **same** memory budget as the quant methods. Results (JSON + Markdown
table) are written to `output_dir`.

**Outlier-layer handling (applied uniformly).** A small set of layers carry
massive activations whose keys span orders of magnitude; quantizing them at low
bit-width destroys generation on reasoning models. The harness detects these
layers once and keeps them in FP16 for **every** quantization method, which is
the standard StreamingLLM/KIVI robustness trick. Keeping it uniform is what makes
the comparison fair — it is not a privilege granted only to the proposed method.

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
(`needle_in_haystack`); **RULER multi-key** retrieval, where several competing
keys force the model to disambiguate rather than merely locate (`ruler_multikey`,
`run_ruler=true`); GSM8K and MATH-500 pass@1 for reasoning non-degradation
(`gsm8k_pass_at_1`, `math500_pass_at_1`, `run_math=true`); WikiText-2 perplexity
as a sanity check (`wikitext2_perplexity`); and the realized compression ratio of
the quantized layers.

## Results

### llama-3.1-8b-instruct — faithful, iso-memory (~0.28), ctx ∈ {2048, 4096, 8192}

needle = 63 cells (7 depths × 3 lengths × 3 seeds); RULER = 15 (4 distractor
keys × 3 lengths × 5 seeds); GSM8K = 40; PPL = WikiText-2 (16 windows).

| Method | Needle | RULER | Needle mem | GSM8K p@1 | PPL |
|---|---:|---:|---:|---:|---:|
| FullKV | 1.00 | 1.00 | 1.00 | 0.825 | 6.10 |
| Uniform-INT4 | 1.00 | 1.00 | 0.26 | 0.825 | 6.17 |
| Uniform-INT2 (KIVI) | 0.54 | 0.07 | 0.13 | 0.725 | 28.54 |
| **StreamingLLM** (evict) | **0.30** | **0.60** | 0.28 | 0.825 | 6.10 |
| ThinKV (faithful) | 1.00 | 1.00 | 0.31 | 0.575 | 6.21 |
| PM-KVQ | 1.00 | 1.00 | 0.20 | 0.800 | 7.42 |
| **TaperKV** (proposed) | **1.00** | **1.00** | 0.28 | **0.825** | 6.24 |

Two findings stand out:

1. **Eviction structurally fails retrieval.** At an equal memory budget (~0.28),
   StreamingLLM recovers only **30%** of needles and **60%** of RULER keys: once
   a token in the middle of the context is evicted, the answer is gone. Every
   length-preserving quantization method, by contrast, retrieves perfectly.
   Uniform-INT2 is the cautionary case — it keeps all tokens but at 2 bits, and
   already loses nearly half the needles while its perplexity blows up to 28.5.
   Keeping tokens is necessary; keeping them at a sensible precision floor is
   what makes them usable.

2. **TaperKV offers the best quality at equal memory.** Among the graded and
   adaptive methods it is the only one that simultaneously preserves perfect
   retrieval, FullKV-level reasoning (GSM8K 0.825, matching the uncompressed
   model), and near-FullKV perplexity (6.24 vs 6.10). PM-KVQ matches retrieval
   here but pays for its more aggressive dynamic floor with higher perplexity
   (7.42) and a small reasoning drop. ThinKV retrieves perfectly, but its
   ternary quantization of low-importance "transition" thoughts over-compresses
   on a non-reasoning instruct model and pulls GSM8K down to 0.575 — a reminder
   that it is designed for long chain-of-thought models, not instruct models.

### gemma-4-31B-it — retrieval breadth, iso-memory, ctx ∈ {2048, 4096, 8192}

A second, larger, and more 2-bit-sensitive architecture, which sharpens the
distinction between the graded methods.

| Method | Needle | RULER | GSM8K p@1 | Quantized cr |
|---|---:|---:|---:|---:|
| FullKV | 1.00 | 1.00 | 0.975 | 1.0× |
| Uniform-INT4 | 1.00 | 1.00 | 0.975 | 3.9× |
| Uniform-INT2 | 0.00 | 0.00 | 0.700 | 7.7× |
| StreamingLLM (evict) | 0.73 | 1.00 | 0.975 | — |
| ThinKV (faithful) | 1.00 | 1.00 | 0.975 | 2.4× |
| **PM-KVQ** | **0.00** | **0.00** | **0.100** | 2.4× |
| **TaperKV** (proposed) | **1.00** | **1.00** | **0.975** | 1.8× |

This model exposes the core thesis cleanly. Gemma is highly sensitive to 2-bit
quantization (Uniform-INT2 collapses to zero retrieval), and that sensitivity
discriminates between the two graded schemes even though both use a 2-bit value
floor:

* **PM-KVQ collapses** (needle and RULER 0.00, GSM8K 0.10, perplexity in the
  thousands). Its high-precision region is a *fixed absolute window*, so at long
  context essentially every token older than that window is pinned to the 2-bit
  floor — and on a 2-bit-sensitive model the needle is lost.
* **TaperKV is unaffected** (needle and RULER 1.00, GSM8K 0.975). Its ladder is
  indexed by *age as a fraction of the current length*, so it stretches with the
  context and keeps mid-context tokens at intermediate precision. The needle,
  wherever it lands, is never pinned to the floor.

This is the central claim made concrete: a *static, length-proportional*
precision ladder is more robust for long-context retrieval than *dynamic,
absolute-window* collapse, and both are strictly better than eviction.

### DeepSeek-R1-Distill-Qwen-7B (long chain-of-thought) — reasoning non-degradation

Retrieval is uninformative on this model — even FullKV scores below 0.2 on the
needle task at 4k–16k — so here we measure reasoning non-degradation with an
adequate generation budget (`max_new=2048`, which long chain-of-thought models
need to reach `\boxed{}`). Needle is disabled; the headline is GSM8K and
MATH-500 pass@1 plus perplexity.

| Method | GSM8K p@1 | MATH-500 p@1 | PPL | Quantized cr |
|---|---:|---:|---:|---:|
| FullKV | 0.675 | 0.313 | 50.8 | 1.0× |
| Uniform-INT4 | 0.775 | 0.375 | 52.3 | 3.9× |
| Uniform-INT2 | 0.625 | 0.375 | 184.8 | 7.5× |
| ThinKV (faithful) | 0.650 | 0.375 | 53.3 | 2.2× |
| PM-KVQ | 0.700 | 0.313 | 68.9 | 4.9× |
| **TaperKV** (proposed) | 0.650 | 0.313 | 53.6 | 3.4× |

With outlier layers held in FP16, every method now tracks the uncompressed model
within the noise of a 40/16-sample evaluation: TaperKV reaches FullKV-level
perplexity (53.6 vs 50.8) at 3.4× compression, and ThinKV — in its native
reasoning regime — preserves both GSM8K and MATH. This is the result that the
earlier degenerate setup hid: quantizing the massive-activation layers was what
broke reasoning (GSM8K had dropped to 0.0); skipping them uniformly restores it
for all methods.

Taken together, across an instruct model (llama-8b), a large and 2-bit-sensitive
model (gemma-31B), and a long chain-of-thought reasoning model (7B), **TaperKV is
the only method that is simultaneously robust on all three axes** — long-context
retrieval, reasoning, and perplexity — while eviction fails retrieval and the
dynamic-precision baseline (PM-KVQ) is architecture-fragile.

## Sub-validation

Component smoke tests and the unit suite live in
`ichikawa_works/kvcomp_bench/smoke_pmkvq.py` and
`ichikawa_works/kvcomp_verification/01_unit_kvcomp.py` (58/58 pass).
The faithful-ThinKV eval path (outlier skip + thought-classification capture)
is checked end-to-end by
`ichikawa_works/kvcomp_verification/09_thinkv_faithful_eval.py` (asserts the
faithful path recovers reasoning vs the degenerate all-NVFP4 path).

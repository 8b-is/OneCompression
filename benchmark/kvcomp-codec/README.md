# HayateQuant reproduction

This directory has one public comparison: `hayatequant` versus `turboquant`.
HayateQuant uses the adopted data-free K/V rates K3/V1, K4/V2, and K5/V3
at average rates 2, 3, and 4. With two-coordinate blocks these are K6/V2,
K8/V4, and K10/V6 code bits. Obsolete variants are not part of the
benchmark registry.

The canonical manifest reproduces the paper's main quality grid:

- seven Llama/Qwen/DeepSeek-distill checkpoints;
- WikiText-2 raw-test perplexity over the complete split with 2048-token
  windows and 1024-token stride, aggregated with a geometric mean;
- normalized zero-shot accuracy on ARC-Easy, ARC-Challenge, PIQA, HellaSwag,
  and WinoGrande, aggregated with an arithmetic mean;
- rates 2, 3, and 4 for both methods;
- vLLM serving on the selected Llama-3.1-8B checkpoint with a 4096-token
  prompt, 128 generated tokens, and three repeats.

No generated result or log belongs in Git. Supply an external output root.

Planning performs a full SHA-256 pass over every selected checkpoint weight
shard and hashes the model configuration and tokenizer metadata. On a shared
filesystem this can take several minutes. It also records the exact dirty or
clean source revision, writes a deterministic `source_snapshot.tar.gz`, and
materializes a content-addressed `source-<sha>` directory below the output
root. Every Slurm job executes that frozen source copy rather than the mutable
checkout. The resulting `source_provenance.json` is attached to every result.

## Plan and submit

```bash
python benchmark/kvcomp-codec/reproduce_hayatequant_paper.py plan \
  --models-root "$MODELS_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --python .venv/bin/python \
  --vllm-python .venv-vllm/bin/python

python benchmark/kvcomp-codec/reproduce_hayatequant_paper.py submit-plan \
  --output-root "$OUTPUT_ROOT" \
  --partition "$SLURM_PARTITION"
```

`submit-plan` verifies and submits the exact frozen plan without hashing all
weights again. The one-step `submit` command remains available when a separate
inspection point is not needed.

Keep the complete output root. It contains the run manifest, model/checkpoint
fingerprints, source attestation, immutable source snapshot, Slurm logs, and
result files needed to audit a paper number. Large weight files are fully
hashed while planning; launch-time checks re-hash small metadata and verify the
recorded size and nanosecond mtime of each weight shard.

Use `--stages ppl`, `zeroshot`, or `speed` for a subset. A small smoke can use
`--zero-limit 4 --prompt-tokens 128 --max-tokens 8 --repeats 1`; limited
zero-shot results are diagnostics, not paper cells.

`--ppl-windows 32` is also available for a capped diagnostic. It is recorded
as `wikitext2_raw_test_capped_window_diagnostic_v1` and can never satisfy the
formal `paper_complete` gate. The reference manuscript labels its main grid
`P-main` and separately labels 32-window development results `P-dev32`; the
runner therefore does not silently equate those contracts.

Zero-shot choices are scored independently with a fresh cache. This preserves
the established single-choice numerical protocol: batching choices can select
different GEMM reduction kernels and change low-bit assignment at rounding
boundaries even for a mathematically token-local codec. The driver writes an
atomic incomplete document after every completed task and resumes matching
documents task-by-task, so a preempted formal run does not discard completed
ARC/PIQA/HellaSwag/WinoGrande work. Console output contains compact scalar
summaries; per-example records remain in the JSON result for auditing.

After all quality jobs finish:

```bash
python benchmark/kvcomp-codec/reproduce_hayatequant_paper.py summarize \
  --output-root "$OUTPUT_ROOT"
```

`summary.json` fails coverage unless every 7×3×2 PPL and zero-shot cell exists.
`run_manifest.json` records resolved model locations, exact commands, source
revision, and every checkpoint artifact identity.

## Serving scope

`vllm_benchmark.py` runs each method/rate in a fresh process and requires
backend-emitted runtime evidence for HayateQuant and TaperKV. HayateQuant uses
its fused store, packed cache, and packed attention backend at rates 2/3/4.
The speed protocol applies compression to every model layer: OneComp removes
vLLM's implicit first/last-two-layer TurboQuant skip while retaining only
explicitly serialized `skip_layers`. Runtime evidence must cover the complete
model layer range. The installed vLLM exposes native TurboQuant presets only at
rates 3 and 4, so the runner does not silently substitute a different codec
for the missing 2-bit serving cell. TaperKV is measured separately and is
never labelled HayateQuant.

TaperKV uses a CUDA-Graph-safe fused quantize-and-scatter update plus an
incremental ladder-boundary regrade kernel. Its current vLLM cache allocator is
still model-dtype semantic storage, which is stated explicitly in runtime
evidence and benchmark results; it must not be reported as packed-memory
compression. Prefix caching is disabled because mutating aged shared pages
would violate request isolation.

The Hugging Face payload benchmark remains available as `codec_benchmark.py`;
`zeroshot_benchmark.py` is the shared downstream evaluator. Both require a
caller-selected output directory.

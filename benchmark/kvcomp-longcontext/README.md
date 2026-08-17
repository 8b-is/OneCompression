# TaperKV long-context checks

TaperKV is a separate proposal from HayateQuant. This directory contains one
standalone correctness benchmark for FullKV, token-granularity TaperKV, and
block-granularity TaperKV. It uses the deterministic synthetic needle protocol
implemented by `onecomp.kvcomp.eval`; optional WikiText-2 evaluation uses the
causal cache-state scorer.

```bash
srun --gres=gpu:1 .venv/bin/python \
  benchmark/kvcomp-longcontext/taperkv_benchmark.py \
  --model "$MODELS_ROOT/llama-3.1-8b-instruct" \
  --output "$OUTPUT_ROOT/taperkv/needle.json"
```

For a quick smoke, use `--context-lengths 256 --depths 0.5 --seeds 1
--max-new-tokens 8`. Set `--ppl-windows 8` for a bounded PPL diagnostic or
another positive window count. Existing outputs fail closed; use `--overwrite`
only when intentional. Results include the exact model-config hash, software
versions, protocol, policies, and cache accounting. Results and Slurm logs must
stay under an external output root and are ignored by Git.

The vLLM model test and speed path are in
`benchmark/kvcomp-codec/vllm_benchmark.py --method taperkv`. Its runtime
evidence explicitly distinguishes semantic GPU regrading from physical packed
cache storage.

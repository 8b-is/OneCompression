# HayateQuant four-recipe production run (2026-07-17)

This directory is the output root for the complete external-model paper
battery. Model weights are not copied into this repository; every job receives
an absolute external model path from the launch manifest.

The four HayateQuant rows are exactly:

- `HQ`: fixed block-VQ recipe;
- `HQ-Frontier`: integer K/V frontier operating point;
- `AdaptiveHQ`: causal split selection over the HQ budget;
- `RotationHQ`: fixed HQ payload with dense rotation.

Each codec, zero-shot, and long-context/reasoning job also includes
`TurboQuant-paper`; long-context jobs retain `FullKV`. FibQuant and BlockQuant
remain quality-only rows where the corresponding paper protocol enables them.

Launch command (paths intentionally external):

```bash
.venv/bin/python benchmark/kvcomp-codec/run_paper_pipeline.py \
  --models-root "$MODELS_ROOT" \
  --manifest benchmark/configs/paper_models.example.json \
  --output-root papers/hayatequant_paper/experiments/results/production-hq4-20260717
```

The production launch used the seven paper models and wrote Slurm logs under
`codec/logs`, `zeroshot/logs`, and `longcontext/logs`. Job IDs are recorded in
the scheduler history; partial JSON files are valid recoverable shards and
must not be merged until their protocol coverage is complete.

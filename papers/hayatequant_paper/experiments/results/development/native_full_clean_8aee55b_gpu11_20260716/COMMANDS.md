# Reproduction commands

The runs used the clean detached worktree
`/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/worktrees/hq-native-diag-8aee55b`
and Python from
`/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab/.venv/bin/python`.

The rate-resolved command template was:

```bash
CUDA_VISIBLE_DEVICES=<gpu> /lustre/home/frj/yuma-ichikawa/conference/onecomp-lab/.venv/bin/python \
  benchmark/kvcomp-codec/hayatequant_vllm_full_native_fair_speed.py \
  --avg-bits <3-or-4> --tokens 4096 --trials 4 --warmups 10 --iters 50 \
  --measure-cudagraph --output <absolute-output-json>
```

Rate 3 used GPU 0 and rate 4 used GPU 1 on `kagura-gpu11`.  The absolute
output paths used at run time remain in `provenance.arguments.output` inside
each raw JSON; that field is authoritative.  The two rates were run
concurrently on separate GPUs.  The parent runner launched one fresh worker
process for every ABBA element.

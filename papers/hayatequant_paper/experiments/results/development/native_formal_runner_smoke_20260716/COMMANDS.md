# Commands

All commands ran from the isolated worktree
`/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/worktrees/native-formal-contract-e13ee54`
through direct SSH to `kagura-gpu11`, using otherwise idle physical GPU 2.

```text
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=.:benchmark/kvcomp-codec <vllm-python> -u benchmark/kvcomp-codec/hayatequant_vllm_full_native_fair_speed.py --avg-bits 3 --tokens 64 --warmups 1 --iters 2 --trials 4 --output papers/hayatequant_paper/experiments/results/development/native_formal_runner_smoke_20260716/rate3.json

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=.:benchmark/kvcomp-codec <vllm-python> -u benchmark/kvcomp-codec/hayatequant_vllm_full_native_fair_speed.py --avg-bits 3 --tokens 32 --warmups 1 --iters 2 --trials 4 --output papers/hayatequant_paper/experiments/results/development/native_formal_runner_smoke_20260716/rate3_gpuenv.json

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=.:benchmark/kvcomp-codec <vllm-python> -u benchmark/kvcomp-codec/hayatequant_vllm_full_native_fair_speed.py --avg-bits 4 --tokens 32 --warmups 1 --iters 2 --trials 4 --output papers/hayatequant_paper/experiments/results/development/native_formal_runner_smoke_20260716/rate4_gpuenv.json
```

The exact Python executable path and its SHA-256 are present in every raw
controller and worker record. `<vllm-python>` above is a documentation label,
not a substituted runtime value.

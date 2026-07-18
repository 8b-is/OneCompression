# Integrated native formal-runner smoke commands

Source commit: `7096e8b374fefb430878e433cc0b18fb3bdca03b` (detached clean worktree).

Host: `kagura-gpu11`. These are development-only four-process ABBA smokes and are intentionally ineligible for a formal claim.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:benchmark/kvcomp-codec /lustre/home/frj/yuma-ichikawa/conference/onecomp-lab/.venv/bin/python -u benchmark/kvcomp-codec/hayatequant_vllm_full_native_fair_speed.py --avg-bits 3 --tokens 64 --warmups 1 --iters 2 --trials 4 --output /lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/onecomp-lab-kvcomp-correctness/papers/hayatequant_paper/experiments/results/development/native_formal_integrated_smoke_20260716/rate3.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:benchmark/kvcomp-codec /lustre/home/frj/yuma-ichikawa/conference/onecomp-lab/.venv/bin/python -u benchmark/kvcomp-codec/hayatequant_vllm_full_native_fair_speed.py --avg-bits 4 --tokens 64 --warmups 1 --iters 2 --trials 4 --output /lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/onecomp-lab-kvcomp-correctness/papers/hayatequant_paper/experiments/results/development/native_formal_integrated_smoke_20260716/rate4.json
```

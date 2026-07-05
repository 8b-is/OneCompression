#!/bin/bash
#SBATCH --job-name=hq-rt-speed-v9
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/kvcomp_hayatequant/logs/hq-rt-speed-v9_%j.log

# v9: fused Triton decode-step kernel (ONECOMP_TRITON_CODEC=1) A/B, same
# single-node single-session protocol as v8.  Each JSON contains its own
# in-session TurboQuant baselines, so v8/v9 rows are independently honest.
set -x
cd /lustre/home/frj/yuma-ichikawa/conference/onecomp-lab
source .venv/bin/activate
nvidia-smi --query-gpu=name,driver_version --format=csv
RES=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/kvcomp_hayatequant/results_speed
MODEL=/lustre/home/frj/yuma-ichikawa/conference/models/llama-3.1-8b-instruct
mkdir -p "$RES"

# --- 0) GPU parity: triton kernel vs eager fused round trip ---------------
python -m pytest tests/onecomp/kvcomp/test_kvcomp.py -q -x \
    -k "triton_step or roundtrip" 2>&1 | tail -5
[ ${PIPESTATUS[0]} -ne 0 ] && exit 1

cd benchmark/kvcomp-longcontext

# --- 1) codec microbench: triton, and triton+CUDA-graph --------------------
ONECOMP_TRITON_CODEC=1 python codec_speed.py --device cuda --heads 32 --seq 4096 \
    --bits 2 3 4 --repeats 20 --out "$RES/codec_speed_v9_triton.json"
ONECOMP_TRITON_CODEC=1 ONECOMP_COMPILE_CODEC=1 python codec_speed.py --device cuda \
    --heads 32 --seq 4096 --bits 2 3 4 --repeats 20 \
    --out "$RES/codec_speed_v9_triton_compiled.json"

# --- 2) end-to-end generation ----------------------------------------------
ONECOMP_TRITON_CODEC=1 python e2e_speed.py --model "$MODEL" --prompt-len 4096 \
    --gen-len 128 --bits 2 3 4 --repeats 3 --out "$RES/e2e_speed_v9_triton.json"
ONECOMP_TRITON_CODEC=1 ONECOMP_COMPILE_CODEC=1 python e2e_speed.py --model "$MODEL" \
    --prompt-len 4096 --gen-len 128 --bits 2 3 4 --repeats 3 \
    --out "$RES/e2e_speed_v9_triton_compiled.json"

echo "ALL_DONE"

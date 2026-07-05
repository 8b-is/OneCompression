#!/bin/bash
#SBATCH --job-name=hq-rt-speed-v10
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/kvcomp_hayatequant/logs/hq-rt-speed-v10_%j.log

# v10: final single-session speed campaign for the paper tables.
#   session A (butterfly): TurboQuant with the FWHT kernel disabled -- the
#     published-implementation rows.
#   session B (serving):   FWHT kernel enabled for the baseline, HayateQuant
#     under its fused serving path (Triton step kernel + CUDA-graph compile).
# Every headline comparison (TQ+FWHT vs HQ) lives inside session B, same
# node, same process, back-to-back -- no cross-session ratios needed.
set -x
cd /lustre/home/frj/yuma-ichikawa/conference/onecomp-lab
source .venv/bin/activate
nvidia-smi --query-gpu=name,driver_version --format=csv
RES=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/kvcomp_hayatequant/results_speed
MODEL=/lustre/home/frj/yuma-ichikawa/conference/models/llama-3.1-8b-instruct
mkdir -p "$RES"

# --- 0) GPU parity: triton kernel vs eager fused round trip ---------------
python -m pytest tests/onecomp/kvcomp/test_kvcomp.py -q -x \
    -k "triton_step or roundtrip" 2>&1 | tail -3
[ ${PIPESTATUS[0]} -ne 0 ] && exit 1

cd benchmark/kvcomp-longcontext

# --- session A: published butterfly baseline --------------------------------
ONECOMP_DISABLE_FHT=1 python codec_speed.py --device cuda --heads 32 --seq 4096 \
    --bits 2 3 4 --repeats 20 --out "$RES/codec_speed_v10_butterfly.json"
ONECOMP_DISABLE_FHT=1 python e2e_speed.py --model "$MODEL" --prompt-len 4096 \
    --gen-len 128 --bits 2 3 4 --repeats 3 --out "$RES/e2e_speed_v10_butterfly.json"

# --- session B: kernel-accelerated baseline vs fused HQ serving path --------
# Headline mode = TRITON + COMPILE: the Triton kernel serves decode steps and
# the CUDA-graph compile serves the prefill LUT path, which is what makes
# HayateQuant win BOTH TTFT and decode throughput in-session (v9 evidence).
ONECOMP_TRITON_CODEC=1 ONECOMP_COMPILE_CODEC=1 python codec_speed.py --device cuda \
    --heads 32 --seq 4096 --bits 2 3 4 --repeats 20 \
    --out "$RES/codec_speed_v10_serving.json"
ONECOMP_TRITON_CODEC=1 ONECOMP_COMPILE_CODEC=1 python e2e_speed.py --model "$MODEL" \
    --prompt-len 4096 --gen-len 128 --bits 2 3 4 --repeats 3 \
    --out "$RES/e2e_speed_v10_serving.json"

# --- session C: Triton-only ablation (appendix) ------------------------------
ONECOMP_TRITON_CODEC=1 python codec_speed.py --device cuda \
    --heads 32 --seq 4096 --bits 2 3 4 --repeats 20 \
    --out "$RES/codec_speed_v10_triton_only.json"
ONECOMP_TRITON_CODEC=1 python e2e_speed.py --model "$MODEL" \
    --prompt-len 4096 --gen-len 128 --bits 2 3 4 --repeats 3 \
    --out "$RES/e2e_speed_v10_triton_only.json"

echo "ALL_DONE"

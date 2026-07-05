#!/bin/bash
#SBATCH --job-name=hq-genqc-v12
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/kvcomp_hayatequant/logs/hq-genqc-v12_%j.log

# v12: served-text verification of the fused serving path (Triton / +graph):
# greedy-generation agreement, serving-cache PPL parity, in-session speedup.
set -x
cd /lustre/home/frj/yuma-ichikawa/conference/onecomp-lab
source .venv/bin/activate
nvidia-smi --query-gpu=name,driver_version --format=csv
RES=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/kvcomp_hayatequant/results_speed
MODEL=/lustre/home/frj/yuma-ichikawa/conference/models/llama-3.1-8b-instruct

python papers/hayatequant_paper/experiments/gen_quality_check.py \
    --model "$MODEL" --bits 2 3 4 --gen-len 128 \
    --out "$RES/gen_quality_check_v12.json"
echo "EXIT_$?"

#!/usr/bin/env bash
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3
#SBATCH --exclude=kagura-gpu11
#SBATCH --job-name=hq_artifact_native
#SBATCH --output=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/onecomp-lab-kvcomp-correctness/papers/hayatequant_paper/experiments/results/development/hq_online_fallback_serialization_20260716/logs/native_%A_%a.out
#SBATCH --error=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/onecomp-lab-kvcomp-correctness/papers/hayatequant_paper/experiments/results/development/hq_online_fallback_serialization_20260716/logs/native_%A_%a.err

set -euo pipefail
repo=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/onecomp-lab-kvcomp-correctness
root=${repo}/papers/hayatequant_paper/experiments/results/development/hq_online_fallback_serialization_20260716
python=/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab/.venv/bin/python
rate=$((SLURM_ARRAY_TASK_ID + 2))
export PYTHONPATH=${repo}
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export KVCOMP_CODEBOOK_DIR=${root}/artifacts
cd "${repo}"
"${python}" -u benchmark/kvcomp-codec/hayatequant_vllm_full_native_smoke.py \
  --avg-bits "${rate}" \
  --recipe hq \
  --head-dim 128 \
  --kv-heads 2 \
  --query-heads 4 \
  --tokens 32 \
  --dtype bfloat16 \
  --seed 0 \
  --warmups 3 \
  --iters 10 \
  --assignment-mode exact \
  --output "${root}/native/hq_rate${rate}_smoke.json"

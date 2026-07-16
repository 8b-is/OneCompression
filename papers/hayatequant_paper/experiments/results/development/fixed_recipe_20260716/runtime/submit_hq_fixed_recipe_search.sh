#!/usr/bin/env bash
# Development-only seven-checkpoint search for one globally frozen HQ recipe.
# Candidate labels are rejected by the formal launch/aggregation boundary.
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-6%7
#SBATCH --job-name=hq_fixed_recipe
#SBATCH --output=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/hayate_iclr2027_refresh_20260715/logs/dev_hq_fixed_recipe/hq_fixed_%A_%a.out
#SBATCH --error=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/hayate_iclr2027_refresh_20260715/logs/dev_hq_fixed_recipe/hq_fixed_%A_%a.err

set -euo pipefail

repo=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/onecomp-lab-kvcomp-correctness
run_root=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/hayate_iclr2027_refresh_20260715
python=/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab/.venv/bin/python
driver=${run_root}/runtime/hq_fixed_recipe_search.py
model_root=/lustre/home/frj/yuma-ichikawa/conference/models

models=(
  Llama-2-7b-hf
  llama-3.1-8b-instruct
  Qwen3-4B
  Qwen3-8B
  DeepSeek-R1-Distill-Llama-8B
  DeepSeek-R1-Distill-Qwen-7B
  DeepSeek-R1-Distill-Qwen-1.5B
)
keys=(llama2 llama31 qwen34 qwen38 r1llama8 r1qwen7 r1qwen15)

index=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
model=${models[$index]}
key=${keys[$index]}
output=${run_root}/results/dev/hq_fixed_recipe_v2_${key}_20260716

export PYTHONPATH=${repo}
export HF_HOME=${run_root}/data/hf_home
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

methods='[TurboQuant-paper,HQ-dev-paper-NC-LSiter-Gauss,HQ-dev-fixed-delta0-Gauss-NC-LSiter,HQ-dev-fixed-delta05-Gauss-NC-LSiter,HQ-dev-fixed-delta05-Beta-NC-LSiter,HQ-dev-fixed-delta05-Gauss-NC-LS,HQ-dev-fixed-delta05-b2-Gauss-NC-LSiter,HQ-dev-fixed-delta05-b2-Beta-NC-LSiter,HQ-dev-fixed-delta1-Beta-NC-LSiter]'

cd "${repo}"
"${python}" -u "${driver}" \
  model_path="${model_root}/${model}" \
  device=cuda:0 \
  dtype=bfloat16 \
  output_dir="${output}" \
  'bits=[2,3,4]' \
  sink_tokens=0 \
  recent_tokens=0 \
  capture_tokens=512 \
  timing_heads=8 \
  timing_seq=4096 \
  timing_iters=20 \
  run_ppl=true \
  ppl_max_length=512 \
  ppl_stride=256 \
  ppl_windows=8 \
  "+methods=${methods}" \
  +skip_outliers=false \
  hydra.run.dir="${output}/hydra"

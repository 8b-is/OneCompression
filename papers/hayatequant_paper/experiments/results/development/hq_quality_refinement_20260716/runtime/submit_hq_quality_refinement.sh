#!/usr/bin/env bash
# Development-only globally predeclared HQ/AdaptiveHQ refinement panel.
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-13%14
#SBATCH --job-name=hq_quality_refine
#SBATCH --output=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/onecomp-lab-kvcomp-correctness/papers/hayatequant_paper/experiments/results/development/hq_quality_refinement_20260716/logs/hq_quality_refine_%A_%a.out
#SBATCH --error=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/onecomp-lab-kvcomp-correctness/papers/hayatequant_paper/experiments/results/development/hq_quality_refinement_20260716/logs/hq_quality_refine_%A_%a.err

set -euo pipefail

repo=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/onecomp-lab-kvcomp-correctness
result_root=${repo}/papers/hayatequant_paper/experiments/results/development/hq_quality_refinement_20260716
run_root=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/hayate_iclr2027_refresh_20260715
python=/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab/.venv/bin/python
driver=${result_root}/runtime/hq_quality_refinement_search.py
model_root=/lustre/home/frj/yuma-ichikawa/conference/models

models=(
  llama-3.1-8b-instruct
  Llama-2-7b-hf
  Qwen3-4B
  Qwen3-8B
  DeepSeek-R1-Distill-Llama-8B
  DeepSeek-R1-Distill-Qwen-1.5B
  DeepSeek-R1-Distill-Qwen-7B
)
keys=(llama31 llama2 qwen34 qwen38 r1llama8 r1qwen15 r1qwen7)

task=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
model_index=$((task % 7))
if (( task < 7 )); then
  kind=fixed
  bits='[2]'
  methods='[TurboQuant-paper,HQ,DEVONLY-HQ-b4-gaussian-auto,DEVONLY-HQ-b4-gaussian-fibinit,DEVONLY-HQ-b4-finite-beta]'
else
  kind=adaptive
  bits='[2,3,4]'
  methods='[DEVONLY-AdaptiveHQ-safe-kw1,DEVONLY-AdaptiveHQ-safe-kw4,DEVONLY-AdaptiveHQ-safe-kw16]'
fi
model=${models[$model_index]}
key=${keys[$model_index]}
model_path=${model_root}/${model}
output=${result_root}/raw/${key}/${kind}
manifest_dir=${result_root}/manifests/${key}/${kind}

mkdir -p "${output}" "${manifest_dir}"
test -d "${model_path}"

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
export HQ_REFINEMENT_KIND=${kind}

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
  echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
  echo "node=${SLURMD_NODENAME}"
  echo "kind=${kind}"
  echo "model=${model}"
  echo "model_path=${model_path}"
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head=$(git -C "${repo}" rev-parse HEAD)"
  if test -n "$(git -C "${repo}" status --porcelain)"; then
    echo "git_dirty=true"
  else
    echo "git_dirty=false"
  fi
  sha256sum \
    "${driver}" \
    "${repo}/benchmark/kvcomp-codec/codec_benchmark.py" \
    "${repo}/onecomp/kvcomp/codec/hayate/codec.py" \
    "${repo}/onecomp/kvcomp/codec/hayate/recipes.py" \
    "${repo}/onecomp/kvcomp/codec/hayate/vq.py" \
    "${repo}/onecomp/kvcomp/eval/ppl.py"
  "${python}" --version
  nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
} > "${manifest_dir}/runtime.txt"

record_exit() {
  local status=$?
  trap - EXIT
  {
    echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit_status=${status}"
  } >> "${manifest_dir}/runtime.txt"
  exit "${status}"
}
trap record_exit EXIT

cd "${repo}"
"${python}" -u "${driver}" \
  model_path="${model_path}" \
  device=cuda:0 \
  dtype=bfloat16 \
  output_dir="${output}" \
  "bits=${bits}" \
  sink_tokens=0 \
  recent_tokens=0 \
  capture_tokens=512 \
  timing_heads=1 \
  timing_seq=64 \
  timing_iters=20 \
  run_ppl=true \
  ppl_max_length=512 \
  ppl_stride=512 \
  ppl_windows=1 \
  "+methods=${methods}" \
  +skip_outliers=false \
  hydra.run.dir="${output}/hydra"

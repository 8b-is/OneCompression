#!/usr/bin/env bash
# Development-only native parity/speed gate for rate-2 HQ candidates.
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-4%5
#SBATCH --job-name=hq_r2_native_gate
#SBATCH --output=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/onecomp-lab-kvcomp-correctness/papers/hayatequant_paper/experiments/results/development/hq_quality_refinement_20260716/logs/hq_r2_native_gate_%A_%a.out
#SBATCH --error=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/onecomp-lab-kvcomp-correctness/papers/hayatequant_paper/experiments/results/development/hq_quality_refinement_20260716/logs/hq_r2_native_gate_%A_%a.err

set -euo pipefail

repo=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/onecomp-lab-kvcomp-correctness
root=${repo}/papers/hayatequant_paper/experiments/results/development/hq_quality_refinement_20260716
python=/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab/.venv/bin/python
smoke=${repo}/benchmark/kvcomp-codec/hayatequant_vllm_full_native_smoke.py
task=${SLURM_ARRAY_TASK_ID:?}
mkdir -p "${root}/native" "${root}/manifests/native_${task}"
export PYTHONPATH=${repo}:${repo}/benchmark/kvcomp-codec
export PYTHONHASHSEED=0

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
  echo "array_task_id=${task}"
  echo "node=${SLURMD_NODENAME}"
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sha256sum \
    "${smoke}" \
    "${root}/runtime/hq_rate2_native_abba.py" \
    "${repo}/vllm_plugins/kvcomp/hayate_full_config.py" \
    "${repo}/vllm_plugins/kvcomp/hayate_full_native.py" \
    "${repo}/vllm_plugins/kvcomp/hayate_full_triton.py"
  nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
} > "${root}/manifests/native_${task}/runtime.txt"

record_exit() {
  local status=$?
  trap - EXIT
  {
    echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit_status=${status}"
  } >> "${root}/manifests/native_${task}/runtime.txt"
  exit "${status}"
}
trap record_exit EXIT

cd "${repo}"
if (( task == 4 )); then
  "${python}" -u "${root}/runtime/hq_rate2_native_abba.py" \
    > "${root}/native/rate2_native_abba.json"
  exit 0
fi

names=(current_hq_b2 hq_b4_gaussian_auto hq_b4_gaussian_fibinit hq_b4_finite_beta)
name=${names[$task]}
args=(
  --avg-bits 2
  --recipe hq
  --assignment-mode exact
  --head-dim 128
  --kv-heads 2
  --query-heads 4
  --tokens 32
  --dtype bfloat16
  --seed 0
  --warmups 3
  --iters 10
  --output "${root}/native/${name}_smoke.json"
)
if (( task > 0 )); then
  args+=(
    --key-code-bits 10
    --key-block-size 4
    --value-code-bits 6
    --value-block-size 4
    --allow-canonical-k10v6-b4
  )
fi
if (( task == 2 )); then
  args+=(--dist gaussian --codebook-init fibquant)
elif (( task == 3 )); then
  args+=(--dist beta --codebook-init auto)
else
  args+=(--dist gaussian --codebook-init auto)
fi
"${python}" -u "${smoke}" "${args[@]}"

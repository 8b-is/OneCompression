#!/usr/bin/env bash
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-5%6
#SBATCH --exclude=kagura-gpu11
#SBATCH --job-name=hq_cb_artifact_fit
#SBATCH --output=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/onecomp-lab-kvcomp-correctness/papers/hayatequant_paper/experiments/results/development/hq_codebook_artifacts_20260716/logs/fit_%A_%a.out
#SBATCH --error=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/onecomp-lab-kvcomp-correctness/papers/hayatequant_paper/experiments/results/development/hq_codebook_artifacts_20260716/logs/fit_%A_%a.err

set -euo pipefail

repo=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/onecomp-lab-kvcomp-correctness
root=${repo}/papers/hayatequant_paper/experiments/results/development/hq_codebook_artifacts_20260716
python=/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab/.venv/bin/python
task=${SLURM_ARRAY_TASK_ID:?}
manifest=${root}/manifests/fit_${task}.txt

mkdir -p "${root}/logs" "${root}/manifests" "${root}/artifacts" "${root}/metadata"
export PYTHONPATH=${repo}
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
unset KVCOMP_CODEBOOK_DIR

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
  echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
  echo "node=${SLURMD_NODENAME}"
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  git -C "${repo}" rev-parse HEAD
  sha256sum "${root}/PREREGISTRATION.md" "${root}/runtime/fit_one_artifact.py" "${repo}/onecomp/kvcomp/codec/hayate/vq.py"
  "${python}" --version
  nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
} > "${manifest}"

record_exit() {
  local status=$?
  trap - EXIT
  {
    echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit_status=${status}"
  } >> "${manifest}"
  exit "${status}"
}
trap record_exit EXIT

cd "${repo}"
"${python}" -u "${root}/runtime/fit_one_artifact.py"

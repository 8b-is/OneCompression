#!/usr/bin/env bash
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%4
#SBATCH --exclude=kagura-gpu11
#SBATCH --job-name=hq_online_audit
#SBATCH --output=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/onecomp-lab-kvcomp-correctness/papers/hayatequant_paper/experiments/results/development/hq_online_fallback_serialization_20260716/logs/audit_%A_%a.out
#SBATCH --error=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/onecomp-lab-kvcomp-correctness/papers/hayatequant_paper/experiments/results/development/hq_online_fallback_serialization_20260716/logs/audit_%A_%a.err

set -euo pipefail
repo=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/onecomp-lab-kvcomp-correctness
root=${repo}/papers/hayatequant_paper/experiments/results/development/hq_online_fallback_serialization_20260716
python=/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab/.venv/bin/python
export PYTHONPATH=${repo}
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
unset KVCOMP_CODEBOOK_DIR
manifest=${root}/manifests/audit_${SLURM_ARRAY_TASK_ID}.txt
{
  echo "job_id=${SLURM_JOB_ID}"
  echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
  echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
  echo "node=${SLURMD_NODENAME}"
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sha256sum "${root}/runtime/audit_regeneration.py" "${root}/runtime/serialize_reference.py" "${repo}/onecomp/kvcomp/codec/hayate/vq.py"
  "${python}" --version
} > "${manifest}"
record_exit() {
  local status=$?
  trap - EXIT
  echo "exit_status=${status}" >> "${manifest}"
  echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${manifest}"
  exit "${status}"
}
trap record_exit EXIT
cd "${repo}"
"${python}" -u "${root}/runtime/audit_regeneration.py"

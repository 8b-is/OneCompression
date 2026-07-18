#!/usr/bin/env bash
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=kagura-gpu11
#SBATCH --job-name=hq_artifact_parity
#SBATCH --output=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/onecomp-lab-kvcomp-correctness/papers/hayatequant_paper/experiments/results/development/hq_online_fallback_serialization_20260716/logs/public_parity_%j.out
#SBATCH --error=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/onecomp-lab-kvcomp-correctness/papers/hayatequant_paper/experiments/results/development/hq_online_fallback_serialization_20260716/logs/public_parity_%j.err

set -euo pipefail
repo=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/onecomp-lab-kvcomp-correctness
root=${repo}/papers/hayatequant_paper/experiments/results/development/hq_online_fallback_serialization_20260716
python=/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab/.venv/bin/python
export PYTHONPATH=${repo}
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
unset KVCOMP_CODEBOOK_DIR
cd "${repo}"
"${python}" -u "${root}/runtime/public_payload_parity.py"

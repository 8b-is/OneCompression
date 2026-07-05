#!/usr/bin/env bash
#SBATCH --job-name=hayate-exp
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.log
# Generic Slurm wrapper for the HayateQuant paper experiments.
#
#   cd papers/hayatequant_paper/experiments
#   sbatch run_experiment.sh bench_kvasym_v8.py [/path/to/model]
#
# Runs any of the bench_*.py scripts in this directory with the repo venv.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
PY=$REPO/.venv/bin/python
SCRIPT=${1:?usage: sbatch run_experiment.sh <bench_script.py> [args...]}
shift || true
cd "$HERE"
mkdir -p results logs
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1
echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-NA} script=$SCRIPT start=$(date -Is)"
$PY -u "$SCRIPT" "$@"
echo "end=$(date -Is)"

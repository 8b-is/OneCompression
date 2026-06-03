#!/usr/bin/env bash
# ============================================================
# TaperKV Paper — Policy comparison benchmark (SLURM)
# Config: benchmark/kvcomp-longcontext/conf/paper_taperkv.yaml
#
# Usage (direct):
#   bash papers/taperkv_paper/scripts/run_policy_bench.sh [MODEL_KEY]
#   MODEL_KEY: llama | r1q7 | r1q14 | r1q15 | all (default)
#
# Usage (SLURM):
#   sbatch --gres=gpu:1 --time=24:00:00 papers/taperkv_paper/scripts/run_policy_bench.sh llama
# ============================================================
set -euo pipefail

REPO=/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab
PY=${REPO}/.venv/bin/python
BENCH=${REPO}/benchmark/kvcomp-longcontext
SCRIPT=${BENCH}/kv_benchmark.py
OUT_DIR=${BENCH}/results_taperkv

MODEL_KEY=${1:-all}

declare -A MODEL_PATHS=(
  [llama]="/lustre/home/frj/yuma-ichikawa/conference/models/Llama-3.1-8B-Instruct"
  [r1q7]="/lustre/home/frj/yuma-ichikawa/conference/models/DeepSeek-R1-Distill-Qwen-7B"
  [r1q14]="/lustre/home/frj/yuma-ichikawa/conference/models/DeepSeek-R1-Distill-Qwen-14B"
  [r1q15]="/lustre/home/frj/yuma-ichikawa/conference/models/DeepSeek-R1-Distill-Qwen-1.5B"
)

mkdir -p "$OUT_DIR"

run_one() {
  local key=$1
  local model_path=${MODEL_PATHS[$key]}
  echo "=== Policy bench: ${key} ==="
  HF_DATASETS_OFFLINE=1 $PY -u "${SCRIPT}" \
    --config-name paper_taperkv \
    model_path="${model_path}" \
    output_dir="${OUT_DIR}" \
    +resume=true \
    device=cuda:0
}

if [[ "$MODEL_KEY" == "all" ]]; then
  for key in llama r1q7 r1q14 r1q15; do
    run_one "$key"
  done
else
  run_one "$MODEL_KEY"
fi

$PY "${REPO}/papers/aggregate_results.py"
echo "Done. Results in ${OUT_DIR}"

#!/usr/bin/env bash
# ============================================================
# HALO Paper — Longcontext benchmark (Needle / RULER / LongBench / PPL)
# Config: benchmark/kvcomp-longcontext/conf/turboquant_paper.yaml
#
# Usage:
#   bash papers/halo_paper/scripts/run_longcontext_bench.sh [MODEL_KEY]
#   MODEL_KEY: llama | ministral | all (default)
# ============================================================
set -euo pipefail

REPO=/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab
PY=${REPO}/.venv/bin/python
BENCH=${REPO}/benchmark/kvcomp-longcontext
SCRIPT=${BENCH}/kv_benchmark.py

MODEL_KEY=${1:-all}

declare -A MODEL_PATHS=(
  [llama]="/lustre/home/frj/yuma-ichikawa/conference/models/Llama-3.1-8B-Instruct"
  [ministral]="/lustre/home/frj/yuma-ichikawa/conference/models/Ministral-8B-Instruct-2410"
)

declare -A RETR_DIRS=(
  [llama]="${BENCH}/results_tq_llama_retr"
  [ministral]="${BENCH}/results_tq_minis_retr"
)

declare -A LB_DIRS=(
  [llama]="${BENCH}/results_tq_llama_lb"
  [ministral]="${BENCH}/results_tq_minis_lb"
)

run_one() {
  local key=$1
  local model_path=${MODEL_PATHS[$key]}

  # ── Retrieval (Needle + RULER + PPL) ──
  echo "=== Retrieval: ${key} ==="
  mkdir -p "${RETR_DIRS[$key]}"
  HF_DATASETS_OFFLINE=1 $PY -u "${SCRIPT}" \
    --config-name turboquant_paper \
    model_path="${model_path}" \
    output_dir="${RETR_DIRS[$key]}" \
    run_longbench=false \
    device=cuda:0

  # ── LongBench ──
  echo "=== LongBench: ${key} ==="
  mkdir -p "${LB_DIRS[$key]}"
  HF_DATASETS_OFFLINE=1 $PY -u "${SCRIPT}" \
    --config-name turboquant_paper \
    model_path="${model_path}" \
    output_dir="${LB_DIRS[$key]}" \
    run_needle=false run_ruler=false run_ppl=false \
    run_longbench=true \
    device=cuda:0
}

if [[ "$MODEL_KEY" == "all" ]]; then
  for key in llama ministral; do
    run_one "$key"
  done
else
  run_one "$MODEL_KEY"
fi

$PY "${REPO}/papers/aggregate_results.py"
echo "Done."

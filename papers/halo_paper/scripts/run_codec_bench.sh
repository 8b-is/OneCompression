#!/usr/bin/env bash
# ============================================================
# HALO Paper — Codec benchmark (PPL + encode latency)
# Reproduces: papers/halo_paper/results/all_results.json
#             → codec_ppl_latency section
#
# Usage:
#   bash papers/halo_paper/scripts/run_codec_bench.sh [MODEL_KEY]
#   MODEL_KEY: llama | ministral | r1q7 | r1q15 | all (default)
# ============================================================
set -euo pipefail

REPO=/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab
PY=${REPO}/.venv/bin/python
BENCH=${REPO}/benchmark/kvcomp-codec
SCRIPT=${BENCH}/codec_benchmark.py

MODEL_KEY=${1:-all}

# ── Model paths ──────────────────────────────────────────────
declare -A MODEL_PATHS=(
  [llama]="/lustre/home/frj/yuma-ichikawa/conference/models/Llama-3.1-8B-Instruct"
  [ministral]="/lustre/home/frj/yuma-ichikawa/conference/models/Ministral-8B-Instruct-2410"
  [r1q7]="/lustre/home/frj/yuma-ichikawa/conference/models/DeepSeek-R1-Distill-Qwen-7B"
  [r1q15]="/lustre/home/frj/yuma-ichikawa/conference/models/DeepSeek-R1-Distill-Qwen-1.5B"
)

declare -A RESULT_DIRS=(
  [llama]="${BENCH}/results_paper_llama"
  [ministral]="${BENCH}/results_paper_ministral"
  [r1q7]="${BENCH}/results_paper_r1q7"
  [r1q15]="${BENCH}/results_paper_r1q15"
)

run_one() {
  local key=$1
  local model_path=${MODEL_PATHS[$key]}
  local out_dir=${RESULT_DIRS[$key]}
  mkdir -p "$out_dir"

  echo "=== Codec bench: ${key} ==="
  HF_DATASETS_OFFLINE=1 $PY -u "${SCRIPT}" \
    --config-name paper_halo \
    model_path="${model_path}" \
    output_dir="${out_dir}" \
    device=cuda:0
}

if [[ "$MODEL_KEY" == "all" ]]; then
  for key in llama ministral r1q7 r1q15; do
    run_one "$key"
  done
else
  run_one "$MODEL_KEY"
fi

# Re-aggregate results
$PY "${REPO}/papers/aggregate_results.py"
echo "Done. Results saved to papers/halo_paper/results/all_results.json"

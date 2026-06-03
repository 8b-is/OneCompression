#!/usr/bin/env bash
# ============================================================
# TaperKV Paper — Reasoning benchmark (AIME / GSM8K / MATH-500 / LiveCodeBench)
# Config: benchmark/kvcomp-longcontext/conf/paper_aime.yaml
#
# Usage:
#   bash papers/taperkv_paper/scripts/run_reasoning_bench.sh [MODEL_KEY] [BENCH_TYPE]
#   MODEL_KEY: r1l8 | r1q7 | r1q14 | r1q15 | qwq | all (default)
#   BENCH_TYPE: basic | aime | lcb | all (default)
#
# SLURM example:
#   sbatch --gres=gpu:1 --time=24:00:00 \
#     papers/taperkv_paper/scripts/run_reasoning_bench.sh r1q14 all
# ============================================================
set -euo pipefail

REPO=/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab
PY=${REPO}/.venv/bin/python
BENCH=${REPO}/benchmark/kvcomp-longcontext
SCRIPT=${BENCH}/kv_benchmark.py

MODEL_KEY=${1:-all}
BENCH_TYPE=${2:-all}

declare -A MODEL_PATHS=(
  [r1l8]="/lustre/home/frj/yuma-ichikawa/conference/models/DeepSeek-R1-Distill-Llama-8B"
  [r1q7]="/lustre/home/frj/yuma-ichikawa/conference/models/DeepSeek-R1-Distill-Qwen-7B"
  [r1q14]="/lustre/home/frj/yuma-ichikawa/conference/models/DeepSeek-R1-Distill-Qwen-14B"
  [r1q15]="/lustre/home/frj/yuma-ichikawa/conference/models/DeepSeek-R1-Distill-Qwen-1.5B"
  [qwq]="/lustre/home/frj/yuma-ichikawa/conference/models/QwQ-32B"
)

declare -A BASIC_DIRS=(
  [r1l8]="${BENCH}/results_aime_basic_r1l8"
  [r1q7]="${BENCH}/results_aime_basic_r1q7"
  [r1q14]="${BENCH}/results_aime_basic_r1q14"
  [r1q15]="${BENCH}/results_aime_basic_r1q15"
  [qwq]="${BENCH}/results_aime_basic_qwq"
)

declare -A AIME_DIRS=(
  [r1l8]="${BENCH}/results_aime_comp_r1l8"
  [r1q7]="${BENCH}/results_aime_comp_r1q7"
  [r1q14]="${BENCH}/results_aime_comp_r1q14_24"
  [r1q15]="${BENCH}/results_aime_comp_r1q15"
  [qwq]="${BENCH}/results_aime_comp_qwq_24"
)

run_basic() {
  local key=$1
  local out_dir=${BASIC_DIRS[$key]}
  mkdir -p "$out_dir"
  echo "=== Basic reasoning (GSM8K + MATH-500): ${key} ==="
  HF_DATASETS_OFFLINE=1 $PY -u "${SCRIPT}" \
    --config-name paper_aime \
    model_path="${MODEL_PATHS[$key]}" \
    output_dir="${out_dir}" \
    run_aime=false run_livecodebench=false \
    +resume=true \
    device=cuda:0
}

run_aime() {
  local key=$1
  local out_dir=${AIME_DIRS[$key]}
  mkdir -p "$out_dir"
  echo "=== AIME 2024: ${key} ==="
  HF_DATASETS_OFFLINE=1 $PY -u "${SCRIPT}" \
    --config-name paper_aime \
    model_path="${MODEL_PATHS[$key]}" \
    output_dir="${out_dir}" \
    run_gsm8k=false run_math=false run_ppl=false run_livecodebench=false \
    aime_years=[2024] \
    +resume=true \
    device=cuda:0
}

run_lcb() {
  local key=$1
  local out_dir=${AIME_DIRS[$key]}
  mkdir -p "$out_dir"
  echo "=== LiveCodeBench: ${key} ==="
  HF_DATASETS_OFFLINE=1 $PY -u "${SCRIPT}" \
    --config-name paper_aime \
    model_path="${MODEL_PATHS[$key]}" \
    output_dir="${out_dir}" \
    run_gsm8k=false run_math=false run_ppl=false run_needle=false run_ruler=false \
    run_aime=false run_livecodebench=true \
    +resume=true \
    device=cuda:0
}

KEYS=($MODEL_KEY)
[[ "$MODEL_KEY" == "all" ]] && KEYS=(r1l8 r1q7 r1q14 r1q15 qwq)

for key in "${KEYS[@]}"; do
  [[ "$BENCH_TYPE" == "all" || "$BENCH_TYPE" == "basic" ]] && run_basic "$key"
  [[ "$BENCH_TYPE" == "all" || "$BENCH_TYPE" == "aime"  ]] && [[ "$key" != "r1l8" ]] && run_aime "$key"
  [[ "$BENCH_TYPE" == "all" || "$BENCH_TYPE" == "lcb"   ]] && run_lcb "$key"
done

$PY "${REPO}/papers/aggregate_results.py"
echo "Done."

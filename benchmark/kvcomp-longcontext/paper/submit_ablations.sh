#!/usr/bin/env bash
# =============================================================================
# submit_ablations.sh
# KV-cache ablation studies: TaperKV (b_min, budget, K/V asymmetry)
#                            HALO (lever ablation)
#
# Covers:
#   E: HALO lever ablation on Ministral-8B           → results_ablation_minis/
#   F: HALO lever + TaperKV asymmetry on R1-Q7       → results_ablation_r1q7/
#   G: TaperKV b_min sensitivity sweep               → results_ablation_bmin_{llama,r1q7}/
#   H: Memory budget trade-off (target_mem sweep)    → results_ablation_budget_*/
#   J: HALO lever + TaperKV asymmetry on R1-L8       → results_ablation_r1l8/
#   M: TaperKV asymmetry + GSM/MATH reasoning        → results_ablation_reason_{r1q7,r1l8}/
#
# After all jobs complete, regenerate evaluation figures with:
#   python benchmark/kvcomp-longcontext/plot_ablations.py
#
# Usage (from onecomp-lab/ root or this directory):
#   bash benchmark/kvcomp-longcontext/paper/submit_ablations.sh
# =============================================================================
set -euo pipefail

# Adjust ROOT to your cluster path ──────────────────────────────────────────
WORK="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../" && pwd )"
PY=${PY:-$WORK/.venv/bin/python}
LC=$WORK/benchmark/kvcomp-longcontext
LOGS=${LOGS:-$LC/paper/logs}
MODELS=${MODELS:-${MODELS_ROOT:-}}
if [ -z "$MODELS" ]; then
  echo "Set MODELS or MODELS_ROOT to external model weights." >&2
  exit 2
fi

mkdir -p "$LOGS"

# Model paths ────────────────────────────────────────────────────────────────
LLAMA=$MODELS/llama-3.1-8b-instruct
R1L8=$MODELS/DeepSeek-R1-Distill-Llama-8B
R1Q7=$MODELS/DeepSeek-R1-Distill-Qwen-7B
MINIS=$MODELS/Ministral-8B-Instruct-2410

# SLURM submission helper ────────────────────────────────────────────────────
sub() {
  local name="$1" tlim="$2"; shift 2
  out=$(sbatch --parsable --job-name="$name" --partition=batch-1gpu --gres=gpu:1 \
        --time="$tlim" --output="$LOGS/${name}_%j.log" \
        --wrap="cd '$LC' && '$PY' -u $*")
  echo "  [$name] -> $out"
}

echo "======================================================================"
echo " submit_ablations.sh  $(date '+%Y-%m-%d %H:%M JST')"
echo "======================================================================"

# ===========================================================================
# Section E: HALO lever ablation — Ministral-8B
#   config: conf/paper_ablation.yaml
#   methods: TurboQuant-{3,4}bit, HALO-{h-noLS,hg-noLS,hg-LS,dense-LS}-{3,4}
#            TaperKV-{asym,sym,revsym}
# ===========================================================================
echo ""
echo "=== E: HALO lever ablation (Ministral-8B) ==="

sub e1-ablation-minis 12:00:00 \
    kv_benchmark.py --config-name paper_ablation \
    model_path="$MINIS" \
    output_dir="$LC/results_ablation_minis" \
    +resume=true

# ===========================================================================
# Section F: HALO lever + TaperKV K/V asymmetry — R1-Distill-Qwen-7B
# ===========================================================================
echo ""
echo "=== F: HALO lever + TaperKV asymmetry (R1-Q7) ==="

sub f1-ablation-r1q7 14:00:00 \
    kv_benchmark.py --config-name paper_ablation \
    model_path="$R1Q7" \
    output_dir="$LC/results_ablation_r1q7" \
    +resume=true

# ===========================================================================
# Section G: TaperKV b_min sensitivity sweep
#   config: conf/paper_ablation_bmin.yaml
#   Sweeps b_min in {2,3,4,5} (3-bit) and {2,4,6} (4-bit)
# ===========================================================================
echo ""
echo "=== G: b_min sensitivity sweep ==="

sub g1-bmin-llama 10:00:00 \
    kv_benchmark.py --config-name paper_ablation_bmin \
    model_path="$LLAMA" \
    output_dir="$LC/results_ablation_bmin_llama" \
    +resume=true

sub g2-bmin-r1q7 12:00:00 \
    kv_benchmark.py --config-name paper_ablation_bmin \
    model_path="$R1Q7" \
    output_dir="$LC/results_ablation_bmin_r1q7" \
    +resume=true

# ===========================================================================
# Section H: Memory budget trade-off
#   config: conf/paper_ablation_budget.yaml  (override target_mem per job)
#   Sweeps target_mem ∈ {0.10, 0.15, 0.20, 0.25, 0.35}
# ===========================================================================
echo ""
echo "=== H: Memory budget trade-off sweep ==="

for budget in 0.10 0.15 0.20 0.25 0.35; do
  bt=$(echo "$budget" | sed 's/0\.//')   # 10, 15, 20, 25, 35
  sub "h-budget-llama-$bt" 10:00:00 \
      kv_benchmark.py --config-name paper_ablation_budget \
      model_path="$LLAMA" \
      target_mem="$budget" \
      output_dir="$LC/results_ablation_budget_llama_$bt" \
      +resume=true
done

for budget in 0.10 0.15 0.20 0.25 0.35; do
  bt=$(echo "$budget" | sed 's/0\.//')
  sub "h-budget-r1q7-$bt" 12:00:00 \
      kv_benchmark.py --config-name paper_ablation_budget \
      model_path="$R1Q7" \
      target_mem="$budget" \
      output_dir="$LC/results_ablation_budget_r1q7_$bt" \
      +resume=true
done

# ===========================================================================
# Section J: HALO lever + TaperKV K/V asymmetry — R1-Distill-Llama-8B
# ===========================================================================
echo ""
echo "=== J: HALO lever + TaperKV asymmetry (R1-L8) ==="

sub j1-ablation-r1l8 12:00:00 \
    kv_benchmark.py --config-name paper_ablation \
    model_path="$R1L8" \
    output_dir="$LC/results_ablation_r1l8" \
    +resume=true

# ===========================================================================
# Section M: TaperKV K/V asymmetry impact on reasoning (GSM8K + MATH)
#   run_needle=false  run_ppl=false   (no retrieval/perplexity overhead)
# ===========================================================================
echo ""
echo "=== M: TaperKV asymmetry + reasoning metrics (R1-Q7 & R1-L8) ==="

sub m1-asym-r1q7-reason 16:00:00 \
    kv_benchmark.py --config-name paper_ablation \
    model_path="$R1Q7" \
    output_dir="$LC/results_ablation_reason_r1q7" \
    run_gsm8k=true gsm8k_limit=100 max_new_reason=4096 \
    run_math=true math_limit=100 max_new_math=4096 \
    run_needle=false run_ppl=false \
    +resume=true

sub m2-asym-r1l8-reason 12:00:00 \
    kv_benchmark.py --config-name paper_ablation \
    model_path="$R1L8" \
    output_dir="$LC/results_ablation_reason_r1l8" \
    run_gsm8k=true gsm8k_limit=100 max_new_reason=4096 \
    run_math=true math_limit=100 max_new_math=4096 \
    run_needle=false run_ppl=false \
    +resume=true

echo ""
echo "======================================================================"
echo " 全ablationジョブ投入完了  $(date '+%Y-%m-%d %H:%M JST')"
echo "======================================================================"

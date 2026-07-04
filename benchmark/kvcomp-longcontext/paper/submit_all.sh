#!/usr/bin/env bash
# ===========================================================================
# Full paper-benchmark battery for the two KV-compression papers, on kagura.
#
#   Paper 1 -- HayateQuant codec (codec quality/speed + retrieval/long-context)
#              proposed: HayateQuant (supersedes HALO)
#              competitors: TurboQuant (arXiv:2504.19874), HALO (prior best)
#   Paper 2 -- TaperKV recipe  (iso-memory retrieval + reasoning)
#              proposed: TaperKV / TurboTaper   competitors: ThinKV, PM-KVQ
#
# Many small jobs (one per model x benchmark slice) so each fits the 24 h
# partition limit and partial results are crash-safe (kv_benchmark.py persists
# the JSON after every method).
#
# Usage:
#   bash submit_all.sh            # submit everything
#   bash submit_all.sh paper1     # HayateQuant codec jobs only
#   bash submit_all.sh paper2     # TaperKV recipe jobs only
#   DRYRUN=1 bash submit_all.sh   # print sbatch lines without submitting
#
# Overridable env:
#   PY       python interpreter   (default: <repo>/.venv/bin/python)
#   MODELS   model directory root (default: .../conference/models)
#   PART     SLURM partition      (default: batch-1gpu)
#   EXCLUDE  nodes to avoid       (default: empty; e.g. kagura-gpu09,kagura-gpu10)
#
# Reproducibility notes (lessons baked into this battery):
#   * run_bench.sh exports HF_DATASETS_OFFLINE=1 -- avoids the HF datasets remote
#     freshness-check hang; every dataset (GSM8K/MATH-500/AIME/WikiText-2/
#     LongBench-E) is pre-cached on lustre.
#   * TaperKV floors are fixed in conf (key floor b_min=4 standard / 3 for the
#     3-bit headline, value floor b_min_value=2). With the old b_min=2 the
#     headline needle was 0.64; b_min>=3 restores needle 1.00 (see ./tests).
#   * R1-Qwen reasoning models run with needle/RULER OFF: the synthetic needle is
#     unreliable on thinking models (FullKV itself retrieves <0.7), so retrieval
#     is claimed on the instruct models (Llama/Ministral) and reasoning
#     non-degradation (GSM8K/MATH/AIME) on the R1 family.
#   * The 14B reasoning model caps context at 8192 (eager attention OOMs at
#     16384) and keeps eager so ThinKV's sparsity calibration stays faithful.
#   * LiveCodeBench is intentionally NOT submitted: its loader is incompatible
#     with datasets>=4.8 (see README "Known-excluded axis").
#   * If jobs fail at import with "CUDA device busy/unavailable" (transient
#     gemlite init contention when many land on one B200 node), set
#     EXCLUDE=<busy nodes> or just resubmit -- pending jobs start cleanly.
# ===========================================================================
set -euo pipefail

# --- repo-relative paths (this script lives in benchmark/kvcomp-longcontext/paper) ---
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
LC="$REPO/benchmark/kvcomp-longcontext"
CODEC="$REPO/benchmark/kvcomp-codec"
WORK="$HERE/run_bench.sh"
LOGS="$HERE/logs"
mkdir -p "$LOGS"

PY=${PY:-$REPO/.venv/bin/python}
MODELS=${MODELS:-/lustre/home/frj/yuma-ichikawa/conference/models}
PART=${PART:-batch-1gpu}
EXCLUDE=${EXCLUDE:-}

# --- models ---
LLAMA=$MODELS/llama-3.1-8b-instruct
MINISTRAL=$MODELS/Ministral-8B-Instruct-2410
R1L8=$MODELS/DeepSeek-R1-Distill-Llama-8B
R1Q15=$MODELS/DeepSeek-R1-Distill-Qwen-1.5B
R1Q7=$MODELS/DeepSeek-R1-Distill-Qwen-7B
R1Q14=$MODELS/DeepSeek-R1-Distill-Qwen-14B
QWQ=$MODELS/QwQ-32B

# Reduced method sets for the big (14B/32B) models so each HF-generate job fits
# the 24 h limit; small models run their config's full method list.
#   CORE7  -- paper_aime reasoning grid (proposed + ThinKV/PM-KVQ/KIVI)
#   CORE11 -- paper_taperkv iso-memory grid (adds eviction + uniform + 3-bit pts)
CORE7='methods=[FullKV,KIVI-INT2,ThinKV,PM-KVQ,TaperKV,TurboTaper,HayateQuant-3bit]'
CORE11='methods=[FullKV,Uniform-INT3,KIVI-INT2,StreamingLLM,ThinKV,PM-KVQ,TurboQuant-3bit,HayateQuant-3bit,TaperKV,TaperKV-3bit,TurboTaper-3bit]'

WHICH=${1:-all}
DRYRUN=${DRYRUN:-0}
NSUB=0

# sub <name> <time> <benchdir> <cmd...>
sub() {
  local name="$1" tlim="$2" bench="$3"; shift 3
  local exarg=""
  [ -n "$EXCLUDE" ] && exarg="--exclude=$EXCLUDE"
  echo "----------------------------------------------------------------"
  echo "[$name] part=$PART time=$tlim"
  echo "        cmd: $*"
  if [ "$DRYRUN" = "1" ]; then return 0; fi
  local out
  out=$(sbatch --parsable --job-name="$name" --partition="$PART" --gres=gpu:1 \
        $exarg --time="$tlim" --output="$LOGS/${name}_%j.log" "$WORK" "$bench" "$@")
  echo "        -> $out"
  NSUB=$((NSUB+1))
}

############################################################################
# PAPER 1 -- HayateQuant codec vs TurboQuant (HALO retained as baseline)
############################################################################
if [ "$WHICH" = all ] || [ "$WHICH" = paper1 ]; then

# 1a. Codec intrinsic quality (codec_benchmark.py, paper_hayatequant): KV-MSE / attn err.
sub p1-hq-llama  02:00:00 "$CODEC" \
    "$PY" -u codec_benchmark.py --config-name paper_hayatequant \
    model_path="$LLAMA" output_dir="$CODEC/results_paper_llama"
sub p1-hq-minis  02:00:00 "$CODEC" \
    "$PY" -u codec_benchmark.py --config-name paper_hayatequant \
    model_path="$MINISTRAL" output_dir="$CODEC/results_paper_ministral"
sub p1-hq-r1q7   02:00:00 "$CODEC" \
    "$PY" -u codec_benchmark.py --config-name paper_hayatequant \
    model_path="$R1Q7" output_dir="$CODEC/results_paper_r1q7"
sub p1-hq-r1q15  02:00:00 "$CODEC" \
    "$PY" -u codec_benchmark.py --config-name paper_hayatequant \
    model_path="$R1Q15" output_dir="$CODEC/results_paper_r1q15"

# 1b. Codec speed x quality Pareto (codec_speed.py, CUDA).
sub p1-speed-synth 01:00:00 "$LC" \
    "$PY" -u codec_speed.py --device cuda --heads 32 --seq 4096 --dim 128 \
    --bits 2 3 4 --repeats 30 --out "$LC/results_paper1/codec_speed/codec_speed_cuda.json"
sub p1-speed-llama 01:00:00 "$LC" \
    "$PY" -u codec_speed.py --device cuda --model "$LLAMA" \
    --bits 2 3 4 --repeats 30 --out "$LC/results_paper1/codec_speed/codec_speed_llama.json"

# 1c. Retrieval grid (turboquant_paper): Needle + RULER + PPL, full bit grid.
sub p1-tq-llama-retr 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name turboquant_paper \
    model_path="$LLAMA" output_dir="$LC/results_paper1/retrieval_llama" run_longbench=false
sub p1-tq-minis-retr 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name turboquant_paper_ministral \
    model_path="$MINISTRAL" output_dir="$LC/results_paper1/retrieval_minis" run_longbench=false

# 1d. LongBench-E downstream generation (Table 1).
sub p1-tq-llama-lb 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name turboquant_paper \
    model_path="$LLAMA" output_dir="$LC/results_paper1/longbench_llama" \
    run_needle=false run_ruler=false run_ppl=false run_longbench=true
sub p1-tq-minis-lb 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name turboquant_paper_ministral \
    model_path="$MINISTRAL" output_dir="$LC/results_paper1/longbench_minis" \
    run_needle=false run_ruler=false run_ppl=false run_longbench=true

# 1e. End-task lever ablation (paper_ablation, legacy HALO levers + TaperKV asymmetry).
sub p1-ablation-llama 16:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_ablation \
    model_path="$LLAMA" output_dir="$LC/results_ablation/halo_llama"

fi

############################################################################
# PAPER 2 -- TaperKV / TurboTaper recipe (iso-memory retrieval + reasoning)
############################################################################
if [ "$WHICH" = all ] || [ "$WHICH" = paper2 ]; then

# 2a. Iso-memory retrieval + reasoning headline (paper_taperkv).
# Llama (instruct): needle/RULER ON (FullKV retrieves perfectly).
sub p2-taper-llama 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_taperkv \
    model_path="$LLAMA" output_dir="$LC/results_paper2/taperkv"

# R1-Qwen (thinking): needle/RULER OFF; still measures GSM8K + MATH + PPL.
NONEEDLE='run_needle=false run_ruler=false'
sub p2-taper-r1q15 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_taperkv \
    model_path="$R1Q15" output_dir="$LC/results_paper2/taperkv" $NONEEDLE
sub p2-taper-r1q7 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_taperkv \
    model_path="$R1Q7" output_dir="$LC/results_paper2/taperkv" $NONEEDLE
sub p2-taper-r1q14 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_taperkv \
    model_path="$R1Q14" output_dir="$LC/results_paper2/taperkv" \
    $NONEEDLE gsm8k_limit=50 math_limit=50 "$CORE11"

# 2b. Reasoning grid (paper_aime): basic = GSM8K + MATH-500 + PPL.
aime_basic='run_livecodebench=false aime_years=[]'
sub p2-basic-r1q15 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_aime model_path="$R1Q15" \
    output_dir="$LC/results_paper2/basic_r1q15" $aime_basic
sub p2-basic-r1l8  24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_aime model_path="$R1L8" \
    output_dir="$LC/results_paper2/basic_r1l8" $aime_basic
sub p2-basic-r1q7  24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_aime model_path="$R1Q7" \
    output_dir="$LC/results_paper2/basic_r1q7" $aime_basic
sub p2-basic-r1q14 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_aime model_path="$R1Q14" \
    output_dir="$LC/results_paper2/basic_r1q14" $aime_basic gsm8k_limit=50 math_limit=50 "$CORE7"
sub p2-basic-qwq   24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_aime model_path="$QWQ" \
    output_dir="$LC/results_paper2/basic_qwq" $aime_basic \
    gsm8k_limit=30 math_limit=30 max_new_reason=6144 max_new_math=6144 "$CORE7"

# 2c. Reasoning grid (paper_aime): comp = AIME-2024 + AIME-2025 (split per year
# for the big models). k tuned with model size (k=8/4/2/1).
aime_only='run_gsm8k=false run_math=false run_ppl=false run_livecodebench=false'
sub p2-aime-r1q15  24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_aime model_path="$R1Q15" \
    output_dir="$LC/results_paper2/comp_r1q15" $aime_only aime_years='[2024,2025]' aime_k=8
sub p2-aime-r1l8   24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_aime model_path="$R1L8" \
    output_dir="$LC/results_paper2/comp_r1l8" $aime_only aime_years='[2024,2025]' aime_k=4
sub p2-aime-r1q7   24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_aime model_path="$R1Q7" \
    output_dir="$LC/results_paper2/comp_r1q7" $aime_only aime_years='[2024,2025]' aime_k=4
sub p2-aime24-r1q14 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_aime model_path="$R1Q14" \
    output_dir="$LC/results_paper2/comp_r1q14_24" $aime_only aime_years='[2024]' aime_k=2 max_new_aime=6144 "$CORE7"
sub p2-aime25-r1q14 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_aime model_path="$R1Q14" \
    output_dir="$LC/results_paper2/comp_r1q14_25" $aime_only aime_years='[2025]' aime_k=2 max_new_aime=6144 "$CORE7"
sub p2-aime24-qwq  24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_aime model_path="$QWQ" \
    output_dir="$LC/results_paper2/comp_qwq_24" $aime_only aime_years='[2024]' aime_k=1 max_new_aime=6144 "$CORE7"
sub p2-aime25-qwq  24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_aime model_path="$QWQ" \
    output_dir="$LC/results_paper2/comp_qwq_25" $aime_only aime_years='[2025]' aime_k=1 max_new_aime=6144 "$CORE7"

fi

echo "================================================================"
if [ "$DRYRUN" = "1" ]; then
  echo "DRYRUN: nothing submitted."
else
  echo "submitted $NSUB jobs. monitor: squeue -u \$USER | grep -E 'p1-|p2-'"
fi

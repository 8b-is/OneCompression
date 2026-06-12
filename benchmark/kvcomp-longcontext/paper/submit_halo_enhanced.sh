#!/usr/bin/env bash
# ===========================================================================
# Enhanced-HALO re-run battery (paper folder), kagura / batch-1gpu / B200.
#
# Re-runs the paper benchmarks for the ENHANCED HALO codec only (the
# role-asymmetric dequant-scale variants), self-contained against the
# references it must beat:
#
#   * HALO-NC-{2,3,4}bit    proposed lever 1: norm-correcting (nc) keys keep the
#                           rotated norm so the attention logit shrinks by
#                           cos(theta) instead of cos^2; least-squares (ls)
#                           values stay MSE-optimal.  Iso-storage to plain HALO.
#   * HALO-NCIT-{2,3,4}bit  lever 1 + lever 2 (ls_iter on values: one Lloyd
#                           alternating-minimisation step, monotone-lower MSE).
#
# References co-run for a direct, iso-bit/iso-memory comparison:
#   FullKV, TurboQuant-{2,3,4}bit, HALO-{2,3,4}bit (the legacy LS codec).
#
# Results land in dedicated dirs (results_halo_enh_*) so this battery does not
# touch the existing paper result JSONs.  kv_benchmark.py persists JSON after
# every method, so a crash/time-out keeps partial results.
#
# Usage:
#   bash submit_halo_enhanced.sh            # submit everything
#   bash submit_halo_enhanced.sh retrieval  # instruct models (Llama/Ministral)
#   bash submit_halo_enhanced.sh reason      # R1/QwQ reasoning models
#   bash submit_halo_enhanced.sh speed       # latency / throughput (codec + e2e)
#   DRYRUN=1 bash submit_halo_enhanced.sh    # print sbatch lines, submit nothing
#
# Overridable env: PY, MODELS, PART, EXCLUDE (same as submit_all.sh).
# ===========================================================================
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
LC="$REPO/benchmark/kvcomp-longcontext"
WORK="$HERE/run_bench.sh"
LOGS="$HERE/logs"
mkdir -p "$LOGS"

PY=${PY:-$REPO/.venv/bin/python}
MODELS=${MODELS:-/lustre/home/frj/yuma-ichikawa/conference/models}
PART=${PART:-batch-1gpu}
EXCLUDE=${EXCLUDE:-}

LLAMA=$MODELS/llama-3.1-8b-instruct
MINISTRAL=$MODELS/Ministral-8B-Instruct-2410
R1L8=$MODELS/DeepSeek-R1-Distill-Llama-8B
R1Q15=$MODELS/DeepSeek-R1-Distill-Qwen-1.5B
R1Q7=$MODELS/DeepSeek-R1-Distill-Qwen-7B
R1Q14=$MODELS/DeepSeek-R1-Distill-Qwen-14B
QWQ=$MODELS/QwQ-32B

# Enhanced HALO + the iso-bit references it must match/beat.
RETR_METHODS='methods=[FullKV,TurboQuant-2bit,TurboQuant-2.5bit,TurboQuant-3bit,TurboQuant-3.5bit,TurboQuant-4bit,HALO-2bit,HALO-3bit,HALO-4bit,HALO-NC-2bit,HALO-NC-2.5bit,HALO-NC-3bit,HALO-NC-3.5bit,HALO-NC-4bit,HALO-NCIT-2bit,HALO-NCIT-2.5bit,HALO-NCIT-3bit,HALO-NCIT-3.5bit,HALO-NCIT-4bit]'
REASON_METHODS='methods=[FullKV,HALO-3bit,HALO-4bit,HALO-NC-3bit,HALO-NC-4bit,HALO-NCIT-3bit,HALO-NCIT-4bit]'

WHICH=${1:-all}
DRYRUN=${DRYRUN:-0}
NSUB=0

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
# Retrieval / PPL / LongBench on the instruct models (needle is meaningful).
############################################################################
if [ "$WHICH" = all ] || [ "$WHICH" = retrieval ]; then

# Llama-3.1-8B-Instruct: needle + RULER + PPL (no LongBench, keep within budget).
sub he-retr-llama 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name turboquant_paper \
    model_path="$LLAMA" output_dir="$LC/results_halo_enh_llama_retr" \
    run_longbench=false "$RETR_METHODS"

# LongBench-E downstream generation (Table 1) on Llama.
sub he-lb-llama 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name turboquant_paper \
    model_path="$LLAMA" output_dir="$LC/results_halo_enh_llama_lb" \
    run_needle=false run_ruler=false run_ppl=false run_longbench=true "$RETR_METHODS"

# Ministral-8B-Instruct-2410: TurboQuant's 2nd model, same retrieval grid.
sub he-retr-minis 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name turboquant_paper_ministral \
    model_path="$MINISTRAL" output_dir="$LC/results_halo_enh_minis_retr" \
    run_longbench=false "$RETR_METHODS"

fi

############################################################################
# Reasoning non-degradation (GSM8K + MATH-500 + PPL) on the R1/QwQ models.
# Needle/RULER OFF (synthetic needle is unreliable on thinking models).
############################################################################
if [ "$WHICH" = all ] || [ "$WHICH" = reason ]; then

basic='run_livecodebench=false aime_years=[] run_needle=false run_ruler=false'

sub he-basic-r1q15 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_aime model_path="$R1Q15" \
    output_dir="$LC/results_halo_enh_basic_r1q15" $basic "$REASON_METHODS"
sub he-basic-r1l8 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_aime model_path="$R1L8" \
    output_dir="$LC/results_halo_enh_basic_r1l8" $basic "$REASON_METHODS"
sub he-basic-r1q7 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_aime model_path="$R1Q7" \
    output_dir="$LC/results_halo_enh_basic_r1q7" $basic "$REASON_METHODS"
sub he-basic-r1q14 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_aime model_path="$R1Q14" \
    output_dir="$LC/results_halo_enh_basic_r1q14" $basic \
    gsm8k_limit=50 math_limit=50 "$REASON_METHODS"
sub he-basic-qwq 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_aime model_path="$QWQ" \
    output_dir="$LC/results_halo_enh_basic_qwq" $basic \
    gsm8k_limit=30 math_limit=30 max_new_reason=6144 max_new_math=6144 "$REASON_METHODS"

fi

############################################################################
# Inference speed: codec-level latency/throughput (prefill S=4096 + decode S=1)
# and end-to-end TTFT / decode tok/s / peak memory on a real model.
############################################################################
if [ "$WHICH" = all ] || [ "$WHICH" = speed ]; then

# Codec-level speed x quality (now includes HALO-NC / HALO-NCIT + decode S=1).
sub he-speed-synth 01:00:00 "$LC" \
    "$PY" -u codec_speed.py --device cuda --heads 32 --seq 4096 --dim 128 \
    --bits 2 3 4 --repeats 50 --out "$LC/results_codec_speed/codec_speed_enh_cuda.json"
sub he-speed-llama 01:00:00 "$LC" \
    "$PY" -u codec_speed.py --device cuda --model "$LLAMA" \
    --bits 2 3 4 --repeats 50 --out "$LC/results_codec_speed/codec_speed_enh_llama.json"

# End-to-end generation latency/throughput (TTFT + decode tok/s + peak mem).
sub he-e2e-llama 02:00:00 "$LC" \
    "$PY" -u e2e_speed.py --model "$LLAMA" --device cuda \
    --prompt-len 4096 --gen-len 128 --bits 2 3 4 --repeats 3 --dtype float16 \
    --out "$LC/results_codec_speed/e2e_llama.json"
sub he-e2e-r1q7 02:00:00 "$LC" \
    "$PY" -u e2e_speed.py --model "$R1Q7" --device cuda \
    --prompt-len 4096 --gen-len 128 --bits 2 3 4 --repeats 3 --dtype float16 \
    --out "$LC/results_codec_speed/e2e_r1q7.json"

fi

echo "================================================================"
if [ "$DRYRUN" = "1" ]; then
  echo "DRYRUN: nothing submitted."
else
  echo "submitted $NSUB jobs. monitor: squeue -u \$USER | grep -E 'he-'"
fi

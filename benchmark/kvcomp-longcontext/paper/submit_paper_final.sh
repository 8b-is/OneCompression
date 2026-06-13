#!/usr/bin/env bash
# ===========================================================================
# Final paper benchmark battery — comprehensive re-run with HALO-NCIT default.
#
# Fills all gaps identified in the June 2026 audit:
#
#   [P1] Paper1 full comparison table (HALO-NCIT + ALL baselines)
#        turboquant_paper.yaml now has HALO-*bit = NC+ls_iter (default).
#        Results land in results_final/ (separate from old results_paper1/).
#
#   [P2] Ministral-8B LongBench with HALO-NCIT (was missing entirely).
#
#   [P3] R1-1.5B AIME comp gap fill (Uniform-INT4 / StreamingLLM / ThinKV /
#        PM-KVQ were missing in both results_paper2/comp_r1q15 and
#        results_aime_comp_r1q15).
#
#   [P4] HALO-enhanced basic R1-1.5B gap fill (HALO-NC-3bit / HALO-NCIT-3bit /
#        HALO-NCIT-4bit were missing in results_halo_enh/basic_r1q15).
#
# Usage:
#   bash submit_paper_final.sh            # submit everything
#   bash submit_paper_final.sh paper1     # P1+P2 only (retrieval + LongBench)
#   bash submit_paper_final.sh gap        # P3+P4 only (gap fills)
#   DRYRUN=1 bash submit_paper_final.sh   # print sbatch lines without submitting
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

# Core methods for the final paper comparison table.
# HALO-*bit now maps to the NC+ls_iter default (validated best).
# Ablation variants (HALO-NC-* / HALO-LS-* / HALO-dense-LS-*) are included
# in turboquant_paper.yaml's full list so they run automatically.
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
# [P1] Final paper1 retrieval+PPL grid (needle + RULER + PPL, all methods).
#      Uses turboquant_paper.yaml which now has HALO-NCIT as default HALO.
#      Output goes to results_final/ to preserve existing results_paper1/.
############################################################################
if [ "$WHICH" = all ] || [ "$WHICH" = paper1 ]; then

# --- Llama-3.1-8B-Instruct: needle + RULER + PPL ---
sub p1f-retr-llama 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name turboquant_paper \
    model_path="$LLAMA" \
    output_dir="$LC/results_final/retrieval_llama" \
    run_longbench=false

# --- Llama-3.1-8B-Instruct: LongBench-E downstream generation ---
sub p1f-lb-llama 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name turboquant_paper \
    model_path="$LLAMA" \
    output_dir="$LC/results_final/longbench_llama" \
    run_needle=false run_ruler=false run_ppl=false run_longbench=true

# --- Ministral-8B-Instruct-2410: needle + RULER + PPL ---
sub p1f-retr-minis 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name turboquant_paper_ministral \
    model_path="$MINISTRAL" \
    output_dir="$LC/results_final/retrieval_minis" \
    run_longbench=false

# [P2] Ministral-8B LongBench with HALO-NCIT (previously entirely missing).
sub p1f-lb-minis 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name turboquant_paper_ministral \
    model_path="$MINISTRAL" \
    output_dir="$LC/results_final/longbench_minis" \
    run_needle=false run_ruler=false run_ppl=false run_longbench=true

fi

############################################################################
# [P3] AIME comp gap fill for R1-1.5B.
#      Missing methods: Uniform-INT4, StreamingLLM, ThinKV, PM-KVQ.
#      kv_benchmark.py appends to existing JSON, so only the missing methods
#      are computed; already-filled entries are skipped automatically.
############################################################################
if [ "$WHICH" = all ] || [ "$WHICH" = gap ]; then

MISSING_R1Q15='methods=[Uniform-INT4,StreamingLLM,ThinKV,PM-KVQ]'
AIME_ONLY='run_gsm8k=false run_math=false run_ppl=false run_livecodebench=false'

# Fill in results_paper2/comp_r1q15
sub p3-comp-r1q15 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_aime \
    model_path="$R1Q15" \
    output_dir="$LC/results_paper2/comp_r1q15" \
    $AIME_ONLY aime_years='[2024,2025]' aime_k=8 \
    "$MISSING_R1Q15"

# Fill in results_aime_comp_r1q15 (legacy dir, same gap)
sub p3-aime-r1q15 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_aime \
    model_path="$R1Q15" \
    output_dir="$LC/results_aime_comp_r1q15" \
    $AIME_ONLY aime_years='[2024,2025]' aime_k=8 \
    "$MISSING_R1Q15"

############################################################################
# [P4] HALO-enhanced basic R1-1.5B gap fill.
#      Missing: HALO-NC-3bit, HALO-NCIT-3bit, HALO-NCIT-4bit.
############################################################################
MISSING_BASIC='methods=[HALO-NC-3bit,HALO-NCIT-3bit,HALO-NCIT-4bit]'
BASIC='run_livecodebench=false aime_years=[] run_needle=false run_ruler=false'

sub p4-basic-r1q15 24:00:00 "$LC" \
    "$PY" -u kv_benchmark.py --config-name paper_aime \
    model_path="$R1Q15" \
    output_dir="$LC/results_halo_enh/basic_r1q15" \
    $BASIC "$MISSING_BASIC"

fi

echo "================================================================"
if [ "$DRYRUN" = "1" ]; then
  echo "DRYRUN: nothing submitted."
else
  echo "submitted $NSUB jobs."
  echo "monitor: squeue -u \$USER | grep -E 'p1f-|p3-|p4-'"
fi

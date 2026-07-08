#!/usr/bin/env bash
# Submit the HayateQuant paper benchmark battery from the onecomp-lab repo.
#
# This script is intentionally a thin orchestrator over the checked-in benchmark
# entry points. It keeps all paper numbers reproducible from the kvcomp branch:
# PPL/distortion, zero-shot ACC, long-context retrieval, reasoning-heavy decode,
# codec speed, end-to-end decode speed, and the native vLLM kernel smoke.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
LC="$REPO/benchmark/kvcomp-longcontext"
PY=${PY:-$REPO/.venv/bin/python}
PYVLLM=${PYVLLM:-$REPO/.venv-vllm/bin/python}
MODELS=${MODELS:-/lustre/home/frj/yuma-ichikawa/conference/models}
OUT=${OUT:-/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/kvcomp_hayatequant}
RUN_ROOT=${RUN_ROOT:-$OUT/paper_final_$(date +%Y%m%d)}
PARTITIONS=${PARTITIONS:-debug,batch-1gpu}
DRYRUN=${DRYRUN:-0}
LOGS="$RUN_ROOT/logs"
mkdir -p "$LOGS"

PAPER_METHODS=(
  TurboQuant-hadamard
  FibQuant
  BlockQuant
  HayateQuant-selected
  HayateQuant-frontier
  HayateQuant-auto
  HayateQuant-K3V1-BQscale
  HayateQuant-K4V2-BQscale
  HayateQuant-K5V3-BQscale
  HayateQuant-FibK3V1-BQscale
  HayateQuant-FibK4V2-BQscale
  HayateQuant-FibK5V3-BQscale
)

submit_wrap() {
  local name="$1" time="$2" mem="$3" cpus="$4" wrap="$5"
  echo "submit $name partitions=$PARTITIONS time=$time"
  if [ "$DRYRUN" = "1" ]; then
    echo "  sbatch --job-name=$name --partition=$PARTITIONS --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem --time=$time --wrap $wrap"
    return 0
  fi
  sbatch --parsable --job-name="$name" --partition="$PARTITIONS" --gres=gpu:1 \
    --cpus-per-task="$cpus" --mem="$mem" --time="$time" \
    --output="$LOGS/${name}_%j.log" --wrap "$wrap"
}

common_env='export TOKENIZERS_PARALLELISM=false; export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}; export PYTHONUNBUFFERED=1'

submit_ppl() {
  if [ "$DRYRUN" = "1" ]; then
    "$PY" "$HERE/hayatequant_cross_model.py" --submit --dry-run \
      --partition "$PARTITIONS" --time 08:00:00 --mem 180G --cpus 8 \
      --output-root "$RUN_ROOT/ppl" \
      --override "methods=[$(IFS=,; echo "${PAPER_METHODS[*]}")]"
    return 0
  fi
  "$PY" "$HERE/hayatequant_cross_model.py" --submit \
    --partition "$PARTITIONS" --time 08:00:00 --mem 180G --cpus 8 \
    --output-root "$RUN_ROOT/ppl" \
    --override "methods=[$(IFS=,; echo "${PAPER_METHODS[*]}")]"
}

submit_zeroshot() {
  local dry_args=()
  [ "$DRYRUN" = "1" ] && dry_args=(--dry-run)
  "$PY" "$HERE/hayatequant_zeroshot.py" --submit "${dry_args[@]}" \
    --partition "$PARTITIONS" --time 1-00:00:00 --mem 180G --cpus 20 \
    --output-root "$RUN_ROOT/zeroshot" \
    --methods "${PAPER_METHODS[@]}"
}

submit_long_reason() {
  for spec in \
    "llama31-8b:$MODELS/llama-3.1-8b-instruct" \
    "ministral-8b:$MODELS/Ministral-8B-Instruct-2410" \
    "qwen3-8b:$MODELS/Qwen3-8B"; do
    key=${spec%%:*}
    model=${spec#*:}
    [ -d "$model" ] || continue
    submit_wrap "hqlong-$key" 1-00:00:00 180G 20 \
      "cd $REPO; $common_env; $PY -u benchmark/kvcomp-longcontext/kv_benchmark.py --config-name hayatequant_fib_blockquant model_path=$model output_dir=$RUN_ROOT/long/$key"
  done

  for spec in \
    "r1-llama-8b:$MODELS/DeepSeek-R1-Distill-Llama-8B" \
    "r1-qwen-7b:$MODELS/DeepSeek-R1-Distill-Qwen-7B" \
    "r1-qwen-1.5b:$MODELS/DeepSeek-R1-Distill-Qwen-1.5B"; do
    key=${spec%%:*}
    model=${spec#*:}
    [ -d "$model" ] || continue
    submit_wrap "hqreason-$key" 1-00:00:00 180G 20 \
      "cd $REPO; $common_env; $PY -u benchmark/kvcomp-longcontext/kv_benchmark.py --config-name hayatequant_fib_blockquant run_needle=false run_ruler=false run_longbench=false run_ppl=true ppl_max_length=2048 ppl_stride=1024 ppl_windows=16 run_gsm8k=true gsm8k_limit=100 max_new_reason=4096 run_math=true math_limit=100 max_new_math=4096 model_path=$model output_dir=$RUN_ROOT/reason/$key"
  done
}

submit_speed_native() {
  submit_wrap "hqspeed-synth" 04:00:00 120G 12 \
    "cd $LC; $common_env; $PY -u codec_speed.py --device cuda --heads 32 --seq 4096 --dim 128 --bits 2 3 4 --repeats 50 --out $RUN_ROOT/speed/codec_speed_synth.json"
  submit_wrap "hqspeed-llama" 06:00:00 180G 12 \
    "cd $LC; $common_env; $PY -u codec_speed.py --device cuda --model $MODELS/llama-3.1-8b-instruct --bits 2 3 4 --repeats 30 --out $RUN_ROOT/speed/codec_speed_llama.json"
  submit_e2e_speed
  submit_native_vllm
}

submit_e2e_speed() {
  submit_wrap "hqe2e-llama" 08:00:00 180G 12 \
    "cd $LC; $common_env; $PY -u e2e_speed.py --device cuda --model $MODELS/llama-3.1-8b-instruct --prompt-len 4096 --gen-len 128 --bits 2 3 4 --repeats 3 --dtype float16 --out $RUN_ROOT/speed/e2e_speed_llama.json"
}

submit_native_vllm() {
  if [ -x "$PYVLLM" ]; then
    submit_wrap "hqnative-vllm" 04:00:00 160G 16 \
      "cd $REPO; $common_env; $PYVLLM -u benchmark/kvcomp-codec/hayatequant_vllm_native_smoke.py --bits 3 4 --bench-iters 100 --output $RUN_ROOT/native/smoke.json"
  fi
}

summarize() {
  "$PY" "$HERE/hayatequant_cross_model.py" --summarize \
    --output-root "$RUN_ROOT/ppl" \
    --summary-out "$RUN_ROOT/ppl/summary.md" || true
  "$PY" "$HERE/hayatequant_zeroshot.py" --summarize \
    --output-root "$RUN_ROOT/zeroshot" \
    --summary-out "$RUN_ROOT/zeroshot/summary.md" || true
}

usage() {
  cat <<'EOF'
Usage: submit_hayatequant_paper.sh [all|ppl|zeroshot|long|speed|e2e-speed|native|summarize]

Environment:
  OUT=/lustre/.../kvcomp_hayatequant
  RUN_ROOT=/lustre/.../kvcomp_hayatequant/paper_final_YYYYMMDD
  MODELS=/lustre/.../models
  PARTITIONS=debug,batch-1gpu
  DRYRUN=0
  PY=/path/to/.venv/bin/python
  PYVLLM=/path/to/.venv-vllm/bin/python
EOF
}

cmd=${1:-all}
case "$cmd" in
  all)
    submit_ppl
    submit_zeroshot
    submit_long_reason
    submit_speed_native
    ;;
  ppl) submit_ppl ;;
  zeroshot) submit_zeroshot ;;
  long|reason) submit_long_reason ;;
  speed) submit_speed_native ;;
  e2e-speed) submit_e2e_speed ;;
  native) submit_native_vllm ;;
  summarize) summarize ;;
  -h|--help|help) usage ;;
  *) usage; exit 2 ;;
esac

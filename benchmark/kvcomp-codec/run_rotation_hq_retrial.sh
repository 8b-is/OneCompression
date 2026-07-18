#!/usr/bin/env bash
# Reproducible split-disjoint Rotation HQ development measurement.
set -euo pipefail

REPO=$(git -C "$(dirname "${BASH_SOURCE[0]}")/../.." rev-parse --show-toplevel)
RESULT_ROOT=${HQ_ROTATION_RESULT_ROOT:?set HQ_ROTATION_RESULT_ROOT}
MODEL_KEY=${HQ_ROTATION_MODEL_KEY:?set HQ_ROTATION_MODEL_KEY}
MODEL_PATH=${HQ_ROTATION_MODEL_PATH:?set HQ_ROTATION_MODEL_PATH}
AVERAGE_BITS=${HQ_ROTATION_AVERAGE_BITS:-3}
CALIBRATION_TEXT=${HQ_ROTATION_CALIBRATION_TEXT:-"${RESULT_ROOT}/calibration/calibration_text.txt"}
HF_HOME_DIR=${HQ_ROTATION_HF_HOME:-"${RESULT_ROOT}/hf_home"}
OUT_DIR="${RESULT_ROOT}/models/${MODEL_KEY}/r${AVERAGE_BITS}"
TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/rotation-hq-retrial.XXXXXX")
FINALIZED=0

cleanup() {
  local status=$?
  if test "$FINALIZED" -eq 0; then
    rm -rf "$TMP_DIR"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM HUP

test -f "$CALIBRATION_TEXT"
test -d "$MODEL_PATH"
test ! -e "$OUT_DIR" || {
  echo "refusing to overwrite existing result: $OUT_DIR" >&2
  exit 2
}
git -C "$REPO" diff --quiet -- || {
  echo "Rotation HQ retrial requires a clean tracked source tree" >&2
  exit 2
}
git -C "$REPO" diff --cached --quiet || {
  echo "Rotation HQ retrial requires no staged tracked edits" >&2
  exit 2
}

mkdir -p "$TMP_DIR/benchmark"
python_bin=${PYTHON_BIN:-"${REPO}/.venv/bin/python"}
test -x "$python_bin"

git -C "$REPO" rev-parse HEAD > "$TMP_DIR/source_commit.txt"
sha256sum \
  "$REPO/benchmark/kvcomp-codec/run_rotation_hq_retrial.sh" \
  "$REPO/benchmark/kvcomp-codec/hayatequant_cayley_calibrate.py" \
  "$REPO/benchmark/kvcomp-codec/codec_benchmark.py" \
  "$REPO/onecomp/kvcomp/codec/hayate/codec.py" \
  "$REPO/onecomp/kvcomp/codec/hayate/calibration.py" \
  "$CALIBRATION_TEXT" > "$TMP_DIR/source_manifest.sha256"

printf '%q ' env "PYTHONPATH=${REPO}" "$python_bin" \
  "$REPO/benchmark/kvcomp-codec/hayatequant_cayley_calibrate.py" \
  --model-path "$MODEL_PATH" --output "$TMP_DIR/rotation_hq.pt" \
  --avg-bits "$AVERAGE_BITS" --no-codebook-refit --max-tokens 2048 \
  --layer-stride 4 --max-vectors 100000 --holdout-frac 0.1 --steps 32 \
  --lr 0.0005 --batch-vectors 8192 --refit-every 16 --seed 0 \
  --dtype bfloat16 --device auto --fit-device cuda --text-file "$CALIBRATION_TEXT" \
  > "$TMP_DIR/calibration_command.txt"
printf '\n' >> "$TMP_DIR/calibration_command.txt"

env PYTHONPATH="$REPO" "$python_bin" \
  "$REPO/benchmark/kvcomp-codec/hayatequant_cayley_calibrate.py" \
  --model-path "$MODEL_PATH" --output "$TMP_DIR/rotation_hq.pt" \
  --avg-bits "$AVERAGE_BITS" --no-codebook-refit --max-tokens 2048 \
  --layer-stride 4 --max-vectors 100000 --holdout-frac 0.1 --steps 32 \
  --lr 0.0005 --batch-vectors 8192 --refit-every 16 --seed 0 \
  --dtype bfloat16 --device auto --fit-device cuda --text-file "$CALIBRATION_TEXT" \
  > "$TMP_DIR/calibration.stdout.log" 2> "$TMP_DIR/calibration.stderr.log"

printf '%q ' env "PYTHONPATH=${REPO}" "HF_HOME=${HF_HOME_DIR}" \
  "HF_DATASETS_OFFLINE=1" "TRANSFORMERS_OFFLINE=1" \
  "HQ_ROTATION_CALIB_${AVERAGE_BITS}BIT=${TMP_DIR}/rotation_hq.pt" "$python_bin" \
  "$REPO/benchmark/kvcomp-codec/codec_benchmark.py" \
  "model_path=${MODEL_PATH}" "output_dir=${TMP_DIR}/benchmark" \
  "bits=[${AVERAGE_BITS}]" "methods=[HQ,RotationHQ]" \
  "capture_tokens=2048" "timing_iters=5" "ppl_max_length=2048" \
  "ppl_stride=1024" "ppl_windows=8" "sample_seed=0" > "$TMP_DIR/benchmark_command.txt"
printf '\n' >> "$TMP_DIR/benchmark_command.txt"

env PYTHONPATH="$REPO" HF_HOME="$HF_HOME_DIR" HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 "HQ_ROTATION_CALIB_${AVERAGE_BITS}BIT=${TMP_DIR}/rotation_hq.pt" \
  "$python_bin" "$REPO/benchmark/kvcomp-codec/codec_benchmark.py" \
  "model_path=${MODEL_PATH}" "output_dir=${TMP_DIR}/benchmark" \
  "bits=[${AVERAGE_BITS}]" "methods=[HQ,RotationHQ]" \
  "capture_tokens=2048" "timing_iters=5" "ppl_max_length=2048" \
  "ppl_stride=1024" "ppl_windows=8" "sample_seed=0" \
  > "$TMP_DIR/benchmark.stdout.log" 2> "$TMP_DIR/benchmark.stderr.log"

mkdir -p "$OUT_DIR"
mv "$TMP_DIR"/* "$OUT_DIR/"
(
  cd "$OUT_DIR"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
FINALIZED=1
rmdir "$TMP_DIR"
trap - EXIT INT TERM HUP

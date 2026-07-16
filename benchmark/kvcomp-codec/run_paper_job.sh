#!/usr/bin/env bash
# Reproducible worker wrapper for every HayateQuant paper benchmark.
set -euo pipefail
umask 077

usage() {
  printf '%s\n' \
    'Usage: run_paper_job.sh --repo DIR --python PY --run-root DIR --work-root DIR' \
    '       --job-key KEY --attempt-id ID --stdout-log FILE --stderr-log FILE' \
    '       --model DIR --input-manifest FILE --input-manifest-sha256 HEX' \
    '       [--smoke-approval-marker FILE --smoke-approval-marker-sha256 HEX' \
    '        --smoke-validated-plan-sha256 HEX]' \
    '       --expected-result FILE' \
    '       [--dataset-file FILE ...]' \
    '       -- COMMAND [ARGS ...]'
}

REPO=
PYTHON=
RUN_ROOT=
WORK_ROOT=
JOB_KEY=
ATTEMPT_ID=
STDOUT_LOG=
STDERR_LOG=
MODEL=
INPUT_MANIFEST=
INPUT_MANIFEST_SHA256=
SMOKE_APPROVAL_MARKER=
SMOKE_APPROVAL_MARKER_SHA256=
SMOKE_VALIDATED_PLAN_SHA256=
EXPECTED_RESULT=
DATASET_FILES=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo) REPO=$2; shift 2 ;;
    --python) PYTHON=$2; shift 2 ;;
    --run-root) RUN_ROOT=$2; shift 2 ;;
    --work-root) WORK_ROOT=$2; shift 2 ;;
    --job-key) JOB_KEY=$2; shift 2 ;;
    --attempt-id) ATTEMPT_ID=$2; shift 2 ;;
    --stdout-log) STDOUT_LOG=$2; shift 2 ;;
    --stderr-log) STDERR_LOG=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --input-manifest) INPUT_MANIFEST=$2; shift 2 ;;
    --input-manifest-sha256) INPUT_MANIFEST_SHA256=$2; shift 2 ;;
    --smoke-approval-marker) SMOKE_APPROVAL_MARKER=$2; shift 2 ;;
    --smoke-approval-marker-sha256) SMOKE_APPROVAL_MARKER_SHA256=$2; shift 2 ;;
    --smoke-validated-plan-sha256) SMOKE_VALIDATED_PLAN_SHA256=$2; shift 2 ;;
    --expected-result) EXPECTED_RESULT=$2; shift 2 ;;
    --dataset-file) DATASET_FILES+=("$2"); shift 2 ;;
    --) shift; break ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown wrapper argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$REPO" ] || [ -z "$PYTHON" ] || [ -z "$RUN_ROOT" ] || \
   [ -z "$WORK_ROOT" ] || [ -z "$JOB_KEY" ] || [ -z "$ATTEMPT_ID" ] || \
   [ -z "$STDOUT_LOG" ] || [ -z "$STDERR_LOG" ] || \
   [ -z "$MODEL" ] || [ -z "$INPUT_MANIFEST" ] || \
   [ -z "$INPUT_MANIFEST_SHA256" ] || [ -z "$EXPECTED_RESULT" ] || \
   [ "$#" -eq 0 ]; then
  usage >&2
  exit 2
fi
case "${CODE_REPO_COMMIT:-}" in
  ''|*[!0-9a-f]*)
    printf '%s\n' \
      'CODE_REPO_COMMIT must be a lowercase 40-hex commit fixed at submission time' >&2
    exit 2
    ;;
esac
if [ "${#CODE_REPO_COMMIT}" -ne 40 ]; then
  printf '%s\n' \
    'CODE_REPO_COMMIT must be a lowercase 40-hex commit fixed at submission time' >&2
  exit 2
fi
case "$JOB_KEY" in
  *[!A-Za-z0-9._-]*) printf 'Unsafe job key: %s\n' "$JOB_KEY" >&2; exit 2 ;;
esac
case "$ATTEMPT_ID" in
  ''|*[!A-Za-z0-9._-]*) printf 'Unsafe attempt ID: %s\n' "$ATTEMPT_ID" >&2; exit 2 ;;
esac
[ -x "$PYTHON" ] || { printf 'Python executable is missing: %s\n' "$PYTHON" >&2; exit 2; }
REPO=$(cd "$REPO" && pwd)
LIFECYCLE="$REPO/benchmark/kvcomp-codec/paper_job_lifecycle.py"
[ -f "$LIFECYCLE" ] || { printf 'Paper lifecycle helper is missing: %s\n' "$LIFECYCLE" >&2; exit 2; }
RUN_ROOT=$(realpath -m -- "$RUN_ROOT")
WORK_ROOT=$(realpath -m -- "$WORK_ROOT")
RAW_EXPECTED_RESULT=$EXPECTED_RESULT
EXPECTED_RESULT=$(realpath -m -- "$EXPECTED_RESULT")
case "$JOB_KEY" in
  smoke-*) JOB_PHASE=smoke ;;
  formal-*) JOB_PHASE=formal ;;
  gpu11-*) JOB_PHASE=smoke ;;
  *) printf 'Job key has no recognized result phase: %s\n' "$JOB_KEY" >&2; exit 2 ;;
esac
RESULT_PHASE_ROOT=$(realpath -m -- "$RUN_ROOT/results/$JOB_PHASE")
case "$EXPECTED_RESULT" in
  "$RESULT_PHASE_ROOT/"*) ;;
  *) printf 'Expected result escapes its phase root: %s\n' "$EXPECTED_RESULT" >&2; exit 2 ;;
esac
RUN_ID=${SLURM_JOB_ID:-local-$$}
[ -z "${SLURM_RESTART_COUNT:-}" ] || RUN_ID+="-restart-${SLURM_RESTART_COUNT}"
case "$RUN_ID" in
  *[!A-Za-z0-9._-]*) printf 'Unsafe run ID: %s\n' "$RUN_ID" >&2; exit 2 ;;
esac
LOG_JOB_ID=${SLURM_JOB_ID:-$RUN_ID}
LOG_JOB_NAME=${SLURM_JOB_NAME:-$JOB_KEY}
case "$LOG_JOB_NAME" in
  *[!A-Za-z0-9._-]*) printf 'Unsafe scheduler job name: %s\n' "$LOG_JOB_NAME" >&2; exit 2 ;;
esac
STDOUT_LOG=${STDOUT_LOG//%x/$LOG_JOB_NAME}
STDOUT_LOG=${STDOUT_LOG//%j/$LOG_JOB_ID}
STDERR_LOG=${STDERR_LOG//%x/$LOG_JOB_NAME}
STDERR_LOG=${STDERR_LOG//%j/$LOG_JOB_ID}
case "$STDOUT_LOG$STDERR_LOG" in
  *%*) printf '%s\n' 'Unresolved scheduler placeholder in log path' >&2; exit 2 ;;
esac
STDOUT_LOG=$(realpath -m -- "$STDOUT_LOG")
STDERR_LOG=$(realpath -m -- "$STDERR_LOG")
case "$STDOUT_LOG" in "$RUN_ROOT/logs/$JOB_PHASE/"*) ;; *)
  printf 'Worker stdout log escapes repo-local phase logs: %s\n' "$STDOUT_LOG" >&2; exit 2 ;;
esac
case "$STDERR_LOG" in "$RUN_ROOT/logs/$JOB_PHASE/"*) ;; *)
  printf 'Worker stderr log escapes repo-local phase logs: %s\n' "$STDERR_LOG" >&2; exit 2 ;;
esac
PROVENANCE="$RUN_ROOT/provenance/${JOB_KEY}_${RUN_ID}.json"
WORKER_RECORD="$RUN_ROOT/job_records/$JOB_PHASE/$JOB_KEY/$ATTEMPT_ID/worker.json"

"$PYTHON" "$LIFECYCLE" validate-root \
  --source-repo "$REPO" --run-root "$RUN_ROOT" \
  --expected-commit "$CODE_REPO_COMMIT" >/dev/null
mkdir -p "$(dirname "$WORKER_RECORD")" "$RUN_ROOT/provenance"

WORKER_STAGE=wrapper_start
WORKER_REASON=worker_exit
WORKER_FINALIZED=0
write_worker_receipt() {
  local status=$1 reason=$2 exit_code=${3:-}
  local -a receipt=(
    "$PYTHON" "$LIFECYCLE" record
    --source-repo "$REPO" --run-root "$RUN_ROOT"
    --source-commit "$CODE_REPO_COMMIT"
    --path "$WORKER_RECORD" --kind worker --status "$status"
    --phase "$JOB_PHASE" --job-key "$JOB_KEY" --attempt-id "$ATTEMPT_ID"
    --expected-result "$EXPECTED_RESULT" --stage "$WORKER_STAGE"
    --reason-code "$reason" --provenance "$PROVENANCE"
    --stdout-log "$STDOUT_LOG" --stderr-log "$STDERR_LOG"
  )
  [ -z "$exit_code" ] || receipt+=(--exit-code "$exit_code")
  [ -z "${SLURM_JOB_ID:-}" ] || receipt+=(--slurm-job-id "$SLURM_JOB_ID")
  "${receipt[@]}"
}
worker_on_exit() {
  local exit_code=$?
  trap - EXIT HUP INT TERM
  if [ "$WORKER_FINALIZED" != 1 ]; then
    [ "$exit_code" -ne 0 ] || exit_code=1
    write_worker_receipt failed "$WORKER_REASON" "$exit_code" >/dev/null || {
      printf 'Could not persist terminal worker failure receipt: %s\n' \
        "$WORKER_RECORD" >&2
      [ "$exit_code" -ne 0 ] || exit_code=125
    }
  fi
  exit "$exit_code"
}
worker_on_signal() {
  local signal_name=$1 exit_code=$2
  WORKER_STAGE=signal
  WORKER_REASON="signal_${signal_name}"
  exit "$exit_code"
}
write_worker_receipt worker_started worker_invoked >/dev/null
trap worker_on_exit EXIT
trap 'worker_on_signal hup 129' HUP
trap 'worker_on_signal int 130' INT
trap 'worker_on_signal term 143' TERM

WORKER_STAGE=argument_validation
WORKER_REASON=argument_validation_failed
case "$INPUT_MANIFEST_SHA256" in
  *[!0-9a-f]*)
    printf '%s\n' '--input-manifest-sha256 must be lowercase hexadecimal' >&2
    exit 2
    ;;
esac
if [ "${#INPUT_MANIFEST_SHA256}" -ne 64 ]; then
  printf '%s\n' '--input-manifest-sha256 must contain exactly 64 hexadecimal digits' >&2
  exit 2
fi
case "$JOB_KEY" in
  formal-*)
    if [ -z "$SMOKE_APPROVAL_MARKER" ] || \
       [ -z "$SMOKE_APPROVAL_MARKER_SHA256" ] || \
       [ -z "$SMOKE_VALIDATED_PLAN_SHA256" ]; then
      printf '%s\n' \
        'Formal jobs require the submitter-bound smoke marker and validated-plan digests' >&2
      exit 2
    fi
    for SMOKE_DIGEST in "$SMOKE_APPROVAL_MARKER_SHA256" \
      "$SMOKE_VALIDATED_PLAN_SHA256"; do
      case "$SMOKE_DIGEST" in
        *[!0-9a-f]*|'')
          printf '%s\n' 'Formal smoke binding digests must be lowercase SHA-256' >&2
          exit 2
          ;;
      esac
      [ "${#SMOKE_DIGEST}" -eq 64 ] || {
        printf '%s\n' 'Formal smoke binding digests must contain 64 hexadecimal digits' >&2
        exit 2
      }
    done
    if [ -z "${PAPER_SMOKE_APPROVAL_MARKER:-}" ] || \
       [ -z "${PAPER_SMOKE_APPROVAL_MARKER_SHA256:-}" ] || \
       [ -z "${PAPER_SMOKE_VALIDATED_PLAN_SHA256:-}" ]; then
      printf '%s\n' 'Formal job environment omitted its submitter-bound smoke identity' >&2
      exit 2
    fi
    if [ "$PAPER_SMOKE_APPROVAL_MARKER" != "$SMOKE_APPROVAL_MARKER" ] || \
       [ "$PAPER_SMOKE_APPROVAL_MARKER_SHA256" != "$SMOKE_APPROVAL_MARKER_SHA256" ] || \
       [ "$PAPER_SMOKE_VALIDATED_PLAN_SHA256" != "$SMOKE_VALIDATED_PLAN_SHA256" ]; then
      printf '%s\n' 'Formal job argv/environment smoke identities disagree' >&2
      exit 2
    fi
    ;;
  *)
    if [ -n "$SMOKE_APPROVAL_MARKER" ] || \
       [ -n "$SMOKE_APPROVAL_MARKER_SHA256" ] || \
       [ -n "$SMOKE_VALIDATED_PLAN_SHA256" ]; then
      printf '%s\n' 'Smoke approval bindings are valid only for formal jobs' >&2
      exit 2
    fi
    ;;
esac
if [ -L "$RAW_EXPECTED_RESULT" ] || [ -L "$EXPECTED_RESULT" ] || \
   [ -e "$EXPECTED_RESULT" ]; then
  printf 'Expected result already exists; refusing overwrite/resume race: %s\n' \
    "$EXPECTED_RESULT" >&2
  exit 2
fi

# Submission-side squeue checks are intentionally fail-fast but are not an
# atomic concurrency primitive.  Hold one advisory lock derived from the
# canonical result path for the worker lifetime, so concurrent launchers with
# different job names still cannot write the same paper result.
command -v flock >/dev/null 2>&1 || {
  printf '%s\n' 'flock is required for per-result paper-job locking' >&2
  exit 2
}
command -v sha256sum >/dev/null 2>&1 || {
  printf '%s\n' 'sha256sum is required for per-result paper-job locking' >&2
  exit 2
}
case "$WORK_ROOT" in
  "$RUN_ROOT"|"$RUN_ROOT/"*)
    printf '%s\n' 'WORK_ROOT must remain external to repo-local RUN_ROOT' >&2
    exit 2
    ;;
esac
case "$RUN_ROOT" in
  "$WORK_ROOT/"*)
    printf '%s\n' 'repo-local RUN_ROOT must not be nested below WORK_ROOT' >&2
    exit 2
    ;;
esac
RESULTS_WORKTREE_ROOT=$(realpath -m -- "$RUN_ROOT/../../../../..")
case "$WORK_ROOT" in
  "$RESULTS_WORKTREE_ROOT"|"$RESULTS_WORKTREE_ROOT/"*)
    printf '%s\n' 'WORK_ROOT caches must remain external to the results Git worktree' >&2
    exit 2
    ;;
esac
RESULT_LOCK_ROOT=$(realpath -m -- "$WORK_ROOT/locks/results")
case "$RESULT_LOCK_ROOT" in
  "$WORK_ROOT/"*) ;;
  *) printf 'Result lock root escapes work root: %s\n' "$RESULT_LOCK_ROOT" >&2; exit 2 ;;
esac
mkdir -p "$RESULT_LOCK_ROOT"
RESULT_LOCK_LINE=$(printf '%s\0' "$EXPECTED_RESULT" | sha256sum)
RESULT_LOCK_ID=${RESULT_LOCK_LINE%% *}
case "$RESULT_LOCK_ID" in
  *[!0-9a-f]*|'') printf '%s\n' 'Could not derive a result lock identity' >&2; exit 2 ;;
esac
[ "${#RESULT_LOCK_ID}" -eq 64 ] || {
  printf '%s\n' 'Could not derive a result lock identity' >&2
  exit 2
}
RESULT_LOCK="$RESULT_LOCK_ROOT/$RESULT_LOCK_ID.lock"
exec 9>"$RESULT_LOCK"
if ! flock -n 9; then
  printf 'Another worker already owns the result lock: %s\n' "$EXPECTED_RESULT" >&2
  exit 2
fi
# Close the check/lock race: a previous writer may have completed immediately
# before this process acquired the lock.
if [ -L "$RAW_EXPECTED_RESULT" ] || [ -L "$EXPECTED_RESULT" ] || \
   [ -e "$EXPECTED_RESULT" ]; then
  printf 'Expected result appeared before lock acquisition: %s\n' \
    "$EXPECTED_RESULT" >&2
  exit 2
fi
INPUT_MANIFEST=$(realpath -e -- "$INPUT_MANIFEST")
manifest_count=0
for file in "${DATASET_FILES[@]}"; do
  [ "$(realpath -e -- "$file")" = "$INPUT_MANIFEST" ] && manifest_count=$((manifest_count + 1))
done
if [ "$manifest_count" -ne 1 ]; then
  printf '%s\n' \
    '--input-manifest must also be supplied exactly once as --dataset-file' >&2
  exit 2
fi
JOB_TMP="$WORK_ROOT/tmp/${JOB_KEY}_${RUN_ID}"
PAPER_HOME="$WORK_ROOT/runtime/home"
mkdir -p "$JOB_TMP" "$RUN_ROOT/provenance" "$PAPER_HOME"

# Defence in depth for direct/local invocations.  Slurm and the SSH launcher
# already use explicit environment allow-lists, but worker children must never
# inherit access tokens, cloud credentials, or experiment-service API keys.
unset GITHUB_TOKEN GH_TOKEN GITLAB_TOKEN GITLAB_ACCESS_TOKEN GIT_TOKEN \
  GIT_ASKPASS SSH_ASKPASS SSH_AUTH_SOCK HF_TOKEN HUGGING_FACE_HUB_TOKEN \
  WANDB_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY NPM_TOKEN PYPI_TOKEN \
  AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE \
  AZURE_STORAGE_KEY AZURE_CLIENT_SECRET GOOGLE_APPLICATION_CREDENTIALS \
  GOOGLE_API_KEY GOOGLE_OAUTH_ACCESS_TOKEN KRB5CCNAME X509_USER_PROXY NETRC

export PYTHONPATH="$REPO"
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=disabled
export HF_HOME=${HF_HOME:-$WORK_ROOT/data/hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HOME="$PAPER_HOME"
export TMPDIR="$JOB_TMP"
export TMP="$JOB_TMP"
export TEMP="$JOB_TMP"
export TRITON_CACHE_DIR="$JOB_TMP/triton"
export TORCHINDUCTOR_CACHE_DIR="$JOB_TMP/torchinductor"
export MPLCONFIGDIR="$JOB_TMP/matplotlib"
export PAPER_PROVENANCE_PATH="$PROVENANCE"
if [ -n "${KVCOMP_CODEBOOK_DIR:-}" ]; then
  printf '%s\n' \
    'KVCOMP_CODEBOOK_DIR is forbidden for formal jobs; use the tracked codebook artifacts' >&2
  exit 2
fi
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# Formal timing owns every GPU GRES on one batch-8gpu node so no other GPU
# process can perturb the measurement.  Keep that allocation visible in the
# immutable preflight record, but expose exactly the first assigned B200 to the
# benchmark process.  This must happen before paper_preflight imports torch.
case "$JOB_KEY" in
  formal-speed-*)
    [ "${SLURM_JOB_PARTITION:-}" = batch-8gpu ] || {
      printf '%s\n' 'Formal speed requires SLURM_JOB_PARTITION=batch-8gpu' >&2
      exit 2
    }
    case "${SLURM_GPUS_ON_NODE:-}" in
      8|gpu:nvidia_b200:8) ;;
      *) printf 'Formal speed did not receive eight B200 GRES: %s\n' \
           "${SLURM_GPUS_ON_NODE:-unset}" >&2; exit 2 ;;
    esac
    [ "${SLURM_CPUS_PER_TASK:-}" = 160 ] || {
      printf 'Formal speed did not receive 160 CPUs per task: %s\n' \
        "${SLURM_CPUS_PER_TASK:-unset}" >&2
      exit 2
    }
    case "${SLURM_JOB_CPUS_PER_NODE:-}" in
      160|160\(x1\)) ;;
      *) printf 'Formal speed did not receive 160 CPUs on its node: %s\n' \
           "${SLURM_JOB_CPUS_PER_NODE:-unset}" >&2; exit 2 ;;
    esac
    [ "${SLURM_MEM_PER_NODE:-}" = 2457600 ] || {
      printf 'Formal speed did not receive 2400G node memory: %s MiB\n' \
        "${SLURM_MEM_PER_NODE:-unset}" >&2
      exit 2
    }
    SPEED_ALLOCATION_CVD=${CUDA_VISIBLE_DEVICES:-}
    SPEED_ALLOCATION_JOB_GPUS=${SLURM_JOB_GPUS:-}
    IFS=',' read -r -a SPEED_VISIBLE_DEVICES <<< "$SPEED_ALLOCATION_CVD"
    IFS=',' read -r -a SPEED_JOB_GPUS <<< "$SPEED_ALLOCATION_JOB_GPUS"
    [ "${#SPEED_VISIBLE_DEVICES[@]}" -eq 8 ] && \
      [ "${#SPEED_JOB_GPUS[@]}" -eq 8 ] || {
      printf 'Formal speed allocation is not eight-GPU: CVD=%s JOB_GPUS=%s\n' \
        "$SPEED_ALLOCATION_CVD" "$SPEED_ALLOCATION_JOB_GPUS" >&2
      exit 2
    }
    declare -A SPEED_DEVICE_SET=()
    for SPEED_DEVICE in "${SPEED_VISIBLE_DEVICES[@]}"; do
      case "$SPEED_DEVICE" in
        ''|*[!A-Za-z0-9._:-]*)
          printf 'Unsafe CUDA device identity in formal speed allocation: %s\n' \
            "$SPEED_DEVICE" >&2
          exit 2
          ;;
      esac
      [ -z "${SPEED_DEVICE_SET[$SPEED_DEVICE]+present}" ] || {
        printf 'Repeated CUDA device in formal speed allocation: %s\n' "$SPEED_DEVICE" >&2
        exit 2
      }
      SPEED_DEVICE_SET[$SPEED_DEVICE]=1
    done
    export PAPER_SPEED_ALLOCATION_CUDA_VISIBLE_DEVICES="$SPEED_ALLOCATION_CVD"
    export PAPER_SPEED_MEASUREMENT_CUDA_VISIBLE_DEVICE="${SPEED_VISIBLE_DEVICES[0]}"
    export CUDA_VISIBLE_DEVICES="${SPEED_VISIBLE_DEVICES[0]}"
    ;;
esac

# Re-run the full semantic smoke validator inside every formal allocation.
# The fixed marker/plan digests are also checked independently by
# paper_preflight.py below, which records their exact file identities in the
# immutable provenance envelope.  Clear the completion sentinel first so a
# direct invocation cannot inherit a false success claim from its caller.
unset PAPER_SMOKE_VALIDATOR_CHECK_COMPLETED
case "$JOB_KEY" in
  formal-*)
    WORKER_STAGE=smoke_gate
    WORKER_REASON=smoke_gate_failed
    SMOKE_GATE_VALIDATOR="$REPO/benchmark/kvcomp-codec/validate_hayatequant_smoke_gate.py"
    [ -f "$SMOKE_GATE_VALIDATOR" ] || {
      printf 'Formal smoke gate validator is missing: %s\n' "$SMOKE_GATE_VALIDATOR" >&2
      exit 2
    }
    "$PYTHON" "$SMOKE_GATE_VALIDATOR" check \
      --run-root "$RUN_ROOT" \
      --expected-commit "$CODE_REPO_COMMIT" \
      --input-manifest "$INPUT_MANIFEST" \
      --input-manifest-sha256 "$INPUT_MANIFEST_SHA256" \
      --approval-marker "$SMOKE_APPROVAL_MARKER"
    export PAPER_SMOKE_VALIDATOR_CHECK_COMPLETED="${SMOKE_APPROVAL_MARKER_SHA256}:${SMOKE_VALIDATED_PLAN_SHA256}"
    ;;
esac

WORKER_STAGE=preflight
WORKER_REASON=preflight_failed
PREFLIGHT=(
  "$PYTHON" "$REPO/benchmark/kvcomp-codec/paper_preflight.py"
  --repo "$REPO"
  --expected-commit "$CODE_REPO_COMMIT"
  --output "$PROVENANCE"
)
PREFLIGHT+=(
  --model "$MODEL"
  --input-manifest "$INPUT_MANIFEST"
  --expected-input-manifest-sha256 "$INPUT_MANIFEST_SHA256"
)
case "$JOB_KEY" in
  formal-*)
    PREFLIGHT+=(
      --run-root "$RUN_ROOT"
      --smoke-approval-marker "$SMOKE_APPROVAL_MARKER"
      --expected-smoke-approval-marker-sha256 "$SMOKE_APPROVAL_MARKER_SHA256"
      --expected-smoke-validated-plan-sha256 "$SMOKE_VALIDATED_PLAN_SHA256"
    )
    ;;
esac
for file in "${DATASET_FILES[@]}"; do
  PREFLIGHT+=(--dataset-file "$file")
done
"${PREFLIGHT[@]}"

# A job can wait in Slurm after the submitter's existence check.  Recheck just
# before entering benchmark code so an output created while pending is never
# silently resumed, merged, or overwritten under this job identity.
if [ -L "$RAW_EXPECTED_RESULT" ] || [ -L "$EXPECTED_RESULT" ] || \
   [ -e "$EXPECTED_RESULT" ]; then
  printf 'Expected result appeared while this job was pending: %s\n' \
    "$EXPECTED_RESULT" >&2
  exit 2
fi

printf 'paper_job key=%s run_id=%s commit=%s\n' "$JOB_KEY" "$RUN_ID" "$CODE_REPO_COMMIT"
printf 'paper_job command:'
printf ' %q' "$@"
printf '\n'
cd "$REPO"
WORKER_STAGE=benchmark
WORKER_REASON=benchmark_failed
set +e
"$@"
BENCHMARK_EXIT_CODE=$?
set -e
if [ "$BENCHMARK_EXIT_CODE" -ne 0 ]; then
  exit "$BENCHMARK_EXIT_CODE"
fi

# A zero process status is not scientific success by itself.  The lifecycle
# helper independently requires a regular, non-empty, strict JSON object and
# records its exact byte count and SHA-256 before publishing success.
WORKER_STAGE=result_validation
WORKER_REASON=result_validation_failed
write_worker_receipt succeeded benchmark_and_result_validated 0 >/dev/null
WORKER_FINALIZED=1
trap - EXIT HUP INT TERM
exit 0

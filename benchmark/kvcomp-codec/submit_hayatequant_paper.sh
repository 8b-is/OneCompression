#!/usr/bin/env bash
# Submit the auditable HayateQuant ICLR paper reproduction battery.
#
# Scientific contract
# -------------------
# * A formal launch is accepted only from a completely clean, fixed git HEAD.
# * Every GPU job enters through run_paper_job.sh, which verifies the commit,
#   import path, model metadata, every indexed weight shard by full SHA-256,
#   dataset/protocol files, and CUDA allocation.
# * Hugging Face inputs come only from the pinned, run-local HF_HOME prepared
#   ahead of submission.  This script never downloads datasets or models.
# * Quality comparisons additionally include independent exact-search FibQuant
#   and BlockQuant rows under stable `*-paper` keys.  Neither is submitted to
#   codec/decode/inference speed comparisons because this repo has no packed
#   serving kernel for either baseline. Formal proposal rows remain HQ/AHQ only.
# * TurboQuant-paper is scoped to the Algorithm-1 MSE baseline and is not
#   labelled a reproduction of the under-specified mixed-precision Table-1
#   recipe. Proxy implementations are not external-baseline rows.
# * Smoke and formal outputs/logs are physically separate. Run `smoke`, wait for
#   all eight exact outputs, inspect them, and run `approve-smoke` with the fixed
#   explicit approval statement. Every formal slice revalidates those result
#   hashes and the marker; there is no submit shortcut around the gate.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
LC="$REPO/benchmark/kvcomp-longcontext"
WRAPPER="$HERE/run_paper_job.sh"
INPUT_VALIDATOR="$HERE/validate_hayatequant_paper_inputs.py"
SMOKE_GATE_VALIDATOR="$HERE/validate_hayatequant_smoke_gate.py"
LIFECYCLE="$HERE/paper_job_lifecycle.py"

# These defaults deliberately point at the original checkout's environments:
# a git worktree does not contain independent virtual environments.  The
# wrapper prepends REPO to PYTHONPATH and paper_preflight.py rejects an import
# from any other checkout.
PY=${PY:-/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab/.venv/bin/python}
PYVLLM=${PYVLLM:-/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab/.venv-vllm/bin/python}
MODELS=${MODELS:-/lustre/home/frj/yuma-ichikawa/conference/models}
# RUN_ROOT contains only paper evidence and must live below the tracked results
# directory of a separate linked Git worktree.  WORK_ROOT holds mutable caches,
# temporary compiler state, and pinned input data outside that worktree.
RESULTS_WORKTREE=${RESULTS_WORKTREE:-/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/hayatequant-paper-results-worktree}
PAPER_RUN_ID=${PAPER_RUN_ID:-hayate-iclr2027-refresh-20260716}
RUN_ROOT=${RUN_ROOT:-$RESULTS_WORKTREE/papers/hayatequant_paper/experiments/results/$PAPER_RUN_ID}
WORK_ROOT=${WORK_ROOT:-/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/hayate_iclr2027_refresh_20260715}
DATA_ROOT=${DATA_ROOT:-$WORK_ROOT/data}
HF_HOME=${HF_HOME:-$DATA_ROOT/hf_home}
PINNED_HF_MANIFEST=${PINNED_HF_MANIFEST:-$DATA_ROOT/pinned_hf_datasets_manifest.json}
INPUT_MANIFEST=${INPUT_MANIFEST:-$RUN_ROOT/manifests/hayatequant_paper_inputs.json}
INPUT_MANIFEST_SHA256=
SMOKE_APPROVAL_MARKER=${SMOKE_APPROVAL_MARKER:-$RUN_ROOT/manifests/formal_smoke_approval.json}
FORMAL_SMOKE_APPROVAL=${FORMAL_SMOKE_APPROVAL:-}
SMOKE_APPROVAL_MARKER_SHA256=
SMOKE_VALIDATED_PLAN_SHA256=

PARTITIONS=${PARTITIONS:-batch-1gpu}
GPU_GRES=${GPU_GRES:-gpu:nvidia_b200:1}
# Formal timing reserves every B200 GRES on one batch-8gpu node.  The worker
# narrows the measured process to the first assigned device before importing
# torch, while retaining the full allocation identity in immutable preflight
# provenance.  These cluster-specific resources are part of the protocol.
SPEED_PARTITION=batch-8gpu
SPEED_GPU_GRES=gpu:nvidia_b200:8
SPEED_CPUS=160
SPEED_MEM=2400G
PREFERRED_NODE=${PREFERRED_NODE:-}
SPEED_NODE=${SPEED_NODE:-}
SPEED_NODE_RESOLVED=0
SPEED_NODE_IDLE_VALIDATED=0
SPEED_PREVIOUS_JOB_ID=
SPEED_PLAN_SEQUENCE=0
GPU11_SSH_HOST=${GPU11_SSH_HOST:-kagura-gpu11}
GPU11_DEVICE=${GPU11_DEVICE:-0}
DRYRUN=${DRYRUN:-0}
ALLOW_DIRTY_DRYRUN=${ALLOW_DIRTY_DRYRUN:-1}
SKIP_INPUT_PREFLIGHT=${SKIP_INPUT_PREFLIGHT:-0}
SPEED_REPLICATES=${SPEED_REPLICATES:-5}
EXISTING_RESULT_POLICY=${EXISTING_RESULT_POLICY:-skip}
REQUIRE_SPEED_NODE_IDLE=${REQUIRE_SPEED_NODE_IDLE:-0}
PAPER_HOME="$WORK_ROOT/runtime/home"

declare -A PLANNED_JOB_KEYS=()
declare -A PLANNED_RESULT_PATHS=()
SUBMISSION_SEQUENCE=0
ACTIVE_SUBMISSION_RECORD=
ACTIVE_SUBMISSION_PHASE=
ACTIVE_SUBMISSION_JOB_KEY=
ACTIVE_SUBMISSION_ATTEMPT=
ACTIVE_SUBMISSION_RESULT=
ACTIVE_SUBMISSION_STDOUT=
ACTIVE_SUBMISSION_STDERR=
ACTIVE_SUBMISSION_JOB_ID=

RULER_ROOT="$DATA_ROOT/ruler_v1"
LONGCODE_ROOT="$DATA_ROOT/LongCodeQA"
LLAMA_LONGCODE_CONTEXT="$DATA_ROOT/longcodeqa_context_llama31-8b.json"
QWEN_LONGCODE_CONTEXT="$DATA_ROOT/longcodeqa_context_qwen3-8b.json"

# Seven-model P-main grid from the paper.
MODEL_SPECS=(
  "llama2-7b|Llama-2-7b-hf"
  "llama31-8b|llama-3.1-8b-instruct"
  "qwen3-4b|Qwen3-4B"
  "qwen3-8b|Qwen3-8B"
  "r1-llama-8b|DeepSeek-R1-Distill-Llama-8B"
  "r1-qwen-7b|DeepSeek-R1-Distill-Qwen-7B"
  "r1-qwen-1.5b|DeepSeek-R1-Distill-Qwen-1.5B"
)

# Long-context additions are deliberately cross-family and tokenizer-specific.
LONG_MODEL_SPECS=(
  "llama31-8b|llama-3.1-8b-instruct|llama31-8b"
  "qwen3-8b|Qwen3-8B|qwen3-8b"
)

# slug|result label.  Both external `*-paper` block-codec rows are quality-only.
LONG_METHOD_SPECS=(
  "fullkv|FullKV"
  "turboquant-paper|TurboQuant-paper"
  "fibquant-paper|FibQuant-paper"
  "blockquant-paper|BlockQuant-paper"
  "hq|HQ"
  "adaptivehq|AdaptiveHQ"
)

# Construct the exact long-context recipes from the neutral base config.  This
# avoids inheriting unrelated experimental variants from any paper grid config.
TURBO_VARIANTS='turboquant_variants=[{label:TurboQuant-paper,bits:3,dist:beta,rotation_mode:dense}]'
FIBQUANT_VARIANTS='fibquant_variants=[{label:FibQuant-paper,bits:3}]'
BLOCKQUANT_VARIANTS='blockquant_variants=[{label:BlockQuant-paper,bits:3}]'
HAYATE_VARIANTS='hayatequant_variants=[{label:HQ,bits:3},{label:AdaptiveHQ,bits:3}]'

KV_ALL_OFF=(
  run_needle=false
  run_ruler=false
  run_official_ruler_v1=false
  run_longbench=false
  run_gsm8k=false
  run_math=false
  run_ppl=false
  run_livecodebench=false
)

die() {
  if [ -n "$ACTIVE_SUBMISSION_RECORD" ]; then
    persist_active_submission submission_failed submitter_error 2 >/dev/null 2>&1 || \
      printf 'ERROR: could not persist terminal submission failure receipt: %s\n' \
        "$ACTIVE_SUBMISSION_RECORD" >&2
    ACTIVE_SUBMISSION_RECORD=
  fi
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

warn() {
  printf 'WARNING: %s\n' "$*" >&2
}

persist_active_submission() {
  local status=$1 reason=$2 exit_code=${3:-}
  [ -n "$ACTIVE_SUBMISSION_RECORD" ] || return 2
  local -a receipt=(
    "$PY" "$LIFECYCLE" record
    --source-repo "$REPO" --run-root "$RUN_ROOT"
    --source-commit "$CODE_REPO_COMMIT"
    --path "$ACTIVE_SUBMISSION_RECORD" --kind submission
    --status "$status" --phase "$ACTIVE_SUBMISSION_PHASE"
    --job-key "$ACTIVE_SUBMISSION_JOB_KEY"
    --attempt-id "$ACTIVE_SUBMISSION_ATTEMPT"
    --expected-result "$ACTIVE_SUBMISSION_RESULT"
    --stage submitter --reason-code "$reason"
    --scheduler-stdout "$ACTIVE_SUBMISSION_STDOUT"
    --scheduler-stderr "$ACTIVE_SUBMISSION_STDERR"
  )
  [ -z "$exit_code" ] || receipt+=(--exit-code "$exit_code")
  [ -z "$ACTIVE_SUBMISSION_JOB_ID" ] || \
    receipt+=(--slurm-job-id "$ACTIVE_SUBMISSION_JOB_ID")
  "${receipt[@]}"
}

require_executable() {
  local path=$1
  [ -x "$path" ] || die "required executable is missing: $path"
}

require_model() {
  local path=$1
  [ -d "$path" ] || die "model directory is missing: $path"
  [ -f "$path/config.json" ] || die "model config is missing: $path/config.json"
}

require_export_value() {
  local name=$1 value=$2
  case "$value" in
    *','*|*$'\n'*|*$'\r'*)
      die "$name contains a comma or newline and cannot be encoded safely for sbatch --export"
      ;;
  esac
}

resolve_source_revision() {
  local head dirty
  head=$(git -C "$REPO" rev-parse HEAD)
  if [ -n "${CODE_REPO_COMMIT:-}" ] && [ "$CODE_REPO_COMMIT" != "$head" ]; then
    die "CODE_REPO_COMMIT=$CODE_REPO_COMMIT does not match checked-out HEAD=$head"
  fi
  CODE_REPO_COMMIT=$head
  export CODE_REPO_COMMIT

  dirty=$(git -C "$REPO" status --porcelain --untracked-files=all)
  if [ -n "$dirty" ]; then
    if [ "$DRYRUN" = 1 ] && [ "$ALLOW_DIRTY_DRYRUN" = 1 ]; then
      warn "dry-run plan uses dirty HEAD $head; submitted workers would reject it"
    else
      printf '%s\n' "$dirty" | sed -n '1,40p' >&2
      die "formal/smoke jobs require a completely clean worktree (tracked and untracked)"
    fi
  fi
}

validate_static_environment() {
  local canonical_manifest_root canonical_input_manifest canonical_approval_marker
  local canonical_run_root canonical_work_root canonical_results_worktree
  require_executable "$PY"
  require_executable "$WRAPPER"
  [ -f "$LIFECYCLE" ] || die "missing paper job lifecycle helper: $LIFECYCLE"
  command -v realpath >/dev/null 2>&1 || die "realpath is required for result-path safety"
  [ -f "$HERE/paper_preflight.py" ] || die "missing paper_preflight.py"
  case "$SPEED_REPLICATES" in
    ''|*[!0-9]*) die "SPEED_REPLICATES must be a positive integer" ;;
  esac
  [ "$SPEED_REPLICATES" -ge 1 ] || die "SPEED_REPLICATES must be >= 1"
  if [ "$DRYRUN" != 1 ] && [ "$SPEED_REPLICATES" -ne 5 ]; then
    die "SPEED_REPLICATES must be exactly 5 for the formal inferential design"
  fi
  [ "$GPU_GRES" = gpu:nvidia_b200:1 ] || die \
    "GPU_GRES is fixed to gpu:nvidia_b200:1 for quality and smoke allocations"
  case "$GPU11_DEVICE" in
    ''|*[!0-9]*) die "GPU11_DEVICE must be a non-negative integer" ;;
  esac
  case "$GPU11_SSH_HOST" in
    ''|-*|*[!A-Za-z0-9._-]*) die "GPU11_SSH_HOST is not a safe host name" ;;
  esac
  case "$SPEED_NODE" in
    -*|*[!A-Za-z0-9._-]*) die "SPEED_NODE is not a safe Slurm node name" ;;
  esac
  case "$EXISTING_RESULT_POLICY" in
    skip|fail) ;;
    *) die "EXISTING_RESULT_POLICY must be skip or fail" ;;
  esac
  case "$REQUIRE_SPEED_NODE_IDLE" in 0|1) ;; *) die \
    "REQUIRE_SPEED_NODE_IDLE must be 0 or 1" ;; esac
  canonical_manifest_root=$(realpath -m -- "$RUN_ROOT/manifests")
  canonical_input_manifest=$(realpath -m -- "$INPUT_MANIFEST")
  canonical_approval_marker=$(realpath -m -- "$SMOKE_APPROVAL_MARKER")
  canonical_run_root=$(realpath -m -- "$RUN_ROOT")
  canonical_work_root=$(realpath -m -- "$WORK_ROOT")
  canonical_results_worktree=$(realpath -m -- "$RUN_ROOT/../../../../..")
  case "$canonical_work_root" in
    "$canonical_run_root"|"$canonical_run_root/"*)
      die "WORK_ROOT must remain external to the repo-local RUN_ROOT"
      ;;
  esac
  case "$canonical_run_root" in
    "$canonical_work_root/"*)
      die "repo-local RUN_ROOT must not be nested below external WORK_ROOT"
      ;;
  esac
  if [ "$DRYRUN" != 1 ]; then
    case "$canonical_work_root" in
      "$canonical_results_worktree"|"$canonical_results_worktree/"*)
        die "WORK_ROOT caches must remain external to the results Git worktree"
        ;;
    esac
  fi
  case "$canonical_input_manifest" in
    "$canonical_manifest_root/"*) ;;
    *) die "INPUT_MANIFEST must stay below RUN_ROOT/manifests" ;;
  esac
  case "$canonical_approval_marker" in
    "$canonical_manifest_root/"*) ;;
    *) die "SMOKE_APPROVAL_MARKER must stay below RUN_ROOT/manifests" ;;
  esac
  [ ! -L "$INPUT_MANIFEST" ] || die "refusing symlinked INPUT_MANIFEST"
  [ ! -L "$SMOKE_APPROVAL_MARKER" ] || die "refusing symlinked smoke approval marker"
  require_export_value CODE_REPO_COMMIT "${CODE_REPO_COMMIT:-}"
  require_export_value HF_HOME "$HF_HOME"
  require_export_value HOME "$PAPER_HOME"
  require_export_value PATH "$PATH"
  require_export_value PAPER_SMOKE_APPROVAL_MARKER "$SMOKE_APPROVAL_MARKER"
  [ -z "${LD_LIBRARY_PATH:-}" ] || require_export_value LD_LIBRARY_PATH "$LD_LIBRARY_PATH"
  [ -z "${CUDA_HOME:-}" ] || require_export_value CUDA_HOME "$CUDA_HOME"
  [ -z "${LANG:-}" ] || require_export_value LANG "$LANG"
  for spec in "${MODEL_SPECS[@]}"; do
    require_model "$MODELS/${spec#*|}"
  done
}

validate_results_repository() {
  if [ "$DRYRUN" = 1 ]; then
    warn "dry-run does not write receipts; executable commands require a separate linked results worktree"
    return
  fi
  "$PY" "$LIFECYCLE" validate-root \
    --source-repo "$REPO" \
    --run-root "$RUN_ROOT" \
    --expected-commit "$CODE_REPO_COMMIT" \
    >/dev/null || die \
      "RUN_ROOT is not the exact repo-local run directory in a separate linked results worktree"
}

resolve_preferred_node() {
  [ -n "$PREFERRED_NODE" ] || return 0
  if ! command -v sinfo >/dev/null 2>&1; then
    warn "sinfo is unavailable; ignoring PREFERRED_NODE=$PREFERRED_NODE"
    PREFERRED_NODE=""
    return
  fi
  local states
  states=$(sinfo -h -n "$PREFERRED_NODE" -o '%T' 2>/dev/null || true)
  if [ -z "$states" ] || printf '%s\n' "$states" | tr '[:upper:]' '[:lower:]' | \
      grep -Eq 'down|drain|fail|maint|unknown'; then
    warn "PREFERRED_NODE is unavailable ($PREFERRED_NODE: ${states:-not in Slurm}); submitting unpinned"
    PREFERRED_NODE=""
  fi
}

validate_speed_node_identity() {
  # Quality retry slices must exclude the timing stratum even while that node
  # is MIXED/ALLOCATED.  Validate immutable hardware/partition identity here;
  # the stronger idle/all-resources-free check belongs only to a new timing
  # chain in resolve_speed_node().
  [ "$SPEED_NODE_RESOLVED" = 0 ] || return 0
  if [ "$DRYRUN" = 1 ]; then
    [ -n "$SPEED_NODE" ] || SPEED_NODE=DRYRUN-SAME-B200-NODE
    SPEED_NODE_RESOLVED=1
    printf 'SPEED_NODE_IDENTITY dryrun_node=%s partition=%s gres=%s cpus=192\n' \
      "$SPEED_NODE" "$SPEED_PARTITION" "$SPEED_GPU_GRES"
    return
  fi
  [ -n "$SPEED_NODE" ] || die \
    "formal quality/speed submission requires the explicit SPEED_NODE stratum"
  command -v scontrol >/dev/null 2>&1 || die \
    "scontrol is required to validate the explicit SPEED_NODE identity"
  local identity
  identity=$(
    scontrol show node "$SPEED_NODE" --json | "$PY" -c '
import json, sys
expected_name, expected_partition = sys.argv[1:]
doc = json.load(sys.stdin)
nodes = doc.get("nodes") if isinstance(doc, dict) else None
if doc.get("errors") or not isinstance(nodes, list) or len(nodes) != 1:
    raise SystemExit("Slurm returned no unique speed-node identity")
node = nodes[0]
states = node.get("state")
if (
    node.get("name") != expected_name
    or expected_partition not in (node.get("partitions") or [])
    or node.get("gres") != "gpu:nvidia_b200:8"
    or node.get("cpus") != 192
):
    raise SystemExit("Speed-node name/partition/8xB200/CPU identity mismatch")
state_text = "+".join(states) if isinstance(states, list) and states else "UNREPORTED"
print(
    "state=" + state_text
    + " alloc_cpus=" + str(node.get("alloc_cpus"))
    + " gres_used=" + str(node.get("gres_used"))
)
' "$SPEED_NODE" "$SPEED_PARTITION"
  )
  SPEED_NODE_RESOLVED=1
  printf 'SPEED_NODE_IDENTITY node=%s partition=%s gres=%s cpus=192 %s\n' \
    "$SPEED_NODE" "$SPEED_PARTITION" "$SPEED_GPU_GRES" "$identity"
}

resolve_speed_node() {
  # A fresh timing chain additionally requires a completely idle, unallocated
  # node. Every process then reserves all eight B200 GRES plus 160 CPUs/2400G,
  # uses --exclusive=user, and is afterok-chained.
  validate_speed_node_identity
  [ "$SPEED_NODE_IDLE_VALIDATED" = 0 ] || return 0
  if [ "$DRYRUN" = 1 ]; then
    SPEED_NODE_IDLE_VALIDATED=1
    printf 'SPEED_STRATUM dryrun_node=%s policy=full_8gpu_allocation_afterok_serial quality_jobs=excluded\n' \
      "$SPEED_NODE"
    return
  fi
  scontrol show node "$SPEED_NODE" --json | "$PY" -c '
import json, sys
expected_name = sys.argv[1]
doc = json.load(sys.stdin)
nodes = doc.get("nodes") if isinstance(doc, dict) else None
if doc.get("errors") or not isinstance(nodes, list) or len(nodes) != 1:
    raise SystemExit("Slurm returned no unique speed node for idle validation")
node = nodes[0]
states = node.get("state")
if (
    node.get("name") != expected_name
    or states != ["IDLE"]
    or node.get("alloc_cpus") != 0
    or node.get("alloc_memory") != 0
    or node.get("gres_used")
    not in ("", "N/A", None, "gpu:nvidia_b200:0(IDX:N/A)")
    or node.get("tres_used") not in ("", None)
):
    raise SystemExit("SPEED_NODE is not completely idle and unallocated")
' "$SPEED_NODE"
  SPEED_NODE_IDLE_VALIDATED=1
  printf 'SPEED_STRATUM node=%s policy=full_8gpu_allocation_afterok_serial quality_jobs=excluded runtime_driver_check=aggregate\n' \
    "$SPEED_NODE"
}

ensure_input_manifest() {
  if [ "$SKIP_INPUT_PREFLIGHT" = 1 ]; then
    [ "$DRYRUN" = 1 ] || die "SKIP_INPUT_PREFLIGHT=1 is permitted only with DRYRUN=1"
    warn "dry-run is printing a plan without reading dataset manifests"
    INPUT_MANIFEST_SHA256=$(printf '0%.0s' {1..64})
    return
  fi
  [ -f "$INPUT_VALIDATOR" ] || die "missing input validator: $INPUT_VALIDATOR"
  [ -f "$PINNED_HF_MANIFEST" ] || die \
    "pinned HF manifest is missing: $PINNED_HF_MANIFEST (run the dataset prefetch first)"
  "$PY" "$INPUT_VALIDATOR" \
    --run-root "$RUN_ROOT" \
    --work-root "$WORK_ROOT" \
    --runtime-home "$PAPER_HOME" \
    --models-root "$MODELS" \
    --ruler-root "$RULER_ROOT" \
    --longcodeqa-root "$LONGCODE_ROOT" \
    --longcodeqa-context-manifest "llama31-8b=$LLAMA_LONGCODE_CONTEXT" \
    --longcodeqa-context-manifest "qwen3-8b=$QWEN_LONGCODE_CONTEXT" \
    --hf-home "$HF_HOME" \
    --hf-manifest "$PINNED_HF_MANIFEST" \
    --require-hf-manifest \
    --output "$INPUT_MANIFEST"
  [ -s "$INPUT_MANIFEST" ] || die "input validator did not create $INPUT_MANIFEST"
  INPUT_MANIFEST_SHA256=$(
    "$PY" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
      "$INPUT_MANIFEST"
  )
  case "$INPUT_MANIFEST_SHA256" in
    ''|*[!0-9a-f]*) die "could not SHA-256 fingerprint input manifest: $INPUT_MANIFEST" ;;
  esac
  [ "${#INPUT_MANIFEST_SHA256}" -eq 64 ] || die "invalid input manifest SHA-256 length"
}

prepare_submission() {
  validate_static_environment
  resolve_source_revision
  validate_results_repository
  resolve_preferred_node
  mkdir -p "$RUN_ROOT/logs/smoke" "$RUN_ROOT/logs/formal" \
    "$RUN_ROOT/results/smoke" "$RUN_ROOT/results/formal" \
    "$RUN_ROOT/job_records/smoke" "$RUN_ROOT/job_records/formal" \
    "$(dirname "$INPUT_MANIFEST")" "$PAPER_HOME" "$WORK_ROOT/tmp" "$WORK_ROOT/locks"
  ensure_input_manifest
  export HF_HOME
  printf 'PLAN repo=%s commit=%s run_root=%s work_root=%s hf_home=%s dryrun=%s\n' \
    "$REPO" "$CODE_REPO_COMMIT" "$RUN_ROOT" "$WORK_ROOT" "$HF_HOME" "$DRYRUN"
}

invalidate_smoke_approval() {
  if [ -e "$SMOKE_APPROVAL_MARKER" ]; then
    rm -f -- "$SMOKE_APPROVAL_MARKER"
    warn "invalidated prior formal approval before submitting a new smoke plan"
  fi
}

validate_smoke_and_approve() {
  [ -f "$SMOKE_GATE_VALIDATOR" ] || die "missing smoke gate validator: $SMOKE_GATE_VALIDATOR"
  [ "$FORMAL_SMOKE_APPROVAL" = I_APPROVE_THE_VALIDATED_HAYATEQUANT_SMOKE_PLAN ] || die \
    "set FORMAL_SMOKE_APPROVAL=I_APPROVE_THE_VALIDATED_HAYATEQUANT_SMOKE_PLAN after manual smoke review"
  "$PY" "$SMOKE_GATE_VALIDATOR" approve \
    --run-root "$RUN_ROOT" \
    --expected-commit "$CODE_REPO_COMMIT" \
    --input-manifest "$INPUT_MANIFEST" \
    --input-manifest-sha256 "$INPUT_MANIFEST_SHA256" \
    --approval-marker "$SMOKE_APPROVAL_MARKER" \
    --approval-statement "$FORMAL_SMOKE_APPROVAL"
}

emit_formal_smoke_binding() {
  "$PY" "$HERE/paper_preflight.py" \
    --repo "$REPO" \
    --expected-commit "$CODE_REPO_COMMIT" \
    --run-root "$RUN_ROOT" \
    --input-manifest "$INPUT_MANIFEST" \
    --expected-input-manifest-sha256 "$INPUT_MANIFEST_SHA256" \
    --smoke-approval-marker "$SMOKE_APPROVAL_MARKER" \
    --emit-smoke-approval-binding
}

set_formal_smoke_binding() {
  local before after extra digest
  # Capture the exact immutable identity before and after the semantic check.
  # This closes the interval in which a marker could otherwise be replaced
  # after validation but before sbatch serialises the formal worker argv.
  before=$(emit_formal_smoke_binding)
  "$PY" "$SMOKE_GATE_VALIDATOR" check \
    --run-root "$RUN_ROOT" \
    --expected-commit "$CODE_REPO_COMMIT" \
    --input-manifest "$INPUT_MANIFEST" \
    --input-manifest-sha256 "$INPUT_MANIFEST_SHA256" \
    --approval-marker "$SMOKE_APPROVAL_MARKER"
  after=$(emit_formal_smoke_binding)
  [ "$before" = "$after" ] || die \
    "smoke approval marker or validated plan changed during submission-side check"
  IFS=' ' read -r SMOKE_APPROVAL_MARKER_SHA256 \
    SMOKE_VALIDATED_PLAN_SHA256 extra <<< "$after"
  [ -z "${extra:-}" ] || die "malformed smoke approval binding output"
  for digest in "$SMOKE_APPROVAL_MARKER_SHA256" "$SMOKE_VALIDATED_PLAN_SHA256"; do
    case "$digest" in
      ''|*[!0-9a-f]*) die "malformed smoke approval binding digest" ;;
    esac
    [ "${#digest}" -eq 64 ] || die "malformed smoke approval binding digest length"
  done
  require_export_value PAPER_SMOKE_APPROVAL_MARKER_SHA256 \
    "$SMOKE_APPROVAL_MARKER_SHA256"
  require_export_value PAPER_SMOKE_VALIDATED_PLAN_SHA256 \
    "$SMOKE_VALIDATED_PLAN_SHA256"
}

require_formal_smoke_approval() {
  [ -f "$SMOKE_GATE_VALIDATOR" ] || die "missing smoke gate validator: $SMOKE_GATE_VALIDATOR"
  [ -f "$SMOKE_APPROVAL_MARKER" ] || die \
    "formal submission requires approve-smoke after the exact eight-job smoke plan completes"
  set_formal_smoke_binding
}

prepare_formal_submission() {
  prepare_submission
  require_formal_smoke_approval
}

prepare_formal_quality_submission() {
  prepare_formal_submission
  validate_speed_node_identity
}

# submit_job PHASE KEY WALL MEM CPUS NODE MODEL RESULT N_DATASET DATASET... -- CMD...
submit_job() {
  local phase=$1 key=$2 wall=$3 mem=$4 cpus=$5 node=$6 model=$7 result=$8
  local n_dataset=$9
  shift 9
  case "$n_dataset" in ''|*[!0-9]*) die "internal error: invalid dataset count for $key" ;; esac
  local -a datasets=()
  local i
  for ((i = 0; i < n_dataset; i++)); do
    [ "$#" -gt 0 ] || die "internal error: missing dataset argument for $key"
    datasets+=("$1")
    shift
  done
  [ "${1:-}" = -- ] || die "internal error: missing command separator for $key"
  shift
  [ "$#" -gt 0 ] || die "internal error: empty worker command for $key"

  case "$phase" in smoke|formal) ;; *) die "unsafe phase: $phase" ;; esac
  case "$key" in *[!A-Za-z0-9._-]*) die "unsafe job key: $key" ;; esac
  [ -n "$model" ] || die "every smoke/formal job must declare a pinned model"
  local canonical_phase_root canonical_result
  canonical_phase_root=$(realpath -m -- "$RUN_ROOT/results/$phase")
  canonical_result=$(realpath -m -- "$result")
  case "$result" in
    "$RUN_ROOT/results/$phase/"*) ;;
    *) die "result path for $phase/$key escapes its phase root: $result" ;;
  esac
  case "$canonical_result" in
    "$canonical_phase_root/"*) ;;
    *) die "resolved result path for $phase/$key escapes its phase root: $canonical_result" ;;
  esac
  if [ -n "${PLANNED_JOB_KEYS[$phase-$key]+present}" ]; then
    die "duplicate planned job key: $phase-$key"
  fi
  if [ -n "${PLANNED_RESULT_PATHS[$result]+present}" ]; then
    die "duplicate planned result path: $result"
  fi
  PLANNED_JOB_KEYS["$phase-$key"]=1
  PLANNED_RESULT_PATHS["$result"]=1

  local job_name="hq-${phase:0:1}-$key"
  local full_job_key="$phase-$key"
  SUBMISSION_SEQUENCE=$((SUBMISSION_SEQUENCE + 1))
  local attempt_id
  if [ "$DRYRUN" = 1 ]; then
    attempt_id="dryrun-$SUBMISSION_SEQUENCE"
  else
    attempt_id="$(date -u +%Y%m%dT%H%M%SZ)-submit-$$-$SUBMISSION_SEQUENCE"
  fi
  local attempt_dir="$RUN_ROOT/job_records/$phase/$full_job_key/$attempt_id"
  local submission_record="$attempt_dir/submission.json"
  local scheduler_stdout="$attempt_dir/sbatch.stdout"
  local scheduler_stderr="$attempt_dir/sbatch.stderr"
  local stdout_template="$RUN_ROOT/logs/$phase/%x_%j.out"
  local stderr_template="$RUN_ROOT/logs/$phase/%x_%j.err"
  local -a worker=(
    "$WRAPPER"
    --repo "$REPO"
    --python "$PY"
    --run-root "$RUN_ROOT"
    --work-root "$WORK_ROOT"
    --job-key "$full_job_key"
    --attempt-id "$attempt_id"
    --input-manifest "$INPUT_MANIFEST"
    --input-manifest-sha256 "$INPUT_MANIFEST_SHA256"
    --expected-result "$result"
    --stdout-log "$stdout_template"
    --stderr-log "$stderr_template"
  )
  if [ "$phase" = formal ]; then
    [ -n "$SMOKE_APPROVAL_MARKER_SHA256" ] && \
      [ -n "$SMOKE_VALIDATED_PLAN_SHA256" ] || die \
      "internal error: formal job has no submitter-bound smoke identity"
    worker+=(
      --smoke-approval-marker "$SMOKE_APPROVAL_MARKER"
      --smoke-approval-marker-sha256 "$SMOKE_APPROVAL_MARKER_SHA256"
      --smoke-validated-plan-sha256 "$SMOKE_VALIDATED_PLAN_SHA256"
    )
  fi
  [ -z "$model" ] || worker+=(--model "$model")
  local dataset
  for dataset in "${datasets[@]}"; do
    worker+=(--dataset-file "$dataset")
  done
  worker+=(-- "$@")

  local wrapped
  printf -v wrapped '%q ' "${worker[@]}"
  wrapped=${wrapped% }

  # Explicit allow-list: access tokens and unrelated login-shell state are not
  # propagated to compute nodes.  Slurm supplies its own SLURM_* variables.
  local export_spec
  export_spec="CODE_REPO_COMMIT=$CODE_REPO_COMMIT,HF_HOME=$HF_HOME"
  export_spec+=",HOME=$PAPER_HOME,PATH=$PATH"
  export_spec+=",HF_DATASETS_OFFLINE=1,HF_HUB_OFFLINE=1"
  export_spec+=",TRANSFORMERS_OFFLINE=1"
  export_spec+=",HF_HUB_DISABLE_TELEMETRY=1,WANDB_MODE=disabled"
  export_spec+=",PYTHONHASHSEED=0,PYTHONUNBUFFERED=1"
  export_spec+=",TOKENIZERS_PARALLELISM=false,OMP_NUM_THREADS=$cpus"
  export_spec+=",ONECOMP_COMPILE_CODEC=0,ONECOMP_COMPILE_DECODE_ONLY=0"
  export_spec+=",ONECOMP_TRITON_CODEC=0,ONECOMP_TRITON_PREFILL=0"
  if [ "$phase" = formal ]; then
    export_spec+=",PAPER_SMOKE_APPROVAL_MARKER=$SMOKE_APPROVAL_MARKER"
    export_spec+=",PAPER_SMOKE_APPROVAL_MARKER_SHA256=$SMOKE_APPROVAL_MARKER_SHA256"
    export_spec+=",PAPER_SMOKE_VALIDATED_PLAN_SHA256=$SMOKE_VALIDATED_PLAN_SHA256"
  fi
  [ -z "${LD_LIBRARY_PATH:-}" ] || export_spec+=",LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  [ -z "${CUDA_HOME:-}" ] || export_spec+=",CUDA_HOME=$CUDA_HOME"
  [ -z "${LANG:-}" ] || export_spec+=",LANG=$LANG"

  local allocation_partition=$PARTITIONS allocation_gres=$GPU_GRES
  if [ "$phase" = formal ] && [[ "$key" = speed-* ]]; then
    allocation_partition=$SPEED_PARTITION
    allocation_gres=$SPEED_GPU_GRES
    [ "$cpus" = "$SPEED_CPUS" ] && [ "$mem" = "$SPEED_MEM" ] || die \
      "internal error: formal speed allocation resources changed"
  fi
  local -a sbatch_cmd=(
    sbatch --parsable
    --job-name "$job_name"
    --partition "$allocation_partition"
    --nodes 1
    --ntasks 1
    --gres "$allocation_gres"
    --cpus-per-task "$cpus"
    --mem "$mem"
    --time "$wall"
    --chdir "$REPO"
    --output "$stdout_template"
    --error "$stderr_template"
    --export "$export_spec"
  )
  if [ "$phase" = formal ] && [[ "$key" = speed-* ]]; then
    [ "$SPEED_NODE_RESOLVED" = 1 ] || die \
      "internal error: formal speed job was planned before SPEED_NODE validation"
    [ "$SPEED_NODE_IDLE_VALIDATED" = 1 ] || die \
      "internal error: formal speed job was planned without idle-node validation"
    [ -n "$node" ] && [ "$node" = "$SPEED_NODE" ] || die \
      "internal error: formal speed job must use the validated SPEED_NODE"
    # All eight GRES prevent any GPU workload from sharing the timing node;
    # exclusive=user excludes other users, and afterok gives strict serial,
    # fail-closed execution even though batch-8gpu caps allocations at 160 of
    # the physical node's 192 CPUs and cannot accept bare --exclusive.
    sbatch_cmd+=(--exclusive=user)
    sbatch_cmd+=(--kill-on-invalid-dep=yes)
    if [ -n "$SPEED_PREVIOUS_JOB_ID" ]; then
      sbatch_cmd+=(--dependency "afterok:$SPEED_PREVIOUS_JOB_ID")
    fi
  elif [ "$phase" = formal ]; then
    [ "$SPEED_NODE_RESOLVED" = 1 ] || die \
      "internal error: formal quality job has no validated SPEED_NODE exclusion"
    [ "$node" != "$SPEED_NODE" ] || die \
      "internal error: formal quality job cannot use the reserved SPEED_NODE"
    sbatch_cmd+=(--exclude "$SPEED_NODE")
  fi
  [ -z "$node" ] || sbatch_cmd+=(--nodelist "$node")
  sbatch_cmd+=(--wrap "$wrapped")

  printf 'JOB phase=%s key=%s result=%s model=%s\n' \
    "$phase" "$key" "$result" "${model:-none}"
  if [ "$DRYRUN" = 1 ]; then
    printf 'SBATCH'
    printf ' %q' "${sbatch_cmd[@]}"
    printf '\nWORKER'
    printf ' %q' "${worker[@]}"
    printf '\n'
    if [ "$phase" = formal ] && [[ "$key" = speed-* ]]; then
      SPEED_PLAN_SEQUENCE=$((SPEED_PLAN_SEQUENCE + 1))
      SPEED_PREVIOUS_JOB_ID=$((900000000 + SPEED_PLAN_SEQUENCE))
    fi
    return
  fi

  mkdir -p "$attempt_dir"
  : >"$scheduler_stdout"
  : >"$scheduler_stderr"
  chmod 600 "$scheduler_stdout" "$scheduler_stderr"
  ACTIVE_SUBMISSION_RECORD=$submission_record
  ACTIVE_SUBMISSION_PHASE=$phase
  ACTIVE_SUBMISSION_JOB_KEY=$full_job_key
  ACTIVE_SUBMISSION_ATTEMPT=$attempt_id
  ACTIVE_SUBMISSION_RESULT=$result
  ACTIVE_SUBMISSION_STDOUT=$scheduler_stdout
  ACTIVE_SUBMISSION_STDERR=$scheduler_stderr
  ACTIVE_SUBMISSION_JOB_ID=
  persist_active_submission submission_started submission_started || die \
    "could not create repo-local submission receipt for $phase/$key"

  if [ -L "$result" ]; then
    die "refusing symlinked result path for $phase/$key: $result"
  fi
  if [ -e "$result" ]; then
    if [ "$EXISTING_RESULT_POLICY" = skip ]; then
      persist_active_submission skipped_existing result_already_exists || die \
        "could not finalize skipped submission receipt for $phase/$key"
      ACTIVE_SUBMISSION_RECORD=
      warn "skipping existing result without overwriting: $result"
      return
    fi
    die "result already exists and will not be overwritten: $result"
  fi
  if command -v squeue >/dev/null 2>&1; then
    local queued
    queued=$(squeue -h -u "$(id -un)" -n "$job_name" -o '%A' 2>/dev/null || true)
    [ -z "$queued" ] || die \
      "job $job_name is already queued/running as ${queued//$'\n'/,}; refusing duplicate"
  fi

  local job_id sbatch_status
  if "${sbatch_cmd[@]}" >"$scheduler_stdout" 2>"$scheduler_stderr"; then
    sbatch_status=0
  else
    sbatch_status=$?
  fi
  if [ "$sbatch_status" -ne 0 ]; then
    sed -n '1,80p' "$scheduler_stderr" >&2 || true
    persist_active_submission submission_failed sbatch_failed "$sbatch_status" || die \
      "could not finalize sbatch failure receipt for $phase/$key"
    ACTIVE_SUBMISSION_RECORD=
    die "sbatch failed for $phase/$key; remaining submissions were stopped"
  fi
  job_id=$(tr -d '\r\n' <"$scheduler_stdout")
  job_id=${job_id%%;*}
  case "$job_id" in
    ''|*[!0-9]*) die "sbatch returned an invalid job ID for $phase/$key" ;;
  esac
  ACTIVE_SUBMISSION_JOB_ID=$job_id
  persist_active_submission submitted sbatch_accepted 0 || die \
    "could not finalize submitted receipt for $phase/$key"
  ACTIVE_SUBMISSION_RECORD=
  if [ "$phase" = formal ] && [[ "$key" = speed-* ]]; then
    case "$job_id" in
      ''|*[!0-9]*) die "sbatch returned an invalid speed predecessor job ID: $job_id" ;;
    esac
    SPEED_PREVIOUS_JOB_ID=$job_id
  fi
  printf 'SUBMITTED phase=%s key=%s job_id=%s result=%s\n' \
    "$phase" "$key" "$job_id" "$result"
}

common_datasets() {
  # Kept as a helper only for documentation; calls use explicit arrays because
  # Bash cannot safely return an array containing arbitrary paths.
  : "$INPUT_MANIFEST" "$PINNED_HF_MANIFEST"
}

long_base_command() {
  # Marker function documenting the shared CLI contract; commands below use
  # arrays so every Hydra override remains one argv element.
  : "$TURBO_VARIANTS" "$FIBQUANT_VARIANTS" "$BLOCKQUANT_VARIANTS" "$HAYATE_VARIANTS"
}

submit_ppl_formal() {
  local spec key model_name model out result
  local -a datasets=("$INPUT_MANIFEST" "$PINNED_HF_MANIFEST")
  for spec in "${MODEL_SPECS[@]}"; do
    key=${spec%%|*}
    model_name=${spec#*|}
    model="$MODELS/$model_name"
    out="$RUN_ROOT/results/formal/ppl/$key"
    result="$out/results_${model_name}.json"
    local -a cmd=(
      "$PY" -u "$HERE/codec_benchmark.py"
      --config-name paper_hayatequant
      "model_path=$model"
      "output_dir=$out"
      # BlockQuant-paper is intentionally requested in the shared config but
      # codec_benchmark emits it only at its sole declared rate-3 operating
      # point; the rate-2/rate-4 formal cells therefore do not exist.
      'methods=[TurboQuant-paper,FibQuant-paper,BlockQuant-paper,HQ,AdaptiveHQ]'
      'bits=[2,3,4]'
      run_ppl=true
      ppl_windows=null
      ppl_max_length=2048
      ppl_stride=1024
      sample_seed=2027
      skip_outliers=false
      sink_tokens=0
      recent_tokens=0
    )
    submit_job formal "ppl-$key" 1-00:00:00 220G 12 "" "$model" "$result" \
      "${#datasets[@]}" "${datasets[@]}" -- "${cmd[@]}"
  done
}

submit_mechanism_formal() {
  # Causal block-VQ control. Both HQ controls use the Beta+dense+RMS-equivalent
  # K=V=r contract matched to the TurboQuant Algorithm-1 MSE baseline; only
  # block_size changes between the two HQ controls. TurboQuant itself stores
  # the mathematically equivalent L2 scale, not RMS.
  # TQ-paper and both formal proposals already come from the main PPL jobs at the same
  # fixed commit, so this adds only the two missing full-text evaluations.
  local spec key rest model_name model out result
  local -a datasets=("$INPUT_MANIFEST" "$PINNED_HF_MANIFEST") cmd=()
  for spec in "${LONG_MODEL_SPECS[@]}"; do
    key=${spec%%|*}
    rest=${spec#*|}
    model_name=${rest%%|*}
    model="$MODELS/$model_name"
    out="$RUN_ROOT/results/formal/mechanism/$key"
    result="$out/results_${model_name}.json"
    cmd=(
      "$PY" -u "$HERE/codec_benchmark.py"
      --config-name paper_hayatequant
      "model_path=$model"
      "output_dir=$out"
      'methods=[HayateQuant-scalar-matched,HayateQuant-blockvq-matched]'
      'bits=[2,3,4]'
      capture_tokens=2048
      timing_iters=10
      run_ppl=true
      ppl_windows=null
      ppl_max_length=2048
      ppl_stride=1024
      sample_seed=2027
      skip_outliers=false
      sink_tokens=0
      recent_tokens=0
    )
    submit_job formal "mechanism-$key" 1-00:00:00 220G 12 "" \
      "$model" "$result" "${#datasets[@]}" "${datasets[@]}" -- "${cmd[@]}"
  done
}

submit_zeroshot_formal() {
  local spec key model_name model task_spec task dataset_size shard_size
  local bit start limit suffix result
  local -a datasets=("$INPUT_MANIFEST" "$PINNED_HF_MANIFEST") cmd=()
  # task|exact valid-example count|rows/job.  Intervals are deterministic in
  # the evaluator's validated-example order.  HellaSwag is deliberately split
  # five ways; every task is checked for exact [0,N) coverage at aggregation.
  local -a task_specs=(
    'arc_easy|2376|2500'
    'arc_challenge|1172|2500'
    'piqa|1838|2500'
    'hellaswag|10042|2500'
    'winogrande|1267|2500'
  )
  for spec in "${MODEL_SPECS[@]}"; do
    key=${spec%%|*}
    model_name=${spec#*|}
    model="$MODELS/$model_name"
    for task_spec in "${task_specs[@]}"; do
      task=${task_spec%%|*}
      task_spec=${task_spec#*|}
      dataset_size=${task_spec%%|*}
      shard_size=${task_spec#*|}
      for ((start = 0; start < dataset_size; start += shard_size)); do
        limit=$shard_size
        [ $((start + limit)) -le "$dataset_size" ] || limit=$((dataset_size - start))

        # FullKV is evaluated exactly once per model/task/row interval, not once
        # per bit.  The aggregate uses exact correct counts, never averages of
        # rounded shard accuracies.
        suffix="full-${task}-s${start}-n${limit}"
        result="$RUN_ROOT/results/formal/zeroshot/zeroshot_${model_name}_${suffix}.json"
        cmd=(
          "$PY" -u "$HERE/zeroshot_benchmark.py"
          --model-path "$model"
          --output-root "$RUN_ROOT/results/formal/zeroshot"
          --output-suffix "$suffix"
          --device cuda:0 --dtype bfloat16
          --bits 3
          --tasks "$task"
          --methods FullKV
          --start "$start" --limit "$limit"
          --sample-seed 2027
          --sink-tokens 0 --recent-tokens 0
          --no-skip-outliers
        )
        submit_job formal "zs-$key-full-$task-s$start" 1-00:00:00 220G 16 "" \
          "$model" "$result" "${#datasets[@]}" "${datasets[@]}" -- "${cmd[@]}"

        # Formal compressed quality methods share one model load per rate.
        # BlockQuant exists only at its paper-declared rate-3 matrix slot;
        # neither external exact-search baseline enters a speed job.
        for bit in 2 3 4; do
          local -a quality_methods=(TurboQuant-paper FibQuant-paper HQ AdaptiveHQ)
          if [ "$bit" -eq 3 ]; then
            quality_methods=(TurboQuant-paper FibQuant-paper BlockQuant-paper HQ AdaptiveHQ)
          fi
          suffix="b${bit}-${task}-s${start}-n${limit}"
          result="$RUN_ROOT/results/formal/zeroshot/zeroshot_${model_name}_${suffix}.json"
          cmd=(
            "$PY" -u "$HERE/zeroshot_benchmark.py"
            --model-path "$model"
            --output-root "$RUN_ROOT/results/formal/zeroshot"
            --output-suffix "$suffix"
            --device cuda:0 --dtype bfloat16
            --bits "$bit"
            --tasks "$task"
            --methods "${quality_methods[@]}"
            --start "$start" --limit "$limit"
            --sample-seed 2027
            --sink-tokens 0 --recent-tokens 0
            --no-skip-outliers
          )
          submit_job formal "zs-$key-b$bit-$task-s$start" 1-00:00:00 220G 16 "" \
            "$model" "$result" "${#datasets[@]}" "${datasets[@]}" -- "${cmd[@]}"
        done
      done
    done
  done
}

submit_ruler_formal() {
  local spec key rest model_name data_key model length method_spec method_slug method
  local data_root manifest out result
  local -a cmd=() datasets=()
  for spec in "${LONG_MODEL_SPECS[@]}"; do
    key=${spec%%|*}
    rest=${spec#*|}
    model_name=${rest%%|*}
    data_key=${rest#*|}
    model="$MODELS/$model_name"
    manifest="$RULER_ROOT/$data_key/manifest.json"
    for length in 4096 8192 16384; do
      data_root="$RULER_ROOT/$data_key/$length"
      datasets=("$INPUT_MANIFEST" "$manifest")
      for method_spec in "${LONG_METHOD_SPECS[@]}"; do
        method_slug=${method_spec%%|*}
        method=${method_spec#*|}
        out="$RUN_ROOT/results/formal/ruler/$key/$length/$method_slug"
        result="$out/results_${model_name}.json"
        cmd=(
          "$PY" -u "$LC/kv_benchmark.py"
          --config-name benchmark_longcontext
          "model_path=$model"
          "output_dir=$out"
          "methods=[$method]"
          "$TURBO_VARIANTS"
          "$FIBQUANT_VARIANTS"
          "$BLOCKQUANT_VARIANTS"
          "$HAYATE_VARIANTS"
          'uniform_bits=[]' 'kivi_bits=[]' 'halo_variants=[]' 'codec_variants=[]'
          thinkv.auto_configure=false
          skip_outliers=false
          sink_tokens=0
          recent_tokens=0
          "${KV_ALL_OFF[@]}"
          run_official_ruler_v1=true
          "official_ruler_data_root=$data_root"
          "official_ruler_manifest_path=$manifest"
          'official_ruler_tasks=[]'
          "official_ruler_max_context=$length"
          official_ruler_start=0
          official_ruler_limit=null
          attn_implementation=sdpa
          +resume=false
        )
        submit_job formal "ruler-$key-$length-$method_slug" 1-00:00:00 220G 16 "" \
          "$model" "$result" "${#datasets[@]}" "${datasets[@]}" -- "${cmd[@]}"
      done
    done
  done
}

submit_longbench_formal() {
  local spec key rest model_name model prompt_budget task_spec task task_n
  local method_spec method_slug method out result
  local -a datasets=("$INPUT_MANIFEST" "$PINNED_HF_MANIFEST") cmd=()
  local -a task_specs=(
    'qasper|224' 'multifieldqa_en|150' 'hotpotqa|300' '2wikimqa|300'
    'gov_report|300' 'multi_news|294' 'trec|300' 'triviaqa|300'
    'samsum|300' 'passage_count|300' 'passage_retrieval_en|300'
    'lcc|300' 'repobench-p|300'
  )
  for spec in "${LONG_MODEL_SPECS[@]}"; do
    key=${spec%%|*}
    rest=${spec#*|}
    model_name=${rest%%|*}
    model="$MODELS/$model_name"
    case "$key" in
      llama31-8b)
        # Every pinned row plus its task output fits below 50K. A 64K raw
        # prompt budget therefore preserves the complete dataset unchanged.
        prompt_budget=65536
        ;;
      qwen3-8b)
        # Selected per task below from its official output budget.
        prompt_budget=
        ;;
      *) die "missing LongBench prompt budget for $key" ;;
    esac
    for task_spec in "${task_specs[@]}"; do
      task=${task_spec%%|*}; task_n=${task_spec#*|}
      if [ "$key" = qwen3-8b ]; then
        # Qwen3 advertises 40,960 positions. Keep a 16-token wrapper/seam
        # reserve in addition to each task's official generation budget.
        case "$task" in
          gov_report|multi_news) prompt_budget=40432 ;; # 40960 - 512 - 16
          qasper|samsum) prompt_budget=40816 ;;         # 40960 - 128 - 16
          multifieldqa_en|trec|lcc|repobench-p)
            prompt_budget=40880 ;;                      # 40960 - 64 - 16
          *) prompt_budget=40912 ;;                     # 40960 - 32 - 16
        esac
      fi
      for method_spec in "${LONG_METHOD_SPECS[@]}"; do
        method_slug=${method_spec%%|*}
        method=${method_spec#*|}
        out="$RUN_ROOT/results/formal/longbench/$key/$task/$method_slug"
        result="$out/results_${model_name}.json"
        cmd=(
          "$PY" -u "$LC/kv_benchmark.py"
          --config-name benchmark_longcontext
          "model_path=$model" "output_dir=$out" "methods=[$method]"
          "$TURBO_VARIANTS" "$FIBQUANT_VARIANTS" "$BLOCKQUANT_VARIANTS" "$HAYATE_VARIANTS"
          'uniform_bits=[]' 'kivi_bits=[]' 'halo_variants=[]' 'codec_variants=[]'
          thinkv.auto_configure=false
          skip_outliers=false
          sink_tokens=0
          recent_tokens=0
          "${KV_ALL_OFF[@]}"
          run_longbench=true
          "longbench_tasks=[$task]"
          "longbench_max_length=$prompt_budget"
          longbench_limit=null
          attn_implementation=sdpa
          +resume=false
        )
        submit_job formal "longbench-$key-$task-n$task_n-$method_slug" 1-00:00:00 220G 20 "" \
          "$model" "$result" "${#datasets[@]}" "${datasets[@]}" -- "${cmd[@]}"
      done
    done
  done
}

submit_math_formal() {
  local spec key rest model_name model method_spec method_slug method start out result
  local -a datasets=("$INPUT_MANIFEST" "$PINNED_HF_MANIFEST") cmd=()
  for spec in "${LONG_MODEL_SPECS[@]}"; do
    key=${spec%%|*}
    rest=${spec#*|}
    model_name=${rest%%|*}
    model="$MODELS/$model_name"
    for method_spec in "${LONG_METHOD_SPECS[@]}"; do
      method_slug=${method_spec%%|*}
      method=${method_spec#*|}
      for ((start = 0; start < 500; start += 50)); do
        out="$RUN_ROOT/results/formal/math500/$key/$method_slug/s$start"
        result="$out/results_${model_name}.json"
        cmd=(
          "$PY" -u "$LC/kv_benchmark.py"
          --config-name benchmark_longcontext
          "model_path=$model" "output_dir=$out" "methods=[$method]"
          "$TURBO_VARIANTS" "$FIBQUANT_VARIANTS" "$BLOCKQUANT_VARIANTS" "$HAYATE_VARIANTS"
          'uniform_bits=[]' 'kivi_bits=[]' 'halo_variants=[]' 'codec_variants=[]'
          thinkv.auto_configure=false
          skip_outliers=false
          sink_tokens=0
          recent_tokens=0
          "${KV_ALL_OFF[@]}"
          run_math=true
          "math_start=$start"
          math_limit=50
          max_new_math=1024
          sample_seed=2027
          math_grader=math_verify
          math_verify_version=0.9.0
          math500_revision=6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be
          attn_implementation=sdpa
          +resume=false
        )
        submit_job formal "math500-$key-$method_slug-s$start" 1-00:00:00 220G 16 "" \
          "$model" "$result" "${#datasets[@]}" "${datasets[@]}" -- "${cmd[@]}"
      done
    done
  done
}

submit_longcodeqa_formal() {
  local key model_name model bracket context_limit method_spec method_slug method
  local data_file context_file bracket_count shard_size start limit out result
  local -a cmd=() datasets=()
  # Only brackets that fit the unmodified model context are submitted.
  local -a feasible=(
    'llama31-8b|llama-3.1-8b-instruct|32K|131072|113|16'
    'llama31-8b|llama-3.1-8b-instruct|64K|131072|76|8'
    'llama31-8b|llama-3.1-8b-instruct|128K|131072|92|4'
    'qwen3-8b|Qwen3-8B|32K|40960|113|8'
  )
  local item rest
  for item in "${feasible[@]}"; do
    key=${item%%|*}; rest=${item#*|}
    model_name=${rest%%|*}; rest=${rest#*|}
    bracket=${rest%%|*}; rest=${rest#*|}
    context_limit=${rest%%|*}; rest=${rest#*|}
    bracket_count=${rest%%|*}; shard_size=${rest#*|}
    model="$MODELS/$model_name"
    data_file="$LONGCODE_ROOT/LQA/$bracket.json"
    if [ "$key" = llama31-8b ]; then
      context_file=$LLAMA_LONGCODE_CONTEXT
    else
      context_file=$QWEN_LONGCODE_CONTEXT
    fi
    datasets=("$INPUT_MANIFEST" "$data_file" "$context_file")
    for method_spec in "${LONG_METHOD_SPECS[@]}"; do
      method_slug=${method_spec%%|*}
      method=${method_spec#*|}
      for ((start = 0; start < bracket_count; start += shard_size)); do
        limit=$shard_size
        [ $((start + limit)) -le "$bracket_count" ] || limit=$((bracket_count - start))
        out="$RUN_ROOT/results/formal/longcodeqa/$key/$bracket/$method_slug/s$start"
        result="$out/results_${model_name}.json"
        cmd=(
          "$PY" -u "$LC/kv_benchmark.py"
          --config-name benchmark_longcontext
          "model_path=$model" "output_dir=$out" "methods=[$method]"
          "$TURBO_VARIANTS" "$FIBQUANT_VARIANTS" "$BLOCKQUANT_VARIANTS" "$HAYATE_VARIANTS"
          'uniform_bits=[]' 'kivi_bits=[]' 'halo_variants=[]' 'codec_variants=[]'
          thinkv.auto_configure=false
          skip_outliers=false
          sink_tokens=0
          recent_tokens=0
          "${KV_ALL_OFF[@]}"
          +run_longcodeqa=true
          "+longcodeqa_data_root=$LONGCODE_ROOT"
          "+longcodeqa_brackets=[$bracket]"
          "+longcodeqa_max_context=$context_limit"
          +longcodeqa_max_new=16
          "+longcodeqa_start=$start"
          "+longcodeqa_limit=$limit"
          +longcodeqa_revision=989d5eff750d65a72c522e47f8f745ef2e22906b
          attn_implementation=sdpa
          +resume=false
        )
        submit_job formal "longcode-$key-$bracket-$method_slug-s$start" \
          1-00:00:00 220G 16 "" "$model" "$result" "${#datasets[@]}" \
          "${datasets[@]}" -- "${cmd[@]}"
      done
    done
  done
}

submit_speed_formal() {
  [ -f "$HERE/hayatequant_fair_speed.py" ] || die "missing fair speed harness"
  if [ "$DRYRUN" != 1 ] && [ "$SPEED_REPLICATES" -ne 5 ]; then
    die "formal fair-speed requires exactly five independent process replicates"
  fi
  resolve_speed_node
  local spec key rest model_name model bits compile replicate seed node out result
  local -a datasets=("$INPUT_MANIFEST") cmd=()
  for spec in "${LONG_MODEL_SPECS[@]}"; do
    key=${spec%%|*}; rest=${spec#*|}; model_name=${rest%%|*}
    model="$MODELS/$model_name"
    for bits in 2 3 4; do
      for compile in off decode-only; do
        for ((replicate = 0; replicate < SPEED_REPLICATES; replicate++)); do
          seed=$((20260715 + replicate))
          # Every inferential speed process is fixed to one physical node so
          # GPU family and driver cannot vary across the pooled matrix.
          node=$SPEED_NODE
          out="$RUN_ROOT/results/formal/speed/$key/b$bits/$compile"
          result="$out/rep${replicate}.json"
          cmd=(
            "$PY" -u "$HERE/hayatequant_fair_speed.py"
            --model "$model"
            --local-files-only
            --device cuda:0
            --dtype float16
            --avg-bits "$bits"
            --mode both
            --micro-seq-len 4096
            --decode-seq-len 1
            --quality-query-len 8
            --micro-warmup 8
            --micro-repeats 40
            --micro-inner-loops 1
            --bootstrap-resamples 10000
            --e2e-prompt-len 4096
            --e2e-gen-len 64
            --e2e-warmup 1
            --e2e-repeats 5
            --sink-tokens 0
            --recent-tokens 0
            --attention-implementation sdpa
            --compile-mode "$compile"
            --fused-backend triton
            --seed "$seed"
            --output "$result"
          )
          submit_job formal "speed-$key-b$bits-$compile-r$replicate" 08:00:00 \
            "$SPEED_MEM" "$SPEED_CPUS" "$node" "$model" "$result" "${#datasets[@]}" \
            "${datasets[@]}" -- "${cmd[@]}"
        done
      done
    done
  done
}

submit_native_smoke() {
  require_executable "$PYVLLM"
  local model="$MODELS/llama-3.1-8b-instruct"
  local result="$RUN_ROOT/results/smoke/native/hayate_native_hybrid.json"
  local -a datasets=("$INPUT_MANIFEST")
  local -a cmd=(
    "$PYVLLM" -u "$HERE/hayatequant_vllm_native_smoke.py"
    --bits 3 4
    --head-dim 128
    --kv-heads 8
    --q-heads 32
    --tokens 4096
    --key-scale-mode ub
    --codebook-init auto
    --bench-iters 100
    --input-dtype float16
    --verify-optimized-store
    --output "$result"
  )
  # This is explicitly a native hybrid/kernel smoke, not a full-HQ serving row.
  submit_job smoke native-hybrid 04:00:00 220G 16 "$PREFERRED_NODE" "$model" \
    "$result" "${#datasets[@]}" "${datasets[@]}" -- "${cmd[@]}"
}

# Direct SSH fallback for kagura-gpu11 while that host is Slurm DOWN/DRAIN.
# This remains a smoke/retry path only.  It uses the same worker wrapper and
# explicit environment allow-list as Slurm jobs; formal paper rows stay under
# the scheduler so resource ownership and accounting remain auditable.
run_ssh_worker() {
  local key=$1 python=$2 model=$3 result=$4 n_dataset=$5
  shift 5
  local -a datasets=()
  local i
  for ((i = 0; i < n_dataset; i++)); do
    datasets+=("$1"); shift
  done
  [ "${1:-}" = -- ] || die "internal SSH worker separator missing"
  shift
  SUBMISSION_SEQUENCE=$((SUBMISSION_SEQUENCE + 1))
  local attempt_id
  if [ "$DRYRUN" = 1 ]; then
    attempt_id="dryrun-ssh-$SUBMISSION_SEQUENCE"
  else
    attempt_id="$(date -u +%Y%m%dT%H%M%SZ)-ssh-$$-$SUBMISSION_SEQUENCE"
  fi
  local full_job_key="gpu11-$key"
  local attempt_dir="$RUN_ROOT/job_records/smoke/$full_job_key/$attempt_id"
  local submission_record="$attempt_dir/submission.json"
  local scheduler_stdout="$attempt_dir/ssh.stdout"
  local scheduler_stderr="$attempt_dir/ssh.stderr"
  local log="$RUN_ROOT/logs/smoke/gpu11-${key}.log"
  local -a worker=(
    "$WRAPPER" --repo "$REPO" --python "$python" --run-root "$RUN_ROOT"
    --work-root "$WORK_ROOT" --job-key "$full_job_key" --attempt-id "$attempt_id"
    --stdout-log "$log" --stderr-log "$log" --input-manifest "$INPUT_MANIFEST"
    --input-manifest-sha256 "$INPUT_MANIFEST_SHA256"
    --expected-result "$result" --model "$model"
  )
  local dataset
  for dataset in "${datasets[@]}"; do worker+=(--dataset-file "$dataset"); done
  worker+=(-- "$@")
  local -a remote=(
    env -i
    "CODE_REPO_COMMIT=$CODE_REPO_COMMIT"
    "HF_HOME=$HF_HOME"
    "HOME=$PAPER_HOME"
    "PATH=$PATH"
    HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled
    PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
    ONECOMP_COMPILE_CODEC=0 ONECOMP_COMPILE_DECODE_ONLY=0
    ONECOMP_TRITON_CODEC=0 ONECOMP_TRITON_PREFILL=0
  )
  [ -z "${LD_LIBRARY_PATH:-}" ] || remote+=("LD_LIBRARY_PATH=$LD_LIBRARY_PATH")
  [ -z "${CUDA_HOME:-}" ] || remote+=("CUDA_HOME=$CUDA_HOME")
  [ -z "${LANG:-}" ] || remote+=("LANG=$LANG")
  remote+=("CUDA_VISIBLE_DEVICES=$GPU11_DEVICE" "${worker[@]}")
  local remote_line
  printf -v remote_line '%q ' "${remote[@]}"
  remote_line=${remote_line% }
  printf -v remote_line '%s > %q 2>&1' "$remote_line" "$log"
  printf 'SSH_JOB host=%s gpu=%s key=%s result=%s log=%s\n' \
    "$GPU11_SSH_HOST" "$GPU11_DEVICE" "$key" "$result" "$log"
  if [ "$DRYRUN" = 1 ]; then
    printf 'SSH ssh -o BatchMode=yes %q %q\n' "$GPU11_SSH_HOST" "$remote_line"
    printf 'WORKER'; printf ' %q' "${worker[@]}"; printf '\n'
    return
  fi
  mkdir -p "$attempt_dir"
  : >"$scheduler_stdout"
  : >"$scheduler_stderr"
  chmod 600 "$scheduler_stdout" "$scheduler_stderr"
  ACTIVE_SUBMISSION_RECORD=$submission_record
  ACTIVE_SUBMISSION_PHASE=smoke
  ACTIVE_SUBMISSION_JOB_KEY=$full_job_key
  ACTIVE_SUBMISSION_ATTEMPT=$attempt_id
  ACTIVE_SUBMISSION_RESULT=$result
  ACTIVE_SUBMISSION_STDOUT=$scheduler_stdout
  ACTIVE_SUBMISSION_STDERR=$scheduler_stderr
  ACTIVE_SUBMISSION_JOB_ID=
  persist_active_submission submission_started submission_started >/dev/null || die \
    "could not create repo-local SSH submission receipt for $key"
  local ssh_status
  if ssh -o BatchMode=yes -o ConnectTimeout=10 "$GPU11_SSH_HOST" "$remote_line" \
      >"$scheduler_stdout" 2>"$scheduler_stderr"; then
    ssh_status=0
  else
    ssh_status=$?
  fi
  if [ "$ssh_status" -ne 0 ]; then
    sed -n '1,80p' "$scheduler_stderr" >&2 || true
    persist_active_submission submission_failed ssh_failed "$ssh_status" >/dev/null || die \
      "could not finalize SSH failure receipt for $key"
    ACTIVE_SUBMISSION_RECORD=
    die "SSH worker failed for $key"
  fi
  persist_active_submission submitted ssh_worker_completed 0 >/dev/null || die \
    "could not finalize SSH submission receipt for $key"
  ACTIVE_SUBMISSION_RECORD=
  printf 'SSH_COMPLETE key=%s result=%s log=%s\n' "$key" "$result" "$log"
}

run_gpu11_smoke() {
  require_executable "$PYVLLM"
  command -v ssh >/dev/null 2>&1 || die "ssh is required for gpu11-smoke"
  local tag=${GPU11_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-$$}
  case "$tag" in *[!A-Za-z0-9._-]*) die "unsafe GPU11_RUN_TAG: $tag" ;; esac
  local model="$MODELS/llama-3.1-8b-instruct"
  local speed_result="$RUN_ROOT/results/smoke/gpu11/$tag/fair_speed_b3.json"
  local native_result="$RUN_ROOT/results/smoke/gpu11/$tag/native_hybrid.json"
  local -a datasets=("$INPUT_MANIFEST") speed_cmd=() native_cmd=()
  speed_cmd=(
    "$PY" -u "$HERE/hayatequant_fair_speed.py"
    --model "$model" --local-files-only --device cuda:0 --dtype float16
    --avg-bits 3 --mode micro --micro-seq-len 512
    --micro-warmup 2 --micro-repeats 5 --bootstrap-resamples 100
    --attention-implementation sdpa --compile-mode off --fused-backend triton
    --output "$speed_result"
    --sink-tokens 0 --recent-tokens 0
  )
  run_ssh_worker "speed-$tag" "$PY" "$model" "$speed_result" \
    "${#datasets[@]}" "${datasets[@]}" -- "${speed_cmd[@]}"
  native_cmd=(
    "$PYVLLM" -u "$HERE/hayatequant_vllm_native_smoke.py"
    --bits 3 4 --head-dim 128 --kv-heads 8 --q-heads 32 --tokens 1024
    --key-scale-mode ub --codebook-init auto --bench-iters 50
    --input-dtype float16 --verify-optimized-store --output "$native_result"
  )
  run_ssh_worker "native-$tag" "$PYVLLM" "$model" "$native_result" \
    "${#datasets[@]}" "${datasets[@]}" -- "${native_cmd[@]}"
}

submit_smoke() {
  local small="$MODELS/Qwen3-0.6B"
  local llama="$MODELS/llama-3.1-8b-instruct"
  require_model "$small"
  local -a common=("$INPUT_MANIFEST" "$PINNED_HF_MANIFEST") cmd=()
  local result out manifest data_file

  out="$RUN_ROOT/results/smoke/ppl"
  result="$out/results_Qwen3-0.6B.json"
  cmd=(
    "$PY" -u "$HERE/codec_benchmark.py" --config-name paper_hayatequant
    "model_path=$small" "output_dir=$out"
    'methods=[TurboQuant-paper,FibQuant-paper,BlockQuant-paper,HQ,AdaptiveHQ]' 'bits=[3]'
    capture_tokens=128 timing_seq=128 timing_iters=2
    run_ppl=true ppl_windows=2 ppl_max_length=512 ppl_stride=256
    skip_outliers=false sink_tokens=0 recent_tokens=0
  )
  submit_job smoke ppl 01:00:00 \
    120G 8 "" "$small" "$result" "${#common[@]}" "${common[@]}" -- "${cmd[@]}"

  result="$RUN_ROOT/results/smoke/zeroshot/zeroshot_Qwen3-0.6B_smoke.json"
  cmd=(
    "$PY" -u "$HERE/zeroshot_benchmark.py"
    --model-path "$small" --output-root "$RUN_ROOT/results/smoke/zeroshot"
    --output-suffix smoke --bits 3 --tasks piqa
    --methods FullKV TurboQuant-paper FibQuant-paper BlockQuant-paper HQ AdaptiveHQ
    --include-fullkv
    --limit 4 --sample-seed 2027
    --sink-tokens 0 --recent-tokens 0 --no-skip-outliers
  )
  submit_job smoke zeroshot 01:00:00 120G 8 "" "$small" "$result" \
    "${#common[@]}" "${common[@]}" -- "${cmd[@]}"

  manifest="$RULER_ROOT/llama31-8b/manifest.json"
  out="$RUN_ROOT/results/smoke/ruler"
  result="$out/results_llama-3.1-8b-instruct.json"
  cmd=(
    "$PY" -u "$LC/kv_benchmark.py" --config-name benchmark_longcontext
    "model_path=$llama" "output_dir=$out"
    'methods=[FullKV,TurboQuant-paper,FibQuant-paper,BlockQuant-paper,HQ,AdaptiveHQ]'
    "$TURBO_VARIANTS" "$FIBQUANT_VARIANTS" "$BLOCKQUANT_VARIANTS" "$HAYATE_VARIANTS"
    'uniform_bits=[]' 'kivi_bits=[]' 'halo_variants=[]' 'codec_variants=[]'
    thinkv.auto_configure=false "${KV_ALL_OFF[@]}"
    run_official_ruler_v1=true
    "official_ruler_data_root=$RULER_ROOT/llama31-8b/4096"
    "official_ruler_manifest_path=$manifest"
    'official_ruler_tasks=[niah_single_1]'
    official_ruler_max_context=4096 official_ruler_limit=2
    skip_outliers=false sink_tokens=0 recent_tokens=0
    attn_implementation=sdpa +resume=false
  )
  local -a ruler_data=("$INPUT_MANIFEST" "$manifest")
  submit_job smoke ruler 02:00:00 180G 10 "" "$llama" "$result" \
    "${#ruler_data[@]}" "${ruler_data[@]}" -- "${cmd[@]}"

  out="$RUN_ROOT/results/smoke/longbench"
  result="$out/results_llama-3.1-8b-instruct.json"
  cmd=(
    "$PY" -u "$LC/kv_benchmark.py" --config-name benchmark_longcontext
    "model_path=$llama" "output_dir=$out"
    'methods=[FullKV,TurboQuant-paper,FibQuant-paper,BlockQuant-paper,HQ,AdaptiveHQ]'
    "$TURBO_VARIANTS" "$FIBQUANT_VARIANTS" "$BLOCKQUANT_VARIANTS" "$HAYATE_VARIANTS"
    'uniform_bits=[]' 'kivi_bits=[]' 'halo_variants=[]' 'codec_variants=[]'
    thinkv.auto_configure=false "${KV_ALL_OFF[@]}"
    run_longbench=true 'longbench_tasks=[passage_retrieval_en]'
    longbench_limit=1 longbench_max_length=4096
    skip_outliers=false sink_tokens=0 recent_tokens=0
    attn_implementation=sdpa +resume=false
  )
  submit_job smoke longbench 02:00:00 180G 10 "" "$llama" "$result" \
    "${#common[@]}" "${common[@]}" -- "${cmd[@]}"

  out="$RUN_ROOT/results/smoke/math500"
  result="$out/results_Qwen3-0.6B.json"
  cmd=(
    "$PY" -u "$LC/kv_benchmark.py" --config-name benchmark_longcontext
    "model_path=$small" "output_dir=$out"
    'methods=[FullKV,TurboQuant-paper,FibQuant-paper,BlockQuant-paper,HQ,AdaptiveHQ]'
    "$TURBO_VARIANTS" "$FIBQUANT_VARIANTS" "$BLOCKQUANT_VARIANTS" "$HAYATE_VARIANTS"
    'uniform_bits=[]' 'kivi_bits=[]' 'halo_variants=[]' 'codec_variants=[]'
    thinkv.auto_configure=false "${KV_ALL_OFF[@]}"
    run_math=true math_limit=2 max_new_math=256
    math_start=0 sample_seed=2027
    math_grader=math_verify math_verify_version=0.9.0
    math500_revision=6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be
    skip_outliers=false sink_tokens=0 recent_tokens=0
    attn_implementation=sdpa +resume=false
  )
  submit_job smoke math500 02:00:00 120G 10 "" "$small" "$result" \
    "${#common[@]}" "${common[@]}" -- "${cmd[@]}"

  data_file="$LONGCODE_ROOT/LQA/32K.json"
  out="$RUN_ROOT/results/smoke/longcodeqa"
  result="$out/results_llama-3.1-8b-instruct.json"
  cmd=(
    "$PY" -u "$LC/kv_benchmark.py" --config-name benchmark_longcontext
    "model_path=$llama" "output_dir=$out"
    'methods=[FullKV,TurboQuant-paper,FibQuant-paper,BlockQuant-paper,HQ,AdaptiveHQ]'
    "$TURBO_VARIANTS" "$FIBQUANT_VARIANTS" "$BLOCKQUANT_VARIANTS" "$HAYATE_VARIANTS"
    'uniform_bits=[]' 'kivi_bits=[]' 'halo_variants=[]' 'codec_variants=[]'
    thinkv.auto_configure=false "${KV_ALL_OFF[@]}"
    +run_longcodeqa=true "+longcodeqa_data_root=$LONGCODE_ROOT"
    '+longcodeqa_brackets=[32K]' +longcodeqa_max_context=131072
    +longcodeqa_max_new=16 +longcodeqa_start=0 +longcodeqa_limit=1
    skip_outliers=false sink_tokens=0 recent_tokens=0
    attn_implementation=sdpa +resume=false
  )
  local -a longcode_data=("$INPUT_MANIFEST" "$data_file" "$LLAMA_LONGCODE_CONTEXT")
  submit_job smoke longcode 02:00:00 180G 10 "" "$llama" "$result" \
    "${#longcode_data[@]}" "${longcode_data[@]}" -- "${cmd[@]}"

  result="$RUN_ROOT/results/smoke/speed/fair_speed_llama_b3.json"
  local -a speed_data=("$INPUT_MANIFEST")
  cmd=(
    "$PY" -u "$HERE/hayatequant_fair_speed.py"
    --model "$llama" --local-files-only --avg-bits 3 --dtype float16
    --mode micro --micro-seq-len 512 --micro-warmup 2 --micro-repeats 5
    --bootstrap-resamples 100 --attention-implementation sdpa
    --compile-mode off --fused-backend triton
    --sink-tokens 0 --recent-tokens 0
    --output "$result"
  )
  submit_job smoke speed 01:00:00 180G 8 "$PREFERRED_NODE" "$llama" "$result" \
    "${#speed_data[@]}" "${speed_data[@]}" -- "${cmd[@]}"

  submit_native_smoke
}

summarize_results() {
  mkdir -p "$RUN_ROOT/summaries"
  "$PY" "$HERE/aggregate_hayatequant_paper_shards.py" \
    --run-root "$RUN_ROOT" \
    --speed-replicates "$SPEED_REPLICATES" \
    --output "$RUN_ROOT/summaries/coverage_checked_aggregates.json"
}

usage() {
  printf '%s\n' \
    'Usage: submit_hayatequant_paper.sh COMMAND' \
    '' \
    'Submission commands:' \
    '  smoke        Capped end-to-end gate plus native hybrid smoke (inspect first)' \
    '  approve-smoke Validate all eight smoke outputs and write an explicit approval marker' \
    '  formal       Full PPL, corrected zero-shot, RULER, LongBench-E, MATH-500,' \
    '               feasible LongCodeQA, mechanism controls, and fair speed' \
    '  ppl          Formal seven-model complete WikiText-2' \
    '  mechanism    Formal two-model exact block-VQ mechanism controls' \
    '  zeroshot     Formal seven-model five-task corrected full splits' \
    '  ruler        Formal RULER v1: 2 models x 4K/8K/16K x 6 quality methods' \
    '  longbench    Formal complete 13-task LongBench-E on Llama/Qwen' \
    '  math         Formal complete MATH-500 + Math-Verify on Llama/Qwen' \
    '  longcodeqa   Formal Llama 32/64/128K and Qwen 32K' \
    '  speed        Formal paired fair-speed matrix; requires explicit idle SPEED_NODE' \
    '  native       Native hybrid/kernel smoke only (never labelled full HQ)' \
    '  gpu11-smoke  Direct SSH fair/native smoke fallback while gpu11 is Slurm-drained' \
    '' \
    'Local commands:' \
    '  preflight    Validate all model/dataset manifests; submit nothing' \
    '  speed-node-preflight Validate SPEED_NODE identity; set REQUIRE_SPEED_NODE_IDLE=1 for timing readiness' \
    '  summarize    Coverage-check and aggregate every completed formal axis' \
    '  formal-plan  DRYRUN-only source plan; cannot submit and does not approve smoke' \
    '  help         Show this message' \
    '' \
    'Fixed defaults:' \
    "  RUN_ROOT=$RUN_ROOT" \
    "  RESULTS_WORKTREE=$RESULTS_WORKTREE" \
    "  WORK_ROOT=$WORK_ROOT" \
    "  DATA_ROOT=$DATA_ROOT" \
    "  PY=$PY" \
    "  PYVLLM=$PYVLLM" \
    "  HF_HOME=$HF_HOME" \
    "  PINNED_HF_MANIFEST=$PINNED_HF_MANIFEST" \
    '' \
    'Approval: run smoke, inspect it, then set FORMAL_SMOKE_APPROVAL to the documented fixed statement and run approve-smoke.' \
    'RUN_ROOT must be papers/hayatequant_paper/experiments/results/RUN_ID in a separate linked Git worktree at the exact source commit.' \
    'Create it with: git worktree add --detach "$RESULTS_WORKTREE" "$(git rev-parse HEAD)"; caches stay in WORK_ROOT.' \
    'Controls: PARTITIONS, PREFERRED_NODE, SPEED_NODE, DRYRUN=1; formal SPEED_REPLICATES is fixed at 5.' \
    'Formal speed policy: idle SPEED_NODE, batch-8gpu all-GRES allocation, exclusive=user, afterok serial chain; full-plan quality jobs exclude it.' \
    'SSH fallback: GPU11_SSH_HOST, GPU11_DEVICE, GPU11_RUN_TAG.' \
    'For source-only plan tests, DRYRUN=1 SKIP_INPUT_PREFLIGHT=1 is allowed.'
}

command=${1:-help}
case "$command" in
  help|-h|--help) usage ;;
  preflight)
    validate_static_environment
    resolve_source_revision
    [ "$DRYRUN" != 1 ] || die \
      "preflight writes a manifest and is unavailable in DRYRUN mode"
    validate_results_repository
    mkdir -p "$(dirname "$INPUT_MANIFEST")" "$PAPER_HOME"
    ensure_input_manifest
    printf 'validated inputs: %s\n' "$INPUT_MANIFEST"
    ;;
  speed-node-preflight)
    validate_static_environment
    if [ "$REQUIRE_SPEED_NODE_IDLE" = 1 ]; then
      resolve_speed_node
    else
      validate_speed_node_identity
    fi
    ;;
  summarize) summarize_results ;;
  smoke)
    prepare_submission
    invalidate_smoke_approval
    submit_smoke
    ;;
  approve-smoke)
    prepare_submission
    validate_smoke_and_approve
    ;;
  formal)
    prepare_formal_submission
    resolve_speed_node
    # Submit the complete all-GRES timing chain first so the scheduler sees
    # the clean-node reservation before any quality work is queued.
    submit_speed_formal
    submit_ppl_formal
    submit_mechanism_formal
    submit_zeroshot_formal
    submit_ruler_formal
    submit_longbench_formal
    submit_math_formal
    submit_longcodeqa_formal
    ;;
  ppl)
    prepare_formal_quality_submission
    submit_ppl_formal
    ;;
  mechanism)
    prepare_formal_quality_submission
    submit_mechanism_formal
    ;;
  zeroshot)
    prepare_formal_quality_submission
    submit_zeroshot_formal
    ;;
  ruler)
    prepare_formal_quality_submission
    submit_ruler_formal
    ;;
  longbench)
    prepare_formal_quality_submission
    submit_longbench_formal
    ;;
  math)
    prepare_formal_quality_submission
    submit_math_formal
    ;;
  longcodeqa)
    prepare_formal_quality_submission
    submit_longcodeqa_formal
    ;;
  speed)
    prepare_formal_submission
    submit_speed_formal
    ;;
  native)
    prepare_submission
    submit_native_smoke
    ;;
  formal-plan)
    [ "$DRYRUN" = 1 ] || die "formal-plan is source inspection only and requires DRYRUN=1"
    [ "$SKIP_INPUT_PREFLIGHT" = 1 ] || die \
      "formal-plan is read-only and requires SKIP_INPUT_PREFLIGHT=1"
    prepare_submission
    # Source-only plan inspection has no approved smoke artifacts.  Fixed,
    # visibly synthetic digests keep the worker/export contract inspectable;
    # every executable formal command obtains real digests from the validated
    # marker instead.
    SMOKE_APPROVAL_MARKER_SHA256=$(printf '0%.0s' {1..64})
    SMOKE_VALIDATED_PLAN_SHA256=$(printf '1%.0s' {1..64})
    resolve_speed_node
    submit_speed_formal
    submit_ppl_formal
    submit_mechanism_formal
    submit_zeroshot_formal
    submit_ruler_formal
    submit_longbench_formal
    submit_math_formal
    submit_longcodeqa_formal
    ;;
  gpu11-smoke)
    prepare_submission
    run_gpu11_smoke
    ;;
  *) usage >&2; die "unknown command: $command" ;;
esac

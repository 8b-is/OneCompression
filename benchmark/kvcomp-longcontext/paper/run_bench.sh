#!/usr/bin/env bash
# Generic kvcomp paper-benchmark worker (SLURM / kagura).
#
# Usage (via sbatch from submit_all.sh):
#   sbatch <slurm-opts> run_bench.sh <BENCH_DIR> <cmd...>
#
# All SLURM directives (--job-name/--partition/--gres/--time/--output) are
# supplied on the sbatch command line by submit_all.sh; this body just cd's into
# the benchmark dir and execs the requested command, logging host/start/end for
# post-hoc accounting.
set -euo pipefail

BENCH_DIR="$1"; shift
cd "$BENCH_DIR"

echo "=================================================================="
echo "[run_bench] host=$(hostname)  jobid=${SLURM_JOB_ID:-NA}  gpu=${CUDA_VISIBLE_DEVICES:-NA}"
echo "[run_bench] bench_dir=$BENCH_DIR"
echo "[run_bench] cmd: $*"
echo "[run_bench] started: $(date -Is)"
echo "=================================================================="

# Avoid HF tokenizer fork warnings.
export TOKENIZERS_PARALLELISM=false
# Use the local datasets cache and skip the remote freshness check, which hangs
# when the HF metadata API is slow (all required datasets are pre-cached on
# lustre). LiveCodeBench jobs override this to 0 since that set is loaded from HF
# at runtime. Overridable so a fresh machine can set HF_DATASETS_OFFLINE=0.
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

"$@"
rc=$?

echo "=================================================================="
echo "[run_bench] finished: $(date -Is)  exit_code=$rc"
echo "=================================================================="
exit $rc

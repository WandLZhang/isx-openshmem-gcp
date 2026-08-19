#!/usr/bin/env bash
# Submit an ISx64 run that survives the two failure modes measured in this study.
#
#   1. Transport instability. Connection establishment fails 55-65% of the time above
#      32 PEs per node (results/rxm-connection-limit.md). A failed run
#      self-aborts after exhausting a 2^30 retry budget.
#   2. Host maintenance. H4D runs onHostMaintenance=TERMINATE and cannot live-migrate,
#      so a maintenance event on any node kills the job. Checkpointing costs more than
#      the run at this size, so the answer is to retry
#      (results/operations.md).
#
# Both are handled the same way: pre-flight the fabric so a doomed run fails in seconds
# instead of minutes, then retry.
#
#   ./run_isx64.sh <nodes> <pes_per_node> <keys_per_pe> [attempts]
set -uo pipefail

NODES=${1:?nodes}
PPN=${2:?pes per node}
KEYS=${3:?keys per pe}
ATTEMPTS=${4:-5}

BIN=${ISX64_BIN:-$HOME/isx64win}
PREFIX=${SOS_PREFIX:-$HOME/isx}
OUT=${ISX64_OUT:-$HOME/isx64_results}
mkdir -p "$OUT"

export PATH="$PREFIX/bin:$PATH" LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"
export SHMEM_OFI_PROVIDER=psm3
export PSM3_ALLOW_ROUTERS=1
export PSM3_UUID=$(printf '%08x-0000-0000-0000-000000000000' "${SLURM_JOB_ID:-1}")
export SHMEM_SYMMETRIC_SIZE=${SHMEM_SYMMETRIC_SIZE:-8G}
# GID 0 is link-local on this NIC and has no route. See results/adaptive-routing.md.
export FI_VERBS_GID_IDX=${FI_VERBS_GID_IDX:-1}
# Stops data CQ traffic starving connection-management progress.
export FI_OFI_RXM_CQ_EQ_FAIRNESS=${FI_OFI_RXM_CQ_EQ_FAIRNESS:-1}

# A run that completes takes seconds. Anything much longer is the failure path, and
# waiting out the retry budget wastes the allocation.
TIMEOUT=${ISX64_TIMEOUT:-300}

preflight () {
  # Two-PE one-sided put across the allocation. If this fails the fabric is not usable
  # and the full run has no chance, so fail in seconds rather than minutes.
  timeout 60 srun -N2 --ntasks-per-node=1 --mpi=pmi2 --export=ALL \
      "$BIN" 65536 1 "$OUT/preflight.$$" >/dev/null 2>&1
}

for a in $(seq 1 "$ATTEMPTS"); do
  echo "=== attempt $a of $ATTEMPTS ==="

  if ! preflight; then
    echo "    pre-flight failed, fabric not usable. Not spending the allocation."
    sleep 15
    continue
  fi

  T0=$(date +%s)
  LOG="$OUT/run.a${a}"
  timeout "$TIMEOUT" srun -N"$NODES" --ntasks-per-node="$PPN" --mpi=pmi2 --export=ALL \
      "$BIN" "$KEYS" 1 "$LOG" > "$LOG.stdout" 2>&1
  RC=$?
  T1=$(date +%s)

  if grep -qi PASSED "$LOG.stdout" 2>/dev/null; then
    echo "    PASSED in $((T1-T0)) s, log $LOG.stdout"
    exit 0
  fi

  if [ $RC -eq 124 ]; then
    echo "    timed out after ${TIMEOUT}s"
  elif grep -qi "retry limit exceeded" "$LOG.stdout" 2>/dev/null; then
    echo "    transport gave up after $((T1-T0)) s (retry limit)"
  else
    echo "    failed rc=$RC after $((T1-T0)) s"
  fi
done

echo "no attempt succeeded in $ATTEMPTS tries"
exit 1

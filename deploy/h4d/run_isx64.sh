#!/usr/bin/env bash
# Submit an ISx64 run that survives host maintenance.
#
# H4D runs onHostMaintenance=TERMINATE and cannot live-migrate, so a maintenance event on
# any node kills the job. Checkpointing costs more than the run at this size, so the answer
# is to retry (results/operations.md). Pre-flight the fabric first so a doomed run fails in
# seconds instead of spending the allocation.
#
#   ./run_isx64.sh <nodes> <pes_per_node> <keys_per_pe> [attempts]
set -uo pipefail

NODES=${1:?nodes}
PPN=${2:?pes per node}
KEYS=${3:?keys per pe}
ATTEMPTS=${4:-5}

# `make -C src/cpu` writes bin/isx64_win at the repo root. Older notes said to copy it to
# ~/bin/isx64win, so accept both and say which paths were tried if neither is there.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [ -n "${ISX64_BIN:-}" ]; then
  BIN=$ISX64_BIN
elif [ -x "$REPO/bin/isx64_win" ]; then
  BIN=$REPO/bin/isx64_win
elif [ -x "$HOME/bin/isx64win" ]; then
  BIN=$HOME/bin/isx64win
else
  echo "no isx64_win binary. Tried:" >&2
  echo "  \$ISX64_BIN        (${ISX64_BIN:-unset})" >&2
  echo "  $REPO/bin/isx64_win" >&2
  echo "  $HOME/bin/isx64win" >&2
  echo "Build it with: source /opt/isx/env.sh && make -C $REPO/src/cpu" >&2
  exit 2
fi

# Must match PREFIX in 01_build_sos.sh.
PREFIX=${SOS_PREFIX:-/opt/isx}
OUT=${ISX64_OUT:-$HOME/isx64_results}
mkdir -p "$OUT"

export PATH="$PREFIX/bin:$PATH" LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"
export SHMEM_OFI_PROVIDER=psm3
export PSM3_ALLOW_ROUTERS=1
export PSM3_UUID=$(printf '%08x-0000-0000-0000-000000000000' "${SLURM_JOB_ID:-1}")
# The windowed exchange holds one window slot per PE, about 0.5 MB, independent of the
# dataset. 1G is ample at 192 PEs per node.
export SHMEM_SYMMETRIC_SIZE=${SHMEM_SYMMETRIC_SIZE:-1G}
export SHMEM_BOOTSTRAP=${SHMEM_BOOTSTRAP:-PMI}
# Some providers size internal address tables from this. Set it to the rank count so it is
# never below. Untested above 2 nodes.
export FI_UNIVERSE_SIZE=${FI_UNIVERSE_SIZE:-$((NODES * PPN))}

# 1.4 TB on two nodes takes 275 s, and time scales with keys per node. Raise this for a
# large run rather than letting a healthy job get killed.
TIMEOUT=${ISX64_TIMEOUT:-1800}

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
  else
    echo "    failed rc=$RC after $((T1-T0)) s"
  fi
done

echo "no attempt succeeded in $ATTEMPTS tries"
exit 1

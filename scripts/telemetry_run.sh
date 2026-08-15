#!/bin/bash
# Deliverable 3: network-utilisation telemetry around an ISx64 run.
#
# Snapshots the irdma0 hardware counters on every node before and after the run and
# reports the deltas. This is the only direct measurement of what the fabric actually
# did, as opposed to what the application thinks it asked for.
#
# The counters that matter:
#   ip4OutOctets / ip4InOctets   bytes actually on the wire
#   InRdmaWrites                 one-sided writes landing on this node, which is the
#                                mechanism the whole study is about
#   cnpSent / cnpHandled         congestion notification points. Non-zero means the
#                                fabric is signalling congestion, which is one of the
#                                two causes the study asks us to distinguish
#   InProtoErrors                protocol errors, the other cause
#
# Retries because the stack completes about 30% of the time; see
# results/BLOCKER_reproducibility_20260814.md.
#
#   bash telemetry_run.sh [keys_per_pe] [pes_per_node] [nodes] [attempts]
set -uo pipefail

P=${PREFIX:-$HOME/isx}
export PATH=$P/bin:$PATH LD_LIBRARY_PATH=$P/lib:${LD_LIBRARY_PATH:-}
export SHMEM_OFI_PROVIDER="verbs;ofi_rxm" SHMEM_SYMMETRIC_SIZE=${SHMEM_SYMMETRIC_SIZE:-1G}

KEYS=${1:-4194304}; PPN=${2:-16}; N=${3:-2}; TRIES=${4:-8}
PES=$((PPN*N))
BIN=${BIN:-$HOME/isx64bin}
HW=/sys/class/infiniband/irdma0/ports/1/hw_counters
COUNTERS="ip4InOctets ip4OutOctets InRdmaWrites InRdmaReads InRdmaSends cnpSent cnpHandled cnpIgnored InProtoErrors CRC_errors"

snap() {
  srun -N"$N" --ntasks-per-node=1 --overlap bash -c '
    for c in '"$COUNTERS"'; do
      v=$(cat '"$HW"'/$c 2>/dev/null || echo 0)
      echo "$(hostname) $c $v"
    done' 2>/dev/null
}

GB=$(awk -v k="$KEYS" -v p="$PES" 'BEGIN{printf "%.2f", k*p*8/1e9}')
echo "=== ISx64 telemetry: ${PES} PEs (${PPN}/node x ${N}), ${KEYS} keys/PE, ${GB} GB of keys ==="

for attempt in $(seq 1 "$TRIES"); do
  snap > /tmp/tel_before.txt
  T0=$(date +%s.%N)
  OUT=$(timeout 300 srun -N"$N" --ntasks-per-node="$PPN" --mpi=pmi2 --export=ALL \
          "$BIN" "$KEYS" 1 /dev/null 2>&1)
  T1=$(date +%s.%N)
  if ! echo "$OUT" | grep -q "verification       : PASSED"; then
    echo "  attempt ${attempt}: did not complete"
    continue
  fi
  snap > /tmp/tel_after.txt

  ELAPSED=$(awk -v a="$T0" -v b="$T1" 'BEGIN{printf "%.3f", b-a}')
  TTS=$(echo "$OUT" | grep -oP 'time to solution\s*:\s*\K[0-9.]+')
  A2A=$(echo "$OUT" | grep -oP '^\s+all2all\s+\K[0-9.]+')
  RDX=$(echo "$OUT" | grep -oP '^\s+radix\s+\K[0-9.]+')
  echo "  completed on attempt ${attempt}: wall=${ELAPSED}s TTS=${TTS}s all2all=${A2A}s radix=${RDX}s"
  echo

  TEL_A2A="$A2A" TEL_KEYS="$KEYS" TEL_PES="$PES" TEL_RDX="$RDX" python3 "$(dirname "$0")/telemetry_report.py"
  exit 0
done

echo "no attempt completed in ${TRIES} tries"
exit 1

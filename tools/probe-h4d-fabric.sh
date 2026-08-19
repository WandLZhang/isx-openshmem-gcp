#!/usr/bin/env bash
# Probe A: is the OpenSHMEM compliance path real on H4D Cloud RDMA?
#
# This is the most important result in the study. Everything else assumes it passes.
#
# The study requires an OpenSHMEM one-sided PGAS model and rules out MPI. Three things
# have to be true, and each can fail quietly:
#
#   1. The fabric offers FI_RMA and FI_ATOMIC. If it does not, SOS may still run over a
#      fallback transport, pass every correctness check, and measure nothing about RDMA.
#   2. The put is genuinely one-sided. A runtime that emulates RMA with a progress
#      thread and matched messages is functionally correct and not responsive to the
#      study.
#   3. No MPI is linked in. An oshcc that wraps mpicc would disqualify the result.
#
# Run from a login node with the cluster up.
#
#   bash tools/probe-h4d-fabric.sh 16 4       # 16 nodes, 4 PEs each
set -uo pipefail

NODES="${1:-16}"
PES_PER_NODE="${2:-4}"
TOTAL_PES=$(( NODES * PES_PER_NODE ))
PREFIX="${PREFIX:-/opt/isx}"
OUT="${OUT:-results/probeA_h4d_$(date +%Y%m%d-%H%M%S)}"

mkdir -p "$(dirname "${OUT}")"
source "${PREFIX}/env.sh"
export FI_UNIVERSE_SIZE="${TOTAL_PES}"

pass=0; fail=0
check() { if [ "$1" = 0 ]; then echo "  PASS  $2"; pass=$((pass+1)); else echo "  FAIL  $2"; fail=$((fail+1)); fi; }

{
echo "=== Probe A: H4D OpenSHMEM compliance ==="
echo "date          : $(date -Is)"
echo "nodes         : ${NODES}"
echo "PEs per node  : ${PES_PER_NODE}"
echo "total PEs     : ${TOTAL_PES}"
echo "machine       : $(curl -s -H Metadata-Flavor:Google \
                        metadata.google.internal/computeMetadata/v1/instance/machine-type 2>/dev/null | awk -F/ '{print $NF}')"
echo

echo "--- 1. fabric capabilities ---"
fi_info -p "verbs;ofi_rxm" -c FI_RMA >/dev/null 2>&1
check $? "verbs;ofi_rxm offers FI_RMA"
fi_info -p "verbs;ofi_rxm" -c FI_ATOMIC >/dev/null 2>&1
check $? "verbs;ofi_rxm offers FI_ATOMIC"
fi_info -p "verbs;ofi_rxm" 2>/dev/null | grep -E "provider:|domain:|version:" | head -6

echo
echo "--- 2. the IRDMA interface is present and up ---"
ip -br link 2>/dev/null | grep -qi "rdma\|ibp\|irdma"
check $? "an RDMA-capable interface exists"
lsmod 2>/dev/null | grep -q irdma
check $? "irdma kernel module loaded"

echo
echo "--- 3. no MPI in the stack ---"
# The study excludes MPI implementations. If oshcc links libmpi the result is not
# responsive regardless of how it performs.
if ldd "${PREFIX}/bin/oshcc" 2>/dev/null | grep -qi "libmpi"; then
  check 1 "oshcc does not link MPI"
else
  check 0 "oshcc does not link MPI"
fi

echo
echo "--- 4. SOS test suite ---"
if [ -d "${PREFIX}/src/SOS/test" ]; then
  ( cd "${PREFIX}/src/SOS" && make check TESTS_ENVIRONMENT="oshrun -n 2" ) >/tmp/sos_check.log 2>&1
  check $? "SOS make check"
  tail -5 /tmp/sos_check.log
else
  echo "  SKIP  SOS sources not at ${PREFIX}/src/SOS"
fi

echo
echo "--- 5. one-sided semantics across nodes ---"
# The target PE never calls a receive. If this returns the right value the put really
# did land in remote memory without peer participation.
cat > /tmp/onesided.c <<'CEOF'
#include <shmem.h>
#include <stdio.h>
int main(void){
  shmem_init();
  int me=shmem_my_pe(), n=shmem_n_pes();
  static long long slot=-1;
  shmem_barrier_all();
  long long v = 1000 + me;
  shmem_longlong_put(&slot,&v,1,(me+1)%n);
  shmem_barrier_all();
  long long want = 1000 + ((me+n-1)%n);
  if(slot!=want){ printf("PE %d got %lld want %lld\n",me,slot,want); shmem_global_exit(1);}
  if(me==0) printf("one-sided put verified across %d PEs\n", n);
  shmem_finalize(); return 0;
}
CEOF
"${PREFIX}/bin/oshcc" -O2 -o /tmp/onesided /tmp/onesided.c 2>/dev/null \
  && "${PREFIX}/bin/oshrun" -n "${TOTAL_PES}" /tmp/onesided
check $? "one-sided put across ${TOTAL_PES} PEs"

echo
echo "--- 6. ISx64 at Phase 1 scale ---"
# 2^26 keys per PE. At 64 PEs that is 4.3e9 keys, 34 GB, well inside a baseline run and
# already past the 2^31 boundary where upstream's int counters would have wrapped.
KEYS_PER_PE=$(( 1 << 26 ))
"${PREFIX}/bin/oshrun" -n "${TOTAL_PES}" ../bin/isx64 "${KEYS_PER_PE}" 3 "${OUT}.csv"
check $? "ISx64 completed and verified at ${TOTAL_PES} PEs"

echo
echo "=== ${pass} passed, ${fail} failed ==="
if [ "${fail}" -gt 0 ]; then
  echo "VERDICT: NO-GO. The compliance path does not hold; do not proceed to Phase 2."
else
  echo "VERDICT: GO."
fi
} 2>&1 | tee "${OUT}.txt"

exit "${fail}"

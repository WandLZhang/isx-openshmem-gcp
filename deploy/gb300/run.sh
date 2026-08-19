#!/usr/bin/env bash
# Run ISx-NVSHMEM at a target scale. One process per GPU.
#   bash run.sh smoke     one node, 4 GPUs, seconds
#   bash run.sh 10        102 nodes, 408 GPUs, 100 TB
#   bash run.sh full      1024 nodes, 4096 GPUs, 1 PB
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="$HERE/isx_nvshmem"
[ -x "$BIN" ] || { echo "build first: bash $HERE/build.sh" >&2; exit 1; }

export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-8G}"

case "${1:-smoke}" in
  smoke) NODES=1;    KEYS=4194304     ;;   # correctness, seconds
  10)    NODES=102;  KEYS=30637254902 ;;   # 100 TB over 408 GPUs
  full)  NODES=1024; KEYS=30517578125 ;;   # 1 PB over 4096 GPUs
  *) echo "usage: $0 {smoke|10|full}" >&2; exit 1 ;;
esac

echo "nodes=$NODES gpus=$((NODES*4)) keys/gpu=$KEYS"
if command -v srun >/dev/null 2>&1 && [ -n "${SLURM_JOB_ID:-}${SLURM_CLUSTER_NAME:-}" ]; then
  srun -N "$NODES" --ntasks-per-node=4 "$BIN" "$KEYS" 1
else
  nvshmrun -n $((NODES*4)) "$BIN" "$KEYS" 1
fi

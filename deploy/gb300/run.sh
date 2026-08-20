#!/usr/bin/env bash
# Run ISx-NVSHMEM at a target scale. One process per GPU.
#   bash run.sh smoke     one node, 4 GPUs, seconds
#   bash run.sh 10        108 nodes, 432 GPUs, 100 TB
#   bash run.sh full      1026 nodes, 4104 GPUs, 1 PB
#
# Node counts are multiples of 18 because A4X capacity is allocated in fixed 18-node
# NVLink domains. 18 nodes is one NVL72.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="$HERE/isx_nvshmem"
[ -x "$BIN" ] || { echo "build first: bash $HERE/build.sh" >&2; exit 1; }

export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-32G}"
# RoCE v2 with the IPv4-mapped GID. Index 0 is RoCE v1 link-local and does not route.
export NVSHMEM_IB_GID_INDEX="${NVSHMEM_IB_GID_INDEX:-3}"

case "${1:-smoke}" in
  smoke) NODES=1;    KEYS=4194304     ;;   # correctness, seconds
  10)    NODES=108;  KEYS=28935185185 ;;   # 100 TB over 432 GPUs
  full)  NODES=1026; KEYS=30458089181 ;;   # 1 PB over 4104 GPUs
  *) echo "usage: $0 {smoke|10|full}" >&2; exit 1 ;;
esac

echo "nodes=$NODES gpus=$((NODES*4)) keys/gpu=$KEYS"
if command -v srun >/dev/null 2>&1 && [ -n "${SLURM_JOB_ID:-}${SLURM_CLUSTER_NAME:-}" ]; then
  srun -N "$NODES" --ntasks-per-node=4 "$BIN" "$KEYS" 1
else
  nvshmrun -n $((NODES*4)) -ppn 4 "$BIN" "$KEYS" 1
fi

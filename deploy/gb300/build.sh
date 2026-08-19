#!/usr/bin/env bash
# Build ISx-NVSHMEM. Run on any node with CUDA 12 and an NVIDIA driver.
#   ARCH=sm_100 bash build.sh      # Blackwell, the default
#   ARCH=sm_80  bash build.sh      # A100, for a smoke test
set -euo pipefail
ARCH="${ARCH:-sm_100}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/../../src"

# libnvshmem3-static supplies libnvshmem_device.a; without it the link fails on
# nvshmemi_init_thread. libhwloc15 is a runtime dependency of nvshmrun.
sudo apt-get update -qq
sudo apt-get install -y -qq libnvshmem3-dev-cuda-12 libnvshmem3-static-cuda-12 \
                            libnvshmem3-cuda-12 libhwloc15

NV_INC=$(dirname "$(find /usr -name nvshmem.h 2>/dev/null | head -1)")
NV_LIB=$(dirname "$(find /usr -name 'libnvshmem_device.a' 2>/dev/null | head -1)")
[ -n "$NV_INC" ] && [ -n "$NV_LIB" ] || { echo "NVSHMEM not found after install" >&2; exit 1; }

export PATH=/usr/local/cuda/bin:$PATH
nvcc -O3 -std=c++17 -arch="$ARCH" -rdc=true \
     -I"$NV_INC" -I"$SRC/cpu" \
     "$SRC/gpu/isx_nvshmem.cu" \
     -L"$NV_LIB" -lnvshmem_host -lnvshmem_device -lcudart -lcurand \
     -o "$HERE/isx_nvshmem"
echo "built $HERE/isx_nvshmem  (arch=$ARCH)"

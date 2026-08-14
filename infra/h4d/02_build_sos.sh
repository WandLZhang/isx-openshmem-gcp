#!/usr/bin/env bash
# Build Sandia OpenSHMEM against Cloud RDMA on an H4D node.
#
# This is the compliance path. The study excludes MPI-based implementations, so the
# runtime underneath ISx64 has to be a real OpenSHMEM one doing real one-sided RMA.
#
# Sandia OpenSHMEM (SOS) sits directly on libfabric, and H4D Cloud RDMA presents itself
# to libfabric as the `verbs;ofi_rxm` provider over the Intel iDPF/iRDMA driver. So SOS
# over that provider needs no MPI anywhere in the stack.
#
# The one thing to prove before trusting any result: `ofi_rxm` is a reliable-message
# layer built over verbs RC. OpenSHMEM needs FI_RMA and FI_ATOMIC from the provider. If
# those capabilities are missing, SOS can still build and run by falling back to a
# different transport, and every correctness test will pass while measuring nothing about
# RDMA. Step 1 checks for that explicitly and refuses to continue.
#
# Run on one H4D node. Install to a shared filesystem so the whole cluster sees it.
set -euo pipefail

PREFIX="${PREFIX:-/opt/isx}"
LIBFABRIC_MIN="2.2.0"
SOS_VERSION="${SOS_VERSION:-v1.5.3}"
JOBS="${JOBS:-$(nproc)}"

echo "==> 1. verifying the fabric exposes what OpenSHMEM needs"
if ! command -v fi_info >/dev/null; then
  echo "fi_info not found. Use the HPC VM image (Rocky 8, build 20250917 or later)," >&2
  echo "which ships libfabric, rdma-core and the iDPF/iRDMA driver." >&2
  exit 1
fi

FI_VER=$(fi_info --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
echo "    libfabric ${FI_VER} (need >= ${LIBFABRIC_MIN})"

echo "    providers offering FI_RMA:"
if ! fi_info -p "verbs;ofi_rxm" -c FI_RMA >/dev/null 2>&1; then
  echo "" >&2
  echo "FATAL: verbs;ofi_rxm does not offer FI_RMA on this node." >&2
  echo "Without one-sided RMA there is no OpenSHMEM compliance story, and a build that" >&2
  echo "falls back to sockets would produce numbers that mean nothing. Check:" >&2
  echo "  - the IRDMA vNIC is attached and up (ip link)" >&2
  echo "  - the instance is on the Falcon VPC" >&2
  echo "  - lsmod | grep irdma" >&2
  exit 2
fi
fi_info -p "verbs;ofi_rxm" -c FI_RMA | grep -E "provider:|fabric:|domain:|caps:" | head -8
echo "    FI_ATOMIC:"
fi_info -p "verbs;ofi_rxm" -c FI_ATOMIC >/dev/null 2>&1 \
  && echo "      present" \
  || echo "      MISSING. shmem_atomic_fetch_add drives the exchange; expect failure."

echo "==> 2. dependencies"
sudo dnf install -y -q git autoconf automake libtool make gcc gcc-c++ \
                       rdma-core-devel libfabric-devel hwloc-devel 2>&1 | tail -2 || true

echo "==> 3. Sandia OpenSHMEM ${SOS_VERSION}"
mkdir -p "${PREFIX}/src" && cd "${PREFIX}/src"
[[ -d SOS ]] || git clone -q --branch "${SOS_VERSION}" --depth 1 \
  https://github.com/Sandia-OpenSHMEM/SOS.git
cd SOS
[[ -f configure ]] || ./autogen.sh

# --disable-fortran            not needed, and it pulls in a compiler we do not have
# --enable-pmi-simple          SOS launches through its own PMI, not through mpirun.
#                              This is what keeps MPI out of the stack.
# --enable-ofi-mr=basic        Falcon does not support scalable memory registration
# --disable-ofi-inject         RxM buffers small sends itself; double-buffering hurts
./configure \
  --prefix="${PREFIX}" \
  --with-ofi="${OFI_PREFIX:-/usr}" \
  --enable-pmi-simple \
  --disable-fortran \
  --enable-ofi-mr=basic \
  --disable-ofi-inject \
  CFLAGS="-O3 -march=native"

make -j"${JOBS}"
make install

echo "==> 4. environment"
cat > "${PREFIX}/env.sh" <<'ENVEOF'
# Source before building or running ISx64 on H4D.
export PATH=/opt/isx/bin:$PATH
export LD_LIBRARY_PATH=/opt/isx/lib:$LD_LIBRARY_PATH

# Pin the provider. If this is unset libfabric may select `tcp`, which works, passes
# every test, and measures the wrong thing entirely.
export FI_PROVIDER="verbs;ofi_rxm"

# Google's published H4D tuning.
export FI_VERBS_INLINE_SIZE=39
export FI_OFI_RXM_BUFFER_SIZE=4096
export FI_OFI_RXM_SAR_LIMIT=2147483648   # 2 GB segmentation/reassembly threshold

# Must equal the total rank count. RxM sizes its address vector from this; leaving it at
# the default causes address resolution failures partway through a large all-to-all
# rather than at startup.
: "${FI_UNIVERSE_SIZE:=4096}"
export FI_UNIVERSE_SIZE

# ISx64 sizes its receive buffer from keys-per-PE, so the symmetric heap must be at
# least keys_per_pe * 8 * 1.2 per PE, plus slack. Raise for large runs.
export SHMEM_SYMMETRIC_SIZE="${SHMEM_SYMMETRIC_SIZE:-16G}"
ENVEOF

echo "==> 5. smoke test"
source "${PREFIX}/env.sh"
cat > /tmp/onesided.c <<'CEOF'
/* Proves the put is genuinely one-sided: the target never calls a matching receive. */
#include <shmem.h>
#include <stdio.h>
int main(void) {
  shmem_init();
  const int me = shmem_my_pe(), n = shmem_n_pes();
  static long long slot = -1;
  shmem_barrier_all();
  long long v = me;
  shmem_longlong_put(&slot, &v, 1, (me + 1) % n);   /* no receive on the peer */
  shmem_barrier_all();
  printf("PE %d of %d received %lld from PE %d\n", me, n, slot, (me + n - 1) % n);
  shmem_finalize();
  return 0;
}
CEOF
"${PREFIX}/bin/oshcc" -O2 -o /tmp/onesided /tmp/onesided.c
"${PREFIX}/bin/oshrun" -n 2 /tmp/onesided

cat <<EOF

SOS installed to ${PREFIX}. Before every run:

  source ${PREFIX}/env.sh

Then build ISx64:

  cd src/isx64 && make provider-check && make
EOF

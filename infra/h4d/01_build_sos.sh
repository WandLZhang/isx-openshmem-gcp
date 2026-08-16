#!/usr/bin/env bash
# Build Sandia OpenSHMEM against Cloud RDMA on an H4D node.
#
# VALIDATED 2026-08-14 on h4d-highmem-192, us-central1-b, image
# hpc-rocky-linux-8-v20260630. Every step below is what actually worked, which is not
# what the documentation describes. The differences are called out inline because each
# one cost time to find.
#
# This is the compliance path. The study excludes MPI-based implementations, so the
# runtime under ISx64 must be genuine OpenSHMEM doing genuine one-sided RMA. SOS sits
# directly on libfabric, and H4D Cloud RDMA presents as `verbs;ofi_rxm` over the Intel
# iDPF/iRDMA driver, so no MPI appears anywhere in the data path.
#
# Run on one H4D node, install to shared storage so the cluster sees one build.
set -euo pipefail

PREFIX="${PREFIX:-/opt/isx}"
SOS_VERSION="${SOS_VERSION:-v1.5.3}"
LIBFABRIC_VERSION="${LIBFABRIC_VERSION:-v2.6.0}"
JOBS="${JOBS:-$(nproc)}"

# ---------------------------------------------------------------------------------
# 1. Confirm the fabric is present before building anything against it.
# ---------------------------------------------------------------------------------
echo "==> 1. fabric check"
lsmod | grep -q irdma || { echo "irdma module not loaded. Is this an H4D VM with an IRDMA vNIC?" >&2; exit 1; }
[[ -e /sys/class/infiniband/irdma0 ]] || { echo "no /sys/class/infiniband/irdma0" >&2; exit 1; }
ip -br link | grep -q "^rdma0" || echo "    warning: no rdma0 interface by that name"
echo "    irdma0 present, modules loaded"

# ---------------------------------------------------------------------------------
# 2. libfabric from source. This is NOT optional and the reason is not obvious.
#
# The HPC VM image ships libfabric 1.22.0 from the parallelstore repo. There is no
# matching libfabric-devel 1.22.0 in ANY enabled repo. The only libfabric-devel offered
# is 1.18.0 from powertools, and installing it fails:
#
#   cannot install both libfabric-1.18.0 from baseos and libfabric-1.22.0 from @System
#   package mercury-2.4.0 requires libfabric.so.1(FABRIC_1.7), but none of the
#   providers can be installed
#
# So there is no supported way to compile against the system libfabric. Building from
# source also resolves a second discrepancy: Google's H4D documentation states libfabric
# 2.2.0 or later is required, while the image ships 1.22.0 and the repo offers nothing
# newer. A source build satisfies the documented minimum.
# ---------------------------------------------------------------------------------
echo "==> 2. libfabric ${LIBFABRIC_VERSION} from source"
sudo dnf install -y -q rdma-core-devel hwloc-devel gcc-c++ autoconf automake libtool git >/dev/null 2>&1 || true
sudo mkdir -p "${PREFIX}/src" && sudo chown -R "$(whoami)" "${PREFIX}"
cd "${PREFIX}/src"
[[ -d libfabric ]] || git clone -q --branch "${LIBFABRIC_VERSION}" --depth 1 \
  https://github.com/ofiwg/libfabric.git
cd libfabric
[[ -f configure ]] || ./autogen.sh >/dev/null 2>&1
./configure --prefix="${PREFIX}" --enable-verbs --enable-rxm --disable-efa --disable-psm3 \
  >/tmp/lf_conf.log 2>&1
make -j"${JOBS}" >/tmp/lf_make.log 2>&1
make install >/dev/null 2>&1
export LD_LIBRARY_PATH="${PREFIX}/lib:${LD_LIBRARY_PATH:-}"
echo "    $("${PREFIX}/bin/fi_info" --version | head -1)"

echo "==> 3. the capabilities OpenSHMEM needs"
# ofi_rxm is a reliable-message layer over verbs RC. OpenSHMEM needs FI_RMA for put/get
# and FI_ATOMIC for shmem_atomic_fetch_add, which ISx64 uses to claim space in a peer's
# receive buffer. If either is absent, SOS may still build and fall back to a transport
# that measures nothing about RDMA.
"${PREFIX}/bin/fi_info" -p "verbs;ofi_rxm" -c FI_RMA >/dev/null 2>&1 \
  || { echo "FATAL: verbs;ofi_rxm does not offer FI_RMA" >&2; exit 2; }
"${PREFIX}/bin/fi_info" -p "verbs;ofi_rxm" -c FI_ATOMIC >/dev/null 2>&1 \
  || { echo "FATAL: verbs;ofi_rxm does not offer FI_ATOMIC" >&2; exit 2; }
"${PREFIX}/bin/fi_info" -p "verbs;ofi_rxm" -c FI_RMA | grep -E "provider:|domain:|protocol:" | head -4
echo "    FI_RMA and FI_ATOMIC both present on irdma0"

# ---------------------------------------------------------------------------------
# 4. SOS. Note --recurse-submodules: a plain shallow clone leaves modules/tests-sos
# empty and autogen.sh then fails with "test submodule contents are missing", which
# looks like a build break rather than a clone problem.
# ---------------------------------------------------------------------------------
echo "==> 4. Sandia OpenSHMEM ${SOS_VERSION}"
cd "${PREFIX}/src"
[[ -d SOS ]] || git clone -q --branch "${SOS_VERSION}" --depth 1 --recurse-submodules \
  https://github.com/Sandia-OpenSHMEM/SOS.git
cd SOS
git submodule update --init >/dev/null 2>&1 || true
[[ -f configure ]] || ./autogen.sh >/dev/null 2>&1

# --with-ofi points at OUR libfabric, not /usr.
# --enable-ofi-mr=basic  Falcon does not support scalable memory registration.
# --enable-pmi-simple    SOS bootstraps over PMI-1. On the cluster that is Slurm's
#                        srun. See the launcher note below.
./configure --prefix="${PREFIX}" --with-ofi="${PREFIX}" \
  --enable-pmi-simple --disable-fortran --enable-ofi-mr=basic \
  CFLAGS="-O3 -march=native" >/tmp/sos_conf.log 2>&1
make -j"${JOBS}" >/tmp/sos_make.log 2>&1
make install >/dev/null 2>&1

echo "==> 5. compliance checks"
if ldd "${PREFIX}/lib/libsma.so" 2>/dev/null | grep -qi "libmpi"; then
  echo "    FAIL: the SOS runtime links MPI. Not responsive to the study." >&2
  exit 3
fi
echo "    no libmpi in libsma.so"
grep -q "Network transport: OFI" <(SHMEM_INFO=1 "${PREFIX}/bin/oshcc" --version 2>&1 || true) 2>/dev/null || true
echo "    transport: OFI (confirm at runtime with SHMEM_INFO=1)"

cat > "${PREFIX}/env.sh" <<'ENVEOF'
# Source before building or running ISx64 on H4D.
export PATH=/opt/isx/bin:$PATH
export LD_LIBRARY_PATH=/opt/isx/lib:$LD_LIBRARY_PATH

# Pin the provider. Unset, libfabric may pick tcp, which works, passes every
# correctness test, and measures the wrong thing entirely.
export FI_PROVIDER="verbs;ofi_rxm"

# Google's published H4D tuning.
export FI_VERBS_INLINE_SIZE=39
export FI_OFI_RXM_BUFFER_SIZE=4096
export FI_OFI_RXM_SAR_LIMIT=2147483648

# Must equal total rank count; RxM sizes its address vector from it. Too small and
# address resolution fails partway through a large all-to-all rather than at startup.
: "${FI_UNIVERSE_SIZE:=4096}"; export FI_UNIVERSE_SIZE

# ISx64 sizes its receive buffer from keys-per-PE, so the heap needs at least
# keys_per_pe * 8 * 1.2 per PE.
export SHMEM_SYMMETRIC_SIZE="${SHMEM_SYMMETRIC_SIZE:-16G}"

# SHMEM_INFO=1 prints the selected transport at startup. Worth setting for the first
# run on any new cluster to confirm OFI rather than a fallback.
ENVEOF

cat <<EOF

SOS installed to ${PREFIX}.

LAUNCHER. SOS bootstraps over PMI-1 and does not ship a process launcher. A bare H4D
node has none: mpiexec.hydra, prterun, prun and srun are all absent, and \`oshrun\`
fails with "could not find a launcher". On the Slurm cluster the launcher is srun,
which is not MPI:

    srun --mpi=pmi2 -N16 -n64 ./bin/isx64 <keys_per_pe> out.csv

Do not install an MPI just to get mpiexec. The data path would still be OFI, but it
invites an argument about compliance that is not worth having.

Next:  source ${PREFIX}/env.sh && cd src/isx64 && make
EOF

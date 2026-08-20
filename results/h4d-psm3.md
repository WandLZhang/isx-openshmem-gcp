# H4D on PSM3

2026-08-19. Two `h4d-highmem-192`, us-east1-b. libfabric 2.6.0 built with `--enable-psm3`,
Sandia OpenSHMEM with one source patch.

PSM3 is the fabric provider Google qualifies for H4D. It implements no connection
management, so nothing in job setup grows with the square of the process count.

## Connection scaling

Round 0 of `tools/repro-rxm-livelock.c` pays for connection setup.

| PEs/node | total PEs | round 0 |
|---:|---:|---:|
| 4 | 8 | 0.021 s |
| 16 | 32 | 0.248 s |
| 32 | 64 | 0.382 s |
| 64 | 128 | 0.759 s |

From 16 to 64 PEs per node, a 4x increase in processes and a 16x increase in
`PEs_per_node x total_PEs`, round 0 grows **3.1x**. It tracks the process count rather than
the connection count, because there are no connections to establish.

## Reproducibility

ISx64, 20 consecutive unattended runs at each shape.

| PEs/node | validated |
|---:|---|
| 64 | **20/20** |
| **192** | **20/20** |

## Density

192 PEs per node is one PE per vCPU, the full shape. It holds. 1M keys per PE, 3 runs each.

| PEs/node | total PEs | validated | time |
|---:|---:|---|---:|
| 32 | 64 | 3/3 | 0.380 s |
| 64 | 128 | 3/3 | 0.383 s |
| 96 | 192 | 3/3 | 0.708 s |
| 128 | 256 | 3/3 | 0.733 s |
| **192** | **384** | **3/3** | 2.632 s |

This changes the node count for the endpoint requirement. An endpoint is one PE, so 4,096
endpoints needs **22 nodes** at this density rather than 128 at 32 PEs per node.

## Scale reached

192 PEs per node on two nodes. Every run validated.

| total | per node | time | exchange share | rate |
|---:|---:|---:|---:|---:|
| 25.8 GB | 13 GB | 6.8 s | 88% | 3.78 GB/s |
| 103.1 GB | 52 GB | 22.4 s | 84% | 4.61 GB/s |
| 412.3 GB | 206 GB | 84.3 s | 86% | 4.89 GB/s |
| 1,000.0 GB | 500 GB | 194.0 s | 87% | 5.16 GB/s |
| 1,200.0 GB | 600 GB | 235.8 s | 88% | 5.09 GB/s |
| **1,400.0 GB** | **700 GB** | **274.5 s** | 88% | **5.10 GB/s** |

1,400 GB is 1,414 GB resident at the 2.02x footprint against about 1,440 GB usable, so it
is the ceiling for two nodes.

Rate rises to 5.1 GB/s and holds there over the top 3.4x of the range. Density is worth
**3.4x**: the same two machines ran 1.50 GB/s at 32 PEs per node.

| PEs/node | largest | rate |
|---:|---:|---:|
| 32 | 1,099.5 GB | 1.50 GB/s |
| **192** | **1,400.0 GB** | **5.10 GB/s** |

The exchange is 84-88% of runtime throughout. On 200 Gbps RoCE the network is the bound,
which is the opposite of the GPU path where bucketing dominates and the exchange is 10%.

## Three changes were needed

**1. Build libfabric with PSM3.** The provider is not in a default build. Configure with
`--enable-psm3` and install `libuuid-devel`, which PSM3 links and SOS then needs too.

**2. Patch SOS to tolerate a provider without shared transmit contexts.** PSM3 does not
implement STX. SOS binds one unconditionally and aborts at `transport_ofi.c:606` with
`fi_ep_bind STX to endpoint failed`. `SHMEM_OFI_STX_MAX` cannot turn it off; SOS overrides
the value and auto-sets it.

A shared STX lets several contexts share transmit resources. A provider that refuses one
is still usable with a private transmit context per endpoint, so the bind can degrade
rather than fail:

```c
if (ctx->stx_idx >= 0) {
    ret = fi_ep_bind(ctx->ep, &shmem_transport_ofi_stx_pool[ctx->stx_idx].stx->fid, 0);
    if (ret == -FI_EINVAL || ret == -FI_ENOSYS) {
        RAISE_WARN_STR("provider rejected shared TX context; continuing without it");
        ctx->stx_idx = -1;
        ret = 0;
    }
    OFI_CHECK_RETURN_STR(ret, "fi_ep_bind STX to endpoint failed");
}
```

Patch in `deploy/h4d/sos-psm3-stx.patch`. Worth sending upstream: it makes SOS work with
any provider lacking STX, not only PSM3.

**3. Set `PSM3_ALLOW_ROUTERS=1`.** H4D presents each RDMA NIC as its own `/32`:

```
$ ip -o -4 addr show dev enp0s9
192.168.64.5/32
```

PSM3 compares subnets before connecting and refuses a peer it considers off-subnet:

```
Trying to connect from irdma0 port 1 (subnet 192.168.64.2/32) to a node
(IP=192.168.64.5/32 QP=697) on a different subnet 192.168.64.5/32
```

`PSM3_ALLOW_ROUTERS=1` disables the check. `PSM3_SUBNETS=192.168.64.0/18` does not work;
it fails earlier in heap init.

## Configuration that works

```bash
export SHMEM_OFI_PROVIDER=psm3
export PSM3_ALLOW_ROUTERS=1
export PSM3_UUID=$(printf '%08x-0000-0000-0000-000000000000' "$SLURM_JOB_ID")
export SHMEM_SYMMETRIC_SIZE=1G
export SHMEM_BOOTSTRAP=PMI
```

The windowed exchange holds one window slot per PE, so the symmetric heap is fixed at
about 0.5 MB per PE and does not grow with the dataset. 1G is ample at 192 PEs per node;
64G is wasteful.

`oshcc` embeds an RPATH that overrides `LD_LIBRARY_PATH`. Two SOS installs on one machine
will silently load the wrong one, so confirm with `ldd` before trusting any comparison
between builds.

`oshrun` aborts with `could not find a launcher` on a bare VM, because SOS wraps whichever
launcher `configure` found and a plain image has none. `mpich` supplies `mpiexec.hydra`,
which speaks PMI-1. That is process launch only; the data path is OpenSHMEM over libfabric
with no MPI in it.

Every rank in a job must share one `PSM3_UUID`. Deriving it from `SLURM_JOB_ID` gives that
for free and keeps concurrent jobs apart.

## Capabilities

PSM3 accepts every hint SOS sets, checked one field at a time: `FI_RMA`, `FI_ATOMIC`,
both together, plus `FI_RMA_EVENT`, basic and scalable MR, and manual and auto progress.
It advertises `FI_RMA`, `FI_ATOMIC` and `FI_RMA_EVENT`, and reports
`mode: [FI_CONTEXT]` and `addr_format: FI_ADDR_PSMX3`.

```
provider: psm3
    fabric: RoCE-192.168.64.2/32
    domain: irdma0
    version: 401.10
    protocol: FI_PROTO_PSMX3
```

## What this does not change

Capacity for the petabyte. 1 PB in memory needs about 1,403 H4D nodes in one zone, which is
several times the unallocated pool of the best zone, so the full data target stays out of
reach on H4D at any provider. Density does not help, because that bound is memory rather
than endpoints. See the scale table in the root README.

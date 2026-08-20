# H4D on PSM3

2026-08-19. Two `h4d-highmem-192`, us-east1-b. libfabric 2.6.0 built with `--enable-psm3`,
Sandia OpenSHMEM with one source patch.

PSM3 is the fabric provider Google qualifies for H4D. Earlier work in this repository used
`verbs;ofi_rxm`. The two behave differently enough that every transport conclusion changes.

## Connection scaling

Round 0 of `tools/repro-rxm-livelock.c` pays for connection setup. Its growth rate is the
measurement that matters.

| PEs/node | total PEs | `verbs;ofi_rxm` | `psm3` |
|---:|---:|---:|---:|
| 4 | 8 | 0.026 s | 0.021 s |
| 16 | 32 | 0.368 s | 0.248 s |
| 32 | 64 | — | 0.382 s |
| 64 | 128 | **6.193 s** | **0.759 s** |

From 16 to 64 PEs per node, a 4x increase in processes and a 16x increase in
`PEs_per_node × total_PEs`:

- `verbs;ofi_rxm` round 0 grows **16.8x**, tracking the connection count
- `psm3` round 0 grows **3.1x**, tracking the process count

The superlinear term is gone. PSM3 implements no connection management, so there is no
per-peer state to establish and nothing that grows with the square of the job.

## Reproducibility

ISx64, 64 PEs per node, 8,388,608 keys per PE, 20 consecutive unattended runs.

| provider | validated |
|---|---|
| `verbs;ofi_rxm` | 7/20 and 9/20 |
| **`psm3`** | **20/20** |

## Scale reached

32 PEs per node on two nodes, `SHMEM_SYMMETRIC_SIZE=64G`. Every run validated.

| total | per node | time | exchange share | rate |
|---:|---:|---:|---:|---:|
| 68.7 GB | 34 GB | 44.7 s | 91% | 1.54 GB/s |
| 274.9 GB | 137 GB | 178.3 s | 91% | 1.54 GB/s |
| 549.8 GB | 275 GB | 364.2 s | 89% | 1.51 GB/s |
| **1,099.5 GB** | **550 GB** | **730.7 s** | 89% | **1.50 GB/s** |

Rate is flat across a 16x data range, so the exchange is bandwidth-limited rather than
degrading with size. 550 GB of keys per node is 1,111 GB resident at the 2.02x footprint,
77% of the 1,440 GB usable.

The largest run on `verbs;ofi_rxm` was 8.59 GB. This is **128x** that, on the same two
machines.

The exchange is 89% of runtime throughout. On 200 Gbps RoCE the network is the bound,
which is the opposite of the GPU path where bucketing dominates and the exchange is 8%.

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
export SHMEM_SYMMETRIC_SIZE=8G
```

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

Capacity. 1 PB in memory needs about 1,403 H4D nodes in one zone, which is several times
the unallocated pool of the best zone, so the full target stays out of reach on H4D at any
provider. See the scale table in the root README.

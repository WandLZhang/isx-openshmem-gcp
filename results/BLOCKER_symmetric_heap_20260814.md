# Dominant blocker: the OpenSHMEM symmetric heap will not grow past ~2 GB per PE

Measured 2026-08-14 on two `h4d-highmem-192-lssd` in `us-east1-b`, 8 PEs per node,
libfabric 2.6.0, Sandia OpenSHMEM 1.5.3, `verbs;ofi_rxm` on `irdma0`.

This is the constraint that decides whether a petabyte in-memory sort is possible on H4D.
It is not.

## The measurement

A trivial program: `shmem_init()`, one cross-node `shmem_longlong_put`, `shmem_finalize()`.
Nothing varies except `SHMEM_SYMMETRIC_SIZE`.

| heap per PE | heap per node (8 PEs) | init + run | result |
|---|---|---:|---|
| 512 MB | 4 GB | 3 s | **PASS** 16/16 |
| 2 GB | 16 GB | 1 s | **PASS** 16/16 |
| 8 GB | 64 GB | 180 s | **HANG** 0/16 |
| 16 GB | 128 GB | 180 s | **HANG** 0/16 |

The ceiling sits between 2 GB and 8 GB per PE. Past it the program does not error, it
stops. Sampled during a hang, the process sits at **0.1% CPU** with the allocation
resident: it is blocked, not working.

The node has 1,488 GB. At the point of failure only 128 GB of it is being asked for, so
this is not a memory-capacity limit.

## Why

The provider rejects scalable memory registration (`mr_mode = 0`), which forces SOS to
build with `--enable-ofi-mr=basic`, meaning `FI_MR_ALLOCATED`. Under that mode the
**entire symmetric heap is pinned and registered with the NIC at `shmem_init()`**, before
any data moves. Registering tens of gigabytes per node through `irdma` does not complete
in any reasonable time.

The two constraints compound. The provider's refusal of scalable MR is what makes the
heap size a hard limit rather than a tuning knob.

## What it costs at petabyte scale

ISx needs roughly 1.2x its per-PE key array resident in the symmetric heap, because that
is the receive buffer peers write into.

At a working ceiling of 2 GB per PE:

```
keys per PE      = 2 GB / 1.2 / 8 B   =~ 2.2e8 keys
1 PB             = 1.25e14 keys
PEs required     = 1.25e14 / 2.2e8    =~ 570,000 PEs
```

And from `isx64_h4d_scaling_20260814.md`, a second independent wall caps usable PEs at
**32 per node**. Together:

```
nodes required   = 570,000 / 32       =~ 17,800 h4d-highmem nodes
```

Against a available H4D fleet machines, and a the largest single-zone pool. The
requirement is short by roughly **20x the entire fleet**, and by **60x** the largest zone
in which Cloud RDMA can exist at all.

For the 4,096-endpoint target alone, ignoring capacity: 4,096 PEs at 32 per node is 128
nodes, and 4,096 x 2.2e8 keys is about **7 TB**, not 1 PB.

## The two requirements are in direct conflict on this fabric

- Reaching **1 PB in memory** wants few PEs each holding a lot. The heap ceiling forbids
  it: no PE may hold more than ~2 GB of symmetric memory.
- Reaching **4,096 endpoints** wants many PEs per node. The retry wall forbids it: no node
  may run more than 32 PEs.

Each constraint pushes against the other, and neither is a tuning parameter. This is the
core Failure Analysis result for the H4D path.

## What was ruled out

- **Put size.** The initial hypothesis was that a ~500 MB single `shmem_uint64_put` was
  stalling. Disproved: with the heap raised to 24 GB, even a 2 MB put hung, and the same
  2 MB put had completed in 0.088 s with a 512 MB heap.
- **Node memory.** 128 GB requested against 1,488 GB available.
- **Shared transmit contexts.** `SHMEM_OFI_STX_AUTO=1` and `SHMEM_OFI_STX_MAX=8` change
  nothing at the PEs-per-node wall.

## Untested mitigations

Worth trying before declaring the path closed, in rough order of promise.

1. **Huge pages.** `SHMEM_SYMMETRIC_HEAP_USE_HUGE_PAGES=1`. Registration cost scales with
   page count, so 2 MB pages could cut it by ~512x. This is the single most likely fix and
   was not reached before the cluster ran out of time.
2. **On-demand paging.** If `irdma` supports `FI_MR_ALLOCATED` with ODP, registration need
   not pin up front.
3. **Fewer, larger PEs.** Fewer PEs per node means a larger per-PE heap within the same
   per-node registration budget. It trades directly against the endpoint count.
4. **Keep bulk data outside the symmetric heap.** Only the receive buffer must be
   symmetric. A design that streams through a small symmetric window, rather than sizing
   the window to the whole dataset, would decouple dataset size from heap size entirely.
   This is the most invasive change and the most likely to actually reach a petabyte.

Option 4 is the honest recommendation for a redesign. It also moves the benchmark further
from stock ISx, which is a tradeoff the customer should decide rather than us.

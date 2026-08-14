# Dominant blocker: aggregate symmetric heap caps at ~32 GB per node

Measured 2026-08-14 on two `h4d-highmem-192-lssd` in `us-east1-b`. libfabric 2.6.0,
Sandia OpenSHMEM 1.5.3, `verbs;ofi_rxm` on `irdma0`. Nodes have 1,464 GB usable and Slurm
reports `RealMemory=1478672` with `DefMemPerNode=UNLIMITED`, so nothing here is a Slurm
memory cap.

This is the constraint that decides whether a petabyte in-memory sort is possible on H4D.

## The measurement

A trivial program: `shmem_init()`, one cross-node `shmem_longlong_put`, `shmem_finalize()`.
Only the heap geometry varies.

| PEs/node | heap per PE | heap per node | time | result |
|---:|---:|---:|---:|---|
| 1 | 32 GB | **32 GB** | 4 s | PASS |
| 2 | 16 GB | **32 GB** | 1 s | PASS |
| 8 | 4 GB | **32 GB** | 2 s | PASS |
| 4 | 8 GB | **32 GB** | 150 s | FAIL (anomaly, see below) |
| 1 | 64 GB | 64 GB | 3 s | FAIL |
| 2 | 32 GB | 64 GB | 150 s | FAIL |
| 4 | 16 GB | 64 GB | 150 s | FAIL |
| 8 | 8 GB | 64 GB | 150 s | FAIL |

**The limit is aggregate per node, not per PE.** A single PE can hold a 32 GB heap. Eight
PEs can hold 4 GB each. Both are 32 GB per node and both work. Every 64 GB-per-node
configuration fails regardless of how it is divided.

The 4 x 8 GB row breaks the pattern and is most likely the same intermittent failure seen
elsewhere in this study (8 PEs passed standalone and failed inside a sweep). It should be
re-run before the boundary is quoted as exactly 32 GB.

Two distinct failure modes appear. 64 GB on one PE fails fast, in 3 s, which looks like a
refused allocation. The multi-PE 64 GB cases hang until the 150 s timeout, which matches
the registration stall described below.

## Correction to the earlier version of this document

This file previously reported the ceiling as **~2 GB per PE**. That was wrong. Every test
behind it was run at 8 PEs per node, so per-PE and per-node size were confounded: 8 x 2 GB
passed and 8 x 8 GB failed, and the per-PE number was the one that got written down. The
1-PE and 2-PE rows above, which separate the two variables, were not run until later.

The corrected ceiling is roughly 16x more generous than reported, and the conclusion for
the petabyte target still holds, but the arithmetic changes materially.

## Why

The provider rejects scalable memory registration (`mr_mode = 0`), forcing
`--enable-ofi-mr=basic`, which means `FI_MR_ALLOCATED`: the whole symmetric heap is pinned
and registered with the NIC during `shmem_init()`, before any data moves. There appears to
be a per-node registration budget somewhere between 32 and 64 GB.

Huge pages do not lift it. With 49,152 x 2 MB pages (96 GB) actually reserved on each
node and `SHMEM_SYMMETRIC_HEAP_USE_HUGE_PAGES=1`, 8 x 8 GB still failed, this time with an
`oom_kill`. An earlier huge-page run was vacuous because `HugePages_Total` was 0 and SOS
silently fell back to 4 KB pages, which is worth knowing: **the huge-page setting fails
silently when no pages are reserved.**

## What it costs at petabyte scale

ISx needs about 1.2x its per-PE key array resident in the symmetric heap, because that is
the buffer peers write into.

```
symmetric heap per node   = 32 GB
keys per node             = 32 GB / (1.2 x 8 B)  =~ 3.33e9 keys  (26.7 GB of keys)
1 PB                      = 1.25e14 keys
nodes required            = 1.25e14 / 3.33e9     =~ 37,500 nodes
```

Against a far beyond any single-zone H4D pool. Cloud
RDMA cannot span zones, so the relevant comparison is the zone: short by roughly **43x**.

The endpoint target lands in a different place. From
`isx64_h4d_scaling_20260814.md`, usable PEs cap at 32 per node, so 4,096 endpoints needs
**128 nodes** — which is obtainable. But 128 nodes x 26.7 GB is about **3.4 TB**, not 1 PB.

## The two requirements pull in opposite directions

- **1 PB in memory** wants a large symmetric heap. Capped at 32 GB per node.
- **4,096 endpoints** wants many PEs per node. Capped at 32 PEs per node.

Satisfying both simultaneously on this fabric is not a matter of tuning. At the measured
limits, the largest configuration that satisfies the endpoint requirement sorts about
3.4 TB, which is 0.3% of the target.

## Untested mitigations, in order of promise

1. **Keep bulk data outside the symmetric heap.** Only the receive buffer must be
   symmetric. A design that streams the exchange through a small fixed symmetric window,
   rather than sizing the window to the whole dataset, decouples dataset size from the
   heap ceiling entirely. This is the only option that plausibly reaches a petabyte, and
   it is a substantial departure from stock ISx.
2. **On-demand paging.** If `irdma` supports ODP, registration need not pin up front and
   the per-node budget may not apply.
3. **1 GB huge pages** rather than 2 MB, reducing page count another 512x. The 2 MB test
   hit `oom_kill` rather than a registration stall, so the mechanism may differ.
4. **Re-run the 4 x 8 GB anomaly** to confirm the boundary is 32 GB and not lower.

Option 1 is the honest recommendation. It changes what is being benchmarked, so it is the
customer's decision rather than ours.

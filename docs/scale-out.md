# Scale-out arithmetic

Node counts for both targets, derived from measured per-node numbers rather than
estimates. Whoever gets the capacity should not have to re-derive any of this.

## Inputs, all measured

| quantity | value | source |
|---|---|---|
| Peak resident / key array | 2.02x | allocations in `src/cpu/isx64_win.c` |
| H4D usable memory per node | 1,440 GB | 1,464 GB total, less OS and symmetric heap |
| GB300 memory per node | 2,076 GB | 1,116 GB HBM + 960 GB Grace, coherent |
| GB200 memory per node | 1,628 GB | 744 GB HBM + 884 GB Grace |
| H4D aggregate rate | 1.51 GB/s on 2 nodes | flat across a 64x data range |
| GPU aggregate rate | 12.61 GB/s on 2 A100 | flat across a 16x data range |

An endpoint is one OpenSHMEM PE on H4D and one GPU on GB200 and GB300.

## Node counts

Keys per node is `usable / 2.02`. Nodes is the larger of the memory requirement and the
endpoint requirement.

| machine | keys/node | 1 PB | 100 TB | endpoints at the 1 PB count |
|---|---:|---:|---:|---:|
| `h4d-highmem-192` | 713 GB | 1,403 | 140 | 44,896 PE |
| `a4x-maxgpu-4g` (GB300) | 1,027 GB | 1,024 | 102 | 4,096 GPU |
| `a4x-highgpu-4g` (GB200) | 806 GB | 1,241 | 124 | 4,964 GPU |

GB300 at 1 PB is bound by the endpoint requirement rather than by memory: 1,024 nodes
gives 4,096 GPUs and 2.13 PB against the 2.02 PB needed.

## Against available capacity

"Free" is the largest single-zone unallocated pool.

| machine | free | 1 PB | 100 TB |
|---|---:|---|---|
| H4D | 562 | **impossible**, needs 2.5x what exists free | fits, 25% of pool |
| GB300 | 6,814 | fits, 15% of pool | fits, 1.5% |
| GB200 | 1,082 | short 1.1x | fits, 11% |

**H4D cannot reach 1 PB under any capacity grant.** 1,403 nodes in one zone exceeds the
unallocated pool of every zone worldwide, and a grant does not create machines that are
already allocated. The largest H4D run on record is 192 nodes.

Free-pool size is not obtainability. H4D returned `ZONE_RESOURCE_POOL_EXHAUSTED` in a zone
the supply data showed as mostly free, so confirm the machines separately from the quota.

## Run time

At the measured 1.51 GB/s per two H4D nodes, 100 TB across 140 nodes projects to roughly
16 minutes. The projection is linear in node count and unvalidated above two nodes; the
all-to-all gets harder as endpoints grow, so treat it as a floor.

The run being minutes rather than hours decides three things. Checkpointing is not worth
building, because writing 1 PB takes longer than sorting it. Retry is the right failure
strategy. And the bill is dominated by how long machines are held rather than by the sort.

## H4D configuration

```bash
export SHMEM_OFI_PROVIDER=psm3
export PSM3_ALLOW_ROUTERS=1
export PSM3_UUID=$(printf '%08x-0000-0000-0000-000000000000' "$SLURM_JOB_ID")
export SHMEM_SYMMETRIC_SIZE=64G

srun -N140 --ntasks-per-node=32 --mpi=pmi2 --export=ALL \
     ./isx64win 22321428571 1 results/run
```

`keys_per_pe` is `total_keys / (nodes * 32)`. 100 TB is 1.25e13 keys, so 140 nodes at 32
PEs gives 2.79e9 per PE.

SOS needs `deploy/h4d/sos-psm3-stx.patch` before it will start on PSM3. The blueprint
node count is `h4d_cluster_size` in `deploy/h4d/isx-slurm-h4d.yaml`, and Cloud RDMA cannot
cross zones.

## Memory work, if a node count needs reducing

Peak is 2.02x today, already down from 2.30x. Two changes remain, both local to
`src/cpu/isx64_win.c`:

1. **Streamed exchange.** Release `send` incrementally as the exchange drains it while
   `recv` fills, so the two never both hold a full copy. Takes 2.02x to about 1.15x, which
   would put 1 PB on H4D at 799 nodes. Still beyond the free pool.
2. **Single-slot symmetric window.** `src/cpu/isx64_stream.c` already does this: because
   the exchange rotates destinations, every PE receives from exactly one sender per step,
   so the window is `WINDOW * 8` bytes rather than `NUM_PES * WINDOW * 8`. At 25,600 PEs
   that is 131 KB instead of 3.36 GB per PE. See `docs/streamed-exchange.md`.

Neither changes the conclusion for H4D at 1 PB.

## What must be re-measured at real node counts

Everything here above two nodes is arithmetic. In order:

1. **32 PEs/node on 4 nodes**, which tests whether anything is specific to a node pair.
2. **The scaling curve at 8, 16, 32 nodes**, holding keys per node constant. That gives
   the weak-scaling slope the projection assumes.
3. **A 20-run reproducibility set** at the largest stable shape.
4. **Then the target run.**

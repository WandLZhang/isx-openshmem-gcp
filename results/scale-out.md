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
| H4D aggregate rate | 5.10 GB/s on 2 nodes | 192 PEs/node, flat over the top 3.4x |
| GPU aggregate rate | 67.15 GB/s on 8 H200 | flat across a 64x data range |

An endpoint is one OpenSHMEM PE on H4D and one GPU on GB200 and GB300. H4D runs 192 PEs
per node, one per vCPU, validated 20/20.

## Node counts

Keys per node is `usable / 2.02`. Nodes is the larger of the memory requirement and the
endpoint requirement. A4X capacity is allocated in fixed 18-node NVLink domains, so both
A4X rows round up to a multiple of 18.

| machine | keys/node | 1 PB | 100 TB | endpoints at the 1 PB count |
|---|---:|---:|---:|---:|
| `h4d-highmem-192` | 713 GB | 1,403 | 140 | 269,376 PE |
| `a4x-maxgpu-4g` (GB300) | 1,027 GB | 1,026 | 108 | 4,104 GPU |
| `a4x-highgpu-4g` (GB200) | 806 GB | 1,242 | 126 | 4,968 GPU |

GB300 at 1 PB is bound by the endpoint requirement rather than by memory: 1,026 nodes is
57 NVL72 domains, 4,104 GPUs, and 2.13 PB against the 2.02 PB needed.

## Topology

Cluster Director exposes three levels: a sub-block is one rack behind a single top-of-rack
switch, one hop; a block is sub-blocks on non-blocking fabric, at most two hops; a cluster
is blocks. It publishes that hierarchy to Slurm and GKE, and a reservation is what makes it
visible.

Request placement with the capacity, not after.

| machine | mechanism | limit |
|---|---|---:|
| A4X, A4X Max | group placement, `--collocation=collocated --gpu-topology=1x72` | one NVL72 domain per policy |
| A3 Ultra | workload policy, `maxTopologyDistance=1` (sub-block) | 22 instances |
| A3 Ultra | workload policy, `maxTopologyDistance=2` (block) | 256 instances |
| H4D | Compute Engine packs instances to minimise hops | — |

This matters for the GB300 exchange. NVLink spans 72 GPUs, so at 4,104 endpoints 56/57 of
the all-to-all leaves its domain. Domain-aware bucketing needs the topology, and Cluster
Director is where it comes from.

## Against available capacity

Cloud RDMA and NVLink are both intra-zone, so the whole job lands in one zone and the
comparison is against that zone's unallocated pool.

| machine | 1 PB | 100 TB |
|---|---|---|
| H4D | **impossible**, several times the best zone's free pool | fits |
| GB300 | fits, comfortably | fits |
| GB200 | just short | fits |

**H4D cannot reach 1 PB under any capacity grant.** 1,403 nodes in one zone exceeds the
unallocated pool of every zone worldwide, and a grant does not create machines that are
already allocated. The largest H4D run on record is 192 nodes.

Free-pool size is not obtainability. H4D returned `ZONE_RESOURCE_POOL_EXHAUSTED` in a zone
the supply data showed as mostly free, so confirm the machines separately from the quota.

## Endpoints on their own

4,096 endpoints does not need the data target. At 192 PEs per node it needs **22 nodes**,
which also holds 15.7 TB. 22 nodes is 4,224 vCPU, a far smaller quota ask than the 26,880
the 100 TB memory target needs.

| goal | nodes | vCPU | endpoints | data held |
|---|---:|---:|---:|---:|
| 4,096 endpoints | 22 | 4,224 | 4,224 | 15.7 TB |
| 100 TB | 140 | 26,880 | 26,880 | 100 TB |

## Run time

At the measured 5.10 GB/s per two H4D nodes, 100 TB across 140 nodes projects to roughly
5 minutes. The projection is linear in node count and unvalidated above two nodes; the
all-to-all gets harder as endpoints grow, so treat it as a floor.

The run being minutes rather than hours decides three things. Checkpointing is not worth
building, because writing 1 PB takes longer than sorting it. Retry is the right failure
strategy. And the bill is dominated by how long machines are held rather than by the sort.

## H4D configuration

```bash
export SHMEM_OFI_PROVIDER=psm3
export PSM3_ALLOW_ROUTERS=1
export PSM3_UUID=$(printf '%08x-0000-0000-0000-000000000000' "$SLURM_JOB_ID")
export SHMEM_SYMMETRIC_SIZE=1G
export SHMEM_BOOTSTRAP=PMI

srun -N140 --ntasks-per-node=192 --mpi=pmi2 --export=ALL \
     ./isx64win 465029017 1 results/run
```

`keys_per_pe` is `total_keys / (nodes * 192)`. 100 TB is 1.25e13 keys, so 140 nodes at 192
PEs gives 4.65e8 per PE.

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
   the exchange rotates destinations, every PE receives from one sender per step,
   so the window is `WINDOW * 8` bytes rather than `NUM_PES * WINDOW * 8`. At 25,600 PEs
   that is 131 KB instead of 3.36 GB per PE. See `results/windowed-exchange.md`.

Neither changes the conclusion for H4D at 1 PB.

## What must be re-measured at real node counts

Everything here above two nodes is arithmetic. In order:

1. **192 PEs/node on 4 nodes**, which tests whether anything is specific to a node pair.
2. **The scaling curve at 8, 16, 32 nodes**, holding keys per node constant. That gives
   the weak-scaling slope the projection assumes.
3. **A 20-run reproducibility set** at the largest stable shape.
4. **Then the target run.**

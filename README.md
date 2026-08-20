# ISx64: 64-bit integer sort over OpenSHMEM on Google Cloud

A port of [ISx](https://github.com/ParRes/ISx), the Sandia integer sort, to 64-bit keys,
with implementations for CPU, GPU and TPU, provisioning recipes, and measurements.

ISx is a distributed bucket sort. Each process generates its own keys, sends every key to
the process that owns its value range, sorts what arrives, and checks the boundary against
its neighbour. It measures interconnect behaviour under an irregular all-to-all rather
than compute.

The study target is 1 PB of `uint64` keys across 4,096 or more endpoints, sorted in
memory, using one-sided RMA. MPI is out of scope.

## Results

| | CPU, H4D | GPU, H200 | TPU, v6e |
|---|---|---|---|
| Model | OpenSHMEM over PSM3 | NVSHMEM over NVLink | `jax.lax.all_to_all` |
| Largest validated | **1,400 GB**, 384 PEs, 2 nodes | **137.4 GB**, 8 GPUs, 1 node | **12.9 GB**, 4 chips |
| Reproducibility | **20/20** at 192 PEs/node | every run | every run, within 0.1% |
| Aggregate rate | 5.10 GB/s | 67.15 GB/s | 0.25 GB/s |
| Flat across | top 3.4x of the range | 64x data range | 6x data range |
| Dominant phase | exchange, 89% | bucket, 82% | bucket, 97% |

All three validate and weak-scale. H4D is interconnect-bound on 200 Gbps RoCE. The GPU and
TPU paths are bound by bucketing, with their exchanges at 10% and 0.1%.

TPU is not responsive to the OpenSHMEM requirement, since there is no remote put. It
answers one thing the others cannot: the ICI exchange runs at **200 GB/s** and is 0.1% of
the step, so a TPU sort number that puts one timer around "exchange and bucket sorting" is
reporting bucket time. Replacing the bucket scatter with a gather made it 9.0x faster.
See [results/tpu.md](results/tpu.md).

**Multi-node NVSHMEM does not work on Google Cloud RoCE with the packaged build.** The
fabric is fine: `ib_write_bw` moves 45 GB/s host to host on one of eight NICs. GPU memory
registration is the blocker. `ibv_reg_mr` on a device pointer fails with errno 14 because
`nvidia_peermem` cannot insert against the inbox `ib_core`, while `ibv_reg_dmabuf_mr` on
the same buffer succeeds and the packaged NVSHMEM has no dmabuf setting to reach it.
Root cause, evidence and three fixes in [results/gpu-nvshmem.md](results/gpu-nvshmem.md).

### The fabric provider decides everything on H4D

Earlier work here used `verbs;ofi_rxm`. PSM3 is the provider Google qualifies for H4D, and
switching removes the failure that bounded the study.

| | `verbs;ofi_rxm` | `psm3` |
|---|---|---|
| Round-0 growth, 16 → 64 PEs/node | 16.8x | **3.1x** |
| Validated runs at 64 PEs/node | 7/20, 9/20 | **20/20** |
| Highest working density | 32 PEs/node | **192 PEs/node**, 20/20 |
| Largest validated dataset | 8.59 GB | **1,400 GB** |
| Aggregate rate | 1.50 GB/s | **5.10 GB/s** |

`ofi_rxm` opens a connection per peer, so round-0 cost grows with
`PEs_per_node × total_PEs` and stops making progress near 8,192 connections per node.
PSM3 implements no connection management, so nothing grows with the square of the job.
The three changes required to run SOS on PSM3 are in [results/h4d-psm3.md](results/h4d-psm3.md).

## Can the criteria be reached

| | H4D, OpenSHMEM/PSM3 | TPU v6e, JAX | GB300, NVSHMEM |
|---|---|---|---|
| Correctness | met, 1,400 GB | met, 12.9 GB | met on 8 x H200, 1 node |
| Reproducibility | met, 20/20 | met, within 0.1% | not run multi-node |
| Performance stability | met, 5.10 GB/s flat | met, 0.25 GB/s flat | met on 1 node |
| Scale, >1 PB | **no, supply** | **no, architecture** | **yes**, 1,026 nodes |
| Scale, 4,096 endpoints | **yes**, 22 nodes | **no on v6e**, yes on TPU7x | **yes**, 4,104 GPUs |
| OpenSHMEM one-sided PGAS | met | **no, architecture** | qualifies, blocked across nodes |
| Connectionless fabric | met, PSM3 | n/a | unproven |
| Per-packet adaptive routing | **no, by design** | n/a, ICI is a static torus | unknown, ConnectX-8 untested |
| Operational plausibility | met | met | package shipped, one blocker |

Three different reasons sit behind the four "no" cells.

**Supply.** H4D at 1 PB needs 1,403 nodes in one zone, because Cloud RDMA cannot cross
zones, and that is more than any zone holds unallocated. The machines would work if they
existed. The largest H4D run on record is 192 nodes.

**Architecture.** TPU has no PGAS and no remote put, so the one-sided requirement cannot be
met. A job cannot span slices over ICI either, so the largest slice that exists, TPU7x at
9,216 chips, holds 1.77 PB against the 2.02 PB the sort needs resident. No grant changes
either fact.

**Design.** Falcon uses multipath subflows rather than per-packet spraying, because
per-packet routing reorders packets and RoCE treats reordering as loss. Zero out-of-order
arrivals across every run. See [results/adaptive-routing.md](results/adaptive-routing.md).

**A blocker, not a wall.** GB300 reaches the full target at 1,026 nodes. Two things sit in
front of it: `nvidia-gb300` is not allowlisted on this project, and multi-node NVSHMEM does
not work on Cloud RoCE. `ibdevx` already registers GPU memory through dmabuf and then
segfaults on the first cross-node put.

### Node counts

Peak resident memory is 2.02x the key array, measured. Usable memory per node is the total
less about 48 GB for OS and symmetric heap. Verdicts compare the node count against the
unallocated pool of the best single zone.

| machine | mem/node | endpoints/node | 1 PB + 4,096 ep | 100 TB |
|---|---:|---:|---|---|
| `h4d-highmem-192` | 1,440 GB | 192 PE | 1,403 nodes, impossible | 140 nodes, fits |
| `a4x-maxgpu-4g` (GB300) | 2,076 GB | 4 GPU | **1,026 nodes, fits** | 108 nodes, fits |
| `a4x-highgpu-4g` (GB200) | 1,628 GB | 4 GPU | 1,242 nodes, just short | 126 nodes, fits |

A4X capacity comes in fixed 18-node NVLink domains, so both A4X rows round up to a multiple
of 18. 1,026 nodes is 57 domains and 4,104 GPU endpoints. A4X Max refuses Spot, so it needs
a reservation, which is also what makes Cluster Director topology visible.

**4,096 endpoints on H4D costs 22 nodes.** 192 PEs per node validated 20/20, and 22 nodes
still hold 15.7 TB. The 100 TB run at 140 nodes gives 26,880 endpoints.

Free-pool size is not obtainability. H4D returned `ZONE_RESOURCE_POOL_EXHAUSTED` in a zone
the supply data showed as mostly free.

Steps for a team with capacity are in [docs/handoff.md](docs/handoff.md).

## Running it

### Locally, no cluster

`tests/shmem_stub.h` implements the OpenSHMEM calls ISx64 uses for a single PE.

```bash
gcc -O2 -std=c11 -Wall -Wextra -DNDEBUG -I tests -I src/cpu \
    -o bin/isx64_stub src/cpu/isx64.c src/cpu/pcg_basic.c -lm
./bin/isx64_stub 4000000 2 /tmp/isx64.log        # expect: verification : PASSED
```

At one PE every put is a local `memcpy`, so this exercises the sort and nothing about a
fabric.

### On H4D

`deploy/h4d/` provisions a Slurm cluster with Cloud RDMA and builds the runtime.
`01_build_sos.sh` records the steps the vendor documentation omits and applies
`sos-psm3-stx.patch`, without which SOS will not start on PSM3.

```bash
bash deploy/h4d/00_setup_project.sh          # org policy, quota, APIs
bash deploy/h4d/01_build_sos.sh              # libfabric + SOS, on a compute node
bash deploy/h4d/run_isx64.sh <nodes> <pes_per_node> <keys_per_pe>
```

`run_isx64.sh` pre-flights the fabric before spending the allocation, then retries. A
failed attempt otherwise costs 237 seconds waiting out the retry budget.

### On GB300

`deploy/gb300/` is self-contained for a capacity team: machine shape, both target sizes
with node counts and keys-per-GPU worked out, the capacity ask, and the build
dependencies that are easy to miss.

```bash
bash deploy/gb300/build.sh                   # ARCH=sm_100 by default
bash deploy/gb300/run.sh smoke               # one node, 4 GPUs, seconds
bash deploy/gb300/run.sh 10                  # 108 nodes, 100 TB
bash deploy/gb300/run.sh full                # 1026 nodes, 1 PB
```

## Layout

```
src/cpu/          OpenSHMEM implementation, three exchange schedules
src/gpu/          NVSHMEM implementation
src/tpu/          JAX implementation, plus a phase-split benchmark
deploy/h4d/       provision, build and run on H4D
deploy/gb300/     handoff package for a GB300 allocation
docs/             architecture, porting notes, scale-out arithmetic
results/          measurements, one file per topic; raw/ holds json and logs
tools/            probes, a standalone reproducer, and the dmabuf registration test
tests/            single-PE shim for local correctness
```

## Detail

| topic | file |
|---|---|
| PSM3, and what it takes to run SOS on it | `results/h4d-psm3.md` |
| The `ofi_rxm` connection limit | `results/rxm-connection-limit.md` |
| GPU on H200 and A100, and the multi-node blocker | `results/gpu-nvshmem.md` |
| Adaptive routing evidence | `results/adaptive-routing.md` |
| TPU: ICI at 200 GB/s, and a 9x bucket fix | `results/tpu.md` |
| Operational readiness, six tests | `results/operations.md` |
| **What a team with capacity does next** | `docs/handoff.md` |
| Scaling to a petabyte | `docs/scale-out.md` |
| Porting ISx to 64 bits | `docs/porting.md` |
| Requirements verbatim, with status | `GOAL.md` |

Upstream issues filed: [ofiwg/libfabric#12673](https://github.com/ofiwg/libfabric/issues/12673),
[Sandia-OpenSHMEM/SOS#1239](https://github.com/Sandia-OpenSHMEM/SOS/issues/1239).

## Attribution

ISx is by Ulf Hanebutte and Jacob Hemstad, Copyright (c) 2015 Intel Corporation, BSD
3-clause. See [LICENSE-ISx](LICENSE-ISx). Upstream:
[github.com/ParRes/ISx](https://github.com/ParRes/ISx). Paper: "ISx, a Scalable Integer
Sort for Co-design in the Exascale Era", PGAS 2015.

Changes here are marked `ISX64` in the source.

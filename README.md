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

| | CPU, H4D | GPU, A100 | TPU, v6e |
|---|---|---|---|
| Model | OpenSHMEM over PSM3 | NVSHMEM over NVLink | `jax.lax.all_to_all` |
| Largest validated | **549.8 GB**, 64 PEs | 8.59 GB, 2 GPUs | not run, see below |
| Reproducibility | **20/20** at 64 PEs/node | 3/3 | — |
| Aggregate rate | 1.51 GB/s | 12.61 GB/s | — |
| Dominant phase | exchange, 89% | bucket, 80% | — |

Both paths validate and weak-scale cleanly: H4D holds 1.51 GB/s across a 64x data range,
the GPU 12.6 GB/s across 16x. H4D is interconnect-bound on 200 Gbps RoCE. The GPU path is
bound by bucketing, with the exchange at 8% over NVLink.

The TPU implementation is written and its key generation corrected, but no slice was
granted. 141 combinations of type, size and zone were tried across spot and on-demand.

### The fabric provider decides everything on H4D

Earlier work here used `verbs;ofi_rxm`. PSM3 is the provider Google qualifies for H4D, and
switching removes the failure that bounded the study.

| | `verbs;ofi_rxm` | `psm3` |
|---|---|---|
| Round-0 growth, 16 → 64 PEs/node | 16.8x | **3.1x** |
| Validated runs at 64 PEs/node | 7/20, 9/20 | **20/20** |
| Largest validated dataset | 8.59 GB | **549.8 GB** |

`ofi_rxm` opens a connection per peer, so round-0 cost grows with
`PEs_per_node × total_PEs` and stops making progress near 8,192 connections per node.
PSM3 implements no connection management, so nothing grows with the square of the job.
The three changes required to run SOS on PSM3 are in [results/h4d-psm3.md](results/h4d-psm3.md).

## Can the target be reached

Peak resident memory is 2.02x the key array, measured. Usable memory per node is the total
less about 48 GB for OS and symmetric heap. "Free" is the largest single-zone unallocated
pool.

| machine | mem/node | endpoints/node | 100%: 1 PB + 4,096 ep | 10%: 100 TB + 410 ep |
|---|---:|---:|---|---|
| `h4d-highmem-192` | 1,440 GB | 32 PE | **1,403 nodes vs 562 free — impossible** | 140 nodes — fits |
| `a4x-maxgpu-4g` (GB300) | 2,076 GB | 4 GPU | **1,024 nodes vs 6,814 free — fits** | 102 nodes — fits |
| `a4x-highgpu-4g` (GB200) | 1,628 GB | 4 GPU | 1,241 nodes vs 1,082 free — short 1.1x | 124 nodes — fits |

**The full target is physically unreachable on H4D.** It needs 1,403 nodes in one zone and
no zone holds that many unallocated anywhere. This is not a quota question, because a
grant cannot produce machines that are already allocated. The largest H4D run on record is
192 nodes.

**GB300 is the only family where the full target fits**, at 1,024 nodes, about 15% of the
free pool in the largest zone. GB200 misses by 17%.

**All three reach 10%.** On H4D, 100 TB needs 140 nodes for memory, and 140 nodes at 32 PEs
per node gives 4,480 endpoints, so a 10% data run on H4D clears the *full* endpoint
requirement.

One caveat throughout: free-pool size is not obtainability. H4D returned
`ZONE_RESOURCE_POOL_EXHAUSTED` in a zone the supply data showed as mostly free.

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
bash deploy/gb300/run.sh 10                  # 102 nodes, 100 TB
bash deploy/gb300/run.sh full                # 1024 nodes, 1 PB
```

## What is not met

**Per-packet adaptive routing.** Falcon uses multipath subflows rather than per-packet
spraying, by design: per-packet routing reorders packets and RoCE treats reordering as
loss. Zero out-of-order arrivals were measured across every run. See
[results/adaptive-routing.md](results/adaptive-routing.md).

**Connectionless fabric semantics.** PSM3 has no connection management, which meets this
in substance. `verbs;ofi_rxd`, the provider that advertises it explicitly, cannot complete
SOS startup.

**Scale.** 549.8 GB validated against 1 PB. Reachable on GB300, not on H4D.

## Layout

```
src/cpu/          OpenSHMEM implementation, three exchange schedules
src/gpu/          NVSHMEM implementation
src/tpu/          JAX implementation
deploy/h4d/       provision, build and run on H4D
deploy/gb300/     handoff package for a GB300 allocation
docs/             architecture, porting notes, scale-out arithmetic
results/          measurements, one file per topic; raw/ holds json and logs
tools/            probes and a standalone reproducer
tests/            single-PE shim for local correctness
```

## Detail

| topic | file |
|---|---|
| PSM3, and what it takes to run SOS on it | `results/h4d-psm3.md` |
| The `ofi_rxm` connection limit | `results/rxm-connection-limit.md` |
| GPU validation on A100 | `results/gpu-nvshmem.md` |
| Adaptive routing evidence | `results/adaptive-routing.md` |
| Operational readiness, six tests | `results/operations.md` |
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

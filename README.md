# ISx64: 64-bit integer sort over OpenSHMEM on Google Cloud

A port of [ISx](https://github.com/ParRes/ISx), the Sandia integer sort, to 64-bit keys,
with implementations for CPU, GPU and TPU, provisioning recipes, and measurements.

ISx is a distributed bucket sort. Each process generates its own keys, sends every key to
the process that owns its value range, sorts what arrives, and checks the boundary against
its neighbour. It measures interconnect behaviour under an irregular all-to-all rather
than compute.

The study target is 1 PB of `uint64` keys across 4,096 or more endpoints, sorted in
memory, using one-sided RMA. MPI is out of scope.

Everything below is measured. `results/` holds the raw records and `GOAL.md` has the
customer's requirements verbatim.

**Contents.** [Results](#results) · [What was measured](#what-was-measured) ·
[How it was built](#how-it-was-built) · [Running it](#running-it) ·
[Taking it to the finish line](#taking-it-to-the-finish-line) · [Layout](#layout)

## Results

| | H4D, OpenSHMEM/PSM3 | TPU, JAX | GB300, NVSHMEM |
|---|---|---|---|
| Correctness | met, 1,400 GB | met, 12.9 GB on v6e-4 | met on 8 x H200, 1 node |
| Reproducibility | met, 20/20 | met, within 0.1% | not run multi-node |
| Performance stability | met, 5.10 GB/s flat | met, 0.25 GB/s flat | met on 1 node |
| Scale, >1 PB | **no, supply** | **no, architecture**, short by 6% | **yes**, 1,026 nodes |
| Scale, 100 TB | **yes**, 140 nodes | **yes on v5p**, about 2,048 chips | **yes**, 108 nodes |
| Scale, 4,096 endpoints | **yes**, 22 nodes | **yes on v5p**, a 16x16x16 slice | **yes**, 4,104 GPUs |
| OpenSHMEM one-sided PGAS | met | **no, architecture** | qualifies, blocked across nodes |
| Connectionless fabric | met, PSM3 | n/a | unproven |
| Per-packet adaptive routing | **no, by design** | n/a, ICI is a static torus | unknown, ConnectX-8 untested |
| Operational plausibility | met | met | package shipped, one blocker |

Three different reasons sit behind the four "no" cells, and only one of them is permanent.

**Supply.** H4D at 1 PB needs 1,403 nodes in one zone, because Cloud RDMA cannot cross
zones, and that is more than any zone holds unallocated. The machines would work if they
existed. The largest H4D run on record is 192 nodes.

**Architecture.** TPU has no PGAS and no remote put, so the one-sided requirement cannot be
met on any generation. That is the permanent one.

The scale rows are close rather than categorical, and the generation matters. A job cannot
span slices over ICI, so slice size is the ceiling. v6e tops out at 256 chips and reaches
neither target. **v5p reaches both the endpoint count and 100 TB**: 95 GiB per chip, a
16x16x16 slice is 4,096 chips exactly, and it holds about 207 TB of keys at the measured
2.02x footprint. 1 PB needs 2,020 TB resident, and the largest TPU7x topology holds
1,900 TB, so it misses by 6%. A footprint below 1.9x would close that gap.

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

### What today's quota buys

Measured against the study project, 2026-08-20.

| family | quota | buys | largest run |
|---|---|---:|---|
| `h4d-highmem-192` | 500 vCPU, `CPUS_PER_VM_FAMILY` | 2 nodes | 1,400 GB, 384 PEs |
| `a3-ultragpu-8g` (H200) | 64 preemptible GPUs per region | 8 nodes | 137.4 GB, 8 GPUs, 1 node |
| `a4-highgpu-8g` (B200) | 64 preemptible GPUs per region | 8 nodes | not run |
| `a4x-highgpu-4g` (GB200) | entitled, no capacity | 0 | not run |
| `a4x-maxgpu-4g-metal` (GB300) | **accelerator not allowlisted** | 0 | not run |
| TPU v6e | 512 on-demand, 1,536 preemptible per zone | a full v6e-256 pod | 12.9 GB, 4 chips |

Two of these are not quota problems. GB300 fails on an accelerator allowlist before quota
is consulted. TPU has ample quota and no capacity, and its ceiling is slice size rather
than supply.

## What was measured

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
the step. A single timer wrapped around the exchange and the sorts together is therefore
not an ICI measurement, whichever of the two dominates. Replacing the bucket scatter with a
gather made the step 9.0x faster. See [results/tpu.md](results/tpu.md).

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
| Round-0 growth, 16 to 64 PEs/node | 16.8x | **3.1x** |
| Validated runs at 64 PEs/node | 7/20, 9/20 | **20/20** |
| Highest working density | 32 PEs/node | **192 PEs/node**, 20/20 |
| Largest validated dataset | 8.59 GB | **1,400 GB** |
| Aggregate rate | 1.50 GB/s | **5.10 GB/s** |

`ofi_rxm` opens a connection per peer, so round-0 cost grows with
`PEs_per_node x total_PEs` and stops making progress near 8,192 connections per node.
PSM3 implements no connection management, so nothing grows with the square of the job.
The three changes required to run SOS on PSM3 are in [results/h4d-psm3.md](results/h4d-psm3.md).

## How it was built

Deliverable 4: the design decisions, the tradeoffs between scale and performance, and where
topology-aware programming helps.

### Choosing the machine family

The study names NVSHMEM on H200 over Quantum-2 InfiniBand. That configuration does not
exist on Google Cloud, for two independent reasons: there is no InfiniBand, only RoCE over
Titanium, and H200 supply is effectively zero. The requirement as written is unbuildable
here at any budget.

Selecting on what the fabric must do rather than on the named hardware:

| family | RDMA | memory/node | why it was or was not chosen |
|---|---|---:|---|
| **h4d-highmem-192** | Cloud RDMA, Falcon and iRDMA | 1,488 GB | **chosen.** Only CPU family with RDMA, and the only one obtainable without a capacity request |
| a4x (GB200, GB300) | RoCE | 960 GB + 186 GB HBM/GPU | the scale path; needs a reservation and an accelerator allowlist |
| a3-ultra (H200) | RoCE | 3,072 GB | used as the NVLink stand-in for GB300 |
| x4, m4 | none | up to 32 TB | fails the fabric requirement outright |
| TPU v6e, v7x | ICI, not RDMA | 32 or 192 GB HBM/chip | no PGAS; compiler collectives only |

h4d is the compliance path and a4x is the scale path.

Capacity reporting disagreed with real requests three times. Absolute pool size predicted
nothing; how much of a zone was already committed predicted well. Confirm a zone by trying
to create in it rather than by reading a number. On-demand in us-east1-b granted
immediately while DWS Flex in that same zone reported exhausted, because they are separate
inventory.

Cloud RDMA is zone-locked. The Falcon network profile is per zone, so a retry elsewhere
means rebuilding the RDMA VPC, the nodeset, the controller and the login node. The Falcon
zone list is not the H4D zone list either: us-west4-c holds a large H4D pool and has no
Falcon profile, so it cannot do Cloud RDMA at all.

### The 64-bit port

Upstream ISx declares `typedef int KEY_TYPE` over a 2^28 key space and warns only that the
SHMEM API calls must change. Three deeper problems:

**Counter width.** Every bucket size, offset and index is `int`. At the target scale each PE
holds about 1.75e10 keys, which overflows a signed 32-bit counter roughly 8x, silently.
Sizes go negative, the prefix scan wraps, and the program still reports success.

**The local sort.** `count_local_keys()` histograms into an array of `BUCKET_WIDTH` ints,
which is a counting sort and is tractable only because the key space is small. At
`MAX_KEY = 2^60` that array would be about 1.4 PB per PE. The local phase became an LSD
radix sort: linear in received keys, independent of key space, and what the study specifies.

**The symmetric buffer.** Upstream hardcodes a 268-million-key receive buffer. Because the
write is one-sided, overrunning it corrupts a peer's heap rather than raising an error, and
the symptom appears later on a different rank. ISx64 sizes it from keys-per-PE and
bounds-checks every offset before the put.

Correctness is provable without a cluster. `tests/shmem_stub.h` implements the handful of
OpenSHMEM calls for one PE, so the sort, the counter widths and the verification run on a
laptop. That caught real bugs before any spend. Change-by-change detail is in
[results/porting.md](results/porting.md).

### Two changes to the OpenSHMEM runtime

Both are required and neither is documented.

**`--enable-ofi-mr=basic`.** SOS defaults to scalable memory registration. The H4D provider
rejects it and `fi_getinfo` returns no data.

**`--enable-hard-polling`.** SOS enables target counters unless this is set, which adds
`FI_RMA_EVENT` to the hints. `verbs;ofi_rxm` on `irdma0` supports neither `FI_RMA_EVENT` nor
`FI_FENCE`, so `shmem_init()` aborts with `Transport init failed (-61)`.

libfabric also has to be built from source. The HPC VM image ships 1.22.0 with no matching
`-devel` in any repo, and the only `-devel` offered conflicts with `mercury`.

A third change is a source patch rather than a configure flag. PSM3 does not implement
shared transmit contexts and SOS binds one unconditionally, so it aborts at
`transport_ofi.c:606`. `deploy/h4d/sos-psm3-stx.patch` lets the bind degrade instead of
failing, and it is worth sending upstream because it fixes SOS for any provider without STX.

### Two walls that belonged to the provider

Both were measured on `verbs;ofi_rxm` and both are gone on PSM3. That they belonged to the
provider rather than to H4D is the study's main transport finding.

**32 PEs per node.** Past it, every configuration died with
`Operation retry limit exceeded (1073741824)`. The wall tracked PEs per node rather than
total, and `SHMEM_OFI_STX_AUTO=1` and `SHMEM_OFI_STX_MAX=8` did not move it. Cause is
connection establishment: `ofi_rxm` opens a connection per peer, so setup cost grows with
`PEs_per_node × total_PEs` and stalls near 8,192 connections per node. On PSM3 the wall does
not exist and 192 PEs per node validates 20/20.

**About 32 GB of symmetric heap per node.** Aggregate rather than per PE: one PE with 32 GB
works, eight PEs with 4 GB each works, every 64 GB-per-node arrangement fails. Cause is
basic MR pinning the whole heap at `shmem_init()`, and huge pages do not lift it. Stock ISx
puts the receive buffer in that heap, which capped the dataset at about 26.7 GB of keys per
node and put 1 PB at roughly 37,500 nodes. `src/cpu/isx64_win.c` keeps one window slot per
PE in the heap and the dataset in ordinary memory, so the heap is about 0.5 MB per PE and
does not grow with the data.

### The TPU path

TPUs handle 64-bit integers despite documentation stating `int32` only. Measured on 4 v5e
chips: `uint64` is real on device, 2^63-1 survives a round trip, and `jax.lax.sort` returns
ordered output at a 1.48x penalty against int32.

The constraint is structural. A PE under PGAS puts whatever it has. A collective needs an
agreed shape before the compiler can emit it, so the exchange must either pad every bucket
or use `ragged_all_to_all`, and padding costs bandwidth on the padding.

### Topology-aware programming

It helps in three places. **Placement**, because Cloud RDMA and NVLink both stop at the zone
boundary and there is no fallback for a stocked-out zone. **PE-to-node mapping**, because
the exchange dominates past 64 PEs; on PSM3 the answer is one PE per vCPU. And **rotating
the all-to-all start offset** so all PEs do not target rank 0 first, retained from upstream.

Cluster Director publishes the hierarchy to Slurm and GKE: a sub-block is one rack behind a
single top-of-rack switch, a block is sub-blocks on non-blocking fabric, a cluster is
blocks. A reservation is what makes it visible.

It would not have helped with either wall above. Both were properties of the provider's
connection and registration model rather than of how work is arranged across a topology.
Changing the provider fixed them; rearranging ranks would not have.

Where it matters next is GB300. NVLink spans 72 GPUs, so at 4,104 endpoints 56 of every 57
bytes in the all-to-all cross RoCE. Making the bucket assignment domain-aware keeps most
keys inside the domain that generated them, and that is a change to the routing prefix in
`compute_dest` rather than to the exchange.

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

## Taking it to the finish line

What a team with capacity does next, per family.

### H4D

The only path that satisfies the OpenSHMEM one-sided requirement.

**Met on 2 nodes:** correctness at 1,400 GB, 20/20 reproducibility at 192 PEs per node,
5.10 GB/s flat over the top 3.4x of the range. 192 PEs per node is one PE per vCPU.

**Not met:** scale. `CPUS_PER_VM_FAMILY` caps H4D at 500 vCPU and all three H4D shapes are
192 vCPU, so self-service buys two nodes and no smaller shape exists to work around it.
Increments of 1,000 through 5,000 all return `COMMON_QUOTA_CONSUMER_OVERRIDE_TOO_HIGH`.

1. **Get quota.** Two sizes, and the smaller one is worth asking for first.

   | goal | nodes | vCPU |
   |---|---:|---:|
   | 4,096 endpoints, 15.7 TB | 22 | 4,224 |
   | 100 TB, 26,880 endpoints | 140 | 26,880 |

   22 nodes clears the endpoint criterion outright. Both need a capacity escalation.
2. **Confirm the zone separately.** Quota grants permission and does not reserve machines.
   us-central1-a and us-central1-b both returned `ZONE_RESOURCE_POOL_EXHAUSTED` while quota
   was sufficient.
3. **Provision.** `deploy/h4d/00_setup_project.sh`, then set `h4d_cluster_size` in
   `deploy/h4d/isx-slurm-h4d.yaml`. Cloud RDMA cannot cross zones.
4. **Build.** `deploy/h4d/01_build_sos.sh`, which applies `sos-psm3-stx.patch`.
5. **Climb, do not jump.** 4 nodes, then 8, 16 and 32 at constant keys per node. That gives
   the weak-scaling slope the projection assumes. Everything above two nodes here is
   arithmetic.
6. **20-run reproducibility set** at the largest stable shape.
7. **Target run.** 140 nodes, 192 PEs per node, 4.65e8 keys per PE. At 5.10 GB/s per node
   pair this projects to about 5 minutes, which is a floor.

Two launcher notes for anyone running outside Slurm. `oshrun` aborts with
`could not find a launcher` on a bare VM; `mpich` supplies `mpiexec.hydra`, which speaks the
PMI-1 that SOS bootstraps against, and that is process launch only. On Rocky with OS Login,
`~/.ssh/authorized_keys` is ignored, so add the launching node's key with
`gcloud compute os-login ssh-keys add`.

### TPU

**Met on 4 chips:** correctness at 12.88 GB, repeat runs within 0.1%, 0.25 GB/s flat across
a 6x range. The ICI exchange runs at 200 GB/s and is 0.1% of the step, while bucketing is
97.4%.

A job cannot span slices over ICI, so slice size is the ceiling and no grant moves it.
Which generation you pick decides three of the four rows.

| | v6e | v5p | TPU7x |
|---|---|---|---|
| HBM per chip | 32 GiB | 95 GiB | 192 GiB |
| Largest schedulable slice | 256 chips | 6,144 chips | 2,048 listed, 9,216 by topology |
| 100 TB | **no**, needs 121 full pods | **yes**, about 2,048 chips | yes |
| 1 PB | **no** | **no**, 310 TB at 6,144 chips | **no**, 1,900 TB raw against 2,020 TB needed |
| 4,096 endpoints | **no**, max slice is 256 | **yes**, a 16x16x16 slice | yes |
| OpenSHMEM one-sided | **no**, no PGAS on TPU | same | same |

**v5p is the generation to use.** A 16x16x16 slice is 4,096 chips exactly, which meets the
endpoint criterion, and it holds about 207 TB of keys at the measured 2.02x footprint, so it
clears 100 TB with headroom in the same run. v5p supports multi-host slice node pools in
GKE, and slices of 1,024 chips and similar have been obtained that way in unrelated work,
so this is an obtainability data point rather than a spec sheet claim.

1 PB stays out of reach. It needs 2,020 TB resident and the largest TPU7x topology holds
1,900 TB, so the miss is 6% rather than an order of magnitude. A footprint below 1.9x would
close it, and `results/windowed-exchange.md` has the design for 1.15x.

To run it: submit on-demand queued resources and wait, because Spot did not convert in 141
attempts while on-demand landed 2 of 7. Switch the exchange to ragged `all_to_all`, since
the padded buffer is held twice and is why the current code uses about 3.2 GB of a 32 GB
chip. Then re-measure the phase split across more than one host, because 200 GB/s is 4
chips sharing a host and says nothing about inter-host ICI.

None of this makes TPU responsive to the brief. It has no remote put.

### GB300

**Met:** validated on 8 x H200 in one node to 137.44 GB at 67.15 GB/s, flat across a 64x
data range.

**Not met, and the thing to fix first:** multi-node NVSHMEM does not work on Cloud RoCE
with the packaged build.

The fabric works. `ib_write_bw` moves 45.0 GB/s host to host on one of eight NICs, all 8
HCAs `PORT_ACTIVE` at MTU 4096. Each GPU has its own RDMA NIC, `gpu0rdma0` through
`gpu7rdma0`, each on its own /32.

Registration is where it stops. On the same buffer, `tools/dmabuf_reg_test.c`:

```
ibv_reg_dmabuf_mr  on rocep145s0: SUCCESS (errno=0)
ibv_reg_mr (peermem) on rocep145s0: FAILED (errno=14)
```

`nvidia_peermem` cannot insert. The module file is present but it registers against
`ib_register_peer_memory_client`, which exists in MOFED's `ib_core` and not in the inbox one
this image ships.

Four NVSHMEM remote transports, four distinct failures:

| transport | failure |
|---|---|
| `ibrc` | registers with `ibv_reg_mr`, needs peermem. `ibv_poll_cq completion status 4, local protection error` |
| `ibgda` | maps the NIC doorbell into GPU BAR space. `cudaHostRegister with IoMemory failed, error=800`. Virtualized MRDMA does not expose it |
| `ibdevx` | registers through dmabuf and succeeds, then segfaults on the first cross-node put. Reproduced with a 25-line program, so this is the transport rather than ISx |
| `ucx` | 1.18.x with CUDA, verbs and rdmacm. Reads the NVSHMEM heap as host memory: `failed to register address 0xa20000000 (host) ... Input/output error` |

`ibdevx` is the near miss, the only transport that both registers GPU memory correctly here
and reaches the data path.

1. **Settle registration on two nodes before booking anything.** An hour of work. Build the
   image, run `tools/dmabuf_reg_test.c`, then a 2-node NVSHMEM job. Ranked by effort: chase
   the `ibdevx` segfault with NVIDIA, build NVSHMEM from source against dmabuf, or move to a
   MOFED image so `nvidia_peermem` inserts and stock `ibrc` works.
2. **Get the accelerator allowlisted.** `nvidia-gb300` is absent from
   `gcloud compute accelerator-types list` on this project, so creation fails before
   capacity is consulted. `nvidia-gb200` is entitled separately.
3. **Reserve.** The family refuses Spot outright, and on-demand A4X GB200 returned
   `ZONE_RESOURCE_POOL_EXHAUSTED` in both zones tested.
4. **Request dense placement with the reservation.** `--collocation=collocated
   --gpu-topology=1x72`, one NVL72 domain per policy. Node counts must be multiples of 18.
5. **Run.** `deploy/gb300/run.sh 10` then `full`, 108 and 1,026 nodes.
6. **Watch the exchange share.** NVLink spans 72 GPUs, so at 4,104 endpoints 56/57 of the
   all-to-all crosses RoCE. If it dominates, make the bucket assignment domain-aware using
   the Cluster Director topology. That changes the routing prefix in `compute_dest` rather
   than the exchange.

Both GB300 verdicts assume step 1 succeeds. If no NVSHMEM transport can be made to work on
Cloud RoCE, the GPU path is limited to one NVL72 domain, 72 GPUs and about 37 TB, and the
full target becomes unreachable on every family.

## Layout

```
src/cpu/          OpenSHMEM implementation, three exchange schedules
src/gpu/          NVSHMEM implementation
src/tpu/          JAX implementation, plus a phase-split benchmark
deploy/h4d/       provision, build and run on H4D
deploy/gb300/     handoff package for a GB300 allocation
results/          measurements and reference, one file per topic; raw/ holds logs
tools/            probes, a standalone reproducer, and the dmabuf registration test
tests/            single-PE shim for local correctness
```

## Detail

| topic | file |
|---|---|
| Requirements verbatim, with status | `GOAL.md` |
| PSM3, and what it takes to run SOS on it | `results/h4d-psm3.md` |
| GPU on H200, and the multi-node blocker | `results/gpu-nvshmem.md` |
| TPU: ICI at 200 GB/s, and a 9x bucket fix | `results/tpu.md` |
| Adaptive routing evidence | `results/adaptive-routing.md` |
| Operational readiness, six tests | `results/operations.md` |
| Telemetry H4D does and does not expose | `results/telemetry.md` |
| The `ofi_rxm` connection limit | `results/rxm-connection-limit.md` |
| The `CPUS_PER_VM_FAMILY` cap | `results/h4d-capacity.md` |
| Memory footprint, and getting it below 2.02x | `results/windowed-exchange.md` |
| Scaling to a petabyte | `results/scale-out.md` |
| Porting ISx to 64 bits, change by change | `results/porting.md` |

Upstream issues filed: [ofiwg/libfabric#12673](https://github.com/ofiwg/libfabric/issues/12673),
[Sandia-OpenSHMEM/SOS#1239](https://github.com/Sandia-OpenSHMEM/SOS/issues/1239).

## Attribution

ISx is by Ulf Hanebutte and Jacob Hemstad, Copyright (c) 2015 Intel Corporation, BSD
3-clause. See [LICENSE-ISx](LICENSE-ISx). Upstream:
[github.com/ParRes/ISx](https://github.com/ParRes/ISx). Paper: "ISx, a Scalable Integer
Sort for Co-design in the Exascale Era", PGAS 2015.

Changes here are marked `ISX64` in the source.

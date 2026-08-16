# ISx64: 64-bit integer sort over OpenSHMEM on Google Cloud

A port of [ISx](https://github.com/ParRes/ISx), the Sandia integer sort, to 64-bit keys,
with provisioning recipes and measurements on Google Cloud H4D nodes using Cloud RDMA.

ISx is a distributed bucket sort. It measures unstructured all-to-all bisection bandwidth
rather than compute. The target for this study is more than 1 PB of `uint64` keys across
4,096 or more endpoints, using OpenSHMEM one-sided RMA. MPI is out of scope.

## Results

The study asked whether Google Cloud can run an OpenSHMEM PGAS workload that sorts more
than 1 PB across 4,096 or more endpoints. On H4D, it cannot today. Four requirements are
met and four are not.

| requirement | result | why |
|---|---|---|
| OpenSHMEM PGAS, no MPI | **met** | Sandia OpenSHMEM on libfabric. No MPI in the data path |
| RDMA one-sided Get/Put/Atomics | **met** | Cross-node put verified; the target posts no receive |
| Correctness | **met** | Validated to 2,147,483,648 keys across 128 endpoints |
| Performance stability | **met** | Three inflection points identified and quantified |
| Reproducibility | **not met** | 34% of runs complete at 64 processes per node, over 120 runs |
| Scale | **not met** | 128 endpoints and 17 GB reached, against 4,096 and 1 PB |
| Connectionless fabric | **not met** | The working provider uses reliable connections |
| Per-packet adaptive routing | **not met** | The fabric routes per subflow, not per packet |

All five contract deliverables are complete: source, provisioning recipe, execution
artifacts, architectural narrative and failure analysis.

### What limits it

**One software defect explains both Reproducibility and Connectionless.** The libfabric
`verbs;ofi_rxm` provider fails to establish connections reliably once a node holds
several thousand of them. The cost of the first exchange round scales with connections
per node, not with bytes moved, and runs that fail do so during that round. The
connectionless provider that would avoid the problem entirely, `verbs;ofi_rxd`, cannot
complete OpenSHMEM startup on this hardware. Both are filed upstream with reproducers, one
of which uses libfabric's own test binary.

**Capacity is what limits Scale.** H4D quota is fixed at 500 vCPU per
region and every self-service increase is refused. All three H4D machine shapes are 192
vCPU, so a project is capped at two nodes. Reaching 4,096 endpoints needs 128 nodes;
1 PB needs about 800.

**The adaptive routing requirement and the fabric disagree on approach.** The requirement
describes Ultra Ethernet behaviour. Google's Falcon transport deliberately chose multipath
subflows over per-packet spraying, because per-packet routing reorders packets and RoCE
treats reordering as loss. Falcon's published results claim up to 8x lower completion
times than the alternative. The fabric uses all available paths and reacts to congestion. It does not do so per
packet.

**Operational readiness has a second, independent problem.** H4D cannot live-migrate, so
a host maintenance event ends a run. This is unaffected by any of the above and would
remain after the software defect is fixed. At 800 nodes a run has 800 chances to be
interrupted. Retrying is the right answer at this run length, and
`infra/h4d/run_isx64.sh` implements it.

### Actions

| action | depends on |
|---|---|
| Approve H4D quota of 24,576 vCPU in one region, 128 nodes | capacity approval |
| Confirm the zone holds the machines. Quota grants permission and reserves nothing | capacity team |
| Resolve the connection establishment defect | libfabric and Sandia OpenSHMEM maintainers |
| Decide whether subflow multipath satisfies the adaptive routing requirement | customer |
| Reconsider the scale target. No published ISx run exceeds about 96 endpoints or 26 GB, and this study already passed both | customer |

`SCALE_OUT.md` holds the node counts, memory arithmetic and configuration for a petabyte
run, so no part of it needs deriving again once capacity is granted.

## Implementations

| path | model | transport |
|---|---|---|
| `src/isx64/isx64.c` | OpenSHMEM one-sided RMA | Sandia OpenSHMEM over libfabric |
| `src/isx64/isx64_win.c` | as above, windowed exchange | fixed symmetric heap, any dataset size |
| `src/isx64/isx64_stream.c` | as above, one destination per step | single-slot symmetric window |
| `src/isx-jax/isx_jax.py` | `jax.lax.all_to_all` on TPU | ICI. A compiler collective, not PGAS |

The JAX version is included for comparison against tensor hardware. It is not OpenSHMEM
and does not satisfy the study's programming-model requirement.

## Build and run locally

`tests/shmem_stub.h` implements the OpenSHMEM calls ISx64 uses for a single PE, so the
sort and its verification run without a cluster.

```bash
gcc -O2 -std=c11 -Wall -Wextra -DNDEBUG -I tests -I src/isx64 \
    -o bin/isx64_stub src/isx64/isx64.c src/isx64/pcg_basic.c -lm
./bin/isx64_stub 4000000 2 /tmp/isx64.log
```

Expected tail:

```
  Max key value      : 1152921504606846976 (2^60)
  verification       : PASSED
```

At one PE every put is a local `memcpy`, so a stub run measures the sort and nothing
about the fabric.

## Build and run on a cluster

```bash
cd src/isx64
make provider-check     # confirm the RDMA provider exposes FI_RMA
make                    # builds with oshcc
make check              # 4 PEs, single node
```

Run `make provider-check` before trusting any result. An OpenSHMEM build that falls back
to a TCP provider passes every correctness test while measuring nothing about RDMA.

On H4D, set these before running:

```bash
export SHMEM_OFI_PROVIDER="verbs;ofi_rxm"
export FI_VERBS_GID_IDX=1              # GID 0 is link-local on this NIC
export FI_OFI_RXM_CQ_EQ_FAIRNESS=1
```

`infra/h4d/` provisions the cluster. `infra/h4d/01_build_sos.sh` builds the runtime and
records five steps the vendor documentation omits.

## Measured results

Two `h4d-highmem-192` nodes in us-east1-b, libfabric 2.6.0, Sandia OpenSHMEM 1.5.3.

### Correctness

| PEs | keys | verification |
|---:|---:|---|
| 4 | 4,194,304 | PASSED |
| 64 | 1,073,741,824 | PASSED |
| 128 | 2,147,483,648 | PASSED |

### Reproducibility

Five runs per configuration, 16,777,216 keys per PE.

| PEs/node | validated |
|---:|---|
| 32 | 3/5 |
| 64 | 1/5 |

Runs that fail do so in the transport, not in verification. See
[results/ROOTCAUSE_connection_establishment.md](results/ROOTCAUSE_connection_establishment.md).

### Exchange variants

32 PEs per node, 64 PEs, 16,777,216 keys per PE.

| build | rounds | time | rate | symmetric window per PE |
|---|---:|---:|---:|---|
| `isx64_win` | 17 | 2.050 s | 4.19 GB/s | `NUM_PES × WINDOW × 8` |
| `isx64_stream` | 1,088 | 2.474 s | 3.47 GB/s | `WINDOW × 8` |

### Bandwidth

| measurement | value |
|---|---|
| 64-PE all-to-all, 8.59 GiB per node | 4.44 GB/s |
| `fi_pingpong` 1 MB, `verbs;ofi_rxm` | 8.15 GB/s |
| `fi_pingpong` 1 MB, `verbs;ofi_rxd` | 5.52 GB/s |
| STREAM triad, single core | 32.7 GB/s |
| link nominal | 25 GB/s |

No byte counter on H4D tracks RoCE traffic. See
[results/D3_telemetry.md](results/D3_telemetry.md).

### TPU 64-bit support

Four TPU v5e chips, JAX 0.9.0
([raw result](results/probeB_tpu_int64_v5e_20260814.json)). Cloud TPU documentation states
that the TPU supports `tf.int32` and that 64-bit values must be converted. That does not
apply to JAX with `jax_enable_x64`.

| test | result |
|---|---|
| `jax_enable_x64` yields a device uint64 array | yes |
| keys above 2^32 survive a round trip | yes, to 2^63-1 |
| `jax.lax.sort` accepts uint64 | yes |
| uint64 sort cost against int32 | 1.48x |
| `jax.lax.all_to_all` moves uint64 over ICI | yes, 175 GB/s off-chip |

```bash
python3 probes/tpu_int64.py --n 4194304
```

## Next steps

H4D was chosen from what was available. Starting instead from what the workload needs
changes the recommendation.

### What the workload requires

Strip the benchmark framing and three requirements remain.

**2 PB of simultaneously addressable memory.** Every key is resident and nothing streams
to disk. A bucket sort needs the input, a destination-ordered copy and a receive buffer,
which careful implementation brings to about 2x the dataset. This port measures 2.02x.

**A global address space with one-sided access.** Each process writes into a peer's memory
without the peer participating.

**Bisection bandwidth.** The shuffle moves essentially the whole dataset across the fabric
once.

### What that implies

Memory sets the machine count, and it is not close. At 1.5 TB per node, 2 PB needs 1,340
nodes; at 4 TB, 500. The lower bound is hundreds of machines whatever the family.

Bandwidth sets the run time. At the 4.44 GB/s per node measured here, a thousand nodes
finish the shuffle in about four minutes.

**The run is minutes, not hours.** That decides three things. Checkpointing is not worth
building, because writing 1 PB takes longer than sorting it. Retry is the correct failure
strategy. And the bill is dominated by how long machines are held rather than by the sort.

### Where the argument turns

One-sided RMA has to be implemented by something, and the options are not equivalent.

Inside an NVL72 rack, 72 GPUs share a hardware-coherent address space. A remote write is a
memory operation. NVSHMEM implements OpenSHMEM semantics directly on it, and no part of
the libfabric or verbs stack is involved.

Over a network it is OpenSHMEM on libfabric on verbs on RoCE. Four layers, each with its
own failure modes. **Every defect in this repository's failure analysis is in that stack**,
and none of them can arise in a coherent domain, because there is no connection to
establish.

The requirements name `nvshmem_put64_nblock` alongside `shmem_put64`, so the GPU path is
in scope. That removes the last reason to prefer the CPU stack.

| requirement | CPU + RoCE | GPU + NVSHMEM |
|---|---|---|
| One-sided put, get, atomics | yes, through four layers | yes, natively |
| Connectionless | no, and the connectionless provider is unusable | inside a rack there are no connections |
| Per-packet adaptive routing | no, the transport uses subflows | inside a rack, no routing |

Three fabric requirements the CPU path fails are answered by moving the workload inside a
coherent domain, because they describe problems that only exist on a network.

### The recommendation

**Sort inside NVLink domains and treat the cross-rack network as the thing to avoid.**

Scaling past one rack reintroduces the network, so the design question is how much of the
shuffle stays inside racks. For a bucket sort with deterministic destinations this is
answerable: **make the bucket assignment rack-aware.** Choose key ranges so most keys land
in the rack that generated them, and only the residue crosses the network. This is a
change to the routing prefix, one function, and it converts a flat all-to-all into a
hierarchical one. It is also the topology-aware programming Deliverable 4 asks about.

What stays hard is obtaining hundreds of accelerator nodes in one zone. No engineering
removes that, it is the same constraint for any family, and it should be the first
conversation rather than the last.

## Detail behind the results

| topic | file |
|---|---|
| Connection establishment defect, with reproducer | `results/ROOTCAUSE_connection_establishment.md` |
| Adaptive routing evidence | `results/adaptive_routing.md` |
| Operational readiness, six tests | `results/operational_plausibility.md` |
| Telemetry available on H4D | `results/D3_telemetry.md` |
| Scaling to a petabyte | `SCALE_OUT.md` |
| Requirements, verbatim, with status | `GOAL.md` |

Upstream issues: [ofiwg/libfabric#12673](https://github.com/ofiwg/libfabric/issues/12673),
[Sandia-OpenSHMEM/SOS#1239](https://github.com/Sandia-OpenSHMEM/SOS/issues/1239).

## Porting notes

Upstream ISx uses `typedef int KEY_TYPE` with a 2^28 key space. Widening the type is not
sufficient. Every bucket counter, offset and key-space index is also `int`, and at
petabyte scale these overflow silently, producing a reported successful sort on corrupted
data. Two changes go beyond type widths:

- Upstream's `count_local_keys()` allocates `BUCKET_WIDTH * sizeof(int)` and histograms
  into it. With a 2^60 key space over 7,000 buckets that allocation is 1.4 PB per
  endpoint. ISx64 uses an LSD radix sort, which is linear in received keys and independent
  of key space.
- Upstream hardcodes `KEY_BUFFER_SIZE (1uLL<<28uLL)`. One-sided writes carry no bounds
  check on the target, so an overrun corrupts a peer's heap. ISx64 sizes the buffer from
  keys per PE and checks each offset before the put.

[PORTING.md](PORTING.md) lists the rest.

## Repository layout

```
src/isx64/       OpenSHMEM implementation, three exchange variants
src/isx-jax/     JAX/TPU implementation
probes/          capability probes to run before provisioning
repro/           standalone reproducer for the transport failure
tests/           single-PE shim for local correctness
infra/           provisioning recipes
results/         measurements and failure analysis
docs/            architectural narrative and design notes
```

## Attribution

ISx is by Ulf Hanebutte and Jacob Hemstad, Copyright (c) 2015 Intel Corporation, BSD
3-clause. See [LICENSE-ISx](LICENSE-ISx). Upstream:
[github.com/ParRes/ISx](https://github.com/ParRes/ISx). Original paper: "ISx, a Scalable
Integer Sort for Co-design in the Exascale Era", PGAS 2015.

Changes in this repository are marked `ISX64` in the source.

# ISx64: 64-bit integer sort over OpenSHMEM on Google Cloud

A port of [ISx](https://github.com/ParRes/ISx), the Sandia integer sort, to 64-bit keys,
with provisioning recipes and measurements on Google Cloud H4D nodes using Cloud RDMA.

ISx is a distributed bucket sort. It measures unstructured all-to-all bisection bandwidth
rather than compute. The target for this study is more than 1 PB of `uint64` keys across
4,096 or more endpoints, using OpenSHMEM one-sided RMA. MPI is out of scope.

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

`infra/h4d/` provisions the cluster. `infra/h4d/02_build_sos.sh` builds the runtime and
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

## Limits

**Scale.** The study target is not demonstrated here. This project reached 128 endpoints
and 17 GB. H4D quota is capped at 500 vCPU per region without an approved escalation,
which is two nodes. [SCALE_OUT.md](SCALE_OUT.md) gives the node counts, memory arithmetic
and configuration for 1 PB across 25,600 endpoints.

**Reproducibility.** Runs complete between 20% and 60% of the time above 32 PEs per node.
The cause is `ofi_rxm` connection establishment, which scales as
`PEs_per_node × total_PEs`. Filed as
[ofiwg/libfabric#12673](https://github.com/ofiwg/libfabric/issues/12673) and
[Sandia-OpenSHMEM/SOS#1239](https://github.com/Sandia-OpenSHMEM/SOS/issues/1239).

**Connectionless fabric.** The study requires connectionless semantics. `verbs;ofi_rxm`
uses reliable connections. `verbs;ofi_rxd` is connectionless but cannot complete SOS
startup, and its RMA path reaches the retry limit at 2 PEs.

**Memory.** ISx64 holds 2.02x the key array resident at peak. `docs/streamed_exchange_design.md`
describes the change that would reduce this to about 1.15x.

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

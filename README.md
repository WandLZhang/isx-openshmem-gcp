# ISx64: 64-bit integer sort over OpenSHMEM on Google Cloud

A feasibility study. The question is whether a public cloud can run a capability-class
workload that treats thousands of nodes as one globally addressable memory. The proxy is
[ISx](https://github.com/ParRes/ISx), the Sandia integer sort, which stresses unstructured
all-to-all bisection bandwidth rather than compute.

The target is a sort of more than 1 PB of 64-bit keys across 4,096 or more endpoints,
using OpenSHMEM one-sided RMA. MPI is out of scope.

Two implementations are compared:

- **`src/isx64`** — Sandia OpenSHMEM over libfabric on CPU nodes with Cloud RDMA. True
  one-sided `shmem_put` and `shmem_atomic_fetch_add` against remote symmetric heaps.
- **`src/isx-jax`** — the same bucket sort on TPU using `jax.lax.all_to_all`, which is a
  compiler-driven collective rather than PGAS. Not OpenSHMEM, and the study says so; the
  question is whether tensor hardware executes the equivalent work faster.

Status: the port and the an internal process probes. Phase 1 baseline runs are in progress.

---

## Findings so far

### Upstream ISx is a 32-bit program, and the port is not a typedef

`SHMEM/params.h` upstream declares `typedef int KEY_TYPE` with a key space of 2^28, and
its comment warns only that "you will have to change the SHMEM API calls used". That
understates it. Every bucket counter, offset and key-space index in the program is also
`int`. At 1 PB spread over roughly 7,000 endpoints each endpoint holds about 1.75e10
keys, which overflows a signed 32-bit counter by about 8x. The overflow is silent: sizes
go negative, offsets wrap, and the program reports a successful sort on corrupted data.

Two changes go beyond widening types:

**The local sort had to be replaced.** Upstream's `count_local_keys()` allocates
`BUCKET_WIDTH * sizeof(int)` and histograms into it. That is a counting sort, tractable
only because the key space is 2^28. With a 2^60 key space over 7,000 buckets,
`BUCKET_WIDTH` is about 1.8e14 and that allocation would be 1.4 PB per endpoint. ISx64
uses an LSD radix sort, which is linear in received keys and independent of key space.

**The symmetric receive buffer had to become dynamic.** Upstream hardcodes
`KEY_BUFFER_SIZE (1uLL<<28uLL)`. At petabyte scale a PE receives 65x that. Because the
write is one-sided there is no bounds check on the target, so the overrun corrupts a
peer's heap rather than raising an error. ISx64 sizes the buffer from keys-per-PE and
checks every offset before the put.

See [PORTING.md](PORTING.md) for the full list.

### TPUs do 64-bit integers, despite the documentation

Cloud TPU documentation still states that "the TPU only supports tf.int32" and that
64-bit values must be converted. That is either stale or specific to TensorFlow. Measured
on four TPU v5e chips with JAX 0.9.0 ([raw result](results/probeB_tpu_int64_v5e_20260814.json)):

| question | result |
|---|---|
| `jax_enable_x64` gives a real uint64 array on device | yes, `dtype=uint64` |
| keys above 2^32 survive a round trip | yes, including 2^63-1 |
| `jax.lax.sort` accepts uint64 and returns ordered output | yes |
| uint64 sort cost against int32 | **1.48x** |
| `jax.lax.all_to_all` moves uint64 over ICI | yes, 175 GB/s off-chip, multiset preserved |

A 1.48x penalty is a real cost and not a blocker. The truncation risk was the one that
mattered, because a silent downcast would have produced sorted, verified, wrong output.
It does not occur. `probes/tpu_int64.py` re-runs this on any TPU.

Reproduce:

```bash
python3 probes/tpu_int64.py --n 4194304
```

### What bounds the TPU path is memory, not arithmetic

ISx holds roughly 2.5x the key array resident. Sorting 1 PB therefore needs about 2.5 PB
of memory. TPU HBM does not reach that inside one interconnect domain, so the TPU
implementation is expected to be reported as bounded by capacity rather than by
bandwidth. That is a finding, not a failure, and it is the reason the CPU path carries
the scale target.

---

## Correctness without a cluster

The algorithm can be validated on a laptop before anyone pays for hardware.
`tests/shmem_stub.h` implements the handful of OpenSHMEM calls ISx64 uses for a single
PE, so the sort, the counter widths and the verification all run locally:

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

This proves the sort is correct. It proves nothing about the fabric, because at one PE
every put is a local memcpy. Never quote a performance number from a stub run.

## Building for a cluster

```bash
cd src/isx64
make provider-check     # confirm the RDMA provider exposes FI_RMA
make                    # builds with oshcc
make check              # 4 PEs, single node
```

`make provider-check` matters. An OpenSHMEM build that silently falls back to a TCP
provider will pass every correctness test and measure nothing about RDMA, which would
make the result non-responsive to the study.

## Repository layout

```
src/isx64/       the OpenSHMEM implementation and its port notes
src/isx-jax/     the JAX/TPU implementation
probes/          an internal process probes that run before the expensive work
tests/           single-PE shim for local correctness
infra/           provisioning recipes
results/         raw measurements
docs/            architectural narrative
```

## Attribution

ISx is by Ulf Hanebutte and Jacob Hemstad, Copyright (c) 2015 Intel Corporation,
BSD 3-clause. See [LICENSE-ISx](LICENSE-ISx). Upstream:
[github.com/ParRes/ISx](https://github.com/ParRes/ISx). Original paper: "ISx, a Scalable
Integer Sort for Co-design in the Exascale Era", PGAS 2015.

Changes in this repository are marked `ISX64` in the source.

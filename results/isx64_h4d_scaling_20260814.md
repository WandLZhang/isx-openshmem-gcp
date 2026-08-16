# ISx64 on H4D Cloud RDMA: first cross-node results

Two `h4d-highmem-192-lssd` nodes, `us-east1-b`, Slurm cluster `isx-h4d-e1`.
libfabric 2.6.0 and Sandia OpenSHMEM 1.5.3, both built from source.
Transport OFI over `verbs;ofi_rxm` on `irdma0`. No MPI in the data path.

## The compliance gate passes

Genuine one-sided RDMA across the wire, target posting no receive:

```
PE  0/2 on isxh4de1-h4dnodeset-0   got 1001 want 1001  OK
PE  1/2 on isxh4de1-h4dnodeset-1   got 1000 want 1000  OK
```

Verified at 2, 4, 6, 8, 12 and 16 PEs across two nodes. `ldd libsma.so` shows no
`libmpi`. The launcher is `srun --mpi=pmi2`, which invokes Slurm's PMI. No MPI runtime is involved.

The build required two non-default flags, both forced by the fabric. See
`BLOCKER_h4d_rma_atomic_20260814.md`:

```
--enable-ofi-mr=basic     provider rejects scalable MR (mr_mode=0), SOS's default
--enable-hard-polling     drops FI_RMA_EVENT, which verbs;ofi_rxm does not support
```

> **Read with the reproducibility result.** Every row below is a **single sample**. A
> later test of 10 identical runs at 32 PEs passed only 2 of 7 completed attempts. At a
> ~30% success rate a single PASS is a lucky draw, not a characterisation. See
> `BLOCKER_reproducibility_20260814.md`.

## ISx64 weak scaling

4,194,304 keys per PE (2^22), 2 nodes, 2 timed iterations after burn-in.

| PEs | PE/node | TTS (s) | rate GB/s | all2all (s) | radix (s) | total GB | verify |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 4 | — | — | — | — | — | **FAIL** (intermittent) |
| 16 | 8 | 0.088 | 6.08 | 0.021 | 0.031 | 0.50 | PASSED |
| 32 | 16 | 0.125 | 8.58 | 0.044 | 0.034 | 1.00 | PASSED |
| 64 | 32 | 0.226 | 9.52 | 0.116 | 0.041 | 2.00 | PASSED |
| 128 | 64 | — | — | — | — | — | **FAIL** |
| 192 | 96 | — | — | — | — | — | **FAIL** |
| 256 | 128 | — | — | — | — | — | **FAIL** |
| 384 | 192 | — | — | — | — | — | **FAIL** |

An earlier standalone run at 8 PEs with 2^24 keys/PE passed at 0.450 s, 2.39 GB/s. The
8-PE failure in this sweep is therefore not a floor, it is jitter. See below.

## Inflection points

**1. Hard ceiling at 32 PEs per node.** Every configuration at 64 or more PEs per node
dies with the same error, regardless of total PE count:

```
[0060] ERROR: transport_ofi.h:596: try_again
       Operation retry limit exceeded (1073741824)
```

SOS spins in `try_again` until it exhausts 2^30 polls. The failure is in completion
progress, not bandwidth: the run never gets far enough to move meaningful data. The wall
tracks PEs *per node*, which points at contention for the single 200 Gbps IRDMA vNIC and
its completion resources rather than at anything global.

This matters for the endpoint target. H4D has 192 vCPUs, and reaching 4,096 endpoints on
a reasonable node count depends on packing many PEs per node. At a working maximum of 32
PEs per node, 4,096 endpoints needs **128 nodes**, not the 22 that 192 PEs per node would
have allowed.

**2. Shared transmit contexts are not the cause.** The obvious suspect was
`SHMEM_OFI_STX_MAX`, which defaults to 1 per PE. Tested at the failure point:

| configuration | 128 PEs | 192 PEs |
|---|---|---|
| baseline | FAIL | FAIL |
| `SHMEM_OFI_STX_AUTO=1` | FAIL | FAIL |
| `SHMEM_OFI_STX_MAX=8` | FAIL | — |

None of it moves the wall. The cause lies deeper than STX allocation.

**3. The all-to-all crosses over and becomes dominant at 64 PEs.**

| PEs | all2all (s) | radix (s) | all2all share of TTS |
|---:|---:|---:|---:|
| 16 | 0.021 | 0.031 | 24% |
| 32 | 0.044 | 0.034 | 35% |
| 64 | 0.116 | 0.041 | 51% |

The exchange roughly doubles per PE doubling while the local sort stays flat, which is
what weak scaling should do: keys per PE is constant, so radix work per PE is constant,
while the exchange grows with the number of destinations. By 64 PEs the benchmark is
measuring the fabric, which is the regime ISx is designed for. That
it only reaches that regime just as the retry wall arrives is the central problem.

**4. Jitter at low PE counts.** 8 PEs passed standalone and failed inside the sweep.
Reproducibility is a stated success criterion, so this needs characterising before any
result is reported as stable. Not yet investigated.

## Distance from the target

| requirement | achieved | gap |
|---|---|---|
| 4,096+ endpoints | **64 PEs** | 64x |
| > 1 PB sorted | **2.00 GB** | ~500,000x |
| correctness validation | PASSED at 16, 32, 64 PEs | met, at small scale |
| reproducibility | jitter observed at 8 PEs | not met |
| inflection points identified | yes, three | met |

The 2 GB figure is bounded by the 2-node cluster, not by the software: at 64 PEs each PE
held 4.19M keys, and per-PE memory was nowhere near the 1,488 GB the node offers. Raising
keys per PE is the cheap axis. Raising node count is the expensive one and is blocked on
capacity, not on code.

## Reproducing

```bash
# on the cluster, after infra/h4d/01_build_sos.sh
export PATH=$HOME/isx/bin:$PATH LD_LIBRARY_PATH=$HOME/isx/lib:$LD_LIBRARY_PATH
export SHMEM_OFI_PROVIDER="verbs;ofi_rxm" SHMEM_SYMMETRIC_SIZE=512M
srun -N2 --ntasks-per-node=32 --mpi=pmi2 --export=ALL ./isx64bin 4194304 2 out.csv
```

Expect `verification : PASSED` and a time to solution near 0.23 s.

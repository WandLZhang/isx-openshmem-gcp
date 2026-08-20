# Windowed exchange removes the symmetric-heap ceiling

Measured 2026-08-15 on two `h4d-highmem-192-lssd` in `us-east1-b`, 32 PEs (16 per node).
Implementation: `src/cpu/isx64_win.c`.

Before this change the dataset lived in the symmetric heap, which caps at about 32 GB per
node, so 1 PB needed about 37,500 nodes. Afterwards the heap holds one window slot per PE
and the node count follows ordinary memory instead.

## The idea

Stock ISx sizes its receive buffer to the whole per-PE dataset and puts it in the
symmetric heap. On H4D the provider rejects scalable memory registration, so SOS must pin
and register the entire heap at `shmem_init()`, and that stalls somewhere between 32 and
64 GB per node. Dataset size therefore runs straight into a registration limit.

Only the *landing zone* has to be symmetric. In the windowed version the symmetric
allocation is a fixed window of 16,384 keys per (sender, receiver) pair. The exchange runs
in rounds: every PE puts a chunk into each peer's window slot, barrier, every PE drains
its window into ordinary `malloc` memory, repeat. Ordinary memory needs no registration,
so the dataset is bounded by node RAM instead.

The transfer stays one-sided. Each chunk is a `shmem_uint64_put` directly into a peer's
symmetric memory and its length is a `shmem_longlong_p` into the same peer. The receiver
participates in neither. Chunking and flow control changed. The memory model did not.

## The measurement

`SHMEM_SYMMETRIC_SIZE=256M` for every run. Dataset grows 20x; the heap does not move.

| keys/PE | dataset/PE | total | symmetric/PE | dataset:heap | rounds | TTS | rate | verify |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1.0e7 | 0.08 GB | 2.6 GB | **4.2 MB** | 19x | 20 | 0.86 s | 2.96 GB/s | PASSED |
| 5.0e7 | 0.40 GB | 12.8 GB | **4.2 MB** | 95x | 96 | 4.40 s | 2.91 GB/s | PASSED |
| 2.0e8 | 1.60 GB | 51.2 GB | **4.2 MB** | **381x** | 383 | 17.8 s | 2.88 GB/s | PASSED |

Two things matter here.

**The symmetric footprint is constant.** 4.2 MB per PE at every size, because it is
`NUM_PES x 16384 x 8` and has nothing to do with the dataset. Stock ISx64 at the largest
of these would have needed 1.92 GB of symmetric heap per PE, 31 GB per node, right at the
ceiling.

**Throughput does not degrade.** 2.96, 2.91, 2.88 GB/s across a 20x growth in dataset.
The extra memcpy on the receive side and the two barriers per round cost essentially
nothing at these sizes. The decoupling is not paid for in bandwidth.

## Remaining bounds on 1 PB

With the heap out of the way, the limit is node RAM and the algorithm's footprint.

| footprint | keys/node | nodes for 1 PB | vs 870-node zone |
|---:|---:|---:|---|
| 2.60x as implemented | 563 GB | 1,776 | too many |
| 2.04x with recv slack 1.3 -> 1.02 | 718 GB | 1,393 | too many |
| **1.02x** with in-place MSD radix and streamed send | **1,435 GB** | **697** | design target |

The shipped footprint is 2.02x, which puts 1 PB at 1,403 nodes. The 1.02x row is a design
target, not a measurement.

Three further reductions get from 2.02x to 1.02x:

1. **Receive slack 1.3 -> 1.02.** Keys are uniform, so the per-destination receive count
   concentrates as `1/sqrt(n)`. At 1e9 keys per PE the relative spread is about 3e-5;
   1.3x is wildly conservative and 1.02x is still many standard deviations safe.
2. **In-place MSD radix**, removing the scratch buffer that currently equals the receive
   array. American-flag sort is the standard choice.
3. **Stream generate/bucket/send in chunks**, so the send array never exists in full.
   The PRNG is deterministic and seeded per rank, so keys can be regenerated rather than
   stored.

## Not addressed

**Peak footprint.** 2.02x is what ships. The three reductions above are unimplemented and
`docs/streamed-exchange.md` carries the design.

None of this changes the petabyte verdict for H4D. Even at 1.02x, 1 PB needs 697 nodes in
one zone, which is still more than any zone holds unallocated.

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
`results/windowed-exchange.md` carries the design.

None of this changes the petabyte verdict for H4D. Even at 1.02x, 1 PB needs 697 nodes in
one zone, which is still more than any zone holds unallocated.

---

# Getting the footprint below 2.02x

Goal: get peak resident memory from 2.02x down toward 1.15x, which is 1,403 nodes → 799
for 1 PB. Written before implementing because the obvious approach does not work and the
one that does also fixes a separate problem.

## The direct approach

"Release `send` incrementally as the exchange drains it" does not survive contact with
the current loop. `exchange_windowed` iterates rounds on the outside and destinations on
the inside: in each round every PE sends one window to every destination. So all
destination regions drain **in parallel**, and the unsent remainder stays scattered
across the whole array. Compacting it would cost an O(n) `memmove` per round, and at the
production shape that is 128 rounds over a 39 GB buffer per PE.

`realloc` down cannot help either, since a single `malloc` block cannot release its
middle.

## Inverting the loop

Reverse the nesting, destinations outside and windows inside, and use the rotation that is
already there. At step `i`, PE `p` sends to `(p + i) % n`. Every PE targets a **different**
destination at each step, and therefore every PE receives from **one** sender per
step.

This has a further consequence:

> The symmetric window needs **one** slot per PE, not `NUM_PES` slots.

The heap goes from `NUM_PES × WINDOW × 8` to `WINDOW × 8` per PE. At the target shape of
25,600 PEs that is **3.36 GB → 131 KB per PE**, which removes the 107 GB/node window
problem in `results/scale-out.md` step 3 entirely, and makes the `WINDOW_KEYS_PER_PEER` reduction
recommended there unnecessary.

With destinations on the outside, `send` also drains in destination order, one contiguous
region at a time, so it can be held as `NUM_PES` separate allocations and freed as each
completes. Memory then falls as the exchange proceeds while `recv` grows, holding the sum
near one copy plus slack.

## The cost

Inverting the loop **serialises the all-to-all into `n` steps** where the current code
overlaps every destination within a round. Each step is a single put to a single peer plus
a barrier. That trades injection concurrency for memory, and on a fabric whose measured
problem is connection and completion behaviour under concurrent load, it might help or
hurt, because the current round structure is the concurrent put storm that
`tools/repro-rxm-livelock.c` shows failing.

So this is not purely a memory optimisation. It is a different communication schedule, and
it needs measuring on both axes.

## Order of work

1. Implement the inverted schedule with a single-slot window, keeping `send` as one block.
   Verify correctness, and measure throughput against the current schedule.
2. Only if throughput holds, split `send` into per-destination allocations and free as it
   drains. This step is what reduces the memory.
3. Re-measure stability, because the schedule changed.

Step 1 is where the risk is and it is not a small change. It should be done with the
cluster available for measurement, not written blind.

## Measured

Implemented in `src/cpu/isx64_stream.c`. Two `h4d-highmem-192` nodes, 16,777,216 keys
per PE, five runs per cell.

| PEs/node | build | validated | rounds | time | rate |
|---:|---|---|---:|---:|---:|
| 32 | `isx64_win` | 3/5 | 17 | 2.050 s | 4.19 GB/s |
| 32 | `isx64_stream` | 2/5 | 1,088 | 2.474 s | 3.47 GB/s |
| 64 | `isx64_win` | 1/5 | 9 | 3.584 s | 4.79 GB/s |
| 64 | **`isx64_stream`** | **4/5** | 1,152 | 4.351 s | 3.95 GB/s |

The two sizes disagree. At 32 PEs per node the streamed build is slightly worse, which
sits inside the plus-or-minus-one spread measured elsewhere. At 64 PEs per node it
validates 4/5 against 1/5, which does not.

The direction at 64 PEs per node follows from the root cause. The failure is connection
establishment under concurrent load, and the streamed schedule sends to one destination
per step instead of all destinations at once. Serialising the schedule reduces the
concurrent connection pressure that the windowed version creates. The cost is 18%
throughput, 4.79 GB/s to 3.95 GB/s.

This is a stronger stability result than either environment variable found in this
session. `FI_VERBS_GID_IDX=1` and `FI_OFI_RXM_CQ_EQ_FAIRNESS=1` both improved the
standalone reproducer and left ISx64 unchanged. This changes ISx64 at the size where it
was failing.

Five runs per cell is a small sample, and repeating it reversed the result. Twenty runs
per build at 64 PEs per node give `isx64_stream` 7/20 and `isx64_win` 9/20. The five-run
figures (4/5 against 1/5) were noise in both directions. There is no stability difference
between the two schedules, and the paragraphs above claiming one are wrong.

What holds is the window reduction, which is structural rather than measured, and the 18%
throughput cost.

## Status

Step 1 is done and measured. Steps 2 and 3 are not implemented. The single-slot-window finding applies whether or not the streaming
lands. It reduces the symmetric heap 25,600x at the target shape and removes a step from
the scale-out recipe.

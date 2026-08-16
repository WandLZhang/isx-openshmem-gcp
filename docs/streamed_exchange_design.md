# Streamed exchange: design notes before implementation

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

Reverse the nesting — destinations outside, windows inside — and use the rotation that is
already there. At step `i`, PE `p` sends to `(p + i) % n`. Every PE targets a **different**
destination at each step, and therefore every PE receives from exactly **one** sender per
step.

This has a further consequence:

> The symmetric window needs **one** slot per PE, not `NUM_PES` slots.

The heap goes from `NUM_PES × WINDOW × 8` to `WINDOW × 8` per PE. At the target shape of
25,600 PEs that is **3.36 GB → 131 KB per PE**, which removes the 107 GB/node window
problem in `SCALE_OUT.md` step 3 entirely, and makes the `WINDOW_KEYS_PER_PEER` reduction
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
hurt — the current round structure is exactly the concurrent put storm that
`repro/livelock_repro.c` shows failing.

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

Implemented in `src/isx64/isx64_stream.c`. Two `h4d-highmem-192` nodes, 16,777,216 keys
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

Five runs per cell is a small sample, and repeating it showed why that matters. Twenty
runs of `isx64_stream` at 64 PEs per node gave **7/20**, not 4/5. The paragraph above,
written from the five-run result, overstates the effect. The windowed twenty-run figure
was still running when this was written.

## Status

Step 1 is done and measured. Steps 2 and 3 are not implemented. The single-slot-window finding applies whether or not the streaming
lands. It reduces the symmetric heap 25,600x at the target shape and removes a step from
the scale-out recipe.

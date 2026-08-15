# Streamed exchange: design notes before implementation

Goal: get peak resident memory from 2.02x down toward 1.15x, which is 1,403 nodes → 799
for 1 PB. Written before implementing because the obvious approach does not work and the
one that does also fixes a separate problem.

## Why the obvious approach fails

"Release `send` incrementally as the exchange drains it" does not survive contact with
the current loop. `exchange_windowed` iterates rounds on the outside and destinations on
the inside: in each round every PE sends one window to every destination. So all
destination regions drain **in parallel**, and the unsent remainder stays scattered
across the whole array. Compacting it would cost an O(n) `memmove` per round, and at the
production shape that is 128 rounds over a 39 GB buffer per PE.

`realloc` down cannot help either, since a single `malloc` block cannot release its
middle.

## What does work: invert the loop, and the window collapses

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
   drains. That is the part that actually buys the memory.
3. Re-measure stability, because the schedule changed.

Step 1 is where the risk is and it is not a small change. It should be done with the
cluster available for measurement, not written blind.

## Status

Not implemented. The single-slot-window finding applies whether or not the streaming
lands. It reduces the symmetric heap 25,600x at the target shape and removes a step from
the scale-out recipe.

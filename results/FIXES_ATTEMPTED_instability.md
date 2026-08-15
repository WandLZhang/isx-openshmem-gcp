# Four attempted fixes for the ~30% completion rate, all measured, none successful

The instability is the last blocker that is not gated on quota. This records every fix
tried, what it changed, and what that rules out. Written so the next person does not
repeat any of it.

Baseline to beat: **2 of 7 completed at 32 PEs/node**, and **0 of 3 at 64 PEs/node**.

## Summary

| # | change | 32 PEs/node | 64 PEs/node | verdict |
|---|---|---|---|---|
| 0 | baseline (SOS, mr=basic, hard-polling) | 2/7 (~29%) | 0/3 | — |
| 1 | `--enable-ofi-manual-progress` | **0/3** | 0/3 | **regression**, reverted |
| 2 | `SHMEM_OFI_STX_AUTO=1`, `STX_MAX=8` | — | 0/3 | no effect |
| 3 | `FI_TRANSMIT_COMPLETE` instead of `FI_DELIVERY_COMPLETE` | 2/10 (20%) | 0/3 | no effect on stability, **35% faster when it completes** |
| 4 | OSSS-UCX over UCX 1.18 (replaces libfabric entirely) | **0/10** | 0/3 | segfaults in `shmem_init` |

Four failed fixes. The debugging discipline says three or more means the architecture is
wrong, not the parameter. Stopping here rather than attempting a fifth guess.

## 3. FI_TRANSMIT_COMPLETE — the best-supported hypothesis, and it was wrong

libfabric [#5601](https://github.com/ofiwg/libfabric/issues/5601) says verbs;ofi_rxm
advertises `FI_DELIVERY_COMPLETE` but silently provides only `FI_TRANSMIT_COMPLETE`. SOS
requests exactly the former in `query_for_fabric()`, so the hypothesis was that SOS spins
waiting for a completion semantic the provider never delivers.

Patched `src/transport_ofi.c:1521` to request what the provider actually implements,
rebuilt, retested. Stability is unchanged: 2/10 against a 2/7 baseline is the same rate
inside noise, and the 64 PEs/node wall did not move at all.

**So the completion semantic is not the cause of the livelock.** #5601 remains a real
correctness concern for this stack, and worth reporting to the customer on its own terms,
but it is not this bug.

One useful side result: when a run does complete, `FI_TRANSMIT_COMPLETE` gives
**TTS 0.139 s against 0.213 s**, a 35% improvement. That is a legitimate tuning finding
for whenever the stability problem is solved.

## 4. OSSS-UCX — the architectural alternative, blocked on its own bug

The reasoning was sound: every remaining hypothesis lives in libfabric's verbs/rxm path,
so replace the layer. OSSS-UCX is OpenSHMEM over UCX, uses no libfabric, stays
one-sided PGAS, and is not MPI, so it keeps the study responsive.

Built OpenPMIx 5.0.6, UCX 1.18.0 and OSSS-UCX. It runs and it segfaults immediately, on
the two-PE one-sided probe, before any of the interesting work.

**A trap worth recording.** UCX was first built on the Slurm controller, a `c2-standard-4`
with no `irdma0` and no verbs headers. `configure` auto-detects, found neither, and
produced a **TCP-only UCX without warning**:

```
Transport: self / tcp (enp0s9, eth0, lo) / sysv / posix
```

No `rc_verbs`, no `ud_verbs`. That build would have run, produced numbers, and measured
TCP rather than RDMA. Rebuilt on a compute node with `--with-verbs` explicit so a missing
dependency fails loudly, which then gave the right transports:

```
Transport: posix / rc_verbs / self / sysv / tcp / ud_verbs
```

**Build accelerator-adjacent libraries on a node that has the accelerator.** Auto-detected
optional dependencies are how you end up benchmarking the wrong thing.

With RDMA transports present and `UCX_TLS=rc_verbs,self,sm`, `UCX_NET_DEVICES=irdma0:1`,
OSSS-UCX still segfaults in both PEs. Not diagnosed. Most likely a PMIx or UCX version
mismatch rather than anything about H4D, since it fails at init before touching the
fabric.

## What is now known about the failure

From `HYPOTHESES_instability.md` plus the above:

- It is a **livelock**, not a deadlock. PEs sit at 98.4% CPU, state `R`, spinning in
  `try_again`.
- The fabric is clean on completing runs: zero congestion notices, zero protocol errors,
  zero CRC errors, and both NICs pass Google's `irdma_health_check`.
- Bounce buffering is enabled (`mode = 0x0`, no `FI_CONTEXT`), so the EAGAIN path does
  drain the CQ. That theory is dead.
- The wall tracks **PEs per node**, not total PEs, which points at a per-NIC or per-node
  resource rather than anything global.
- It is not the completion semantic, not shared transmit contexts, and not SOS's progress
  strategy.

## What to try next, in order

1. **libfabric debug build with `FI_LOG_LEVEL=debug FI_LOG_PROV=rxm,verbs`.** Still the
   cheapest unblocked diagnostic and still not done; the release build compiles the log
   statements out, which is why earlier logging attempts produced nothing. This is the one
   thing likely to name the stuck operation directly.
2. **Debug the OSSS-UCX segfault.** A stack trace from the core file would say whether it
   is PMIx version skew, which would be quick to fix and would open the whole alternative
   path.
3. **A different node pair.** Still the only way to separate a fabric-wide property from
   two specific machines, and still blocked on `CPUS_PER_VM_FAMILY`.
4. **Report it upstream.** With a livelock reproducer at 64 PEs/node on verbs;ofi_rxm plus
   irdma, this is worth a Sandia-OpenSHMEM or libfabric issue regardless of whether we
   solve it. Both prior issues in this area were closed stale, so a fresh reproducer on
   current hardware has value.

---

## 5. Bounding operations in flight — first crack in the wall, not a fix

The mechanism nothing had yet addressed: `exchange_keys()` issues `NUM_PES` puts back to
back and only quiets at the end. The provider reports a transmit queue depth of **2048**,
and at 64 PEs per node sharing one 200 Gbps NIC the aggregate outstanding count runs far
past it. That is a plausible source of permanent `-FI_EAGAIN`.

Added `ISX64_THROTTLE`: force `shmem_quiet()` every N puts, 0 restoring upstream
behaviour. Swept at 64 PEs/node, the wall that had returned 0/3 under **every** previous
configuration.

| `ISX64_THROTTLE` | 128 PEs | TTS |
|---:|---|---|
| 0 (upstream) | 0/3 | — |
| **1** | **1/3** | 0.576 s |
| 4 | 0/3 | — |
| 16, 64 | run did not finish in the window | — |

**THROTTLE=1 is the first configuration in this entire study to complete at 64 PEs per
node.** Every other attempt, across five different changes, returned 0/3 there.

That is real evidence for the mechanism: the livelock is outstanding-operation
exhaustion, and bounding in-flight operations relieves it.

It is not a fix, and should not be reported as one:

- **1 of 3 is not reliability.** It converts "never" into "occasionally", which is
  progress in understanding and not progress toward a usable configuration.
- **THROTTLE=1 quiets after every single put**, which fully serialises the exchange. TTS
  is 0.576 s against 0.226 s for 64 PEs unthrottled, roughly 2.5x slower, and it still
  fails two runs in three.
- **THROTTLE=4 returns to 0/3**, so there is no smooth knee to tune toward. A parameter
  that only works at its most extreme setting, and then only sometimes, is a symptom being
  suppressed rather than a cause being removed.

The honest reading is that in-flight operation count is *a* contributing factor and not
the whole story. Something else is also wrong, most likely in the provider, which is
consistent with the two upstream issues closed stale.

The throttle is kept in the source because it is the best diagnostic lever found, and
because it is the evidence for the mechanism. It is off by default.

## 6. Raising the queue depth — the targeted fix for the mechanism, and it made things worse

If bounding in-flight operations helps (fix 5), the non-serialising version of that fix is
to raise the queue instead of throttling the application. libfabric exposes this without
a rebuild.

| configuration | 64 PEs/node | 32 PEs/node |
|---|---|---|
| baseline, tx=2048 | 0/3 | 2/10 |
| `FI_OFI_RXM_TX_SIZE/RX_SIZE=16384` | 0/3 | — |
| `+ FI_VERBS_TX_SIZE/RX_SIZE=16384` | 0/3 | — |
| `+ FI_OFI_RXM_MSG_TX_SIZE/RX_SIZE=16384` | 0/3 | — |
| queues 16384 + `THROTTLE=32` | 0/3 | — |
| queues 16384 + `THROTTLE=8` | 0/3 | — |
| queues 16384, throttle off | — | **0/10** |

Raising every queue-depth knob available did not move the wall, and at 32 PEs per node it
turned an already-poor 2/10 into **0/10**. This is the second change in the study to make
the failure worse rather than better, after manual progress.

That result also undercuts the simple reading of fix 5. If the livelock were straightforward
transmit-queue exhaustion, a queue eight times deeper would help and it does not. Whatever
`THROTTLE=1` is doing, it is not simply keeping the queue under its limit; more likely it
is changing timing enough to dodge a race.

## Conclusion after six attempts

| # | change | result |
|---|---|---|
| 1 | `--enable-ofi-manual-progress` | worse |
| 2 | STX auto / STX_MAX=8 | no effect |
| 3 | `FI_TRANSMIT_COMPLETE` | no effect on stability, 35% faster |
| 4 | OSSS-UCX over UCX | segfault at init |
| 5 | throttle in-flight ops | 1/3 at the wall, first ever pass, not a fix |
| 6 | raise all queue depths | no effect, and worse at 32 PEs/node |

Six independent changes across the application, the OpenSHMEM runtime, the provider
configuration and the entire transport library. Two made it worse. One produced a single
completion at a wall nothing else passed. None produced a usable configuration.

**This is not an application tuning problem and it should stop being treated as one.**
The evidence points at the provider or driver: a livelock at 98% CPU, a clean fabric, a
limit that tracks PEs per node, immunity to every queue and completion parameter, and two
upstream issues in exactly this provider pair that were closed without fixes.

The remaining moves are all outside application tuning:

1. **libfabric debug build with `FI_LOG_PROV=rxm,verbs`.** Still undone, still the cheapest
   thing that could name the stuck operation. The release build compiles the log statements
   out, which is why every earlier logging attempt produced nothing.
2. **Upstream report.** There is now a clean reproducer: ISx64 at 64 PEs/node on
   verbs;ofi_rxm over irdma, livelocking in `try_again`, relieved only by
   `shmem_quiet()` after every put. Both related libfabric issues (#5601, #6720) were
   closed stale, so a reproducer on current hardware has standalone value.
3. **A different node pair**, still blocked on `CPUS_PER_VM_FAMILY`.
4. **Fix the OSSS-UCX segfault**, which would open a transport path that shares none of
   this code.

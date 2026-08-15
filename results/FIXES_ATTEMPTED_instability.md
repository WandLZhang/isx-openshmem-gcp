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

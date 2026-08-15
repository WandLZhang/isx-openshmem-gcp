# The ~30% completion rate: evidence, hypotheses, and what would settle each

Working document for the instability that now blocks scale. Written after the memory
ceiling was solved (`BREAKTHROUGH_windowed_exchange_20260815.md`), which promoted this
from second-order to the main blocker.

## Evidence established so far

| observation | how it was obtained | what it rules out |
|---|---|---|
| Fails with `Operation retry limit exceeded (1073741824)` at `transport_ofi.h:596` | run output | — |
| PEs sit at **98.4% CPU, state R, `syscall=running`** during the hang | direct `ps` on the compute node, bypassing Slurm | **not** a blocked deadlock; this is a livelock, spinning in `try_again` |
| Zero congestion notices, zero protocol errors, zero CRC errors | `irdma0` hw_counters before/after a completing run | fabric congestion as the cause, at least on runs that finish |
| Both nodes pass Google's `irdma_health_check`, all port error counters zero | Slurm prolog | grossly degraded NICs |
| Provider `mode = 0x0`, no `FI_CONTEXT` | `fi_getinfo` probe | the theory that bounce buffering is disabled, leaving the EAGAIN path unable to drain the CQ. **Bounce buffers are on and the drain path is taken.** |
| `--enable-ofi-manual-progress` gives 0/3 at 32 PEs/node, where baseline was intermittent | rebuild and retest | SOS's polling strategy as the fix; it is a regression |
| `SHMEM_OFI_STX_AUTO=1`, `STX_MAX=8` change nothing | tested at the wall | shared transmit context exhaustion |
| TX queue depth is 2048, `rx_attr.size` 2048 | `fi_getinfo` probe | — |
| Wall tracks **PEs per node**, not total PEs | scaling sweep | a global/topological cause |

Two fixes have been tried and failed. Per the debugging discipline, a third failed fix
means the architecture is the problem, not the parameter.

## Upstream findings

Two open-in-practice libfabric issues match this stack closely. Both were closed as
stale rather than fixed.

**[ofiwg/libfabric#5601](https://github.com/ofiwg/libfabric/issues/5601) —
"verbs;ofi_rxm: should either stop advertising FI_DELIVERY_COMPLETE or support it"**

> "it seems that the verbs;ofi_rxm provider doesn't actually support
> `FI_DELIVERY_COMPLETE`, but silently provides `FI_TRANSMIT_COMPLETE` semantics instead.
> This can lead to application errors, in my case when an initiating node does a bunch of
> RDMA writes and then continues, believing that the target node must be able to see the
> results of those in its memory when in fact the latter cannot yet do so."

Filed by a Chapel runtime developer, which is the same problem shape: a PGAS runtime doing
one-sided RDMA writes over libfabric.

SOS sets exactly this flag:

```c
tx_attr.op_flags = FI_DELIVERY_COMPLETE;   /* transport_ofi.c, query_for_fabric() */
```

and the bisect confirmed `fi_getinfo` accepts it, which is precisely the "advertises but
does not implement" behaviour the issue describes.

**[ofiwg/libfabric#6720](https://github.com/ofiwg/libfabric/issues/6720) —
"Failed to receive with verbs;ofi_rxm"**

> "it will get stuck in progress"

A progress stall in the same provider pair.

This is the concrete evidence for "software stack immaturity" that the Failure Analysis
deliverable asks us to characterise, with upstream citations rather than assertion.

## Hypotheses, ranked

**H1 — `FI_DELIVERY_COMPLETE` is accepted but not implemented, and the completion SOS
waits for never arrives.** Best supported: matches #5601 exactly, matches the livelock
(SOS spins waiting for a completion semantic the provider does not deliver), matches
worsening with PE count, and matches a clean fabric (the data may already be on the wire).

*Test:* patch `query_for_fabric()` to request `FI_TRANSMIT_COMPLETE`, rebuild, re-run the
10x reproducibility test at 32 PEs/node and the wall at 64.
*Caution:* this weakens the completion guarantee SOS relies on for `shmem_quiet`. If it
fixes the hang, the correct fix is not to ship it as-is but to add explicit remote
completion. Note #5601's implication that we may **already** be getting
`FI_TRANSMIT_COMPLETE` semantics silently, in which case the guarantee is illusory today
and validation has been passing on timing luck.

**H2 — rxm progress stall (#6720).** Plausible and hard to separate from H1 without
provider-level logging.

*Test:* rebuild libfabric with `--enable-debug` and run with `FI_LOG_LEVEL=debug
FI_LOG_PROV=rxm,verbs`. My earlier logging attempt produced nothing, most likely because
the release build compiles the log statements out. This is the cheapest unblocked
diagnostic and should be done first.

**H3 — TX queue exhaustion at 2048 with N-squared operation counts.** ISx64 issues
`NUM_PES` puts plus `NUM_PES` atomics per PE per iteration. EAGAIN is supposed to handle a
full queue gracefully, and bounce buffering is on, so this should self-resolve. Weakened
but not eliminated.

*Test:* instrument `try_again` to count EAGAINs per operation type and print the CQ state
at abort. Tells us which operation is starving.

**H4 — irdma driver specific.** Cannot be separated from H1-H3 on two nodes that always
pass their health check.

*Test:* a different node pair. Blocked on `CPUS_PER_VM_FAMILY`.

## The alternative worth serious consideration

**Replace the runtime, not the flag: OSSS-UCX instead of Sandia OpenSHMEM.**

Every hypothesis above lives in libfabric's verbs/rxm path. OSSS-UCX is an
OpenSHMEM-compliant implementation over UCX, which has its own RC transport and does not
use libfabric at all. UCX is also the better-trodden path on Intel iRDMA hardware.

This keeps the study responsive: OSSS-UCX is OpenSHMEM, it is one-sided PGAS, and it is
not MPI. It swaps out the layer where all the evidence points, and it is far cheaper than
debugging a provider that upstream abandoned twice.

*Test:* build OSSS-UCX plus UCX on one H4D node, run the same one-sided cross-node probe,
then the 10x reproducibility test. If it is stable at 64 PEs/node, that is both the fix
and a clean finding for the report.

## On HuggingFace

Not applicable. HuggingFace hosts models and datasets; there is nothing there for an
OpenSHMEM transport bug. Naming it as a research avenue would be padding.

The useful sources are the ofiwg/libfabric and Sandia-OpenSHMEM/SOS issue trackers, the
`ofiwg` mailing list thread referenced in #5601, and the UCX repository if the OSSS-UCX
route is taken.

## Recommended order

1. libfabric debug build plus provider logging (H2 diagnostic, cheap, unblocked)
2. `FI_TRANSMIT_COMPLETE` experiment (H1, one-line change, decisive either way)
3. OSSS-UCX (architectural alternative, highest expected value if 1 and 2 are inconclusive)
4. Different node pair (H4, blocked on quota)

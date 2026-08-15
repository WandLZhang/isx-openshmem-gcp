# Draft upstream issue — not yet filed

Target: [`ofiwg/libfabric`](https://github.com/ofiwg/libfabric/issues), cross-linked from
[`Sandia-OpenSHMEM/SOS`](https://github.com/Sandia-OpenSHMEM/SOS/issues).

Filing it needs Willis's go-ahead, since it posts publicly under his GitHub account.

---

**Title:** `verbs;ofi_rxm` over Intel irdma: unbounded EAGAIN in RMA write at high
per-node process counts, with a clean fabric

Running Sandia OpenSHMEM 1.5.3 on libfabric 2.6.0 over `verbs;ofi_rxm` on Intel iDPF/iRDMA
NICs, an all-to-all of one-sided `fi_write`s stops making progress once enough processes
per node are injecting concurrently. Every process spins at ~98% CPU (state R, not
blocked) retrying on `-FI_EAGAIN` until SOS gives up at its 2^30 retry limit.

Completions out of attempts, 2 nodes, 192 cores each, 256 KB writes, reproducer below:

| PEs/node | total PEs | quiet after every put | result |
|---:|---:|---|---|
| 16 | 32 | no | 2/3 |
| 16 | 32 | yes | 3/3 |
| 64 | 128 | no | 3/7 |
| 64 | 128 | yes | 0/3 |

It tracks processes **per node**, not total processes, which points at a per-node or
per-NIC resource rather than anything topological.

**The fabric is clean on both passing and failing runs.** `irdma0` hw_counters show
`cnpSent`, `cnpHandled`, `cnpIgnored`, `InProtoErrors` and CRC errors all zero; both nodes
pass the vendor health check with all port error counters at zero. This does not look like
congestion or a degraded link.

**Calling `fi_cntr_wait`-equivalent quiescence more often makes it worse at scale**, which
is the part I find hardest to explain. `shmem_quiet()` after every put helps at 16
PEs/node and is strictly harmful at 64. If the stall were queue-depth exhaustion I would
expect the opposite.

Ruled out, each tested directly:

- **Queue depth.** `FI_OFI_RXM_TX_SIZE`, `FI_OFI_RXM_RX_SIZE`, `FI_VERBS_TX_SIZE`,
  `FI_VERBS_RX_SIZE` and the `RXM_MSG` variants raised to 16384. No help at 64 PEs/node,
  and it made 32 PEs/node worse (2/10 → 0/10). Default `tx_attr.size` is 2048.
- **Shared transmit contexts.** `SHMEM_OFI_STX_AUTO=1`, `SHMEM_OFI_STX_MAX=8`: no change.
- **Completion semantics.** Requesting `FI_TRANSMIT_COMPLETE` instead of
  `FI_DELIVERY_COMPLETE` (per #5601) does not change stability.
- **Progress model.** SOS `--enable-ofi-manual-progress` is a regression: 0/3 where the
  baseline was intermittent.
- **CQ draining on the EAGAIN path.** Provider `mode` is `0x0` with no `FI_CONTEXT`, so
  SOS's bounce buffers are active and it does drain the CQ while retrying.
- **Atomics.** Not required to trigger it. Adding a contended `fi_fetch_atomic` per target
  gives 0/4 vs 3/7, so it aggravates but is not the cause.

Possibly related, both closed as stale rather than fixed: #5601 (`verbs;ofi_rxm` advertises
`FI_DELIVERY_COMPLETE` without implementing it, reported by a Chapel PGAS runtime dev with
the same "RDMA writes then continue" shape) and #6720 ("it will get stuck in progress" on
the same provider pair).

**Reproducer:** `repro/livelock_repro.c` in this repo. About 60 lines, no dependency
beyond an OpenSHMEM install. Every PE writes a fixed 256 KB block into every peer's
symmetric buffer, then barriers, four times. `QUIET_EVERY` and `USE_ATOMIC` env toggles
select the rows in the table above.

```
oshcc -O2 -o livelock_repro livelock_repro.c
export SHMEM_OFI_PROVIDER="verbs;ofi_rxm" SHMEM_SYMMETRIC_SIZE=1G
srun -N2 --ntasks-per-node=64 --mpi=pmi2 --export=ALL ./livelock_repro
```

I can run diagnostics on the affected hardware. One thing I have not managed: a
`--enable-debug` libfabric build with `FI_LOG_LEVEL=debug FI_LOG_PROV=rxm,verbs` emits no
provider output at all, so I have not been able to see inside the progress loop. If
there's a step I'm missing there, that would probably settle this quickly.

**Environment:** libfabric 2.6.0 built from source · Sandia OpenSHMEM 1.5.3
(`--enable-ofi-mr=basic --enable-hard-polling`) · Intel iDPF/iRDMA, 200 Gbps RoCE ·
Rocky Linux 9, HPC image · 2 × 192-core / 1488 GB nodes · Google Cloud H4D.

# Draft upstream issue — not yet filed

Target: [`ofiwg/libfabric`](https://github.com/ofiwg/libfabric/issues), cross-linked from
[`Sandia-OpenSHMEM/SOS`](https://github.com/Sandia-OpenSHMEM/SOS/issues).

Filing it needs Willis's go-ahead, since it posts publicly under his GitHub account.

---

**Title:** `verbs;ofi_rxm` over Intel irdma: connection establishment stops making
progress at ~8k connections per node, and `verbs;ofi_rxd` is unusable as the alternative

## The failure

Sandia OpenSHMEM 1.5.3 on libfabric 2.6.0 over `verbs;ofi_rxm`, Intel iDPF/iRDMA, 2 nodes
of 192 cores. An all-to-all of one-sided 256 KB `fi_write`s stops making progress once
enough processes per node participate. Every process spins at ~98% CPU (state R, not
blocked) retrying on `-FI_EAGAIN` until SOS gives up at its 2^30 retry limit.

It is **connection establishment**, not the data path. The reproducer prints per-round
timings, and round 0 is not like the others:

| PEs/node | total PEs | round 0 | each later round | connections/node |
|---:|---:|---:|---:|---:|
| 4 | 8 | 0.026 s | 0.001 s | 32 |
| 16 | 32 | 0.368 s | 0.007 s | 512 |
| 64 | 128 | 6.193 s | 0.13 s | 8,192 |

Every round moves identical bytes. Round 0 costs 47x a steady-state round and scales with
connections per node (`PEs_per_node × total_PEs`): connections go 512 → 8,192, which is
16.0x, and round 0 goes 0.368 → 6.193 s, which is 16.8x. Runs that fail print no round at
all, so they die inside round 0. The wall tracks processes **per node**, not total
processes, which is what connections-per-node predicts.

Opening the connections one at a time first (an 8-byte write to each peer, each quiesced)
relocates the cost exactly as that implies:

```
warmup 6.269 s
round 0 done (0.098 s)      <- was 6.193 s, a 63x drop
COMPLETED in 0.383 s
```

but does not fix it: 1/5 completions with the warmup against 0/5 without, and the failing
runs now hang inside the warmup. So this is not a thundering herd of concurrent connects.
Establishing this many connections per node is itself unreliable.

## Ruled out

- **QP count.** `max_qp` on irdma0 is 899,068.
- **Queue depth.** `FI_OFI_RXM_TX_SIZE`, `FI_OFI_RXM_RX_SIZE`, `FI_VERBS_TX_SIZE`,
  `FI_VERBS_RX_SIZE` and the `RXM_MSG` variants at 16384. No help at 64 PEs/node, and it
  made 32 PEs/node worse (2/10 → 0/10). Default `tx_attr.size` is 2048.
- **Shared receive contexts.** `FI_OFI_RXM_USE_SRX=1`: 0/5, and 0/5 with the warmup.
- **Shared transmit contexts.** `SHMEM_OFI_STX_AUTO=1`, `SHMEM_OFI_STX_MAX=8`: no change.
- **Completion semantics.** `FI_TRANSMIT_COMPLETE` instead of `FI_DELIVERY_COMPLETE`
  (per #5601) changes nothing.
- **Atomics.** Not needed to trigger it. A contended `fi_fetch_atomic` per target gives
  0/4 against 3/7 without, so it aggravates but is not the cause.
- **Fabric health.** `cnpSent`, `cnpHandled`, `cnpIgnored`, `InProtoErrors` and CRC errors
  all zero on passing and failing runs; both nodes pass the vendor health check with all
  port error counters at zero.

One oddity that may be diagnostic: quiescing after **every** write helps at 16 PEs/node
(3/5 → and 3/3 in an earlier set) and is strictly harmful at 64 (0/3 against 3/7).

## The connectionless provider does not substitute

`verbs;ofi_rxd` is present on the same NIC and is connectionless, which should avoid the
failing resource entirely:

```
provider: verbs;ofi_rxd
    domain: irdma0-dgram
    protocol: FI_PROTO_RXD
    addr_format: FI_ADDR_IB_UD
    caps: FI_MSG, FI_RMA, FI_TAGGED, FI_ATOMIC, ..., FI_RMA_EVENT, ...
```

It advertises `FI_ATOMIC` and `FI_RMA_EVENT`; `verbs;ofi_rxm` advertises neither. Two
things stop it being usable:

1. **`fi_getinfo` refuses `FI_PROGRESS_AUTO`.** Bisecting SOS's hint set one field at a
   time, rxd accepts every field SOS sets (`FI_RMA|FI_ATOMIC`, basic MR, `FI_RM_ENABLED`,
   `mr_key_size=1`, `FI_THREAD_DOMAIN`, `FI_TRANSMIT_COMPLETE`, `inject_size=16`,
   `FI_EP_RDM`) except `domain_attr.data_progress`. `FI_PROGRESS_AUTO` returns
   `-FI_ENODATA`; `FI_PROGRESS_MANUAL` succeeds. Identical across nine requested API
   versions from 1.5 through 2.6. Is auto progress genuinely unsupported for rxd over
   verbs, or is this an over-strict match?

2. **`fi_av_insert` then fails.** With SOS rebuilt for manual progress, `fi_getinfo`
   succeeds and startup fails at the address vector with an insert failure. rxd's
   `addr_format` is `FI_ADDR_IB_UD` where rxm uses `FI_SOCKADDR_IN`. I have not isolated
   this further and would appreciate a pointer.

Net effect on this platform: an OpenSHMEM runtime can have one-sided RMA with atomics
(rxm) or connectionless semantics (rxd), not both.

## Reproducer

`repro/livelock_repro.c` in this repo. About 60 lines, no dependency beyond an OpenSHMEM
install, no sort and no atomics in the default path. Every PE writes a fixed 256 KB block
into every peer's symmetric buffer, then barriers, four times.

```
oshcc -O2 -o livelock_repro livelock_repro.c
export SHMEM_OFI_PROVIDER="verbs;ofi_rxm" SHMEM_SYMMETRIC_SIZE=1G
srun -N2 --ntasks-per-node=64 --mpi=pmi2 --export=ALL ./livelock_repro
WARMUP=1 srun ... ./livelock_repro      # moves the connection cost out of round 0
```

Possibly related, both closed as stale: #5601 (`verbs;ofi_rxm` advertises
`FI_DELIVERY_COMPLETE` without implementing it, reported by a Chapel PGAS developer with
the same one-sided-write shape) and #6720 (progress stall on the same provider pair).

I can run diagnostics on the affected hardware. One gap: a `--enable-debug` libfabric
build with `FI_LOG_LEVEL=debug FI_LOG_PROV=rxm,verbs` emits no provider output, so I have
not been able to see inside the progress or CM loop.

**Environment:** libfabric 2.6.0 from source · Sandia OpenSHMEM 1.5.3
(`--enable-ofi-mr=basic --enable-hard-polling`) · Intel iDPF/iRDMA, 200 Gbps RoCE ·
Rocky Linux 9 HPC image · 2 × 192-core / 1488 GB · Google Cloud H4D.

---

### Note for anyone A/B-testing two SOS builds

`oshcc` embeds an RPATH to its own install `lib`, and **RPATH beats `LD_LIBRARY_PATH`**.
Setting `LD_LIBRARY_PATH` to a second SOS prefix silently keeps loading the first one.
Use `LD_PRELOAD=<prefix>/lib/libsma.so.0` and confirm with `ldd` before trusting a
comparison. This invalidated one round of results here.

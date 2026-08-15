# The instability is connection establishment, and the connectionless alternative is blocked

2026-08-15. This supersedes the ranked hypotheses in `HYPOTHESES_instability.md`. H1
(`FI_DELIVERY_COMPLETE` not implemented) and H3 (TX queue exhaustion) are both wrong.
The failure is not in the data path at all.

## What the timings show

`repro/livelock_repro.c` prints each round. Round 0 is not like the others:

| PEs/node | total PEs | round 0 | each later round | connections/node |
|---:|---:|---:|---:|---:|
| 4 | 8 | 0.026 s | 0.001 s | 32 |
| 16 | 32 | 0.368 s | 0.007 s | 512 |
| 64 | 128 | 6.193 s | 0.13 s | 8,192 |

Round 0 costs 47x a steady-state round at 64 PEs/node, and it scales with
**connections per node**, which is `PEs_per_node × total_PEs`:

- connections 512 → 8,192 is **16.0x**
- round 0 time 0.368 → 6.193 s is **16.8x**

Bytes moved per round are identical in every round, so the extra cost in round 0 is not
data. `ofi_rxm` is connection-oriented over verbs RC and establishes lazily on first
message to each peer, so round 0 pays for every connection.

This is also the first hypothesis that explains the one fact none of the others did:
**the wall tracks PEs per node rather than total PEs.** Connections per node is a
per-node resource that grows with both terms.

When a run fails it prints no round at all, so it dies inside round 0.

## Serialising the connections moves the cost but does not fix the hang

`WARMUP=1` opens every connection first with an 8-byte put and a `shmem_quiet()` after
each, then runs the same exchange. On the one attempt in five that completed:

```
warmup 6.269 s
round 0 done (0.098 s)      <- was 6.193 s
COMPLETED in 0.383 s
```

The 6.2 s moved wholesale into the warmup and round 0 got **63x faster**, which confirms
the attribution. But the completion rate did not improve: 1/5 with warmup against 0/5
without, and the failing runs now hang inside the warmup instead. So the problem is not
too many connections being opened at once. It is that opening roughly 8,000 connections
per node on this provider and NIC is itself unreliable.

Shared receive contexts (`FI_OFI_RXM_USE_SRX=1`) gave 0/5 alone and 0/5 with warmup.

`max_qp` on irdma0 is **899,068**, so this is not a hard QP count limit.

| configuration | 16 PEs/node | 64 PEs/node |
|---|---|---|
| baseline | 3/5 | 0/5 |
| warmup | 2/5 | 1/5 |
| SRX | — | 0/5 |
| SRX + warmup | — | 0/5 |

## The connectionless provider exists, and is blocked by two separate things

The requirement asks for a fabric with **connectionless semantics**. `verbs;ofi_rxm` is
connection-oriented, and connection setup is exactly what fails. H4D does expose a
connectionless provider:

```
provider: verbs;ofi_rxd
    domain: irdma0-dgram
    protocol: FI_PROTO_RXD
    addr_format: FI_ADDR_IB_UD
    caps: FI_MSG, FI_RMA, FI_TAGGED, FI_ATOMIC, ..., FI_RMA_EVENT, ...
```

It advertises **FI_ATOMIC and FI_RMA_EVENT**, which SOS asks for at
`transport_ofi.c:1463-1467`. `verbs;ofi_rxm`, the one that works, advertises neither.

**Blocker 1 — `data_progress`.** SOS gets `-61 ENODATA` on rxd. Bisecting SOS's hint set
one field at a time isolates it to a single field: rxd refuses
`FI_PROGRESS_AUTO` and accepts `FI_PROGRESS_MANUAL`. Everything else SOS sets
(`FI_RMA|FI_ATOMIC`, basic MR, `FI_RM_ENABLED`, `mr_key_size=1`, `FI_THREAD_DOMAIN`,
`FI_TRANSMIT_COMPLETE`, `inject_size=16`) is accepted. Confirmed identical across nine
requested API versions from 1.5 to 2.6, so the libfabric API version SOS asks for is not
involved.

Rebuilding SOS with `--enable-ofi-manual-progress` clears this. Verified from inside
SOS by instrumenting `query_for_fabric` to dump its own hints:

```
[ISX_HINT_DUMP] prov=verbs;ofi_rxd caps=0x14 mr_mode=0x70 data_progress=2 ...
```

`data_progress=2` is `FI_PROGRESS_MANUAL`, and `fi_getinfo` now succeeds.

**Blocker 2 — address vector insertion.** Past `fi_getinfo`, rxd fails at startup:

```
WARN:  transport_ofi.c:1290: populate_av
       av insert failed
ERROR: init.c:492: shmem_internal_heap_postinit
       Transport startup failed (4)
```

rxd's `addr_format` is `FI_ADDR_IB_UD`, not the `FI_SOCKADDR_IN` that rxm uses. SOS
exchanges raw addresses over PMI and inserts them into the AV, and that insert fails.
Not yet isolated further.

## What this means for the study

On H4D, an OpenSHMEM runtime can have RDMA one-sided Get/Put/Atomics **or**
connectionless semantics, not both:

| provider | one-sided RMA + atomics to SOS | connectionless | usable |
|---|---|---|---|
| `verbs;ofi_rxm` | yes | **no**, RC connections | yes, but fails above ~32 PEs/node |
| `verbs;ofi_rxd` | yes on paper, more caps than rxm | **yes**, UD | no, `av insert failed` |

The requirement asks for both at once. That is a fabric-and-stack finding rather than a
tuning problem, and it belongs in the Failure Analysis deliverable.

## A correction to the earlier record

`oshcc` from a SOS install embeds an RPATH to its own `lib`, and **RPATH takes precedence
over `LD_LIBRARY_PATH`**. Every run in this session that was labelled "manual progress"
before the `LD_PRELOAD` fix was silently executing the original auto-progress
`libsma.so`, confirmed with `ldd`:

```
libsma.so.0 => /home/.../isx/lib/libsma.so.0     <- the auto-progress build
```

Two consequences. The manual-progress-on-rxm numbers collected before that point
(16 PEs 3/5, 64 PEs 2/5) are baseline numbers, not manual-progress numbers. And the
earlier claim recorded in `HYPOTHESES_instability.md` and the reproducer header, that
`--enable-ofi-manual-progress` is a regression at 0/3, is **not supported** — that test
was very likely running the auto-progress library too. It is withdrawn rather than
reversed; manual progress on rxm has not been measured cleanly.

Use `LD_PRELOAD=<prefix>/lib/libsma.so.0` to force the intended runtime, and check with
`ldd` before trusting any A/B between two SOS builds.

## Still open

- Why the AV insert fails for `FI_ADDR_IB_UD`. If it is a fixed-size address buffer in
  SOS's PMI exchange, it may be a small patch, and it would deliver a connectionless
  OpenSHMEM that both matches the requirement and avoids the failing resource.
- Whether manual progress on rxm helps, now that it can actually be tested.
- Whether the connection failure is specific to this node pair. Still blocked: 500 vCPU
  is a hard self-service ceiling in all 18 H4D regions and every H4D shape is 192 vCPU,
  so 2 nodes is the maximum here without an approved escalation.

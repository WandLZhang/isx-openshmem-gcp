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

**Blocker 2 — the link-local GID. Found and fixed.** Past `fi_getinfo`, rxd failed at
startup with `populate_av: av insert failed`. Instrumenting the bulk `fi_av_insert` to
retry one address at a time showed each PE inserting **its own** address and failing on
the peer's:

```
[ISX_AV_DUMP] bulk fi_av_insert returned 1, expected 2, addrlen=32
  me=0 pe=0 insert->1  fe 80 00 00 00 00 00 00 40 01 c0 ff fe a8 40 04 ...  <-- SELF
  me=0 pe=1 insert->0  fe 80 00 00 00 00 00 00 40 01 c0 ff fe a8 40 02 ...
  me=1 pe=0 insert->0  fe 80 00 00 00 00 00 00 40 01 c0 ff fe a8 40 04 ...
  me=1 pe=1 insert->1  fe 80 00 00 00 00 00 00 40 01 c0 ff fe a8 40 02 ...  <-- SELF
```

The addresses are exchanged correctly: PE 0 is `fe80::4001:c0ff:fea8:4004` and PE 1 is
`...:4002`, matching each node's `GID[0]`. So SOS's PMI exchange is fine. The problem is
which GID. This NIC is `link_layer: Ethernet`, RoCE v2, and offers two:

```
GID[0]: fe80::4001:c0ff:fea8:4004, RoCE v2     <- link-local, the libfabric default
GID[1]: ::ffff:192.168.64.4,       RoCE v2     <- routable IPv4-mapped
```

A link-local GID has no route, so the UD path cannot build an address handle for a remote
peer, while the local one needs no resolution. Setting **`FI_VERBS_GID_IDX=1`** selects
the routable GID and the AV insert succeeds.

This is not specific to SOS. libfabric's own `fi_pingpong` reproduces it exactly:

```
$ fi_pingpong -p "verbs;ofi_rxd" -e rdm          # default GID index 0
[error] util/pingpong.c:1580: fi_av_insert: number of addresses inserted = 0;
                              number of addresses given = 1

$ FI_VERBS_GID_IDX=1 fi_pingpong -p "verbs;ofi_rxd" -e rdm
  1m   10  =10   20m   0.00s   5521.73 MB/s   189.90 us
```

`verbs;ofi_rxm` is unaffected by the GID index because it connects through `rdma_cm`,
which resolves over IP. For a two-line upstream report: on a RoCE v2 NIC whose GID 0 is
link-local, `verbs;ofi_rxd` is unusable at its default setting, using libfabric's own
test binary.

**Blocker 3 — RMA over rxd. Still open.** With manual progress and the routable GID, rxd
initialises and the provider demonstrably moves data (5,521 MB/s on `fi_pingpong`, about
68% of rxm's 8,154 MB/s at 1 MB). But SOS's one-sided path over rxd exhausts the retry
limit at **2 PEs**, where rxm is untroubled:

```
ERROR: transport_ofi.h:596: try_again
       Operation retry limit exceeded (1073741824)
```

So the remaining failure is narrowed to RMA and atomics on rxd, not to the provider's
basic operation. rxd emulates RMA in software over datagrams, so this is the most likely
place for it to be incomplete on this hardware.

## The fix: use the routable GID on rxm too

The link-local GID turned out not to be only an rxd problem. `FI_VERBS_GID_IDX=1` was
never set for `verbs;ofi_rxm` in any earlier test. With `LD_PRELOAD` making a clean A/B
possible for the first time, seven attempts per cell:

| configuration | 32 PEs/node | 64 PEs/node |
|---|---|---|
| auto progress (baseline) | 4/7 (57%) | 2/7 (29%) |
| **auto progress + `FI_VERBS_GID_IDX=1`** | **6/7 (86%)** | **5/7 (71%)** |
| manual progress | 0/7 | — |
| manual progress + `FI_VERBS_GID_IDX=1` | 0/7 | — |

**This is the first change in ten attempts that has improved stability.** It roughly
doubles the completion rate at 64 PEs/node and takes 32 PEs/node from 57% to 86%.

The mechanism follows from the root cause. rxm connects through `rdma_cm`, and address
resolution is part of establishing each connection. A link-local GID has no route, so
resolution is more work and less deterministic per connection, and at 8,192 connections
per node that amplifies. The routable GID makes each resolution cheap. Connection
establishment is where the failure lives, so it is where a fix should show up.

It also confirms the diagnosis independently: a change that only touches address
resolution should do nothing at all if the problem were in the data path.

Not a complete fix, and it does **not** carry over to the benchmark.

An earlier attempt at this measurement reported 0/5 for both arms. That was wrong:
`isx64win` takes `<keys_per_pe> [iters] <log>` and the log argument was missing, so every
process exited on a usage error and the benchmark never ran. Corrected, with the
benchmark actually executing 1,073,741,824 keys per run at 32 PEs/node:

| | validated |
|---|---|
| baseline | 2/5 |
| `FI_VERBS_GID_IDX=1` | 2/5 |

So the reproducer improves (4/7 → 6/7 at 32 PEs/node, 2/7 → 5/7 at 64) while ISx64 does
not move at all. The two differ in
several ways: ISx64 uses an 8 GB symmetric heap against the reproducer's 1 GB, issues
`shmem_atomic_fetch_add` per destination, runs many more operations per exchange, and adds
key generation and a local sort. Testing those one at a time, with `FI_VERBS_GID_IDX=1` set throughout, identifies the
ingredient as **operation count**:

| symmetric heap | rounds | puts per PE | completions |
|---|---:|---:|---|
| 1 GB | 4 | 512 | 3/5 |
| 1 GB | 32 | 4,096 | 2/5 |
| 1 GB | 128 | **16,384** (what ISx64 issues) | **1/5** |
| 8 GB | 4 | 512 | 3/5 |

Reliability decays monotonically with the number of operations, and at ISx64's operation
count the reproducer drops to 1/5, next to the benchmark's 0/5. Symmetric heap size makes
no difference at all: 8 GB and 1 GB both give 3/5 at the same operation count.

So there is no separate mechanism in ISx64. It fails because it issues 32x more puts, and
whatever the per-operation hazard is, it accumulates. Note this is not a simple
independent per-put failure: 0.6 success at 512 puts would give an unmeasurably small
number at 16,384 if each put were an independent trial, so the hazard is attached to
something coarser than a single operation. Since connections are established once in
round 0 and reused, a candidate worth testing next is whether connections are being torn
down and re-established during a long run.

Manual progress is now cleanly measured as a hard regression (0/7), which reinstates the
earlier claim that had to be withdrawn for being untestable.

## Why the hazard grows with operation count: CM progress is starved by data

Two rxm defaults explain it:

```
FI_OFI_RXM_CM_PROGRESS_INTERVAL   10000 us   between connection-management progress calls
FI_OFI_RXM_CQ_EQ_FAIRNESS         128        data CQ entries read consecutively before
                                             checking whether CM progress is due
```

Connection management only progresses during `fi_cq_read`, and only after either 10 ms
has elapsed or 128 data completions have been consumed. Under a put storm the data CQ is
never empty, so CM progress is starved in proportion to how much data is in flight. That
is the missing link: connections are established once and reused, yet more operations
still means more failures, because operations are what starve the connection manager.

Tested at ISx64's operation count (`ROUNDS=128`, 16,384 puts per PE), 64 PEs/node, with
`FI_VERBS_GID_IDX=1` throughout:

| setting | completions |
|---|---|
| baseline | 1/5 |
| `FI_OFI_RXM_CM_PROGRESS_INTERVAL=1000` | 1/5 |
| `FI_OFI_RXM_CM_PROGRESS_INTERVAL=100` | 1/5 |
| **`FI_OFI_RXM_CQ_EQ_FAIRNESS=1`** | **3/5** |

Lowering the time interval does nothing, which makes sense: the interval is not what is
binding when the CQ always has work. Forcing a CM-progress check after **every** data
completion triples the completion rate at the operation count where the benchmark lives.

This is the second fix found, and unlike the GID setting it was predicted from the
mechanism before it was measured, which is a reasonable check on the diagnosis.

**Verified on the real ISx64, and it does not transfer either.** Five runs per cell,
1,073,741,824 keys:

| | 32 PEs/node | 64 PEs/node |
|---|---|---|
| baseline | 3/5 | 1/5 |
| `FI_VERBS_GID_IDX=1` + `FI_OFI_RXM_CQ_EQ_FAIRNESS=1` | 3/5 | 1/5 |

So both settings that help the reproducer do nothing for the benchmark. Matching the
operation count was not sufficient to make the reproducer model ISx64. What still differs:
ISx64 issues a `shmem_atomic_fetch_add` per destination, its puts are variable-sized
rather than a fixed 256 KB, and it interleaves key generation and a radix sort between
exchanges. One of those, not raw operation count, is what pins the benchmark.

Note also the run-to-run spread at n=5: the same 32 PEs/node baseline measured 2/5 in one
job and 3/5 in another. Cells differing by one count should not be read as signal, and the
2-node results in general need larger samples than this session had time for.

## What this means for the study

On H4D, an OpenSHMEM runtime can have RDMA one-sided Get/Put/Atomics **or**
connectionless semantics, not both:

| provider | one-sided RMA + atomics to SOS | connectionless | usable |
|---|---|---|---|
| `verbs;ofi_rxm` | yes | **no**, RC connections | yes, but fails above ~32 PEs/node |
| `verbs;ofi_rxd` | messaging yes, RMA no | **yes**, UD | no, RMA hits the retry limit at 2 PEs |

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
earlier claim that `--enable-ofi-manual-progress` is a regression had to be withdrawn as
untestable. Re-measured properly with `LD_PRELOAD`, it is **0/7 at 32 PEs/node**, so the
original claim was right; it just had no valid evidence behind it at the time.

Use `LD_PRELOAD=<prefix>/lib/libsma.so.0` to force the intended runtime, and check with
`ldd` before trusting any A/B between two SOS builds.

## Still open

- Why RMA specifically fails over rxd when messaging works. Blockers 1 and 2 are solved,
  so this is the last thing between the study and a connectionless OpenSHMEM that both
  matches the fabric requirement and removes the connection setup that breaks rxm.
- Whether manual progress on rxm helps, now that it can actually be tested.
- Whether the connection failure is specific to this node pair. Still blocked: 500 vCPU
  is a hard self-service ceiling in all 18 H4D regions and every H4D shape is 192 vCPU,
  so 2 nodes is the maximum here without an approved escalation.

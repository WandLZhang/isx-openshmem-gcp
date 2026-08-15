# Tasks against the success criteria

Status as of 2026-08-15. Criteria are from `GOAL.md`. Each is scored on what has been
**measured**, not what is expected to work.

Owner column: **me** = doable with the current cluster and no approvals. **Willis** =
needs a decision or a conversation. **External** = needs someone outside the team.

---

## Goal 1 — Endpoints: ≥ 4,096

**Measured: 64.** Blocked, and the blocker is not technical.

| task | owner | blocker |
|---|---|---|
| Request `CPUS_PER_VM_FAMILY` H4D = **24,576**, single region | Willis → capacity escalation | self-service override is capped at 500 and refuses; `COMMON_QUOTA_CONSUMER_OVERRIDE_TOO_HIGH` |
| Confirm 128 nodes of H4D are actually **free in one zone** | Willis → DWS/capacity owners | quota is permission, capacity is inventory; us-central1-a and -b were both exhausted while quota was fine |
| Re-run the endpoint sweep at 128 nodes once granted | me | waits on the two above |

Arithmetic: 4,096 endpoints ÷ 32 PEs/node (the measured wall) = **128 nodes** = 24,576
vCPU. Cloud RDMA cannot cross zones, so it must be one zone. Largest H4D pool observed
anywhere is comfortably above this, so 128 is physically plausible.

**Note the dependency:** if the 32 PEs/node wall is lifted (Goal 3), the node count needed
drops sharply. At 192 PEs/node it would be 22 nodes and the quota ask nearly disappears.
Goal 3 is therefore worth solving *before* the quota request is sized.

---

## Goal 2 — Storage: > 1 PB sorted in memory

**Measured: 51.2 GB** (windowed) / 2.0 GB (stock). No longer architecturally blocked.

| task | owner | blocker |
|---|---|---|
| Cheap reductions: recv slack 1.3→1.02, in-place bucketize, in-place MSD radix | me | none |
| Streamed exchange: shrink `send` as `recv` grows, so they never both hold a full copy | me | none, but it is the only real design work left here |
| Scale-test on whatever node count Goal 1 delivers | me | Goal 1 |

The symmetric-heap ceiling that made this look impossible is **solved in software**:
`src/isx64/isx64_win.c` holds the heap at 4.2 MB/PE while the dataset grows 381x, with
flat throughput and passing validation.

What remains is arithmetic. Read off the allocations in `isx64_win.c` rather than
estimated, peak resident is the largest of three overlaps:

| overlap | multiple of the key array |
|---|---:|
| `keys` + `send` (lines 224, 242) | 2.00x |
| `send` + `recv` at slack 1.3 (242, 253) | **2.30x** |
| `recv` + radix scratch (253, 88) | 2.30x |

So the implementation is **2.3x**, and the three cheap reductions take it to **2.02x**,
not to 1.02x. They cannot do better, because `send` and `recv` both hold a full copy at
the same time and no amount of in-place work removes that. Going below 2x needs the
streamed exchange: release `send` incrementally as the exchange drains it while `recv`
fills, which holds the sum near one copy plus slack.

Per node, on 1,488 GB with the 4,096-PE window heap and OS taking about 48 GB:

| footprint | keys/node | nodes for 1 PB |
|---|---:|---:|
| 2.30x as implemented | 626 GB | 1,597 |
| 2.02x after the cheap reductions | 713 GB | 1,403 |
| ~1.15x streamed | 1,252 GB | **799** |

**Realistic near-term target is not 1 PB.** At the 128 nodes Goal 1 asks for, even the
streamed footprint gives about 160 TB. 1 PB needs roughly 800 nodes in one zone even
after all the memory work, which is a much larger capacity conversation and should be
framed as Phase 3, not Phase 2.

---

## Goal 3 — Reproducibility: consistent across runs

**Measured: 29% baseline, 71% with `FI_VERBS_GID_IDX=1`** at 64 PEs/node. Still the real
blocker, but it moved for the first time.

**Root cause found**, in `results/ROOTCAUSE_connection_establishment.md`: it is
`ofi_rxm` connection establishment, not the data path. Round 0 costs 47x a steady-state
round and scales as connections-per-node; failing runs die inside round 0.

| task | owner | blocker |
|---|---|---|
| Isolate the rxd `av insert failed` — if SOS's PMI address exchange assumes a fixed address size, `FI_ADDR_IB_UD` may need a small patch | me | none, this is the highest-value item left |
| File the livelock upstream with the reproducer | me, **needs Willis's ok to post publicly** | outward-facing |
| ~~Re-measure manual progress~~ | done | 0/7 at 32 PEs/node, a hard regression |
| Push past 71%: try `FI_VERBS_GID_IDX=1` combined with the staggered warmup, and at 128+ PEs/node | me | none |
| Fix libfabric logging so the CM loop can be seen | me | emits nothing even with `--enable-debug` |
| Test on a different node pair | me | Goal 1 (needs >2 nodes) |
| Raise with the H4D / Cloud RDMA product team | Willis | — |

Nine fixes attempted, in `results/FIXES_ATTEMPTED_instability.md`. None fixed it. The
reproducer is stripped to puts alone, no sort and no atomics, and still fails, so this is
not an ISx problem.

**The connectionless alternative is blocked too.** `verbs;ofi_rxd` exists on this NIC and
advertises more of what SOS wants than rxm does, but refuses `FI_PROGRESS_AUTO` at
`fi_getinfo` and then fails at `fi_av_insert` once that is fixed. So on H4D an OpenSHMEM
runtime gets one-sided RMA with atomics **or** the connectionless semantics the
requirement asks for, not both.

**Until this is fixed, the other two goals are not worth spending capacity on.** A
configuration that completes 30% of the time on 2 nodes completes approximately never on
128, because the failure probability compounds.

---

## Criteria already met

| criterion | evidence |
|---|---|
| OpenSHMEM PGAS, no MPI | SOS on OFI/verbs, no `libmpi` in `libsma.so`, launcher is `srun --mpi=pmi2` |
| RDMA one-sided Get/Put/Atomics | cross-node `shmem_put` verified, target posts no receive; `InRdmaWrites` non-zero in NIC counters |
| ~~Connectionless semantics~~ | **not met.** The working provider is RC-connected. The connectionless one cannot complete SOS startup. |
| Correctness validation | PASSED at 16, 32, 64 PEs — but each was a single sample against a 30% success rate |
| Performance stability / inflection points | three identified and quantified |

---

## Deliverables

| # | deliverable | state | remaining |
|---|---|---|---|
| 1 | Source code + runtime modifications | **done** | — |
| 2 | Provisioning recipe | **done, exercised** | — |
| 3 | Execution artifacts | **partial** | byte-level bandwidth: `ip4InOctets`/`ip4OutOctets` do not track RoCE on irdma0; needs netdev stats or a different counter set |
| 4 | Architectural narrative | **done** | refresh with the windowed result and the scale-precedent finding |
| 5 | Failure analysis | **strong** | this is the best deliverable in the package |

---

## The conversation that should happen before any of the above

**No published ISx run exceeds ~96 PEs or ~26 GB**, on any machine, including
purpose-built supercomputers. The requirement is 43x the endpoints and ~39,000x the data.
The ISx authors state in their own README that it "is not a benchmark" and "has not been
optimized for the features of any particular system."

Our 64-PE result **equals the largest published academic run** (NERSC Cori, 64 PEs, 2
nodes).

| audience | message |
|---|---|
| **Willis's coworker** | H4D is compliance-viable and livelocked past 32 PEs/node. Recommend Phase 2 on a4x/GB200. |
| **The customer** | The scale precedent. Phase 2 as written has no precedent anywhere; propose renegotiating the target or making "first run at this scale" the deliverable. |
| **Capacity escalation** (`the internal capacity escalation path`) | H4D 24,576 vCPU **and** an a4x reservation |
| **DWS owners** (`raquibur@`, `kelvinp@`) | a4x in us-east4-a; a4x is absent from the DWS dashboard |
| **H4D / Cloud RDMA PM** | Three product gaps: no `libfabric-devel` matching the shipped 1.22.0; docs require libfabric 2.2.0 that no repo provides; `verbs;ofi_rxm` advertises `FI_DELIVERY_COMPLETE` it does not implement. Plus the livelock. |
| **libfabric / SOS upstream** | Reproducer. Both prior issues here (#5601, #6720) were closed stale. |

---

## Recommended sequence

1. **Have the scale-precedent conversation.** It changes what Phase 2 should even be, and
   it is free.
2. **Fix or escalate Goal 3.** Nothing else is worth capacity until runs are repeatable.
3. **Do the Goal 2 footprint work.** Cheap, no approvals, and it is the difference between
   1,776 nodes and 697.
4. **Then request quota**, sized by whatever Goal 3 turns out to allow per node.
5. **Stand up a4x in parallel** as the Phase 2 path, since it shares none of the failing
   code and matches the customer's original NVSHMEM intent.

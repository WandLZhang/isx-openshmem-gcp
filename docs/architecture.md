# Architectural narrative

Deliverable 4. Design decisions, the tradeoffs between scale and performance, and where
topology-aware programming does and does not help on Google Cloud.

Every number here is measured. Sources are in `results/`.

---

## 1. The question, restated

Can a public cloud treat thousands of independent nodes as one globally addressable
memory, well enough to run a capability-class PGAS workload? ISx is the proxy: a bucket
sort with no spatial locality, whose cost is dominated by an irregular all-to-all rather
than by arithmetic.

Two paths were built. An OpenSHMEM path on H4D with Cloud RDMA, which is the compliance
answer. And a JAX path on TPU using `jax.lax.all_to_all`, which is not OpenSHMEM and is
offered as the throughput comparison the study asked for.

## 2. Machine family selection

The study names NVSHMEM on H200 over Quantum-2 InfiniBand. That configuration does not
exist on Google Cloud, for two independent reasons: there is no InfiniBand, only RoCE over
Titanium, and H200 supply is effectively zero (single-digit to low-hundreds of chips in the best zone on
2026-07-07). The requirement as written is unbuildable here at any budget.

Selecting on what the fabric must do rather than on the named hardware:

| family | RDMA | memory/node | why it was or was not chosen |
|---|---|---:|---|
| **h4d-highmem-192** | Cloud RDMA (Falcon, iRDMA) | 1,488 GB | **chosen.** Only CPU family with RDMA, and the only one obtainable without a capacity request |
| a4x (GB200) | RoCE | 960 GB + 186 GB HBM/GPU | largest pool on both axes; needs a reservation |
| a3-ultra (H200) | RoCE | 3,072 GB | supply is gone |
| x4, m4 | none | up to 32 TB | fails the fabric requirement outright |
| TPU v7x | ICI, not RDMA | 192 GB HBM/chip | no PGAS; compiler collectives only |

The honest framing is **h4d is the compliance path and a4x is the scale path**.

### Measuring capacity

Three separate times, a capacity dashboard disagreed with a real request. The most
instructive: internal capacity reporting showed us-central1-b well under-subscribed, and four VMs could not
be obtained. On-demand in us-east1-b granted immediately while DWS Flex in that **same
zone** reported exhausted, because they are separate inventory.

What did predict obtainability was the **ratio of schedulable to total hosts** rather than the absolute pool size.
Zones above 90% granted; zones at 46-55% did not. Absolute pool size predicted nothing,
and picking us-central1-b on absolute size cost roughly two hours.

Cloud RDMA is also **zone-locked**: the Falcon network profile is per zone, so a retry in
another zone means rebuilding the RDMA VPC, the nodeset, the controller and the login node.
And the Falcon zone list is not the H4D zone list: us-west4-c is the second-largest H4D
pool in the fleet and has no Falcon profile, so it cannot do Cloud RDMA at all.

## 3. The 64-bit port

Upstream ISx declares `typedef int KEY_TYPE` over a 2^28 key space. The header warns only
that the SHMEM API calls must change. Three deeper problems:

**Counter width.** Every bucket size, offset and index is `int`. At the target scale each
PE holds about 1.75e10 keys, which overflows a signed 32-bit counter roughly 8x, silently.
Sizes go negative, the prefix scan wraps, and the program still reports success.

**The local sort.** `count_local_keys()` histograms into an array of `BUCKET_WIDTH` ints.
This is a counting sort. It is tractable only because the key space is small. At `MAX_KEY = 2^60`
over a few thousand buckets that array would be about 1.4 PB per PE. Counting sort is a
property of a small key space, not of bucket sort, so the local phase became an LSD radix
sort: linear in received keys, independent of key space. This also matches what the study
specifies ("bitonic or radix sort").

**The symmetric buffer.** Upstream hardcodes a 268-million-key receive buffer. Because the
write is one-sided, overrunning it corrupts a *peer's* heap rather than raising an error,
and the symptom appears later on a different rank. ISx64 sizes it from keys-per-PE and
bounds-checks every offset before the put.

Correctness is provable without a cluster: `tests/shmem_stub.h` implements the handful of
OpenSHMEM calls for one PE, so the sort, the counter widths and the verification all run
on a laptop. That caught real bugs before any spend.

## 4. Two platform-specific modifications to the OpenSHMEM runtime

Both are required. Neither is documented.

**`--enable-ofi-mr=basic`.** SOS defaults to scalable memory registration (`mr_mode = 0`).
The H4D provider rejects it and `fi_getinfo` returns no data.

**`--enable-hard-polling`.** SOS enables target counters unless this is set, which adds
`FI_RMA_EVENT` to the hints. `verbs;ofi_rxm` on `irdma0` supports neither `FI_RMA_EVENT`
nor `FI_FENCE`, so `shmem_init()` aborts with `Transport init failed (-61)`. Hard polling
trades provider-side completion counters for CPU polling. This is a performance decision
forced by the fabric, and it is a plausible contributor to the PEs-per-node wall below.

libfabric also has to be built from source. The HPC VM image ships 1.22.0 from the
parallelstore repo with no matching `-devel` in any repo, and the only `-devel` offered
(1.18.0) conflicts with `mercury`. There is no supported way to compile against the system
libfabric.

## 5. Results and the two walls

Cross-node one-sided RDMA works. Two PEs on two physical nodes, each writing into the
other's symmetric heap with no matching receive on the target. No `libmpi` in the runtime;
the launcher is `srun --mpi=pmi2`, which is Slurm's PMI.

ISx64 weak scaling, 4.19M keys per PE, 2 nodes:

| PEs | TTS | rate | all2all | radix | all2all share |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.088 s | 6.08 GB/s | 0.021 s | 0.031 s | 24% |
| 32 | 0.125 s | 8.58 GB/s | 0.044 s | 0.034 s | 35% |
| 64 | 0.226 s | 9.52 GB/s | 0.116 s | 0.041 s | **51%** |

The exchange roughly doubles per PE doubling while the local sort stays flat. That is
correct weak-scaling behaviour. By 64 PEs the benchmark measures the fabric rather than
the CPU, which is the regime it exists to measure. It reaches that regime at the same
point as the first wall.

### Wall 1: 32 PEs per node

Past it, every configuration dies with `Operation retry limit exceeded (1073741824)`. The
wall tracks PEs **per node**, not total, which points at contention for the single
200 Gbps IRDMA vNIC and its completion resources. `SHMEM_OFI_STX_AUTO=1` and
`SHMEM_OFI_STX_MAX=8` do not move it.

Consequence: reaching 4,096 endpoints needs **128 nodes**, not the 22 that 192 PEs per node
would have allowed.

### Wall 2: ~32 GB of symmetric heap per node

Aggregate, not per PE: one PE with 32 GB works, eight PEs with 4 GB each works, and every
64 GB-per-node arrangement fails. Cause is basic MR pinning and registering the whole heap
at `shmem_init()`. Huge pages do not lift it.

Consequence: about 26.7 GB of sortable keys per node, so 1 PB needs roughly **37,500
nodes**, which is far beyond any single-zone H4D pool.

### Interaction between the two walls

1 PB wants a large heap per node. 4,096 endpoints wants many PEs per node. At the measured
limits, the largest configuration satisfying the endpoint requirement sorts about
**3.4 TB** — 0.3% of the target. Neither limit is a tuning parameter.

## 6. The TPU path

TPUs do handle 64-bit integers, despite documentation stating `int32` only. Measured on
4 v5e chips: `uint64` is real on device, 2^63-1 survives a round trip, `jax.lax.sort`
returns ordered output at a 1.48x penalty against int32, and `all_to_all` moves uint64 over
ICI at 175 GB/s.

The constraint is structural. ISx buckets are statistically
even but never exactly even. PGAS does not care: a PE puts whatever it has. A collective
needs an agreed shape before the compiler can emit it, so the exchange must either pad
every bucket or use `ragged_all_to_all`. **That gap is the real difference between the two
paradigms**, and it is measurable rather than rhetorical.

The first implementation verifies correct at 16.7M keys on 4 chips but reports only
0.20 GB/s end to end, against 175 GB/s for the exchange alone on the same hardware. The
network is under 1% of the time; the padded path costs two full sorts per iteration. That
number is an untuned local phase, not a fabric measurement, and `results/` says so.

TPU is ultimately bounded by capacity, not arithmetic: 192 GiB per chip and a maximum slice
of 9,216 chips gives 1.9 PB of HBM, and ISx needs about 2.5x the key array resident.

## 7. Topology-aware programming

It helps in three places, all of which were necessary here:

- **Placement.** Compact placement and single-zone allocation are mandatory, because Cloud
  RDMA does not cross zones. There is no topology-aware fallback for a stocked-out zone.
- **PE-to-node mapping.** With the exchange dominating past 64 PEs and a hard wall at 32
  PEs per node, how ranks map onto NICs is the dominant tunable. 4,096 endpoints on 128
  nodes at 32 PEs each is the only shape that satisfies the endpoint requirement.
- **Rotating the all-to-all start offset**, so all PEs do not target rank 0 first. Retained
  from upstream.

It cannot help with either wall. Both are properties of the NIC and its registration
budget, not of how work is arranged across a topology. This is the substantive difference
from a purpose-built HPC fabric, where per-node injection and registration capacity are
provisioned for exactly this access pattern.

## 8. Recommendation

For the study as specified, on H4D, the measured result is negative. The fabric does
genuine one-sided RDMA and the benchmark verifies correct across nodes, but 1 PB across
4,096 endpoints is out of reach by roughly two orders of magnitude, for reasons that are
fabric properties rather than configuration.

Three things would change that answer, in order of value:

1. **Redesign the exchange to stream through a small symmetric window** rather than sizing
   the window to the dataset. This decouples dataset size from the heap ceiling and is the
   only route to a petabyte on this hardware. It is a real departure from stock ISx, so it
   is the customer's call.
2. **Evaluate a4x (GB200)**, which has both the endpoint count and the memory, and where
   NVSHMEM is native rather than ported. It needs a reservation.
3. **Characterise the jitter** before any result is called reproducible. 8 PEs passed
   standalone and failed inside a sweep. Reproducibility is a stated success criterion and
   is currently unmet.

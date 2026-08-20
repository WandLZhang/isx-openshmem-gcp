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

h4d is the compliance path and a4x is the scale path.

### Measuring capacity

Three separate times, a capacity dashboard disagreed with a real request. The most
instructive: internal capacity reporting showed us-central1-b well under-subscribed, and four VMs could not
be obtained. On-demand in us-east1-b granted immediately while DWS Flex in that **same
zone** reported exhausted, because they are separate inventory.

What predicted obtainability was the **ratio of schedulable to total hosts**. Absolute pool size did not.
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

## 5. Results, and two walls that turned out to be provider-specific

Both walls below were measured on `verbs;ofi_rxm`. Both are gone on PSM3, and knowing that
they belonged to the provider rather than to H4D is the study's main transport finding.

### Wall 1: 32 PEs per node

On `ofi_rxm`, every configuration past 32 PEs per node died with
`Operation retry limit exceeded (1073741824)`. The wall tracked PEs **per node** rather
than total. `SHMEM_OFI_STX_AUTO=1` and `SHMEM_OFI_STX_MAX=8` did not move it.

Cause is connection establishment. `ofi_rxm` opens a connection per peer, so setup cost
grows with `PEs_per_node x total_PEs` and stalls near 8,192 connections per node.

**On PSM3 the wall does not exist.** 192 PEs per node, one per vCPU, validates 20/20.
Round-0 cost grows 3.1x from 16 to 64 PEs per node against 16.8x on `ofi_rxm`, so the
superlinear term is gone. 4,096 endpoints costs 22 nodes.

### Wall 2: about 32 GB of symmetric heap per node

Aggregate, not per PE: one PE with 32 GB works, eight PEs with 4 GB each works, and every
64 GB-per-node arrangement fails. Cause is basic MR pinning and registering the whole heap
at `shmem_init()`. Huge pages do not lift it.

Stock ISx puts the receive buffer in the symmetric heap, which capped the dataset at about
26.7 GB of keys per node and put 1 PB at roughly 37,500 nodes.

**The windowed exchange removes it.** `src/cpu/isx64_win.c` keeps one window slot per PE in
the heap and the dataset in ordinary memory, so the heap is about 0.5 MB per PE and does
not grow with the data. Node count then follows ordinary memory: 713 GB of keys per node,
1,403 nodes for 1 PB.

### What is left after both

Two `h4d-highmem-192` sorted **1,400 GB** at **5.10 GB/s**, 384 PEs, every run validated.
That is 98% of what two nodes hold at the measured 2.02x footprint.

| | `verbs;ofi_rxm` | `psm3` |
|---|---:|---:|
| Working density | 32 PEs/node | 192 PEs/node |
| Largest validated | 8.59 GB | 1,400 GB |
| Rate | 1.50 GB/s | 5.10 GB/s |

The exchange is 84-88% of runtime throughout, so the benchmark measures the fabric rather
than the CPU, which is the regime it exists to measure.

What remains is capacity. 1 PB needs 1,403 nodes in one zone and no zone holds that many
unallocated.

## 6. The TPU path

TPUs do handle 64-bit integers, despite documentation stating `int32` only. Measured on
4 v5e chips: `uint64` is real on device, 2^63-1 survives a round trip, `jax.lax.sort`
returns ordered output at a 1.48x penalty against int32, and `all_to_all` moves uint64 over
ICI at 175 GB/s.

The constraint is structural. ISx buckets are statistically even but never perfectly even.
A PE under PGAS puts whatever it has. A collective needs an agreed shape before the
compiler can emit it, so the exchange must either pad every bucket or use
`ragged_all_to_all`. Padding costs bandwidth on the padding.

Measured on 4 v6e chips: the ICI exchange runs at 200 GB/s and takes 0.1% of the step,
while local bucketing takes 97.4%. End to end the sort reports 0.25 GB/s. The collective
is not the cost; the padded local path is.

TPU is bounded by slice size rather than by supply. A job cannot span slices over ICI. v6e
tops out at 256 chips and 8.2 TB. TPU7x reaches 9,216 chips and 1.77 PB of HBM, against the
2.02 PB that 1 PB of keys needs resident. No grant moves either number.

## 7. Topology-aware programming

It helps in three places:

- **Placement.** Compact placement and single-zone allocation are mandatory, because Cloud
  RDMA and NVLink both stop at the zone boundary. There is no topology-aware fallback for a
  stocked-out zone. Cluster Director publishes the hierarchy to Slurm and GKE: a sub-block
  is one rack behind a single top-of-rack switch, a block is sub-blocks on non-blocking
  fabric, a cluster is blocks. A reservation is what makes it visible.
- **PE-to-node mapping.** The exchange dominates past 64 PEs, so how ranks map onto NICs is
  the dominant tunable. On PSM3 the answer is one PE per vCPU, which puts 4,096 endpoints on
  22 nodes.
- **Rotating the all-to-all start offset**, so all PEs do not target rank 0 first. Retained
  from upstream.

It would not have helped with either wall in section 5. Both were properties of the
provider's connection and registration model rather than of how work is arranged across a
topology. Changing the provider fixed them; rearranging ranks would not have.

Where it will matter next is GB300. NVLink spans 72 GPUs, so at 4,104 endpoints 56 of every
57 bytes in the all-to-all cross RoCE. Making the bucket assignment domain-aware keeps most
keys inside the domain that generated them, and that is a change to the routing prefix in
`compute_dest` rather than to the exchange.

## 8. Recommendation

H4D on PSM3 meets every criterion the hardware can reach. Two nodes sorted 1,400 GB at
5.10 GB/s with 20/20 validation at full density. 4,096 endpoints costs 22 nodes and 100 TB
costs 140, both quota rather than physics.

1 PB on H4D is out of reach. It needs 1,403 nodes in one zone, more than any zone holds
unallocated, and that is supply rather than a fabric property.

Three things follow, in order of value:

1. **Ask for 22 H4D nodes.** It clears the 4,096-endpoint criterion outright, sorts 15.7 TB
   while doing it, and costs 4,224 vCPU. Everything needed to run it is in `deploy/h4d`.
2. **Fix multi-node NVSHMEM on Cloud RoCE.** GB300 is the only family where 1 PB and 4,096
   endpoints land together, at 1,026 nodes. `ibdevx` already registers GPU memory through
   dmabuf and then segfaults on the first cross-node put, so this is one bug rather than a
   redesign. An hour on two nodes settles it. See `results/gpu-nvshmem.md`.
3. **Take per-packet adaptive routing back to the customer.** No Google fabric sprays per
   packet, by design, because RoCE treats reordering as loss. The requirement describes
   Ultra Ethernet. That is a conversation rather than engineering.

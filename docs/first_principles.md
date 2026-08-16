# Sorting a petabyte in memory on Google Cloud, from first principles

This document ignores what was available and asks what the workload needs, then checks
what can supply it. H4D appears near the end as one option among several, which is where
it belongs.

## 1. What the workload actually requires

ISx is a bucket sort. Strip the benchmark framing and three requirements remain.

**Memory.** Every key is resident. Nothing streams to disk. The dataset is 1 PB, and a
bucket sort needs the input, a destination-ordered copy and a receive buffer. Careful
implementation gets the peak to about **2x the dataset**, and this repository's port
measures 2.02x. So the machine must hold **2 PB simultaneously**.

**A global address space with one-sided access.** Each process writes into a peer's memory
without the peer participating. This is the PGAS model, and it is what distinguishes the
workload from a message-passing sort. The fabric must carry remote writes and atomics.

**Bisection bandwidth.** The shuffle moves essentially the whole dataset across the
network exactly once. Roughly 1 PB crosses the fabric.

Everything else in the brief follows from these three.

## 2. What that implies numerically

### Memory sets the machine count

2 PB of addressable memory is the binding constraint, and it is not close. Per-node
capacity decides how many nodes:

| memory per node | nodes for 2 PB |
|---|---:|
| 1.5 TB | 1,340 |
| 2 TB | 1,000 |
| 4 TB | 500 |
| 8 TB | 250 |

Nothing in a cloud catalogue makes this a small cluster. The lower bound is hundreds of
machines, and that is before considering whether they can be obtained in one zone.

### Bandwidth sets the run time

1 PB across the fabric. At an aggregate bisection bandwidth of B:

| aggregate bisection | shuffle time |
|---|---:|
| 100 GB/s | 2.8 hours |
| 1 TB/s | 17 minutes |
| 10 TB/s | 100 seconds |
| 100 TB/s | 10 seconds |

Per node, a 200 Gbps link contributes about 12.5 GB/s of usable bandwidth after protocol
overhead, and this repository measured 4.44 GB/s per node on an all-to-all, which is 36%
of that. A thousand nodes at 4.44 GB/s is 4.4 TB/s aggregate, so the shuffle takes about
four minutes.

**The run is short.** That matters more than it first appears, and section 6 returns to
it.

### The two requirements pull in opposite directions

Memory-dense machines are not bandwidth-dense. A 32 TB memory-optimised instance has no
RDMA fabric at all. A GPU node with a fast fabric has a fraction of the memory per dollar.
Any real answer trades one against the other.

## 3. Where 2 PB of fast memory can live on Google Cloud

Four categories exist, and only two can host this workload.

**Accelerator HBM.** Highest bandwidth, lowest capacity per device, and the fabric is
built in. Sorting inside HBM is the fastest option per byte and needs the most devices.

**Grace or host DRAM attached to accelerators.** On GB200 systems the CPU memory is
coherent with the GPU and sits on the same NVLink domain. This roughly triples capacity
per node against HBM alone, at lower bandwidth.

**CPU nodes with an RDMA NIC.** Highest capacity per node among fabric-attached options.
On Google Cloud this is H4D, at 1,488 GB and 200 Gbps.

**Memory-optimised CPU nodes.** Up to tens of terabytes per node. **These are disqualified**,
because they have no RDMA fabric. A PGAS workload cannot run on them, and the endpoint
count would be in the dozens.

## 4. The fabric question, which is where the model is decided

One-sided RMA has to be implemented by something. Three mechanisms exist and they are not
equivalent.

**NVLink inside a coherent domain.** On an NVL72 rack, 72 GPUs share a hardware-coherent
address space. A remote write is a memory operation, not a network operation. NVSHMEM
implements OpenSHMEM semantics directly on it. Nothing in the libfabric or verbs stack is
involved.

**RDMA over a network, GPU to GPU.** Beyond one rack, NVSHMEM falls back to RoCE through
the network adapters. Still one-sided, an order of magnitude slower than NVLink, and it
reintroduces the network transport stack.

**RDMA over a network, CPU to CPU.** OpenSHMEM over libfabric over verbs over RoCE. Four
layers, each with its own failure modes. This repository's failure analysis is entirely
about defects in that stack, not about the hardware underneath it.

**The number of layers between the application and the wire predicts how much breaks.**
The measured evidence here supports that: connection establishment in libfabric's
`verbs;ofi_rxm` provider fails above a few thousand connections per node, and the
connectionless provider that would avoid it cannot complete startup. Neither defect exists
in a coherent NVLink domain, because there is no connection to establish.

## 5. Reading that against the brief

The requirements document names `shmem_put64` **or** `nvshmem_put64_nblock`. NVSHMEM is
explicitly in scope, which removes the only reason to prefer a CPU stack.

| requirement | CPU + RoCE | GPU + NVSHMEM |
|---|---|---|
| One-sided put, get, atomics | yes, through four layers | yes, natively |
| Connectionless | no, and the connectionless provider is unusable | inside a rack there are no connections |
| Per-packet adaptive routing | no, the transport uses subflows | inside a rack, no routing |
| Memory per endpoint | 1,488 GB per node | HBM plus coherent CPU memory |
| Endpoints per node | as many processes as cores | one per GPU |

Three fabric requirements that the CPU path fails are answered by moving the workload
inside a coherent domain, because the requirements describe problems that only exist on a
network.

## 6. The conclusion the arithmetic points to

**Sort inside NVLink domains, and treat the cross-rack network as the thing to avoid
rather than the thing to benchmark.**

An NVL72 rack is a hardware-coherent 72-GPU address space. This is the closest thing any
cloud offers to the premise in the brief, which is thousands of nodes behaving as one
globally addressable memory. Within it, one-sided RMA is a memory operation and the entire
class of defect documented in this repository does not arise.

Scaling beyond one rack reintroduces the network, so the design question becomes how much
of the shuffle can be kept inside racks. For a bucket sort with a deterministic
destination, this is answerable: **make the bucket assignment rack-aware.** Assign key
ranges so most keys land in the same rack they were generated in, and only the residue
crosses the network. This is a change to one function, the routing prefix, and it converts
a flat all-to-all into a hierarchical one.

That is the topology-aware programming the brief asks about in Deliverable 4, and it is
the single highest-leverage change available.

### What this costs

The run is minutes, not hours. That reframes several things:

- **Checkpointing is not worth building.** Writing 1 PB takes longer than the sort.
- **Retry is the correct failure strategy**, since a lost run costs minutes.
- **The bill is dominated by how long the machines are held, not by the sort.** Hundreds
  of nodes for a fifteen-minute run is cheap. Hundreds of nodes waiting for the slowest to
  provision is not.

### What remains genuinely hard

Obtaining hundreds of accelerator nodes in a single zone. This is not a technical problem
and no engineering removes it. It is the same constraint whichever machine family is
chosen, and it should be the first conversation rather than the last.

## 7. What this means for the study as run

H4D was a reasonable thing to try with what was available, and it produced a clear result:
the CPU plus RoCE plus libfabric path has defects that stop it well before the target
scale, and those defects are now documented upstream with reproducers.

Read from first principles, it was also the hardest of the available paths, because it
puts the most layers between the application and the wire and it answers none of the three
fabric requirements. A GPU path with NVSHMEM answers all three by construction, and the
brief permits it.

The recommendation is to change machine family rather than to continue debugging the
transport.

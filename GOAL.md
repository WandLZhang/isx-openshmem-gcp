# ISx Petascale Sort Benchmark — Requirements

The customer requirements this repository is built against, recorded verbatim so the
implementation can be checked against them rather than against a paraphrase.

## Environment
- **Programming model:** OpenSHMEM-compliant one-sided communication (PGAS).
- **Prohibition:** MPI-based implementations are **not** responsive to this study.
- **Network fabric:** Must support RDMA one-sided operations (Get/Put/Atomics), connectionless semantics, and per-packet adaptive routing.
- **Scale:** Provisioned for **≥ 4,096 network endpoints**, where an endpoint = one TPU chip, HPC compute engine, or GPU.

## Workload
- **Source code:** OpenSHMEM-compliant sorting kernels from established public repositories (e.g., OpenSHMEM.org or Sandia National Lab benchmarks), such as <https://github.com/ParRes/ISx>.
- **Data generation:** Must generate a dataset exceeding **1 PB**, sorted entirely in-memory.

## Success Criteria
| Criterion | Requirement |
| --- | --- |
| **Correctness** | Successful completion with all required validation checks |
| **Reproducibility** | Consistent results across multiple runs using the same configuration |
| **Scale** | Demonstrated sorting of >1 PB of data across 4,096+ endpoints |
| **Performance Stability** | Identification of inflection points where performance degrades or jitter emerges |
| **Operational Plausibility** | The setup must represent a configuration realistically usable in production |

## Required Deliverables
1. **Source Code** — the full implementation, including any platform-specific modifications to the OpenSHMEM runtimes.
2. **Infrastructure Provisioning Recipe** — a step-by-step automated recipe (e.g., Terraform or deployment scripts) letting the Government reproduce the environment without intervention.
3. **Execution Artifacts** — detailed logs, performance measurements (Time to Solution), and telemetry on network utilization and memory bandwidth.
4. **Architectural Narrative** — a report explaining key design decisions, tradeoffs between scale and performance, and the role of topology-aware programming.
5. **Failure Analysis** — if the benchmark fails or degrades at scale, a characterization of the dominant technical blockers (e.g., fabric congestion or software stack immaturity).

---

## Background: What ISx Measures
The **Integer Sort eXtreme (ISx)** benchmark evaluates **large-scale, non-local, high-concurrency memory and interconnect performance**. Unlike compute-heavy workloads (HPL/LINPACK) or dense tensor operations (GEMM), it stresses the **unstructured communication bisection bandwidth** and **Partitioned Global Address Space (PGAS)** performance of supercomputing fabrics.

ISx is a distributed parallel bucket sort on 64-bit unsigned integers (`uint64_t`), operating in three distinct phases.

### Phase 1: Deterministic Parallel Key Generation
- Each Processing Element (PE) generates a subset of 64-bit keys using a pseudo-random number generator (PRNG).
- **Bucket Routing Prefix:** The generator injects a target bucket index into the most significant bits (MSBs) of each key:

  Prefix = TargetPEID ≪ (64 − log₂(TotalPEs))

- This forces keys to be evenly distributed across all PEs while ensuring every key has a single, deterministic destination rank across the cluster.

### Phase 2: Global Unstructured Shuffle (The Stress Test)
- Every PE partitions its local keys into *N* buckets (where *N* = TotalPEs).
- PEs use **one-sided RDMA operations** (`shmem_put64` or `nvshmem_put64_nblock`) to push bucket *i* directly into the symmetric heap memory of remote PE *i*.
- This generates an **irregular, high-concurrency All-to-All network exchange** that saturates the bisection bandwidth and adaptive routing capability of the interconnect.

### Phase 3: Local Bucket Sorting & Boundary Validation
- Each PE sorts its received keys using an in-memory sorting algorithm (e.g., bitonic or radix sort).
- **Validation:** PEs verify global order by checking that the maximum key on PE *i* is less than or equal to the minimum key on PE *i+1*:

  max(Keys_PE i) ≤ min(Keys_PE i+1)

---

## Status, 2026-08-20

Measured on two `h4d-highmem-192`, eight H200 in one node, and four TPU v6e chips.

| requirement | status |
|---|---|
| OpenSHMEM PGAS, no MPI | **met.** SOS on OFI/PSM3, no `libmpi` in `libsma.so` |
| RDMA one-sided Get/Put/Atomics | **met.** Verified cross-node; the target posts no receive |
| Connectionless fabric | **met in substance.** PSM3 implements no connection management |
| Per-packet adaptive routing | **not met, and not achievable.** Falcon uses multipath subflows by design, because per-packet routing reorders and RoCE treats reordering as loss. Zero out-of-order arrivals measured |
| Correctness | **met.** 1,400 GB on H4D, 137.4 GB on 8 H200, 12.9 GB on 4 v6e chips |
| Reproducibility | **met.** 20/20 at 192 PEs per node on PSM3 |
| Performance stability | **met.** H4D 5.10 GB/s, GPU 67.15 GB/s, TPU 0.25 GB/s, each flat across its range |
| >= 4,096 endpoints | **not run, reachable.** 22 H4D nodes at 192 PEs, 1,026 GB300 nodes, or a 16x16x16 TPU v5p slice |
| > 1 PB in memory | **not run, reachable on GB300 only.** 1,026 nodes. H4D needs 1,403 in one zone, more than any zone holds free. TPU misses by 6%, 1,900 TB of HBM in the largest TPU7x topology against 2,020 TB resident |
| Deliverable 1, source | **met.** CPU, GPU and TPU implementations, plus `deploy/h4d/sos-psm3-stx.patch` |
| Deliverable 2, provisioning recipe | **met and exercised.** `deploy/h4d`, plus `deploy/gb300` as a handoff package |
| Deliverable 3, execution artifacts | **met.** Time to solution and phase breakdown throughout, raw logs in `results/raw`. No byte counter on H4D tracks RoCE, so bandwidth is derived |
| Deliverable 4, architectural narrative | **met.** `the README` |
| Deliverable 5, failure analysis | **met.** `results/`, and `the README` for what no grant can fix |

## What reaching the target requires

**4,096 endpoints:** 22 H4D nodes at 192 PEs per node, which also hold 15.7 TB. Below the
192-node H4D record and a 4,224 vCPU quota ask.

**1 PB with 4,096 endpoints:** 1,026 `a4x-maxgpu-4g-metal`, 57 NVL72 domains, 4,104 GPUs.
The only configuration where both land together. `deploy/gb300` is ready to run, and
multi-node NVSHMEM on Cloud RoCE has to be fixed first.

**Per-packet adaptive routing** is the one requirement no Google fabric satisfies. It
describes Ultra Ethernet. This is a conversation with the customer rather than
engineering.

## Scope note

The full-scale demonstration is not in this repository. Reaching it needs a capacity grant
that has not been made. Everything else is measured. `the README` says what a team
with capacity does next, and `results/scale-out.md` carries the arithmetic.

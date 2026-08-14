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

## Status against these criteria

Tracked honestly. See `results/` for the underlying measurements.

| requirement | status |
|---|---|
| OpenSHMEM PGAS, no MPI | **met.** SOS 1.5.3 on OFI/verbs, no `libmpi` in `libsma.so`, launcher is `srun --mpi=pmi2` |
| RDMA one-sided Get/Put/Atomics | **met.** Cross-node `shmem_put` verified, target posts no receive. `FI_RMA_EVENT` and `FI_FENCE` unsupported; both have configure workarounds |
| >= 4,096 endpoints | **not met.** 64 PEs on 2 nodes. Capped at 32 PEs/node by an OFI retry wall |
| > 1 PB in-memory | **not met, and blocked.** 2.00 GB largest verified run. Symmetric heap caps at ~2 GB/PE, see `results/BLOCKER_symmetric_heap_20260814.md` |
| Correctness validation | **met at small scale.** PASSED at 16, 32 and 64 PEs cross-node |
| Reproducibility | **not met.** 8 PEs passed standalone and failed in-sweep; jitter uncharacterised |
| Performance stability / inflection points | **met.** Three identified: 32 PEs/node ceiling, all2all crossover at 64 PEs, low-PE jitter |
| Deliverable 1, source code | **met.** ISx64 uint64 port, `src/isx64`, plus the two SOS build flags H4D requires |
| Deliverable 2, provisioning recipe | **met and exercised.** `infra/h4d`, deployed end to end in a clean project |
| Deliverable 3, execution artifacts | **partial.** TTS and phase breakdown captured; no network or memory-bandwidth telemetry |
| Deliverable 4, architectural narrative | **not started** |
| Deliverable 5, failure analysis | **in progress.** `results/`, including one retracted claim |

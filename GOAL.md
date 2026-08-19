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

## Scope change, 2026-08-15

The Scale criterion was renegotiated during the work. The requirement above is recorded
verbatim and unchanged; this section records what was agreed instead.

`CPUS_PER_VM_FAMILY` for H4D is 500 in every region that offers it, which is two nodes,
and self-service override is refused at every increment from 1,000 to 5,000. All three
H4D shapes are 192 vCPU, so no smaller machine works around it. Demonstrating 4,096
endpoints needs 128 nodes and 1 PB needs roughly 800, neither of which is reachable
without an approved capacity escalation.

The agreed substitute is a handoff: a plan the HPC team can execute when capacity is
granted, precise enough that nobody re-derives the arithmetic. That is `docs/scale-out.md`,
and the escalation itself is drafted. Scale is therefore tracked below as descoped rather
than as failed, and the demonstration remains outstanding.

## Status against these criteria

Tracked honestly. See `results/` for the underlying measurements.

| requirement | status |
|---|---|
| OpenSHMEM PGAS, no MPI | **met.** SOS 1.5.3 on OFI/verbs, no `libmpi` in `libsma.so`, launcher is `srun --mpi=pmi2` |
| RDMA one-sided Get/Put/Atomics | **met.** Cross-node `shmem_put` verified, target posts no receive |
| Per-packet adaptive routing | **not met.** Falcon uses multipath subflows by design, not packet spraying; the verbs layer sees plain RoCE v2 with DCQCN and no path-selection interface; zero out-of-order packets measured across all traffic. Two nodes cannot separate "no AR" from "no AR needed". `results/adaptive-routing.md` |
| Connectionless fabric semantics | **not met.** `verbs;ofi_rxm` uses reliable connections. `verbs;ofi_rxd` is connectionless; two of its three blockers were solved, and its RMA path still reaches the retry limit at 2 PEs. `results/rxm-connection-limit.md` |
| >= 4,096 endpoints | **descoped to a plan.** 128 PEs demonstrated. Needs 128 nodes and 24,576 vCPU. `docs/scale-out.md`, `results/h4d-capacity.md` |
| > 1 PB in-memory | **descoped to a plan.** 17 GB largest verified run. The heap ceiling is solved in software; 1 PB needs about 800 nodes after the memory work. `docs/scale-out.md` |
| Correctness validation | **met when runs complete.** PASSED at 4, 64 and 128 PEs, largest 2,147,483,648 keys. Failures are transport hangs, not wrong answers |
| Reproducibility | **not met, quantified.** 35-45% at 64 PEs/node over 20 runs per arm. Root cause identified as `ofi_rxm` connection establishment; ten fixes measured, none moved the benchmark |
| Operational plausibility | **not met.** Two independent reasons. 35-45% unattended completion, mean 1.6 runs to first failure. And `onHostMaintenance: TERMINATE` with no live migration and no checkpoint in ISx, so a maintenance event on any one of N nodes discards the whole run. Fixing the transport does not fix the second. `results/operations.md` |
| Performance stability / inflection points | **met.** Three identified: 32 PEs/node ceiling, all2all crossover at 64 PEs, low-PE jitter |
| Deliverable 1, source code | **met.** uint64 port with three exchange schedules, plus the SOS build flags H4D requires |
| Deliverable 2, provisioning recipe | **met and exercised.** `infra/h4d`, deployed end to end in a clean project |
| Deliverable 3, execution artifacts | **met, with a documented limitation.** No byte counter on H4D tracks RoCE; bandwidth is derived from payload and wall time. `results/telemetry.md` |
| Deliverable 4, architectural narrative | **met.** `docs/architecture.md` |
| Deliverable 5, failure analysis | **met.** `results/rxm-connection-limit.md`, with reproducers and two upstream issues filed. Includes three retracted claims and why each was wrong |

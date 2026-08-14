# Dominant blocker: the stack succeeds about 30% of the time at a fixed configuration

Measured 2026-08-14, two `h4d-highmem-192-lssd` in `us-east1-b`. Ten identical invocations
of ISx64: 32 PEs (16 per node), 4,194,304 keys per PE, 2 GB symmetric heap, same binary,
same nodes, same environment, back to back in one Slurm allocation.

## The measurement

| run | result |
|---:|---|
| 1 | **FAIL** |
| 2 | PASS, TTS 0.213 s, all2all 0.041 s |
| 3 | **FAIL** |
| 4 | **FAIL** |
| 5 | **FAIL** |
| 6 | **FAIL** |
| 7 | PASS, TTS 0.214 s, all2all 0.043 s |
| 8-10 | not reached before the job's time limit |

**2 of 7 completed runs passed.** Roughly a 30% success rate at a configuration that
passed cleanly when sampled once during the scaling sweep.

When it does complete, timing is tight: 0.213 s against 0.214 s, and all2all 0.041 s
against 0.043 s. The instability is binary — the run either completes with a repeatable
time or it does not complete at all.

## What this invalidates

`isx64_h4d_scaling_20260814.md` reports 16, 32 and 64 PEs as PASSED. Each of those was a
**single sample**. At a ~30% success rate, a single PASS is a lucky draw, not evidence that
the configuration works. The scaling table should be read as "this configuration can
sometimes complete, and when it does, here is the time" rather than as a performance
characterisation.

By the same token, the failures previously attributed to specific thresholds need
re-examination. The 8-PE failure inside the sweep, and the 4 x 8 GB anomaly in the heap
ceiling test, are both consistent with this background failure rate rather than with a
limit at those particular settings. The 64-PEs-per-node and 64-GB-per-node walls are still
real, because they failed at every attempt, but the exact boundaries are softer than a
single-sample table implies.

## Against the success criteria

The study requires "consistent results across multiple runs using the same configuration".
This is the criterion the result speaks to most directly, and it is **not met**. It is
arguably the most consequential finding of the H4D path: a fabric that completes a
petabyte sort 30% of the time is not a production configuration, and "Operational
Plausibility" is also a stated criterion.

## Cause, and what is not yet known

Not established. The failures present as the same `Operation retry limit exceeded
(1073741824)` seen at the PEs-per-node wall, which means SOS spun 2^30 times in
`try_again` waiting for a completion that never arrived.

Candidates, none yet tested:

1. **Hard polling.** `--enable-hard-polling` was forced on because the provider does not
   support `FI_RMA_EVENT`. It replaces provider-side completion counters with CPU polling,
   and it is the most plausible source of a completion-progress race. The fabric leaves no
   choice about the flag, so if this is the cause it is a genuine incompatibility rather
   than a tuning error.
2. **Completion queue depth** under bursty all-to-all traffic.
3. **Node-to-node variation.** All runs used the same two nodes, so a single degraded NIC
   would produce exactly this pattern. Re-running on a different node pair would separate
   a fabric-wide property from a bad host.

### Node health has been checked and is clean

Google's own `irdma_health_check`, installed as a Slurm prolog by the Cluster Toolkit
blueprint, passes on both nodes:

```
isxh4de1-h4dnodeset-0 exit=0   Loopback test PASSED. RDMA health checks passed.
isxh4de1-h4dnodeset-1 exit=0   Loopback test PASSED. RDMA health checks passed.
```

The `irdma0` port error counters are all zero on both nodes.

That weakens candidate 3 considerably: these are not visibly degraded machines. It does
not eliminate it, because the health check is a single-pair loopback test rather than a
multi-PE stress test, and the failures only appear under concurrent all-to-all load. But
the simple explanation is now unlikely, which leaves hard polling as the leading
candidate.

Testing a different node pair is still worth doing and needs a cluster larger than two
nodes to do properly.

## Reproducing

```bash
export PATH=$HOME/isx/bin:$PATH LD_LIBRARY_PATH=$HOME/isx/lib:$LD_LIBRARY_PATH
export SHMEM_OFI_PROVIDER="verbs;ofi_rxm" SHMEM_SYMMETRIC_SIZE=2G
for i in $(seq 1 10); do
  srun -N2 --ntasks-per-node=16 --mpi=pmi2 --export=ALL ./isx64bin 4194304 1 /dev/null \
    | grep -E "time to solution|verification"
done
```

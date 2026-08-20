# Operational plausibility

The brief asks whether "the setup must represent a configuration realistically usable in
production". That is not directly measurable, so it is broken into six tests an operator
would apply.

## A. Does the provisioning recipe reproduce from cold? PASS

`deploy/h4d/` was run end to end in a clean project with no manual intervention, producing
a working Slurm cluster with Cloud RDMA. The Government reproduction requirement in
Deliverable 2 is satisfied by demonstration rather than by assertion.

Two caveats an operator inherits:

- The recipe resets org policies. It detects which constraints actually block the build
  and resets only those (`00_setup_project.sh`), but it still needs
  `orgpolicy.policy.set`, which in a hardened org requires an admin binding.
- One reset is unavoidable: `compute.trustedImageProjects`, because the HPC VM image H4D
  requires lives in `cloud-hpc-image-public`, which default allowlists omit.

## B. Unattended completion rate: PASS

On PSM3, 20 consecutive unattended runs at each shape.

| configuration | completions |
|---|---|
| 64 PEs/node, 128 PEs | **20/20** |
| 192 PEs/node, 384 PEs | **20/20** |

A production scheduler submitting this job would see it complete. Retry is still worth
wrapping for the maintenance case below, not for transport instability.

## C. Mean runs before first failure

No failures in 40 unattended runs, so the rate is below what 40 samples can resolve. A
failure discards the whole sort, since ISx has no intermediate state to resume from. At
this run length that argues for retry rather than checkpointing; see the arithmetic at the
end.

## D. Failure mode: PASS. Clean, bounded, and the node survives

Ten runs at 64 PEs/node with the timeout removed:

| outcome | count | duration |
|---|---:|---|
| self-aborted, retry limit | 9 | 235-244 s, mean 237 s |
| PASSED | 1 | 40 s |

The failure path is well behaved for an operator. It always terminates, it never wedges,
and the duration is tight: 235 to 244 seconds across nine failures, which is the 2^30
retry budget draining at a near-constant rate. Nothing needed a human.

The node is undamaged. A 4-PE run immediately after the nine failures passed in 0.010 s,
and both NICs still reported one active port.

Two consequences for the wrapper in `deploy/h4d/run_isx64.sh`:

- A timeout above 244 s is pointless, since the application aborts itself first. The
  wrapper's 300 s therefore acts as a backstop. The application's own abort does the work.
- **237 seconds per failed attempt is the real operational cost.** At a 60-75% failure
  rate, five attempts average about 15 minutes of allocation burned before a success.
  This is what makes the pre-flight check worth having: it fails in seconds instead.

The single passing run took 40 s, where earlier passing runs at this size took 3.5 to
4.4 s. Time to solution is not stable even on success.

## E. Survives host maintenance: FAIL, by configuration

```
onHostMaintenance: TERMINATE
automaticRestart:  true
provisioningModel: STANDARD
```

H4D cannot live-migrate. A host maintenance event terminates the VM. `automaticRestart`
brings the machine back but the job is gone, and ISx holds everything in memory with no
checkpoint.

The structural problem is that this compounds with scale and duration. A run needs every
one of its N nodes to avoid a maintenance event for the whole run. At the target shape of
800 nodes that is 800 independent opportunities to lose the job, and the run has to
complete inside the gap between maintenance events on the unluckiest node in the fleet.

This is independent of the transport instability and would remain after it is fixed. The
remedy is not checkpointing, for the reason given at the end of this document: the
checkpoint would cost more than the run. It is retry, plus keeping the run short.

## F. Stack maintainability: marginal

An operator has to carry several things the vendor documentation does not mention:

- **libfabric must be built from source.** The HPC VM image ships libfabric 1.22.0 from
  the parallelstore repo, and no matching `libfabric-devel` exists in any enabled repo.
  The only one offered is 1.18.0 from powertools, and installing it conflicts with the
  1.22.0 already present. Documented inline in `01_build_sos.sh`.
- **SOS needs two non-default configure flags** on this fabric,
  `--enable-ofi-mr=basic` and `--enable-hard-polling`, because the provider rejects
  scalable MR and does not support `FI_RMA_EVENT`.
- **Two environment variables are load-bearing and undiscoverable.**
  `FI_VERBS_GID_IDX=1` is required for `verbs;ofi_rxd` and nothing in the failure message
  points at addressing. `FI_OFI_RXM_CQ_EQ_FAIRNESS=1` was found by reading provider
  source.
- **`oshcc` embeds an RPATH that overrides `LD_LIBRARY_PATH`.** Two SOS installs on one
  machine will silently load the wrong one. This invalidated a round of results here.

None of these is fatal. Together they mean the environment is reproducible by the recipe
but not by a reader of the vendor documentation.

## Verdict

**Not production-plausible as it stands**, for two independent reasons.

The transport instability at 35-45% is the immediate one, and it is being pursued
upstream. The maintenance exposure is the structural one: `TERMINATE` with no live
migration, and a failure surface that grows with node count. Fixing the first does not
fix the second, though the second is survivable with retry in a way the first is not.

The provisioning recipe itself is sound and reproduces from cold, so Deliverable 2 stands
regardless of this verdict.

## What would make it plausible

1. Upstream resolution of the connection establishment failure, or a transport that
   avoids it.
2. **A retry wrapper in the Slurm submission.** Cheap, and it is the correct answer to the
   maintenance exposure. See the arithmetic below.
3. A pre-flight fabric check in the prolog, so a run that is going to fail on connection
   setup fails in seconds rather than after the retry budget.

### Why not checkpointing

An earlier version of this document proposed application-level checkpointing as the fix
for section E. The arithmetic does not support it.

At the measured 4.44 GB/s per node, 1 PB across 800 nodes is 1.25 TB per node to move,
which is about 5 minutes of exchange and 10 to 15 minutes end to end. Checkpointing 1 PB
of in-memory state needs 1 PB of storage and, at a generous 1 TB/s aggregate write, about
17 minutes.

**The checkpoint costs more than the run it protects**, and it would also change what ISx
measures, since the benchmark reports time to solution for one uninterrupted sort.

The right response to a maintenance event on a 15-minute run is to rerun it. Exposure is
proportional to node count times run duration, and the run duration here is short enough
that retry dominates. That reasoning would invert for a workload running for hours, which
this one is not.

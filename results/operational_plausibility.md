# Operational plausibility

The brief asks whether "the setup must represent a configuration realistically usable in
production". That is not directly measurable, so it is broken into six tests an operator
would actually apply. Five are answered. One is measuring.

## A. Does the provisioning recipe reproduce from cold? PASS

`infra/h4d/` was run end to end in a clean project with no manual intervention, producing
a working Slurm cluster with Cloud RDMA. The Government reproduction requirement in
Deliverable 2 is satisfied by demonstration rather than by assertion.

Two caveats an operator inherits:

- The recipe resets org policies. It detects which constraints actually block the build
  and resets only those (`00_setup_project.sh`), but it still needs
  `orgpolicy.policy.set`, which in a hardened org requires an admin binding.
- One reset is unavoidable: `compute.trustedImageProjects`, because the HPC VM image H4D
  requires lives in `cloud-hpc-image-public`, which default allowlists omit.

## B. Unattended completion rate: FAIL

| configuration | completions | sample |
|---|---|---|
| 64 PEs/node, 128 PEs | 7/20 and 9/20 | n=20 per arm |
| 32 PEs/node, 64 PEs | 3/5 | n=5, treat as indicative only |

35-45% at the size that matters. A production scheduler submitting this job would see
most submissions fail.

## C. Mean runs before first failure: 1.6

At a per-run failure probability of about 0.62, the expected number of runs before the
first failure is 1/0.62, which is **1.6 runs**. Unattended operation therefore fails
inside the first two attempts on average, and a retry wrapper would need roughly three
attempts to reach 95% confidence of one success.

For a petabyte run this is worse than it looks. ISx has no checkpoint, so a failure
discards the whole sort rather than a stage of it.

## D. Failure mode: measuring

The operationally important question is whether a failed run self-aborts and frees the
allocation, or wedges and needs a human. Every measurement in this study wrapped the run
in `timeout 200`, which masked the natural failure duration. Slurm job 101 removes the
bound and records time to failure over 10 runs, plus whether the node is still usable
afterwards. Result to be appended here.

What is already known: the failure path ends in
`ERROR: transport_ofi.h:596: try_again — Operation retry limit exceeded (1073741824)`,
so it does terminate rather than hang forever. The retry budget is 2^30, and how long
that takes to exhaust is what job 101 measures.

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

This is independent of the transport instability and would remain after it is fixed.
Any production use at this scale needs application-level checkpointing, which upstream
ISx does not have and which this port did not add.

## F. Stack maintainability: marginal

An operator has to carry several things the vendor documentation does not mention:

- **libfabric must be built from source.** The HPC VM image ships libfabric 1.22.0 from
  the parallelstore repo, and no matching `libfabric-devel` exists in any enabled repo.
  The only one offered is 1.18.0 from powertools, and installing it conflicts with the
  1.22.0 already present. Documented inline in `02_build_sos.sh`.
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
upstream. The maintenance exposure is the structural one: `TERMINATE` with no
live migration, no checkpoint in the application, and a failure surface that grows with
node count. Fixing the first does not fix the second.

The provisioning recipe itself is sound and reproduces from cold, so Deliverable 2 stands
regardless of this verdict.

## What would make it plausible

1. Upstream resolution of the connection establishment failure, or a transport that
   avoids it.
2. Application-level checkpointing in ISx64, so a maintenance event costs one stage
   rather than the whole run. This is the only item here with no external dependency, and
   it is a real piece of work rather than a configuration change.
3. A retry wrapper in the Slurm submission, which is cheap and worth having regardless.

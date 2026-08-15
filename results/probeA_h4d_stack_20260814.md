# H4D software stack validation, 2026-08-14

One `h4d-highmem-192` in `us-central1-b`, image `hpc-rocky-linux-8-v20260630`, project
`wz-isx-benchmark`. Object was to de-risk the OpenSHMEM compliance path before deploying
a cluster. Node was deleted after the run.

## Result

The stack works, and the documented build recipe does not. Five things had to change.

| check | result |
|---|---|
| H4D boots with the HPC VM image | yes |
| `irdma0` device present, `idpf` + `irdma` modules loaded | yes |
| `verbs;ofi_rxm` advertises `FI_RMA` on `irdma0` | yes |
| `verbs;ofi_rxm` advertises `FI_ATOMIC` | yes |
| Sandia OpenSHMEM 1.5.3 builds | yes, against a source-built libfabric |
| SOS runtime links `libmpi` | **no**, clean |
| SOS transport | **OFI** |
| one-sided put across nodes | **not tested**, needs 2 nodes and a launcher |

## Documentation errors

**1. `onHostMaintenance` must be TERMINATE, and the error contradicts itself.**

```
Invalid value for field 'resource.scheduling.onHostMaintenance': 'TERMINATE'.
Scheduling must have onHostMaintenance be one of the following valid types:
[TERMINATE]. But was MIGRATE
```

It reports the value as TERMINATE and then says it was MIGRATE. Pass
`--maintenance-policy=TERMINATE` explicitly.

**2. There is no way to compile against the system libfabric.**

The image ships libfabric 1.22.0 from the parallelstore repo. No matching
`libfabric-devel` exists in any enabled repo. The only one offered is 1.18.0 from
powertools, and installing it fails:

```
cannot install both libfabric-1.18.0 from baseos and libfabric-1.22.0 from @System
package mercury-2.4.0 requires libfabric.so.1(FABRIC_1.7), but none of the
providers can be installed
```

So libfabric has to be built from source. This is a required platform-specific
modification, not a preference.

**3. The documented libfabric minimum is unmeetable from packages.**

Google's H4D guidance states libfabric 2.2.0 or later. The image ships 1.22.0 and the
repos offer nothing newer. Building v2.6.0 from source satisfies the requirement and
resolves point 2 at the same time. Verified that the source build still finds `irdma0`:

```
provider: verbs;ofi_rxm
    fabric: IB-0xfe80000000000000
    domain: irdma0
    version: 206.0
    protocol: FI_PROTO_RXM
```

**4. A shallow SOS clone breaks the build in a way that looks unrelated.**

`git clone --depth 1` without `--recurse-submodules` leaves `modules/tests-sos` empty and
`autogen.sh` then fails with "test submodule contents are missing", producing no
`configure` at all.

**5. A bare H4D node has no process launcher.**

`mpiexec.hydra`, `prterun`, `prun` and `srun` are all absent, and `oshrun` fails with
"could not find a launcher". SOS bootstraps over PMI-1. On the Slurm cluster the launcher
is `srun --mpi=pmi2`, which is not MPI. Installing an MPI purely to obtain `mpiexec`
would leave the data path on OFI but invites a compliance argument that is not worth
having.

## Still open

Cross-node one-sided RMA. Everything above is single-node, so it proves the runtime and
the fabric capabilities but not that a `shmem_put` actually traverses the wire into a
peer's symmetric heap. That needs two or more nodes and a launcher, which is the Slurm
cluster. It remains the single most important unproven claim in the study.

## Cost note

Spot was stocked out in us-central1-b at the time of the run, so this used on-demand. The
DWS Dedicated pool for h4d-highmem in that zone is separate inventory from Spot and is
where the real capacity sits.

## SSH timing

Per remote command against this node:

| method | time |
|---|---:|
| `gcloud compute ssh --tunnel-through-iap` | 3.5 s |
| raw ssh, external IP, fresh connection | 0.7 s |
| raw ssh + ControlMaster, external IP | 0.11 s |
| raw ssh + ControlMaster, over IAP | 0.11 s |

Connection reuse is the whole effect. Multiplexed IAP matches multiplexed external, so
there is no reason to attach public IPs or relax `compute.vmExternalIpAccess` for shell
speed. `scripts/hssh.sh` wraps this.

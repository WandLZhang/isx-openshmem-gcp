# H4D Slurm cluster deployment, 2026-08-14

Project `wz-isx-benchmark`, us-central1-b, 4 x `h4d-highmem-192-lssd`, DWS Flex.
Blueprint derived from Cluster Toolkit `examples/hpc-slurm-h4d`.

## Outcome

The cluster stands up. The compute nodes do not, because the zone is out of H4D.

```
Slurm:  isxh4d-h4dnodeset-[0-3]  idle#        (powering up, 4/4)
MIG:    size 0, target 4
DWS:    resize request "initial-resize"  state ACCEPTED  resize_by 4  duration P7D
Error:  ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS
        "Waiting for resources. Currently there are not enough resources
         available to fulfill the request."
```

This is DWS Flex behaving correctly. The request is accepted and queued for up to seven
days, and it bills nothing while it waits. It is not a configuration error and there is
nothing to fix in the blueprint.

## The capacity lesson, for the third time

Internal capacity reporting showed h4d-highmem in us-central1-b at substantially free
capacity. Read naively that is 1,280 free VMs. Four could not be obtained.

The same pattern has now appeared three times in this work:

| what the dashboard said | what happened |
|---|---|
| v6e Spot us-west1-c: 0 used of 15,548 | zone has no flex pool at all, code 3 |
| h4d Spot us-central1-b | `stockout` on a single VM |
| h4d DWS us-central1-b: reported free | `ZONE_RESOURCE_POOL_EXHAUSTED` on 4 VMs |

A capacity dashboard reports pool size. It does not report obtainability, and the two are
not close. The only reliable test is a real request.

## Cloud RDMA is zone-locked, which caps a retry strategy

The usual remedy is to change zone. That is constrained here: the Falcon network profile
is per zone, so an RDMA VPC built for `us-central1-b-vpc-falcon` cannot serve any other
zone. Moving zones means rebuilding the RDMA network, the nodeset, the controller and the
login node.

Falcon-capable zones, which is the complete set of places Cloud RDMA can exist:

```
asia-southeast1-a  europe-west4-b  europe-west4-c  us-central1-a
us-central1-b      us-central1-c   us-east1-b      us-west1-b      us-west4-a
```

Cross-referenced against internal per-zone availability, the fallback order is:

| zone | H4D schedulable | falcon profile |
|---|---:|---|
| us-central1-b | 396 | yes, currently exhausted |
| us-west4-c | 278 | **no** |
| europe-west4-c | 254 | yes |
| us-central1-a | 220 | yes |
| asia-southeast1-a | 106 | yes |
| us-east1-b | 71 | yes |
| us-west4-a | 54 | yes |

us-west4-c is the second-largest H4D pool in the fleet and has no Falcon profile, so it
cannot run Cloud RDMA at all. Machine availability and fabric availability are separate
questions and have to be checked separately.

## Deployment defects found and fixed

Four, in order.

**1. Filestore BASIC_SSD is out of stock in us-central1-b.** The official blueprint
attaches a 2,560 GB Filestore for `/home`. It failed with
`Error code 14 ... does not have enough resources available`. This study does not need a
parallel filesystem; the only shared state is the SOS build and the ISx64 binary, and the
slurm-gcp controller exports `/home` over NFS on its own. Removing the module removed a
dependency that can stock out independently of the compute we care about.

**2. `compute.requireShieldedVm` blocks the slurm-gcp controller.** The module does not
set `shielded_instance_config`, so instance creation fails with
`Constraint constraints/compute.requireShieldedVm violated ... Secure Boot is not enabled`.
Note this is a property of the Terraform module, not of the image, so checking whether
the image supports Shielded VM does not predict it.

**3. The default compute service account has no roles.**
`iam.automaticIamGrantsForDefaultServiceAccounts` is enforced, so the controller SA
carried zero bindings and the setup script died on
`403 Required 'compute.instanceTemplates.get' permission`. The startup script then
reported `Finished running startup scripts` and exited 0, so the only symptom was
`slurmctld inactive` and `sinfo` failing on a missing `cloud.conf`.

Granting explicit roles is better than resetting the org policy, because the recipe stays
reproducible in a hardened environment:

```
roles/compute.instanceAdmin.v1  roles/iam.serviceAccountUser  roles/storage.objectAdmin
roles/logging.logWriter  roles/monitoring.metricWriter  roles/pubsub.admin
```

**IAM must be granted before the first deploy.** slurm-gcp's setup script writes a
completion marker and then refuses to re-run with
`WARNING: Slurm was previously configured, quitting`. Granting roles afterwards does not
recover the node; it has to be recreated.

**4. Do not delete the deployment directory to retry.** `rm -rf` on the Terraform state
while resources exist orphans every one of them, and the next apply fails with a wall of
409 `alreadyExists`. Use `gcluster destroy`. Recovering meant hand-deleting two networks,
two subnets, three firewall rules, a router, two addresses, a disk and five buckets.

## State

The resize request is still queued and costs nothing. Running: one `c2-standard-4`
controller and one `n2-standard-4` login node, roughly $0.35/hr combined.

# CPUS_PER_VM_FAMILY caps the cluster at 2 nodes

Established 2026-08-14 in project `wz-isx-benchmark`.

`CPUS_PER_VM_FAMILY` is the binding metric, not `CPUS`.

## The measurement

Attempting to grow the Slurm nodeset from 2 to 15 nodes, `slurmctld`'s resume path fails:

```
ERROR: bulkInsert API failures: HttpError 403 ...
  "Quota 'CPUS_PER_VM_FAMILY' exceeded. Limit: 500.0 in region us-east1."
ERROR: Resetting nodes isxh4de1-h4dnodeset-[2-14] to power_down.
```

Confirmed directly:

```
us-east1     {'region': 'us-east1',    'vm_family': 'H4D'}  limit=500
us-central1  {'region': 'us-central1', 'vm_family': 'H4D'}  limit=500
```

`h4d-highmem-192` is 192 vCPU. **500 / 192 = 2.6, so 2 nodes.** This is the entire
explanation for a two-node cluster.

## The quota that binds

The obvious quota reads fine and is misleading:

| quota | us-east1 limit | nodes it implies |
|---|---:|---:|
| `CPUS` | 3,000 | 15 |
| `PREEMPTIBLE_CPUS` | 5,000 | 26 |
| **`CPUS_PER_VM_FAMILY` (H4D)** | **500** | **2** |

Earlier notes in this repository said "quota for 15 nodes on-demand" on the basis of
`CPUS = 3000`. That was wrong. For any accelerator-adjacent or specialised family, the
per-family quota binds first and the general `CPUS` quota is irrelevant.

Check it with:

```bash
gcloud alpha services quota list --service=compute.googleapis.com \
  --consumer=projects/PROJECT --filter="metric:cpus_per_vm_family"
```

## Self-service cannot raise it

```
gcloud alpha services quota update ... --value=24576 --force
  reason: COMMON_QUOTA_CONSUMER_OVERRIDE_TOO_HIGH
  metadata: max: '500'
```

The consumer override is itself capped at 500, so this needs a quota increase through
support or the internal capacity escalation path. A console click will not do it.

## Requirement for the target

Density is 192 PEs per node, one per vCPU, validated 20/20 on PSM3.

| goal | endpoints | nodes | H4D vCPU | quota needed |
|---|---:|---:|---:|---:|
| current | 384 | 2 | 384 | 500, have |
| 4,096 endpoints | 4,224 | 22 | 4,224 | **4,224** |
| 100 TB | 26,880 | 140 | 26,880 | **26,880** |
| 1 PB | 269,376 | 1,403 | 269,376 | more than any zone holds free |

All nodes must land in one zone, because Cloud RDMA cannot cross zones.

## Ordering of the blockers

1. **`CPUS_PER_VM_FAMILY` = 500** caps the cluster at 2 nodes. Needs a quota request. This
   is the only one still binding.
2. **PE density** was capped at 32 per node on `verbs;ofi_rxm`. PSM3 runs 192, so this is
   resolved.
3. **Symmetric heap** capped the dataset at about 26.7 GB of keys per node. The windowed
   exchange holds one window slot per PE instead, so this is resolved. See
   `windowed-exchange.md`.

Raising the quota to 4,224 vCPU takes the demonstration from 384 endpoints to 4,224 and
from 1.4 TB to 15.7 TB. Reaching 1 PB needs machines that no zone has free.

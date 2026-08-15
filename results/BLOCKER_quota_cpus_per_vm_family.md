# CPUS_PER_VM_FAMILY caps the cluster at 2 nodes

Established 2026-08-14 in project `wz-isx-benchmark`.

The scale target was never blocked by code, and past the first zone it was not blocked by
capacity either. It is blocked by a quota most people never look at.

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
support or the internal capacity escalation path, not a console click.

## Requirement for the target

| goal | PEs | PEs/node (measured wall) | nodes | H4D vCPU | quota needed |
|---|---:|---:|---:|---:|---:|
| current | 64 | 32 | 2 | 384 | 500 (have) |
| 4,096 endpoints | 4,096 | 32 | 128 | 24,576 | **24,576** |
| 1 PB in memory | — | — | ~37,500 | 7,200,000 | not plausible |

Reaching the endpoint requirement needs a **49x** increase in `CPUS_PER_VM_FAMILY` for
H4D, in a single region, and the nodes must all land in one zone because Cloud RDMA cannot
cross zones. Single-zone H4D pools are large enough for this, so
128 nodes is physically plausible where 37,500 is not.

## Ordering of the blockers

The three limits found in this study apply in sequence, and only the first is currently
binding:

1. **`CPUS_PER_VM_FAMILY` = 500** caps the cluster at 2 nodes. Needs a quota request.
2. **32 PEs per node** caps endpoints at 64 on those 2 nodes. Needs a fix or a workaround
   for the OFI retry-limit wall.
3. **~32 GB symmetric heap per node** caps the dataset at about 26.7 GB of keys per node.
   Needs the streaming redesign described in `BLOCKER_symmetric_heap_20260814.md`.

Raising the quota alone would take the demonstration from 64 endpoints to 4,096 and from
2 GB to roughly 3.4 TB. It would not reach 1 PB; blocker 3 governs that, and no quota
change affects it.

## Recommended ask

Request `CPUS_PER_VM_FAMILY` for **H4D = 24,576 in a single region**, sized for 128
`h4d-highmem-192` nodes in one zone. This is the smallest request that makes the 4,096-endpoint
criterion testable, and it is the request to put through the capacity escalation path.

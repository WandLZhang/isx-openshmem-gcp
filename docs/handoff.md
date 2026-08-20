# Taking this to the finish line

What this project's quota allows today, what was run against it, and the steps a team with
capacity would follow. One section per family. Where a criterion cannot be met by any
grant, that is stated with the reason.

## What today's quota buys

Measured against `wz-isx-benchmark`, 2026-08-19.

| family | quota | buys | largest run |
|---|---|---:|---|
| `h4d-highmem-192` | 500 vCPU, `CPUS_PER_VM_FAMILY` | 2 nodes | 1,400 GB, 384 PEs |
| `a3-ultragpu-8g` (H200) | 64 preemptible GPUs per region | 8 nodes | 137.4 GB, 8 GPUs, 1 node |
| `a4-highgpu-8g` (B200) | 64 preemptible GPUs per region | 8 nodes | not run |
| `a4x-highgpu-4g` (GB200) | entitled, no capacity | 0 | not run |
| `a4x-maxgpu-4g-metal` (GB300) | **accelerator not allowlisted** | 0 | not run |
| TPU v6e | 512 on-demand, 1,536 preemptible per zone | up to a full v6e-256 pod | 12.9 GB, 4 chips |

Two of these are not quota problems. GB300 fails on an accelerator allowlist before quota
is consulted. TPU has ample quota and no capacity, and its ceiling is slice size rather
than supply.

## H4D

The only path that satisfies the OpenSHMEM one-sided requirement, and the only one where
a 10% data run also clears the full endpoint count.

**Met on 2 nodes:** correctness at 1,400 GB, 20/20 reproducibility at 192 PEs per node,
throughput 5.10 GB/s and flat over the top 3.4x of the range. 192 PEs per node is one PE
per vCPU, the full shape.

**Not met:** scale. `CPUS_PER_VM_FAMILY` caps H4D at 500 vCPU, and all three H4D shapes are
192 vCPU, so self-service buys two nodes and no smaller shape exists to work around it.
Increments of 1,000 through 5,000 all return `COMMON_QUOTA_CONSUMER_OVERRIDE_TOO_HIGH`.

### Steps

1. **Get quota.** Two sizes, and the smaller one is worth asking for first.

   | goal | nodes | vCPU |
   |---|---:|---:|
   | 4,096 endpoints, 15.7 TB | 22 | 4,224 |
   | 100 TB, 26,880 endpoints | 140 | 26,880 |

   22 nodes clears the endpoint criterion outright and is a far smaller ask. Both need a
   capacity escalation, because self-service refuses every increment above 500.
2. **Confirm the zone separately.** Quota grants permission and does not reserve machines.
   us-central1-a and us-central1-b both returned `ZONE_RESOURCE_POOL_EXHAUSTED` while quota
   was sufficient.
3. **Provision.** `deploy/h4d/00_setup_project.sh`, then set `h4d_cluster_size` in
   `deploy/h4d/isx-slurm-h4d.yaml`. Cloud RDMA cannot cross zones.
4. **Build.** `deploy/h4d/01_build_sos.sh`. It applies `sos-psm3-stx.patch`, without which
   SOS aborts at `transport_ofi.c:606` on PSM3.
5. **Climb, do not jump.** 4 nodes, then 8, 16 and 32 at constant keys per node. That gives
   the weak-scaling slope the projection assumes. Everything above two nodes in this
   repository is arithmetic.
6. **20-run reproducibility set** at the largest stable shape.
7. **Target run.** 140 nodes, 192 PEs per node, 4.65e8 keys per PE. At the measured
   5.10 GB/s per node pair this projects to about 5 minutes, which is a floor.

### Feasibility

| target | verdict |
|---|---|
| 4,096 endpoints on their own | **reachable at 22 nodes**, which also holds 15.7 TB |
| 100 TB + 4,096 endpoints | **reachable.** 140 nodes for memory, and 140 x 192 PEs gives 26,880 endpoints |
| 1 PB + 4,096 endpoints | **impossible under any grant.** 1,403 nodes in one zone exceeds the unallocated pool of every zone worldwide, and a grant cannot produce machines that are already allocated. The largest H4D run on record is 192 nodes |

Density was the open question and it is now answered. 192 PEs per node validated 20/20, so
the endpoint requirement costs 22 nodes rather than the 128 that 32 PEs per node implied.
Density also bought 3.4x throughput on the same two machines, 1.50 GB/s to 5.10 GB/s.

Two launcher notes for whoever runs this outside Slurm. `oshrun` aborts with
`could not find a launcher` on a bare VM; `mpich` supplies `mpiexec.hydra`, which speaks
the PMI-1 that SOS bootstraps against, and that is process launch only. On Rocky with OS
Login, `~/.ssh/authorized_keys` is ignored, so add the launching node's key with
`gcloud compute os-login ssh-keys add`.

## TPU

Not responsive to the requirement. TPU has no PGAS and no remote put, so the exchange is a
collective. It is here for one measurement the others cannot make.

**Met on 4 chips:** correctness at 12.88 GB, repeat runs within 0.1%, throughput flat at
0.25 GB/s across a 6x range.

**Measured:** the ICI exchange runs at 200 GB/s and is 0.1% of the step. Bucketing is 97.4%.
Replacing the bucket scatter with a gather made it 9.0x faster.

### Feasibility

A job cannot span slices over ICI, so slice size is the ceiling and no grant moves it.

| | v6e | TPU7x |
|---|---|---|
| 100 TB | **impossible.** Needs 121 full v6e-256 pods | reachable, about 1,050 chips |
| 1 PB | **impossible** | **impossible.** 2.02 PB resident against 1.77 PB in the largest slice |
| 4,096 endpoints | **impossible.** Max slice is 256 chips | reachable |
| OpenSHMEM one-sided | **impossible.** No PGAS on TPU | same |

### Steps, if a TPU7x slice is wanted anyway

1. Submit on-demand queued resources and wait. Spot did not convert in 141 attempts across
   six accelerator types; on-demand queued resources landed 2 of 7.
2. `src/tpu/isx_jax.py` needs `check_vma=False` on `shard_map` from JAX 0.6, which
   `_shard_map` already handles.
3. Switch the exchange to ragged `all_to_all`. The padded buffer is held twice and is why
   the current code uses about 3.2 GB of a 32 GB chip. `isx_jax.py` already detects ragged
   support.
4. Re-measure the phase split at more than one host. The 200 GB/s figure is 4 chips sharing
   a host and says nothing about inter-host ICI.

## GB300, and the NVLink shapes that stand in for it

**Met:** the implementation is validated on 8 x H200 in one node to 137.44 GB at
67.15 GB/s, flat across a 64x data range.

**Not met, and this is the thing to fix first:** multi-node NVSHMEM does not work on Google
Cloud RoCE with the packaged build.

### What was ruled in and out

The fabric works. `ib_write_bw` moves 45.0 GB/s host to host on one of eight NICs, with all
8 HCAs `PORT_ACTIVE` at MTU 4096. Each GPU has its own RDMA NIC, `gpu0rdma0` through
`gpu7rdma0`, each on its own /32.

GPU memory registration is where it stops. On the same buffer, `tools/dmabuf_reg_test.c`:

```
ibv_reg_dmabuf_mr  on rocep145s0: SUCCESS (errno=0)
ibv_reg_mr (peermem) on rocep145s0: FAILED (errno=14)
```

`nvidia_peermem` cannot insert. The module file is present but it registers against
`ib_register_peer_memory_client`, which exists in MOFED's `ib_core` and not in the inbox
one this image ships.

Four NVSHMEM remote transports, four distinct failures:

| transport | failure |
|---|---|
| `ibrc` | registers with `ibv_reg_mr`, needs peermem. `ibv_poll_cq completion status 4, local protection error` |
| `ibgda` | maps the NIC doorbell into GPU BAR space. `cudaHostRegister with IoMemory failed, error=800`. Virtualized MRDMA does not expose it |
| `ibdevx` | registers through dmabuf and succeeds, logs `ibv_reg_dmabuf_mr handle ... mr ...`, then segfaults on the first cross-node put. Reproduced with a 25-line program, so this is the transport rather than ISx |
| `ucx` | 1.18.x built with CUDA, verbs and rdmacm. Reads the NVSHMEM heap as host memory: `failed to register address 0xa20000000 (host) ... Input/output error (md supports: host)` |

`ibdevx` is the near miss. It is the only transport that both registers GPU memory
correctly on this hardware and reaches the data path.

### Steps

1. **Settle registration on two nodes before booking anything.** An hour of work. Build the
   image, run `tools/dmabuf_reg_test.c`, then a 2-node NVSHMEM job. Ranked by effort:
   chase the `ibdevx` segfault with NVIDIA, build NVSHMEM from source against dmabuf, or
   move to a MOFED image so `nvidia_peermem` inserts and stock `ibrc` works.
2. **Get the accelerator allowlisted.** `nvidia-gb300` is absent from
   `gcloud compute accelerator-types list` on this project, so creation fails before
   capacity is consulted. `nvidia-gb200` is entitled separately, which means the two are
   granted independently.
3. **Reserve.** The family refuses Spot outright, and on-demand A4X GB200 returned
   `ZONE_RESOURCE_POOL_EXHAUSTED` in both zones tested.
4. **Request dense placement with the reservation.** `--collocation=collocated
   --gpu-topology=1x72`, one NVL72 domain per policy. Node counts must be multiples of 18,
   because A4X capacity is allocated in fixed 18-node NVLink domains.
5. **Run.** `deploy/gb300/run.sh 10` then `full`. 108 and 1,026 nodes.
6. **Watch the exchange share.** NVLink spans 72 GPUs, so at 4,104 endpoints 56/57 of the
   all-to-all crosses RoCE. If it dominates, make the bucket assignment domain-aware using
   the Cluster Director topology. That changes the routing prefix in `compute_dest` rather
   than the exchange.

### Feasibility

| target | verdict |
|---|---|
| 100 TB | reachable, 108 nodes |
| 1 PB + 4,096 endpoints | **reachable, 1,026 nodes and 4,104 GPUs.** The only family where the full target fits |

Both verdicts assume step 1 succeeds. If no NVSHMEM transport can be made to work on Cloud
RoCE, the GPU path is limited to one NVL72 domain, which is 72 GPUs and about 37 TB, and
the full target becomes unreachable on every family.

## Summary

| criterion | H4D | TPU | GB300 |
|---|---|---|---|
| Correctness | met, 1,400 GB | met | met on H200, 1 node |
| Reproducibility | met, 20/20 at full density | met, 0.1% | not run multi-node |
| Performance stability | met | met | met on 1 node |
| Scale, 100 TB | needs quota | impossible on v6e | needs allowlist and reservation |
| Scale, 1 PB | impossible | impossible | reachable |
| 4,096 endpoints | reachable at 22 nodes | impossible on v6e | reachable at 1,026 nodes |
| OpenSHMEM one-sided | met | impossible | met by NVSHMEM, unproven across nodes |
| Operational plausibility | recipe and patch shipped | scripts shipped | package shipped, one blocker open |

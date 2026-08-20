# GB300 run package

Everything needed to run ISx at scale on `a4x-maxgpu-4g-metal`. Written so a capacity team
can execute it without reading the rest of this repository.

## Read this first

One thing will stop the run and it is cheap to settle in advance.

NVSHMEM as packaged by NVIDIA cannot register GPU memory on Google Cloud RoCE NICs.
Measured on 2 x `a3-ultragpu-8g`: `ibv_reg_mr` on a device pointer fails with errno 14,
because `nvidia_peermem` will not insert against the inbox `ib_core`. `ibv_reg_dmabuf_mr`
on the same buffer succeeds, so the hardware is willing and the packaged build cannot ask
for it. Full evidence in [../../results/gpu-nvshmem.md](../../results/gpu-nvshmem.md).

Inside one NVL72 domain the transport is NVLink and this does not apply. It applies to
every byte that crosses domains, which at 4,096 GPUs is 56/57 of the exchange.

**Settle it on 2 nodes before booking 1,026.** Run `tools/dmabuf_reg_test.c` on the image
you intend to use, then a 2-node NVSHMEM job. An hour of work. Three fixes, in order of
effort: build NVSHMEM from source with dmabuf, use a MOFED image, or fall back to NCCL/gIB.

## What you are running

A distributed sort of 64-bit integers across GPUs, using NVSHMEM one-sided RMA. Each GPU
generates its own keys, sends each key to the GPU that owns its value range, sorts what
arrives, and checks the result against its neighbour. It measures interconnect behaviour
under an irregular all-to-all, not compute.

The code is `src/gpu/isx_nvshmem.cu`. Validated on 8 x H200 in one node to 137.44 GB. Never
run on GB300 and never run across nodes.

## Machine and shape

| | |
|---|---|
| Machine type | `a4x-maxgpu-4g-metal` |
| GPUs per node | 4 x NVIDIA B300 |
| HBM per GPU | 279 GB |
| Host memory | 960 GB Grace, coherent over NVLink-C2C |
| Endpoint | one GPU, so 4 per node |
| Provisioning | **on-demand or reservation. Spot is rejected** |
| Zone | single zone. NVLink and Cloud RDMA do not cross zones |

Spot is not a fallback here:

```
$ gcloud compute instances create ... --provisioning-model=SPOT
ERROR: Invalid value for field 'resource.scheduling.preemptible': 'true'.
Preemptible VMs are not supported for this VM family.
```

## Node counts

A4X capacity comes in fixed **18-node NVLink domains**, so any node count must be a
multiple of 18. 18 nodes x 4 GPUs is 72 GPUs, one NVL72 domain.

| target | domains | nodes | GPU endpoints | memory | keys sorted | keys/GPU |
|---|---:|---:|---:|---:|---:|---:|
| **Full** | 57 | **1,026** | 4,104 | 2.13 PB | 1.0 PB | 30,458,089,181 |
| **10%** | 6 | **108** | 432 | 224 TB | 100 TB | 28,935,185,185 |

The full row is sized by the 4,096-endpoint requirement; memory then has headroom. Peak
resident memory is 2.02x the key array, measured, which is where 2.13 PB against 1.0 PB
comes from.

`keys_per_gpu` is `total_keys / gpus`, where 1 PB is 1.25e14 keys and 100 TB is 1.25e13.

## Three gates, in the order you hit them

Tested 2026-08-19 against a project with no A4X entitlement. Expect the first gate before
capacity is ever discussed.

**1. Accelerator allowlist.** `a4x-maxgpu-4g-metal` appears in
`gcloud compute machine-types list` for us-central1-b and us-east4-b, which is misleading.
Creation fails on the accelerator, not the machine:

```
ERROR: Accelerator type 'nvidia-gb300' is not valid in container 'ZONE:2001/PROJECT:...'
```

`gcloud compute accelerator-types list --filter="zone:us-central1-b"` returned
`nvidia-b200`, `nvidia-gb200`, `nvidia-h100-80gb`, `nvidia-h100-mega-80gb` and
`nvidia-h200-141gb`, with no `nvidia-gb300`. The project needs the accelerator type
allowlisted first. A machine type appearing in the list does not mean you can create it.

`a4x-highgpu-4g` (GB200) passes this gate on the same project, so the two are entitled
separately.

**2. Provisioning model.** Spot is refused for the whole family, and on-demand A4X GB200
returned `ZONE_RESOURCE_POOL_EXHAUSTED` in both us-central1-b and us-east4-b. This is a
reservation, not an opportunistic grab.

**3. Capacity and placement.** Ask for **1,026** (or 108) `a4x-maxgpu-4g-metal` in one
zone, as a reservation. `us-east4-a` held the largest unallocated pool at the last check
and the full request fits inside it comfortably.

Request dense placement with the reservation, not after. A reservation is also what makes
topology visible at all.

```bash
# one NVL72 domain per policy
gcloud beta compute resource-policies create group-placement isx-nvl72 \
    --collocation=collocated --gpu-topology=1x72 --region=REGION

gcloud compute instances create ... --resource-policies=isx-nvl72
```

Under GKE, pass `--placement-policy=isx-nvl72` at node pool creation and bind to the
reservation with `--reservation-affinity=specific --reservation=NAME`. Sub-blocks can be
targeted as `NAME/reservationBlocks/BLOCK/reservationSubBlocks/SUB_BLOCK`.

Cluster Director exposes the physical hierarchy to Slurm and GKE: a sub-block is one rack
behind a single top-of-rack switch, one hop; a block is sub-blocks on non-blocking fabric,
at most two hops; a cluster is blocks. That hierarchy is what makes the rack-aware
bucketing below implementable rather than theoretical.

## Build

Any node with CUDA 12 and an NVIDIA driver. The DLVM image
`common-cu129-ubuntu-2204-nvidia-580` works for the build.

```bash
sudo apt-get install -y libnvshmem3-dev-cuda-12 libnvshmem3-static-cuda-12 \
                        libnvshmem3-cuda-12 libhwloc15 \
                        rdma-core libibverbs1 ibverbs-providers ibverbs-utils

nvcc -O3 -std=c++17 -arch=sm_100 -rdc=true \
     -I/usr/include/nvshmem_12 -I../../src/cpu \
     ../../src/gpu/isx_nvshmem.cu \
     -L/usr/lib/x86_64-linux-gnu/nvshmem/12 \
     -lnvshmem_host -lnvshmem_device -lcudart -lcurand \
     -o isx_nvshmem
```

`-arch=sm_100` targets Blackwell. Use `sm_90` for H200 or `sm_80` for A100.

Four things that cost time to find and will apply here:

- `libnvshmem3-static-cuda-12` is a separate package and supplies `libnvshmem_device.a`.
  Without it the link fails on `nvshmemi_init_thread`.
- `libhwloc15` is needed at runtime by `nvshmrun` and is not pulled in automatically.
- `-rdc=true` is required.
- `rdma-core` and `ibverbs-providers` are absent from the DLVM image. Without them NVSHMEM
  reports `Unable to dlopen libibverbs` and aborts before it reaches the fabric.

## Run

One process per GPU.

```bash
export NVSHMEM_SYMMETRIC_SIZE=32G
export NVSHMEM_IB_GID_INDEX=3          # RoCE v2 IPv4. Index 0 is RoCE v1 link-local

srun --ntasks-per-node=4 -N 1026 ./isx_nvshmem 30458089181 1     # full, 1 PB
srun --ntasks-per-node=4 -N  108 ./isx_nvshmem 28935185185 1     # 10%, 100 TB
```

Under Slurm, NVSHMEM bootstraps from PMI, so `srun` is enough. Outside Slurm use
`nvshmrun -n <total_gpus> -ppn 4 --hostfile hosts`, and every host must be reachable by
passwordless SSH.

Smoke test on one node before committing the allocation:

```bash
nvshmrun -n 4 ./isx_nvshmem 4194304 1
```

Under a second, prints `verification : PASSED`.

## What you will see

```
ISx-NVSHMEM  4104 PEs (1 per GPU)
  keys/PE       : 30458089181
  total keys    : 125000318... (1000.00 GB)
=== results ===
  time to solution    : X.XXX s
    generate  ...
    bucket    ...
    exchange  ...
    radix     ...
  aggregate rate      : XX.XX GB/s
  verification        : PASSED
```

`verification` covers three checks: every key landed in the right GPU's range, each GPU's
output is ordered, and no keys were lost globally. A `FAILED` on the last prints the count.

## Expected behaviour, measured on 8 x H200

| total | time | bucket | exchange | radix | rate |
|---:|---:|---:|---:|---:|---:|
| 2.15 GB | 0.033 s | 0.026 | 0.004 | 0.002 | 65.05 GB/s |
| 8.59 GB | 0.130 s | 0.105 | 0.015 | 0.008 | 66.19 GB/s |
| 34.36 GB | 0.513 s | 0.422 | 0.055 | 0.033 | 66.96 GB/s |
| 137.44 GB | 2.047 s | 1.686 | 0.215 | 0.131 | 67.15 GB/s |

Flat at 67 GB/s over a 64x data range, so the code weak-scales inside a node. Bucketing is
82% and the NVLink exchange is 10%.

**Two things H200 in one node cannot show.**

NVLink spans 72 GPUs. At 4,104 endpoints that is 57 domains, and 56/57 of the all-to-all
crosses RoCE instead of NVLink. Expect the exchange share to rise far above 10%. If it
dominates, make the bucket assignment domain-aware so most keys stay in the domain that
generated them. That is a change to the routing prefix in `compute_dest`, not to the
exchange, and Cluster Director supplies the topology to drive it.

Whether ConnectX-8 and MRDMA spray per packet is unknown. On H4D, Falcon uses multipath
subflows rather than per-packet spraying and zero out-of-order arrivals were measured. Do
not assume MRDMA behaves the same.

## If it fails

| symptom | cause |
|---|---|
| `undefined reference to nvshmemi_init_thread` | missing `libnvshmem3-static-cuda-12` |
| `libhwloc.so.15: cannot open shared object file` | missing `libhwloc15` |
| `Unable to dlopen libibverbs` | missing `rdma-core` and `ibverbs-providers` |
| `ibv_poll_cq completion status 5`, `progress_send failed` | GPU memory registration. See the top of this file |
| `cudaHostRegister with IoMemory failed with error=800` | IBGDA wants the NIC doorbell in GPU BAR space. Not available on MRDMA |
| connects but no data moves | `NVSHMEM_IB_GID_INDEX` unset, so RoCE v1 link-local |
| `recv overflow` | receive slack too tight. Raise `1.02` in `isx_nvshmem.cu` |
| `nvshmem_malloc failed` | raise `NVSHMEM_SYMMETRIC_SIZE` |
| hangs at init | every rank must see the same `NVSHMEM_SYMMETRIC_SIZE` |

The symmetric heap is one window slot per PE, `WINDOW_KEYS * 8` bytes, independent of PE
count. It does not grow with the job, so 32G is ample at any scale.

## Contact

Findings and the full study are in the repository root README.

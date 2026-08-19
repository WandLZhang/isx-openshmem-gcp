# GB300 run package

Everything needed to run ISx at scale on `a4x-maxgpu-4g-metal`. Written so a capacity team
can execute it without reading the rest of this repository.

## What you are running

A distributed sort of 64-bit integers across GPUs, using NVSHMEM one-sided RMA. Each GPU
generates its own keys, sends each key to the GPU that owns its value range, sorts what
arrives, and checks the result against its neighbour. It measures interconnect behaviour
under an irregular all-to-all, not compute.

The code is `src/gpu/isx_nvshmem.cu`. It is validated on 2 x A100 over NVLink and has
never run on GB300.

## Machine and shape

| | |
|---|---|
| Machine type | `a4x-maxgpu-4g-metal` |
| GPUs per node | 4 x NVIDIA B300 |
| HBM per GPU | 279 GB |
| Host memory | 960 GB Grace, coherent over NVLink-C2C |
| Endpoint | one GPU, so 4 per node |
| Zone | single zone. NVLink does not cross zones |

Two target sizes. Pick one:

| target | nodes | GPU endpoints | memory available | keys sorted |
|---|---:|---:|---:|---:|
| **Full** | 1,024 | 4,096 | 2.13 PB | 1.0 PB |
| **10%** | 102 | 408 | 212 TB | 100 TB |

The full row is sized by the endpoint requirement; memory then has headroom. Peak resident
memory is 2.02x the key array, which is where the 2.13 PB against 1.0 PB comes from.

## Capacity request

Ask for **1,024** (or 102) `a4x-maxgpu-4g-metal` in one zone. `us-east4-a` held the
largest free pool at the last check, and the full request was about 15% of it. Bare metal,
so this needs a reservation rather than DWS.

## Build

Any node with CUDA 12 and an NVIDIA driver. The DLVM image
`common-cu129-ubuntu-2204-nvidia-580` works as-is.

```bash
# NVSHMEM from the CUDA apt repo
sudo apt-get install -y libnvshmem3-dev-cuda-12 libnvshmem3-static-cuda-12 \
                        libnvshmem3-cuda-12 libhwloc15

nvcc -O3 -std=c++17 -arch=sm_100 -rdc=true \
     -I/usr/include/nvshmem_12 -I../../src/cpu \
     ../../src/gpu/isx_nvshmem.cu \
     -L/usr/lib/x86_64-linux-gnu/nvshmem/12 \
     -lnvshmem_host -lnvshmem_device -lcudart -lcurand \
     -o isx_nvshmem
```

`-arch=sm_100` targets Blackwell. Use `sm_80` to rebuild on A100 for a smoke test.

Three things that cost time to find on A100 and will apply here:

- `libnvshmem3-static-cuda-12` is a separate package and supplies `libnvshmem_device.a`.
  Without it the link fails on `nvshmemi_init_thread`.
- `libhwloc15` is needed at runtime by `nvshmrun` and is not pulled in automatically.
- `-rdc=true` is required.

## Run

One process per GPU.

```bash
export NVSHMEM_SYMMETRIC_SIZE=8G

# keys per GPU = total_keys / n_gpus.  1 PB is 1.25e14 keys.
#   full:  1.25e14 / 4096 =  30,517,578,125
#   10%:   1.25e13 /  408 =  30,637,254,902
srun --ntasks-per-node=4 -N 1024 ./isx_nvshmem 30517578125 1
```

Under Slurm, NVSHMEM bootstraps from PMI, so `srun` is enough. Outside Slurm use
`nvshmrun -n <total_gpus>`.

For a single-node smoke test before committing the allocation:

```bash
nvshmrun -n 4 ./isx_nvshmem 4194304 1
```

That should finish in well under a second and print `verification : PASSED`.

## What you will see

```
ISx-NVSHMEM  4096 PEs (1 per GPU)
  keys/PE       : 30517578125
  total keys    : 125000000000000  (1000.00 GB)
  ...
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
output is ordered, and no keys were lost globally. A `FAILED` on the last one prints the
count.

## Expected behaviour, from the A100 validation

At 2 GPUs, 8.59 GB total, the phases split: bucket 80%, radix 10%, exchange 8%,
generate 1%. Aggregate was flat at 12.6 GB/s across a 16x data increase, which is what
weak scaling should look like.

**Two things to watch that A100 could not show.**

NVLink spans 72 GPUs. At 4,096 endpoints that is 57 NVL72 domains, and 56/57 of the
all-to-all crosses the host RoCE network rather than NVLink. Expect the exchange share to
rise well above 8%. If it dominates, the fix is to make the bucket assignment rack-aware
so most keys stay in the domain that generated them; that is a change to the routing
prefix in `compute_dest`, not to the exchange.

Bucketing is a full `SortPairs` over (destination, key) and was 80% of runtime at small
scale. The exchange needs keys grouped, not sorted within a group, so a partition would be
cheaper. Left as-is because correctness came first.

## If it fails

| symptom | cause |
|---|---|
| `undefined reference to nvshmemi_init_thread` | missing `libnvshmem3-static-cuda-12` |
| `libhwloc.so.15: cannot open shared object file` | missing `libhwloc15` |
| `recv overflow` | receive slack too tight. Raise `1.02` in `isx_nvshmem.cu` |
| `nvshmem_malloc failed` | raise `NVSHMEM_SYMMETRIC_SIZE` |
| hangs at init | check every rank sees the same `NVSHMEM_SYMMETRIC_SIZE` |

The symmetric heap is one window slot per PE, `WINDOW_KEYS * 8` bytes, independent of PE
count. It does not grow with the job, so `8G` is ample at any scale.

## Contact

Findings and the full study are in the repository root README.

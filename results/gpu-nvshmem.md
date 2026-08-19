# ISx-NVSHMEM on 2 x A100, validated

2026-08-19. `a2-highgpu-2g`, two A100-SXM4-40GB connected by **NV12** (12 NVLink lanes),
NVSHMEM 3.7.2, CUDA 12.9, `NVSHMEM_REMOTE_TRANSPORT=none` so all traffic is NVLink.

One PE per GPU. Verification passed on every run.

| keys/PE | total | time | generate | bucket | exchange | radix | rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,194,304 | 0.07 GB | 0.006 s | 0.000 | 0.004 | 0.001 | 0.001 | 11.54 GB/s |
| 33,554,432 | 0.54 GB | 0.043 s | 0.000 | 0.034 | 0.004 | 0.005 | 12.46 GB/s |
| 268,435,456 | 4.29 GB | 0.341 s | 0.003 | 0.274 | 0.029 | 0.035 | 12.61 GB/s |
| 536,870,912 | 8.59 GB | 0.681 s | 0.006 | 0.548 | 0.057 | 0.070 | 12.61 GB/s |

## Against the H4D result at the same key count

The 8.59 GB row is 1,073,741,824 keys, which is the same key count as the largest
repeatedly-validated H4D run.

| | hardware | time | rate | validated |
|---|---|---:|---:|---|
| ISx64 / OpenSHMEM | 64 PEs on 2 x h4d-highmem-192 | 2.050 s | 4.19 GB/s | 2-3 of 5 |
| **ISx-NVSHMEM** | **2 GPUs on 1 a2-highgpu-2g** | **0.681 s** | **12.61 GB/s** | **3 of 3** |

Two A100s beat 64 CPU processes by 3x on time, on a part two generations behind GB300,
and passed every attempt where the H4D configuration passes fewer than half.

## The exchange is not the cost

| phase | share of runtime at 8.59 GB |
|---|---:|
| bucket (CUB `SortPairs` by destination) | **80%** |
| radix (CUB `SortKeys`) | 10% |
| exchange (`nvshmem_uint64_put_nbi`) | **8%** |
| generate | 1% |

The exchange moves 4.29 GB in 0.057 s, which is about 75 GB/s and consistent with NVLink.
Sorting work, meaning bucket plus radix together, is 91% of the runtime.

**This bears directly on the TPU result** reported at 122.6 s for a phase labelled "ICI
Exchange & Bucket Sorting". That label puts one timer over both, and on this hardware the
same two phases split 91/8 in favour of sorting. The reasonable reading is that the TPU
number is dominated by the sort rather than by the interconnect.

## Weak scaling

12.46, 12.61, 12.61 GB/s across a 16x increase in data. Flat, which is what weak scaling
should look like and what the H4D path never produced.

## The obvious optimisation

`bucket` is 80% of the time and it is a full `SortPairs` over (destination, key). The
exchange needs keys grouped by destination, not sorted within a destination. A partition
would do the same job for less. Not done, because correctness came first.

## What this does not show

Nothing about GB300. A100 has 40 GB of HBM against GB300's 279 GB, no Grace coherent
memory, and 12 NVLink lanes against an NVL72 domain. This validates the algorithm, the
NVSHMEM calls and the one-slot window. Performance on GB300 is unmeasured.

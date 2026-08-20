# TPU

`v6e-4`, 4 chips, us-east5-a, JAX 0.6.2. `src/tpu/isx_jax.py`.

TPU has no PGAS. There is no remote put, so the exchange is `jax.lax.all_to_all`, a
collective the compiler schedules. This path is not responsive to the OpenSHMEM
requirement. It is here to answer one question the CPU and GPU paths cannot: what does the
collective model cost on irregular data.

## Getting a slice

Spot did not work. 141 combinations across six accelerator types returned
`WAITING_FOR_RESOURCES` and none converted. On-demand queued resources did: of 7 submitted,
2 landed, both v6e in us-east5-a. Submit on-demand queued resources and wait, rather than
spraying Spot.

## Result

Every run validated: keys land in the right device's range, each device's output is
ordered, and no keys are lost.

| total | per chip | time | rate | verification |
|---:|---:|---:|---:|---|
| 0.54 GB | 0.13 GB | 425 ms | 1.26 GB/s | PASSED |
| 2.15 GB | 0.54 GB | 8,583 ms | 0.25 GB/s | PASSED |
| 6.44 GB | 1.61 GB | 26,043 ms | 0.25 GB/s | PASSED |
| **12.88 GB** | **3.22 GB** | 52,540 ms | 0.25 GB/s | PASSED |

Flat at 0.25 GB/s across a 6x range. Repeat runs land within 0.1% of each other, so the
timing is stable.

25.77 GB does not fit. The padded buffer is `n_dev * cap * 8` bytes per chip and at 1.3x
slack that is 8.4 GB, held twice for send and receive, against 32 GB of HBM per chip.

## Where the time goes

`src/tpu/isx_jax_phases.py` times each phase in its own `jit`. Fusion across boundaries is
lost so the parts do not add to the whole, but the ratio is the answer. At 16.8M keys per
chip:

| phase | seconds | share |
|---|---:|---:|
| generate | 0.018 | 0.6% |
| bucket | 3.245 | **97.4%** |
| exchange | 0.003 | 0.1% |
| sort | 0.064 | 1.9% |

**The ICI exchange moves 0.70 GB at 200 GB/s and is 0.1% of the step.** Local bucketing is
everything else.

This matters for reading other TPU sort numbers. A report that puts one timer around
"exchange and bucket sorting" is almost entirely reporting bucket time, and says nothing
about ICI.

## Making the bucket cheap

The first implementation sorted an array of destinations with `argsort`, then scattered
each key into a `(n_dev, cap)` buffer. TPU has no fast scatter.

In ISx the destination is `key // bucket_width`, so it is the top bits of the key. Sorting
the keys groups them by destination as a side effect. The bucket boundaries are then a
`searchsorted` on the edges, and placement becomes a gather.

| bucket | 16.8M keys/chip |
|---|---:|
| `argsort` on destinations, then scatter | 3.245 s |
| `sort` on keys, then gather | **0.362 s** |

9.0x, with the same keys out, checked. End to end that is 4,754 ms to 425 ms at 0.54 GB and
27,951 ms to 8,583 ms at 2.15 GB. The gain shrinks with size because the final sort of the
padded receive buffer grows.

## Two things to know before running this

**`jax.shard_map` rejects the code without `check_vma=False`.** From JAX 0.6, comparing a
device-varying array against a replicated one raises `Primitive le_to requires varying
manual axes to match`. `jnp.searchsorted` against `jnp.arange(n_dev)` does that. The
comparison is well defined, so turn the check off rather than materialise a varying copy of
every constant. `_shard_map` in both scripts does this and falls back for older JAX.

**Generate per device.** A single global `jax.random.randint` followed by `device_put` is a
one-device operation plus a broadcast, and it caps the dataset at one chip's HBM. Seed each
device with `jax.random.fold_in` inside `shard_map`.

## Can either target be reached on TPU

A TPU slice is one ICI-connected unit and a job cannot span slices over ICI. So the ceiling
is slice size, not supply, and no capacity grant moves it.

| | v6e | TPU7x |
|---|---:|---:|
| HBM per chip | 32 GB | 192 GB |
| Max chips per slice | 256 | 9,216 |
| Raw HBM per slice | 8.2 TB | 1.77 PB |
| Keys held, at the measured 3.2 GB/chip | 0.8 TB | 1.8 TB |
| Keys held, at a 2.02x footprint | 4.1 TB | 876 TB |

**Neither target is reachable on v6e.** 100 TB needs 121 full v6e-256 pods even at the raw
HBM limit, and the largest slice is one pod.

**On TPU7x, 100 TB fits and 1 PB does not.** 100 TB needs about 1,050 chips at a 2.02x
footprint, inside a 2,048-chip slice, and 4,096 endpoints needs a slice at least that
large, so both constraints are satisfied together. 1 PB needs 2.02 PB resident against
1.77 PB of HBM in the largest slice that exists.

Two caveats on the TPU7x row. The 2.02x footprint is the CPU implementation's, measured;
this JAX version runs at about 3.2 GB per 32 GB chip because the padded exchange buffer is
held twice. Ragged `all_to_all` is the fix and is available in this JAX, untested here.
Nothing about TPU7x in this table is measured.

## Against the success criteria

| criterion | status |
|---|---|
| Correctness | met, 4 sizes |
| Reproducibility | met, repeat runs within 0.1% |
| Performance stability | met, flat 0.25 GB/s over 6x |
| Scale | 12.88 GB of 1 PB, on 4 chips |
| OpenSHMEM one-sided | **not met, and not meetable**. TPU has no PGAS |

## What a larger slice would answer

The two open questions both need more chips, not more time.

1. **Does the exchange stay at 200 GB/s past one host?** 4 chips share a host. ICI between
   hosts is the number that matters and this cannot see it.
2. **Does padding stop being affordable?** `cap` is `per_dev / n_dev * slack`, so the
   buffer per chip shrinks as chips grow, but the number of destinations rises. Ragged
   `all_to_all` is available in this JAX and `isx_jax.py` detects it. Comparing padded
   against ragged at 64 or 256 chips is the measurement that says what the collective model
   costs on irregular data.

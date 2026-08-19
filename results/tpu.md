# TPU

Implementation written and corrected. No slice granted, so nothing is measured.

## What the code does

`src/tpu/isx_jax.py` runs the three ISx phases with `jax.lax.all_to_all` for the exchange.
It is a compiler-driven collective rather than PGAS, so it does not satisfy the study's
programming-model requirement. It exists to compare tensor hardware against the two
OpenSHMEM paths.

Key generation was fixed on 2026-08-19. It drew the whole `n_dev * per_dev` array in one
call and let `device_put` scatter it, which is not ISx phase 1 and caps the dataset at
what one chip's HBM holds. It now generates inside `shard_map` with `jax.random.fold_in`
per device index, so each device draws its own stream from one seed and nothing crosses
the interconnect during generation. Deterministic across runs, which the study requires.

**Untested since that change.** The v5e used for the earlier probe is no longer
provisioned.

## Capacity attempt

141 combinations, spot, across six accelerator types and every zone offering each:

| type | zones tried |
|---|---:|
| `v6e-4`, `v6e-8`, `v6e-16` | 20 each |
| `v5litepod-4`, `-8`, `-16` | 27 each |

Every one returned `WAITING_FOR_RESOURCES`. None granted. The run was stopped during the
spot pass and did not reach the on-demand pass.

## Why spot was the wrong mode

Stanford's `create_persistent_tpu.sh` records the reason directly: *"measured v6e Spot
survival was about an hour per slice"*, which is why their persistent development box is
an **on-demand queued resource** rather than spot. Building the hunt on spot alone
repeated a mode already documented as unreliable.

TPUs are obtainable in this org. A v6e-8 is live in `wz-stanford-hie-lab` at
`southamerica-west1-a` as an on-demand QR.

## To get a result

In order of cost:

1. **On-demand queued resource, small slice.** `v6e-4` is enough to verify the key
   generation fix and separate exchange from sort. Use
   `stanford/brian-hie/evo-google-cloud/infra/capacity/spray_tpu_slices.sh`, which loops
   rather than spraying once, because a spot grant failure is a point-in-time answer.
2. **Borrow the Stanford slice.** It is idle between sessions and already on the right
   runtime, `v2-alpha-tpuv6e`. The generic `tpu-ubuntu2204-base` yields a node with no TPU
   access daemon on which `libtpu` never initialises.
3. **Ask the AI ninja team.** They are already running a JAX ISx variant on v7 for this
   engagement.

## The measurement worth taking

A colleague's v6e run reports 122.6 s for a phase labelled "ICI Exchange & Bucket
Sorting", one timer over two phases, and reads 0.76 GB/s off it as an ICI figure.

Two pieces of evidence say that number is the sort, not the network. Probe B measured
`jax.lax.all_to_all` moving uint64 over ICI at **175 GB/s** on four v5e chips, 230x the
reported figure. And on the GPU path, where the same two phases are timed separately, they
split **91% sorting to 8% exchange**.

`src/tpu/isx_jax.py` already separates generate, bucket, exchange and sort. One run at
about 6 GB per chip, matching that per-chip size, settles it.

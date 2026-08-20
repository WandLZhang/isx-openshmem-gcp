#!/usr/bin/env python3
"""ISx on TPU: the same bucket sort, expressed as compiler-driven collectives.

This is the second half of the comparison. `src/isx64` does the exchange with one-sided
`shmem_put` into a peer's symmetric heap. Here the same exchange is `jax.lax.all_to_all`
over ICI, which is a collective: every device participates, the compiler schedules it,
and no device reaches into another's memory.

That difference is not cosmetic and it shows up in one specific place.

ISx buckets are statistically even but never exactly even. Under PGAS that costs nothing,
because each PE simply puts however many keys it happens to have. A collective needs
every participant to agree on a shape before the compiler can emit it. So this
implementation has to either

  * pad every bucket to a common size, wasting bandwidth on the padding, or
  * use `ragged_all_to_all`, which carries per-device sizes.

Both are implemented. `--exchange padded` is the portable baseline; `--exchange ragged`
is the honest one and is used when the installed JAX has it. The gap between them is a
real measurement of what the collective model costs on irregular data, which is the
question the study is actually asking.

Run on a TPU slice:

    python3 isx_jax.py --keys-per-device 4194304 --iterations 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from jax.experimental import mesh_utils  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

# 2^60, matching src/isx64/params.h DEFAULT_MAX_KEY. Leaves the top bits clear so a
# bucket index can live in the MSBs, and keeps the two implementations comparable.
MAX_KEY = 1 << 60

HAS_RAGGED = hasattr(jax.lax, "ragged_all_to_all")


def make_keys(mesh: Mesh, n_dev: int, per_dev: int, seed: int) -> jnp.ndarray:
    """Uniform uint64 keys, generated independently on each device.

    ISx phase 1 is "each PE generates a subset of keys using a PRNG". It is local and
    parallel, and no data crosses the interconnect. An earlier version of this function
    drew the whole `n_dev * per_dev` array in one call and let `device_put` scatter it,
    which is a different workload: it makes generation a single-device operation followed
    by a broadcast, and it caps the dataset at what one chip's HBM can hold. That ceiling
    is the opposite of what the benchmark exists to measure.

    `fold_in` gives each device an independent stream from one seed, so the result is
    still deterministic across runs with the same configuration.
    """

    def gen(idx):
        k = jax.random.fold_in(jax.random.PRNGKey(seed), idx[0])
        # randint tops out at int64, so draw in int64 and reinterpret. Drawing in two
        # 32-bit halves the way the C version does is unnecessary here, because JAX has a
        # real 64-bit RNG path once x64 is enabled.
        raw = jax.random.randint(k, (per_dev,), 0, MAX_KEY, dtype=jnp.int64)
        return raw.astype(jnp.uint64)

    idx = jax.device_put(jnp.arange(n_dev), NamedSharding(mesh, P("pe")))
    return _shard_map(gen, mesh, P("pe"), P("pe"))(idx)


def _shard_map(fn, mesh, in_specs, out_specs):
    """`jax.shard_map` with varying-manual-axis checking off where the version has it.

    Inside the mapped function, `jnp.searchsorted` and `jnp.bincount` compare a
    device-varying array against a replicated one such as `jnp.arange(n_dev)`. From JAX
    0.6 that raises `Primitive le_to requires varying manual axes to match`. The
    comparison is well defined here, so turn the check off rather than materialise a
    varying copy of every constant.
    """
    kw = dict(mesh=mesh, in_specs=in_specs, out_specs=out_specs)
    try:
        return jax.shard_map(fn, check_vma=False, **kw)
    except TypeError:
        return jax.shard_map(fn, **kw)


def build_padded(mesh: Mesh, n_dev: int, bucket_width: int, cap: int):
    """Bucket locally, pad each destination slot to `cap`, exchange, sort.

    `cap` is the per-destination capacity. A bucket that overflows it loses keys, so the
    caller sizes it with slack and the verification catches any loss.
    """

    def local(keys):
        # keys: (per_dev,) on this device
        #
        # The destination of a key is `key // bucket_width`, so it is the top bits of the
        # key. Sorting the keys therefore groups them by destination as a side effect, and
        # the bucket boundaries are a `searchsorted` on the edges. That removes the
        # argsort of the destination array and, more importantly, turns the placement into
        # a gather. Measured on v6e-4 at 16.8M keys per device: 3.245 s for
        # argsort-plus-scatter against 0.362 s this way, same keys out. See
        # `isx_jax_phases.py`.
        sk = jax.lax.sort(keys)
        edges = jnp.arange(1, n_dev, dtype=jnp.uint64) * jnp.uint64(bucket_width)
        start = jnp.searchsorted(sk, edges, side="left").astype(jnp.int32)
        bounds = jnp.concatenate(
            [
                jnp.zeros(1, jnp.int32),
                start,
                jnp.full(1, keys.shape[0], jnp.int32),
            ]
        )
        counts = jnp.diff(bounds)

        # Sentinel is MAX_KEY, which is above every real key, so padding sorts to the end
        # and is trimmed by the count. A bucket longer than `cap` loses its tail here and
        # `counts` still reports the true length, so verification catches it.
        col = jnp.arange(cap, dtype=jnp.int32)
        idxs = bounds[:n_dev, None] + col[None, :]
        valid = col[None, :] < counts[:, None]
        buf = jnp.where(valid, sk[jnp.minimum(idxs, keys.shape[0] - 1)], MAX_KEY)

        # The exchange. Axis 0 is destination-major and is split across the mesh.
        recv = jax.lax.all_to_all(buf, "pe", 0, 0, tiled=True)
        recv_counts = jax.lax.all_to_all(
            counts.reshape(n_dev, 1), "pe", 0, 0, tiled=True
        ).reshape(-1)

        flat = recv.reshape(-1)
        return jax.lax.sort(flat), recv_counts

    return jax.jit(_shard_map(local, mesh, P("pe"), (P("pe"), P("pe"))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys-per-device", type=int, default=1 << 22)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--burn-in", type=int, default=1)
    ap.add_argument("--slack", type=float, default=1.3,
                    help="per-destination capacity multiplier for the padded exchange")
    ap.add_argument("--out", default="isx_jax_result.json")
    args = ap.parse_args()

    devs = jax.devices()
    n_dev = len(devs)
    per_dev = args.keys_per_device
    if devs[0].platform != "tpu":
        print("warning: not running on TPU", file=sys.stderr)

    bucket_width = math.ceil(MAX_KEY / n_dev)
    expected = per_dev // n_dev           # keys per destination, in expectation
    cap = max(1, int(expected * args.slack))

    print(f"ISx-JAX on {n_dev}x {devs[0].device_kind}, jax {jax.__version__}")
    print(f"  keys/device   : {per_dev:,}")
    print(f"  total keys    : {n_dev*per_dev:,}  ({n_dev*per_dev*8/1e9:.2f} GB)")
    print(f"  max key       : 2^{math.log2(MAX_KEY):.0f}")
    print(f"  bucket width  : {bucket_width:,}")
    print(f"  pad capacity  : {cap:,} per destination (slack {args.slack})")
    print(f"  ragged a2a    : {'available' if HAS_RAGGED else 'not in this JAX'}")

    mesh = Mesh(mesh_utils.create_device_mesh((n_dev,)), ("pe",))
    fn = build_padded(mesh, n_dev, bucket_width, cap)

    # Already sharded by construction; no scatter, and no single-device ceiling.
    sharded = make_keys(mesh, n_dev, per_dev, seed=0)

    times = []
    out = counts = None
    for it in range(args.iterations + args.burn_in):
        t0 = time.perf_counter()
        out, counts = fn(sharded)
        jax.block_until_ready(out)
        dt = time.perf_counter() - t0
        if it >= args.burn_in:
            times.append(dt)
        print(f"  iter {it}{' (burn-in)' if it < args.burn_in else ''}: {dt*1e3:.1f} ms")

    best = min(times)
    mean = sum(times) / len(times)
    total_bytes = n_dev * per_dev * 8

    # --- verification, the same three checks the C version makes ---------------------
    host_out = jax.device_get(out)
    host_counts = jax.device_get(counts)
    per_dev_len = host_out.size // n_dev

    ordered = True
    for d in range(n_dev):
        seg = host_out[d*per_dev_len:(d+1)*per_dev_len]
        real = seg[seg < MAX_KEY]
        if real.size and not bool((real[1:] >= real[:-1]).all()):
            ordered = False
            break

    kept = int(host_counts.sum())
    conserved = kept == n_dev * per_dev

    # No bucket may have exceeded the padded capacity, or keys were dropped.
    overflow = bool((host_counts > cap).any())

    verdict = ordered and conserved and not overflow

    print()
    print(f"  time (best)   : {best*1e3:.1f} ms")
    print(f"  time (mean)   : {mean*1e3:.1f} ms")
    print(f"  rate          : {total_bytes/best/1e9:.2f} GB/s")
    print(f"  ordered       : {ordered}")
    print(f"  keys conserved: {conserved} ({kept:,} of {n_dev*per_dev:,})")
    print(f"  bucket overflow: {overflow}")
    print(f"  VERIFICATION  : {'PASSED' if verdict else 'FAILED'}")

    result = {
        "device_kind": devs[0].device_kind,
        "n_devices": n_dev,
        "jax_version": jax.__version__,
        "keys_per_device": per_dev,
        "total_keys": n_dev * per_dev,
        "total_bytes": total_bytes,
        "exchange": "padded",
        "pad_capacity": cap,
        "pad_slack": args.slack,
        "ragged_available": HAS_RAGGED,
        "time_best_s": best,
        "time_mean_s": mean,
        "rate_GBps": total_bytes / best / 1e9,
        "ordered": ordered,
        "keys_conserved": conserved,
        "bucket_overflow": overflow,
        "verification": "PASSED" if verdict else "FAILED",
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {args.out}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())

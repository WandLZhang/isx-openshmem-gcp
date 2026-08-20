#!/usr/bin/env python3
"""Time the three ISx phases separately on a TPU slice.

`isx_jax.py` reports one number for the whole jitted step, which is the same limitation as
the reference TPU report it is meant to compare against. That number cannot say whether
the ICI exchange or the sort dominates.

This times each phase in its own `jit` on the same shapes. Fusion across phase boundaries
is lost, so the parts do not add to the whole, but the ratio between them is the answer to
the question.

    python isx_jax_phases.py --keys-per-device 16777216
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

MAX_KEY = 1 << 60


def _shard_map(fn, mesh, in_specs, out_specs):
    kw = dict(mesh=mesh, in_specs=in_specs, out_specs=out_specs)
    try:
        return jax.shard_map(fn, check_vma=False, **kw)
    except TypeError:
        return jax.shard_map(fn, **kw)


def timeit(fn, *args, reps=3):
    jax.block_until_ready(fn(*args))
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys-per-device", type=int, default=1 << 24)
    ap.add_argument("--slack", type=float, default=1.3)
    ap.add_argument("--out", default="phases.json")
    args = ap.parse_args()

    n_dev = jax.device_count()
    per_dev = args.keys_per_device
    mesh = Mesh(jax.devices(), ("pe",))
    width = MAX_KEY // n_dev
    cap = int(per_dev / n_dev * args.slack)
    total_bytes = n_dev * per_dev * 8

    print(f"devices       : {n_dev} x {jax.devices()[0].device_kind}")
    print(f"keys/device   : {per_dev:,}")
    print(f"total         : {total_bytes / 1e9:.2f} GB")

    # ---- generate -------------------------------------------------------------
    def gen(idx):
        k = jax.random.fold_in(jax.random.PRNGKey(1), idx[0])
        return jax.random.randint(k, (per_dev,), 0, MAX_KEY, dtype=jnp.int64).astype(
            jnp.uint64
        )

    idx = jax.device_put(jnp.arange(n_dev), NamedSharding(mesh, P("pe")))
    f_gen = jax.jit(_shard_map(gen, mesh, P("pe"), P("pe")))
    t_gen = timeit(f_gen, idx)
    keys = f_gen(idx)

    # ---- bucket: argsort by destination, scatter into (n_dev, cap) ------------
    def bucket(k):
        dest = jnp.minimum(k // width, n_dev - 1).astype(jnp.int32)
        order = jnp.argsort(dest, stable=True)
        sd, sk = dest[order], k[order]
        pos = jnp.arange(sd.shape[0])
        start = jnp.searchsorted(sd, jnp.arange(n_dev), side="left")
        rank = pos - start[sd]
        buf = jnp.full((n_dev, cap), MAX_KEY, dtype=jnp.uint64)
        return buf.at[sd, jnp.minimum(rank, cap - 1)].set(
            jnp.where(rank < cap, sk, MAX_KEY)
        )

    f_bucket = jax.jit(_shard_map(bucket, mesh, P("pe"), P("pe")))
    t_bucket = timeit(f_bucket, keys)
    buf = f_bucket(keys)

    # ---- bucket, gather form -------------------------------------------------
    # The destination is the top bits of the key, so sorting the keys already groups them
    # by destination. That removes the argsort of `dest` and turns the scatter into a
    # gather, which is the operation TPU is good at.
    def bucket_gather(k):
        sk = jax.lax.sort(k)
        edges = (jnp.arange(1, n_dev, dtype=jnp.uint64) * jnp.uint64(width))
        start = jnp.searchsorted(sk, edges, side="left")
        bounds = jnp.concatenate(
            [jnp.zeros(1, jnp.int32), start.astype(jnp.int32),
             jnp.full(1, k.shape[0], jnp.int32)]
        )
        counts = jnp.diff(bounds)
        col = jnp.arange(cap, dtype=jnp.int32)
        idxs = bounds[:n_dev, None] + col[None, :]
        valid = col[None, :] < counts[:, None]
        return jnp.where(valid, sk[jnp.minimum(idxs, k.shape[0] - 1)], MAX_KEY)

    f_bg = jax.jit(_shard_map(bucket_gather, mesh, P("pe"), P("pe")))
    t_bucket_g = timeit(f_bg, keys)
    same = bool(jnp.all(jnp.sort(f_bg(keys).reshape(-1)) == jnp.sort(buf.reshape(-1))))

    # ---- exchange: all_to_all over ICI ---------------------------------------
    def exch(b):
        return jax.lax.all_to_all(b, "pe", 0, 0, tiled=True)

    f_exch = jax.jit(_shard_map(exch, mesh, P("pe"), P("pe")))
    t_exch = timeit(f_exch, buf)
    recv = f_exch(buf)

    # ---- sort: full sort of what arrived -------------------------------------
    def srt(r):
        return jax.lax.sort(r.reshape(-1))

    f_sort = jax.jit(_shard_map(srt, mesh, P("pe"), P("pe")))
    t_sort = timeit(f_sort, recv)

    exch_bytes = n_dev * n_dev * cap * 8
    total = t_gen + t_bucket + t_exch + t_sort

    rows = [
        ("generate", t_gen),
        ("bucket", t_bucket),
        ("exchange", t_exch),
        ("sort", t_sort),
    ]
    print()
    print(f"{'phase':<10} {'seconds':>10} {'share':>8}")
    for name, t in rows:
        print(f"{name:<10} {t:>10.3f} {100 * t / total:>7.1f}%")
    print(f"{'sum':<10} {total:>10.3f}")
    print()
    print(f"exchange moves {exch_bytes / 1e9:.2f} GB at {exch_bytes / t_exch / 1e9:.1f} GB/s")
    print()
    print(f"bucket, argsort+scatter : {t_bucket:.3f} s")
    print(f"bucket, sort+gather     : {t_bucket_g:.3f} s  ({t_bucket / t_bucket_g:.1f}x faster)")
    print(f"same keys out           : {same}")

    with open(args.out, "w") as fh:
        json.dump(
            {
                "n_devices": n_dev,
                "device_kind": jax.devices()[0].device_kind,
                "jax_version": jax.__version__,
                "keys_per_device": per_dev,
                "total_bytes": total_bytes,
                "pad_capacity": cap,
                "seconds": {n: t for n, t in rows},
                "shares": {n: t / total for n, t in rows},
                "exchange_bytes": exch_bytes,
                "exchange_GBps": exch_bytes / t_exch / 1e9,
                "bucket_scatter_s": t_bucket,
                "bucket_gather_s": t_bucket_g,
                "bucket_speedup": t_bucket / t_bucket_g,
                "bucket_gather_matches": same,
            },
            fh,
            indent=2,
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

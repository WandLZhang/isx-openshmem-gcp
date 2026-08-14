#!/usr/bin/env python3
"""Probe B: can a TPU sort 64-bit integers at all?

The study asks for a 64-bit integer sort. The HPC path gets that for free because CPUs
and GPUs have 64-bit integer ALUs. TPUs may not: Cloud TPU documentation still states
that "the TPU only supports tf.int32" and that 64-bit values must be converted. If that
is true for the XLA:TPU backend as well, a JAX implementation of ISx is either impossible
or so slow that reporting it as a comparison would be misleading.

This probe answers four questions before anyone writes that implementation:

  1. Does jax_enable_x64 produce a real uint64 array on TPU, or silently downcast?
  2. Does jax.lax.sort accept uint64, and what does it cost against int32?
  3. Does jax.lax.all_to_all move uint64 across chips over ICI?
  4. Does a uint64 round trip preserve values above 2^32?

Question 4 is the one that matters most. A silent downcast to int32 truncates every key
above 4.29e9, and because the sort would still return sorted output and the verification
would still pass, the result would look correct and be wrong.

    python3 tpu_int64.py                  # runs on whatever TPU is attached
    python3 tpu_int64.py --n 33554432     # larger array

Exit code 0 means the TPU path is worth building. Exit code 1 means it is not.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

# Must be set before jax is imported, or the flag has no effect.
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from jax.experimental import mesh_utils  # noqa: E402
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding  # noqa: E402


def timed(fn, *args, reps: int = 5) -> tuple[float, object]:
    """Compile once, then time. Returns seconds per call and the last result."""
    out = jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / reps, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1 << 22, help="keys per device")
    args = ap.parse_args()

    devs = jax.devices()
    result: dict = {
        "jax_version": jax.__version__,
        "platform": devs[0].platform,
        "device_kind": devs[0].device_kind,
        "n_devices": len(devs),
        "keys_per_device": args.n,
    }
    print(f"jax {jax.__version__} on {len(devs)}x {devs[0].device_kind}")

    if devs[0].platform != "tpu":
        print("NOT ON TPU. This probe is meaningless off TPU.", file=sys.stderr)
        return 1

    # --- Q1: does x64 survive on device? -------------------------------------------
    host = jnp.arange(4, dtype=jnp.uint64)
    dev_arr = jax.device_put(host)
    result["q1_dtype_on_device"] = str(dev_arr.dtype)
    result["q1_x64_real"] = dev_arr.dtype == jnp.uint64
    print(f"Q1 dtype on device: {dev_arr.dtype}  ({'real 64-bit' if result['q1_x64_real'] else 'DOWNCAST'})")

    # --- Q4 first, because it invalidates everything else if it fails --------------
    # Values that only exist above the 32-bit boundary. If these come back changed, the
    # backend truncated them and no sort result can be trusted.
    probes = jnp.array(
        [2**32 + 1, 2**40 + 12345, 2**52 - 7, 2**60 - 1, 2**63 - 1], dtype=jnp.uint64
    )
    back = jax.device_get(jax.device_put(probes) + jnp.uint64(0))
    expected = [2**32 + 1, 2**40 + 12345, 2**52 - 7, 2**60 - 1, 2**63 - 1]
    got = [int(x) for x in back]
    result["q4_roundtrip_expected"] = expected
    result["q4_roundtrip_got"] = got
    result["q4_preserves_high_bits"] = got == expected
    print(f"Q4 high-bit round trip: {'PRESERVED' if got == expected else 'TRUNCATED'}")
    if got != expected:
        for e, g in zip(expected, got):
            if e != g:
                print(f"   {e} -> {g}")

    # --- Q2: sort, uint64 against int32 --------------------------------------------
    key = jax.random.PRNGKey(0)
    n = args.n
    try:
        k64 = jax.random.randint(
            key, (n,), 0, 2**62, dtype=jnp.int64
        ).astype(jnp.uint64)
        sort64 = jax.jit(lambda x: jax.lax.sort(x))
        t64, out64 = timed(sort64, k64)
        ok64 = bool(jnp.all(out64[1:] >= out64[:-1]))
        result["q2_sort_uint64_s"] = t64
        result["q2_sort_uint64_ordered"] = ok64
        print(f"Q2 sort uint64  : {t64*1e3:8.2f} ms  ordered={ok64}")
    except Exception as exc:  # noqa: BLE001
        result["q2_sort_uint64_error"] = repr(exc)
        result["q2_sort_uint64_s"] = None
        print(f"Q2 sort uint64  : FAILED {exc!r}")

    k32 = jax.random.randint(key, (n,), 0, 2**30, dtype=jnp.int32)
    sort32 = jax.jit(lambda x: jax.lax.sort(x))
    t32, _ = timed(sort32, k32)
    result["q2_sort_int32_s"] = t32
    print(f"Q2 sort int32   : {t32*1e3:8.2f} ms")
    if result.get("q2_sort_uint64_s"):
        ratio = result["q2_sort_uint64_s"] / t32
        result["q2_uint64_penalty_x"] = ratio
        print(f"Q2 uint64 penalty: {ratio:.2f}x")

    # --- Q3: all_to_all on uint64 over ICI ------------------------------------------
    if len(devs) > 1:
        try:
            nd = len(devs)
            mesh = Mesh(mesh_utils.create_device_mesh((nd,)), ("pe",))

            # Shape matters here and it is easy to get wrong. Under shard_map each shard
            # sees only its own slice, and all_to_all splits that shard's split_axis
            # across the named axis. So the PER-SHARD length of axis 0 must be divisible
            # by the number of devices, not the global length. A global (nd, per) array
            # with in_specs P("pe", None) gives each shard axis 0 of length 1, which
            # fails with "split_axis (1) has to be divisible by the size of the named
            # axis". Use a flat array so each shard holds nd*chunk elements.
            chunk = max(1, n // nd)
            per_shard = nd * chunk
            data = jnp.arange(nd * per_shard, dtype=jnp.uint64)
            sharded = jax.device_put(data, NamedSharding(mesh, P("pe")))

            shard_fn = jax.shard_map(
                lambda x: jax.lax.all_to_all(x, "pe", 0, 0, tiled=True),
                mesh=mesh, in_specs=P("pe"), out_specs=P("pe"),
            )
            ta2a, out = timed(jax.jit(shard_fn), sharded)

            # Each device sends (nd-1)/nd of its shard off-chip.
            moved = data.size * 8 * (nd - 1) / nd
            result["q3_all_to_all_s"] = ta2a
            result["q3_all_to_all_GBps"] = moved / ta2a / 1e9
            result["q3_all_to_all_dtype"] = str(out.dtype)
            result["q3_all_to_all_ok"] = str(out.dtype) == "uint64"
            # An all_to_all is a permutation, so the multiset must be unchanged.
            result["q3_sum_preserved"] = bool(
                jnp.sum(out.astype(jnp.uint64)) == jnp.sum(data)
            )
            print(f"Q3 all_to_all u64: {ta2a*1e3:8.2f} ms  "
                  f"{moved/ta2a/1e9:6.2f} GB/s off-chip  dtype={out.dtype}  "
                  f"sum_preserved={result['q3_sum_preserved']}")
        except Exception as exc:  # noqa: BLE001
            result["q3_all_to_all_error"] = repr(exc)
            print(f"Q3 all_to_all u64: FAILED {exc!r}")
    else:
        result["q3_all_to_all_error"] = "single device, ICI not exercised"
        print("Q3 all_to_all   : skipped, one device")

    # --- verdict ---------------------------------------------------------------------
    blockers = []
    if not result.get("q1_x64_real"):
        blockers.append("jax_enable_x64 does not produce uint64 on device")
    if not result.get("q4_preserves_high_bits"):
        blockers.append("keys above 2^32 are truncated, so any sort result is invalid")
    if result.get("q2_sort_uint64_s") is None:
        blockers.append("jax.lax.sort rejects uint64")
    if len(devs) > 1:
        if result.get("q3_all_to_all_error"):
            blockers.append("all_to_all cannot move uint64 over ICI")
        elif not result.get("q3_sum_preserved"):
            blockers.append("all_to_all changed the data, so the exchange is wrong")

    penalty = result.get("q2_uint64_penalty_x")
    if penalty and penalty > 4:
        blockers.append(f"uint64 sort is {penalty:.1f}x slower than int32, so a TPU "
                        "comparison would flatter the wrong bottleneck")

    result["blockers"] = blockers
    result["verdict"] = "GO" if not blockers else "NO-GO"

    print()
    print(f"VERDICT: {result['verdict']}")
    for b in blockers:
        print(f"  blocker: {b}")

    with open("tpu_int64_probe.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print("wrote tpu_int64_probe.json")

    return 0 if not blockers else 1


if __name__ == "__main__":
    sys.exit(main())

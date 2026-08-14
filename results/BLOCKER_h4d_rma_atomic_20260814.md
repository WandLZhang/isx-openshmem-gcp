# RETRACTED — and the real cause

**This document previously claimed that H4D Cloud RDMA cannot offer `FI_RMA` and
`FI_ATOMIC` on the same endpoint. That claim was wrong. It is retracted in full.**

The original conclusion is kept below the fold because the way it was reached is a trap
worth documenting, and because it was published before it was checked.

## What was actually wrong

The real blocker was **`FI_RMA_EVENT`**, not atomics.

Sandia OpenSHMEM enables target counters unless built with `--enable-hard-polling`:

```c
/* src/transport_ofi.h */
#define ENABLE_TARGET_CNTR 1     /* when ENABLE_HARD_POLLING is off */

/* src/transport_ofi.c, query_for_fabric() */
hints.caps = FI_RMA | FI_ATOMIC;
#if ENABLE_TARGET_CNTR
    hints.caps |= FI_RMA_EVENT;  /* want to use remote counters */
#endif
```

Bisecting the exact hints against the fabric, everything else held at SOS's values:

| requested caps | result |
|---|---|
| `FI_RMA \| FI_ATOMIC` | **OK** |
| `FI_RMA \| FI_ATOMIC \| FI_RMA_EVENT` | No data available |
| `FI_RMA \| FI_ATOMIC \| FI_FENCE` | No data available |

`verbs;ofi_rxm` on H4D `irdma0` supports RMA and atomics together. It does not support
`FI_RMA_EVENT` (remote completion counters) or `FI_FENCE`.

One further attribute also fails, independently: `mr_mode = 0`, meaning scalable memory
registration, which is SOS's **default**. The provider needs basic-mode registration.

## The fix

```bash
./configure --prefix=$PREFIX --with-ofi=$PREFIX \
  --enable-pmi-simple --disable-fortran \
  --enable-ofi-mr=basic \      # provider rejects scalable MR (mr_mode=0)
  --enable-hard-polling        # drops FI_RMA_EVENT from the hints
```

`--enable-hard-polling` turns off target counters, so SOS polls for completion instead of
relying on provider-side counters. That is a performance tradeoff, not a correctness one.

With both flags, cross-node one-sided RMA works:

```
PE  0/2 on isxh4de1-h4dnodeset-0   got 1001 want 1001  OK
PE  1/2 on isxh4de1-h4dnodeset-1   got 1000 want 1000  OK
```

Two PEs on two physical H4D nodes, each writing into the other's symmetric heap with
`shmem_longlong_put` and no matching receive on the target. Verified at 2, 4, 6, 8, 12 and
16 PEs across 2 nodes, all passing.

## How the wrong conclusion was reached

The claim rested on this test:

```
fi_info -p "verbs;ofi_rxm" -c FI_RMA           -> YES
fi_info -p "verbs;ofi_rxm" -c FI_ATOMIC        -> YES
fi_info -p "verbs;ofi_rxm" -c FI_RMA,FI_ATOMIC -> NO      <-- "they cannot coexist"
```

**`fi_info -c` does not accept a comma-separated list of capabilities.** Any value
containing a comma fails to parse and returns nothing, whatever the fabric supports. The
control that would have caught it immediately:

```
fi_info -p "verbs;ofi_rxm" -c FI_MSG,FI_RMA    -> NO
```

`FI_MSG` and `FI_RMA` appear side by side in the provider's own default `caps:` line, so
that combination is certainly supported. It also returned NO. The tool was the problem,
not the fabric.

Two lessons, both of which cost time here:

1. **Every negative capability result needs a positive control.** Run the same query
   shape against something known to work before believing a NO.
2. **`fi_info` is a discovery tool, not a conformance test.** To find out whether a
   specific application's hints are satisfiable, replicate those hints in a few lines of C
   calling `fi_getinfo` directly and bisect the attributes. That is what finally isolated
   `FI_RMA_EVENT`, and it took less time than the guessing did.

A related trap from the same session: run any of these queries from the Slurm controller,
a `c2-standard-4` with no `irdma0`, and every single one returns NO. That looks like a
catastrophic fabric failure and is only a wrong host.

## Correction history

| when | claim | status |
|---|---|---|
| single-node probe | "FI_RMA and FI_ATOMIC both present" | correct, but reached via `fi_info -c` exiting zero, which is not a capability check |
| this document, first version | "RMA and atomics cannot coexist; OpenSHMEM cannot run on H4D" | **wrong, retracted** |
| this document, current | `FI_RMA_EVENT` and scalable MR are unsupported; both have configure workarounds | verified by bisect and by a passing cross-node run |

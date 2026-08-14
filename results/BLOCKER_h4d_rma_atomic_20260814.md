# Dominant blocker: H4D Cloud RDMA cannot offer RMA and atomics on one endpoint

Measured 2026-08-14 on two `h4d-highmem-192-lssd` nodes in `us-east1-b`, Slurm cluster
`isx-h4d-e1`, libfabric 2.6.0 built from source, Sandia OpenSHMEM 1.5.3.

This is the failure-analysis deliverable. It is a fabric capability limit, not a
configuration mistake, and it blocks the OpenSHMEM compliance path as specified.

## The measurement

Run on a compute node with the `irdma0` device present:

| capability requested from `verbs;ofi_rxm` | available |
|---|---|
| `FI_RMA` | **yes** |
| `FI_ATOMIC` | **yes** |
| `FI_MSG` | **yes** |
| `FI_RMA,FI_ATOMIC` | **no** |
| `FI_RMA,FI_ATOMIC,FI_MSG` | **no** |

The same holds for the plain `verbs` provider, so it is not an artifact of the `ofi_rxm`
layer:

| | available |
|---|---|
| `verbs` `FI_RMA,FI_ATOMIC` | **no** |
| `verbs` `FI_RMA,FI_ATOMIC,FI_MSG` | **no** |

Each capability exists in isolation. No single endpoint provides both.

## Why that stops OpenSHMEM

Sandia OpenSHMEM calls `fi_getinfo` once, with hints requesting RMA and atomics together,
because the OpenSHMEM API surface needs both on the same endpoint: `shmem_put` and
`shmem_get` need RMA, and `shmem_atomic_*` needs atomics. The query returns nothing:

```
[0000] WARN:  transport_ofi.c:1531: query_for_fabric
[0000]        OFI transport did not find any valid fabric services (provider=verbs;ofi_rxm)
[0000] ERROR: init.c:466: shmem_internal_heap_postinit
[0000]        Transport init failed (-61)
srun: error: isxh4de1-h4dnodeset-0: task 0: Aborted (core dumped)
```

`shmem_init()` aborts. No OpenSHMEM program runs on this fabric, ISx64 included.

This is specifically what ISx needs atomics for: `shmem_atomic_fetch_add` against a peer's
`receive_offset`, so each sender claims a disjoint range of the target's symmetric heap
before writing into it.

## What was ruled out first

Each of these was tested and is not the cause.

- **MR mode.** SOS rebuilt against all three of `scalable`, `basic` and `rma-event`. All
  three fail with the identical transport init error.
- **Provider selection.** SOS uses `SHMEM_OFI_PROVIDER`, not `FI_PROVIDER`. Setting it
  correctly changes the message from `provider=<auto>` to `provider=verbs;ofi_rxm` and
  changes nothing else.
- **libfabric version.** Built 2.6.0 from source, which the SOS runtime demonstrably
  loads (`ldd libsma.so` resolves to `~/isx/lib/libfabric.so.1`).
- **Device presence.** Both compute nodes expose `irdma0`, and `fi_info` from a compute
  node reports `provider: verbs;ofi_rxm, domain: irdma0`.
- **Node health.** Both nodes reached Slurm `idle` and a 2-node `srun` runs normally.

## A trap worth recording

`fi_info -p "verbs;ofi_rxm" -c FI_ATOMIC` prints provider lines and exits zero. That reads
like a pass and is not one. libfabric returns whatever subset of the request it can
satisfy, so the only reliable check is to request the full combination the application
needs and inspect the returned `caps:` line.

An earlier single-node check in this study recorded "FI_RMA and FI_ATOMIC both present"
on that basis. That conclusion was wrong. Both are present; both together are not.

Also note the capability query must run **on a node that has the RDMA device**. Run from
the Slurm controller, a `c2-standard-4` with no `irdma0`, every query returns NO and the
result looks like a much broader failure than it is.

## Options

**1. Remove the atomic from ISx64.** ISx routing is deterministic, so the receive offsets
do not have to be discovered atomically. Each PE can compute exactly where its bucket
lands in every peer's buffer given the full matrix of per-destination counts. That costs
one small all-to-all of counts before the data exchange, and the bulk data movement stays
one-sided `shmem_put`. This keeps the PGAS character of the benchmark, is a legitimate
"platform-specific modification" under the deliverables, and is the recommended path.

**2. Patch SOS to split the endpoints.** Open one endpoint for RMA and a second for
atomics. This is invasive, touches the OpenSHMEM runtime's transport layer, and the
performance consequence of two endpoints per PE at 4,096 PEs is unknown.

**3. Report the fabric as non-compliant for unmodified OpenSHMEM.** The study requires
"RDMA one-sided operations (Get/Put/Atomics)". H4D Cloud RDMA provides Get, Put and
Atomics, but not addressable through one endpoint, so a standard OpenSHMEM runtime cannot
consume them. That is a defensible finding and should be reported whichever option is
taken.

Option 1 is being implemented. Option 3 is reported regardless, because a customer
bringing their own unmodified OpenSHMEM application would hit this on their first run.

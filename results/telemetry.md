# Telemetry available on H4D

## Network utilisation

Snapshotting every counter in `/sys/class/infiniband/irdma0/ports/1/hw_counters` and
`/sys/class/net/enp0s9/statistics` on node 0, around a 2-node 64-PE exchange that moved
**8.59 GiB** from that node in 1.936 s, **not one counter moved by more than 100 KB**.

RoCE writes are offloaded in hardware, so they bypass the netdev statistics, and this
irdma build does not expose byte counts in its RDMA counters either. `ip4InOctets` and
`ip4OutOctets` do not track RoCE, and no substitute counter exists on this platform.

The workable substitute is derived bandwidth from a known payload and measured wall time,
which is what the benchmark reports anyway:

| source | measurement |
|---|---|
| 64-PE all-to-all, 8.59 GiB per node | 1.936 s → **4.44 GB/s** per node effective |
| `fi_pingpong`, 1 MB messages, `verbs;ofi_rxm` | **8.15 GB/s** |
| `fi_pingpong`, 1 MB messages, `verbs;ofi_rxd` | **5.52 GB/s** |

The link is 200 Gbps = 25 GB/s nominal, so the pingpong ceiling is about 33% of nominal
and the all-to-all achieves about 55% of the pingpong ceiling.

## Memory bandwidth

STREAM-style triad, single core: **32.7 GB/s**.

## Time to solution

Reported per run by `isx64win` and captured in the per-run logs.

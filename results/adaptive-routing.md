# Per-packet adaptive routing on H4D

The brief requires a fabric with "per-packet adaptive routing". H4D does not do it. Three
lines of evidence below, and one limitation that two nodes cannot overcome.

## 1. Falcon does not do per-packet spraying, by design

The VPC network profile is explicit about the transport:

```
$ gcloud compute network-profiles describe us-east1-b-vpc-falcon
description: A VPC network that supports RDMA connectivity over Falcon, compatible
  with Intel NICs
features:
  interfaceTypes: [IRDMA]
  subnetStackTypes: [SUBNET_STACK_TYPE_IPV4_ONLY]
```

Falcon's published design uses **multipath subflows**, not packet spraying. From the
SIGCOMM 2025 paper: a connection is associated with many flows, each flow is one
sender-to-receiver path, and the sending NIC selects a path by writing the flow index
into the **IPv6 Flow Label**. Per-packet spraying is the Ultra Ethernet approach, which
Falcon deliberately did not take, because RoCE treats reordering as loss and per-packet
routing therefore degrades it.

So the requirement as written describes UET behaviour. Falcon reaches the same goal, using
every available path and reacting to congestion on a subset, through a different
mechanism at coarser granularity.

**Note the interaction with the profile above.** Falcon selects paths using the IPv6 Flow
Label, and this network is IPv4-only by its own declaration. Whatever multipath is
available here cannot be using the documented flow-label mechanism.

## 2. The NIC presents RoCE v2, not Falcon, to the RDMA stack

Whatever the VPC does underneath, the verbs layer sees a plain RoCE v2 device:

```
hca_id: irdma0        vendor_id: 0x8086   vendor_part_id: 5212
driver: idpf          link_layer: Ethernet
GID[0]: fe80::4001:c0ff:fea8:4004, RoCE v2
GID[1]: ::ffff:192.168.64.4,       RoCE v2
```

Its counters are the RoCE set, including DCQCN congestion notification
(`cnpSent`, `cnpHandled`, `cnpIgnored`, `RxECNMrkd`) and go-back-N style recovery
(`Nak Sequence Error`, `RetransSegs`). There is no Falcon-specific counter, no multipath
counter, and no per-flow statistic anywhere in
`/sys/class/infiniband/irdma0/ports/1/hw_counters/`.

An application using libfabric verbs on this machine is programming RoCE v2. It has no
interface through which to observe or influence path selection.

## 3. Measured: zero out-of-order arrivals

The counter that would move under multipath delivery is `Rcvd Out of order packets`.
Across every run in this study, including 128-PE all-to-all exchanges moving several GiB
per node:

```
Rcvd Out of order packets     0
Nak Sequence Error            0
Nak Sequence Error Implied    0
RetransSegs                   0
```

Zero reordering is consistent with all packets between a given pair taking one path.

## The limitation this cannot escape

Two nodes in one zone may sit on a single switch, in which case there is one path and no
adaptive routing would engage regardless of whether the fabric supports it. **This
evidence cannot separate "the fabric does not do adaptive routing" from "these two nodes
did not need it."**

Settling that needs a topology with multiple hops and multiple paths, which means more
nodes. It is on the list in `docs/scale-out.md` for when capacity lands.

## Answer for the study

| question | answer |
|---|---|
| Does the fabric do per-packet adaptive routing? | **No.** Falcon uses multipath subflows by design, and per-packet spraying is the UET approach |
| Does the application see any multipath behaviour? | **No.** The verbs layer is RoCE v2 with DCQCN, with no path-selection interface and no multipath counters |
| Was any reordering observed? | **No.** Zero out-of-order packets across all traffic in this study |
| Is per-packet AR achievable on H4D? | **Not through this stack.** It would need a fabric that sprays and a transport that tolerates reordering |

This requirement describes Ultra Ethernet. Google's
Falcon made the opposite choice deliberately, and its own published comparison argues
subflow multipath beats RoCE plus switch-based adaptive routing, reporting up to 8x lower
completion times against CX-7 RoCE under loss. If the underlying goal is "use all
available paths and react to congestion", Falcon addresses it. If the requirement is
literal, H4D does not meet it.

## Sources

- [Falcon: A Reliable, Low Latency Hardware Transport, SIGCOMM 2025](https://dl.acm.org/doi/10.1145/3718958.3754353)
- [Introducing Falcon, Google Cloud Blog](https://cloud.google.com/blog/topics/systems/introducing-falcon-a-reliable-low-latency-hardware-transport)
- [RDMA over Falcon Transport Specification, OCP](https://www.opencompute.org/documents/rdma-over-falcon-spec-v1-1-pdf)

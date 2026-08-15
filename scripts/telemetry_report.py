#!/usr/bin/env python3
"""Turn irdma0 counter snapshots into the network-utilisation table for Deliverable 3.

Reads /tmp/tel_before.txt and /tmp/tel_after.txt, written by telemetry_run.sh as
"<host> <counter> <value>" lines, and reports the deltas across the run.

Bytes on the wire are the honest measure of what the exchange cost. They will exceed the
key payload, because every one-sided write carries headers and because ISx sends
(N-1)/N of each PE's keys off-node. Reporting the ratio makes that visible rather than
letting it inflate an apparent bandwidth number.
"""

import os
import sys

COUNTERS = [
    "ip4OutOctets", "ip4InOctets",
    "InRdmaWrites", "InRdmaReads", "InRdmaSends",
    "cnpSent", "cnpHandled", "cnpIgnored",
    "InProtoErrors", "CRC_errors",
]
LINE_RATE_GBPS = 200.0  # H4D Cloud RDMA, per node


def load(path):
    d = {}
    try:
        for line in open(path):
            parts = line.split()
            if len(parts) == 3:
                try:
                    d[(parts[0], parts[1])] = int(parts[2])
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    return d


def main():
    before, after = load("/tmp/tel_before.txt"), load("/tmp/tel_after.txt")
    if not before or not after:
        print("  no counter snapshots found", file=sys.stderr)
        return 1

    hosts = sorted({h for h, _ in before})
    a2a = float(os.environ.get("TEL_A2A") or 0)
    keys = int(os.environ.get("TEL_KEYS") or 0)
    pes = int(os.environ.get("TEL_PES") or 0)

    short = [h.split("-")[-1] for h in hosts]
    print(f"  {'counter':16s}" + "".join(f"{s:>16s}" for s in short) + f"{'TOTAL':>18s}")
    print("  " + "-" * (16 + 16 * len(hosts) + 18))

    tot = {}
    for c in COUNTERS:
        vals = [after.get((h, c), 0) - before.get((h, c), 0) for h in hosts]
        tot[c] = sum(vals)
        if any(vals):
            print(f"  {c:16s}" + "".join(f"{v:>16,d}" for v in vals) + f"{sum(vals):>18,d}")

    out_gb = tot["ip4OutOctets"] / 1e9
    in_gb = tot["ip4InOctets"] / 1e9
    key_gb = keys * pes * 8 / 1e9

    print()
    print("  network utilisation")
    print(f"    on the wire          : {out_gb:.2f} GB out, {in_gb:.2f} GB in")
    print(f"    key payload          : {key_gb:.2f} GB")
    if key_gb:
        print(f"    wire / payload       : {out_gb / key_gb:.2f}x")
    if a2a > 0 and hosts:
        per_node_gbps = out_gb / a2a / len(hosts) * 8
        print(f"    exchange aggregate   : {out_gb / a2a:.2f} GB/s over {a2a:.3f}s of all2all")
        print(f"    per node             : {per_node_gbps:.1f} Gbps of {LINE_RATE_GBPS:.0f} Gbps "
              f"({per_node_gbps / LINE_RATE_GBPS * 100:.1f}% of line rate)")

    print()
    print("  fabric health")
    cong = tot["cnpSent"] + tot["cnpHandled"]
    print(f"    congestion notices   : sent {tot['cnpSent']:,}, handled {tot['cnpHandled']:,}, "
          f"ignored {tot['cnpIgnored']:,}")
    print(f"    protocol errors      : {tot['InProtoErrors']:,}")
    print(f"    CRC errors           : {tot['CRC_errors']:,}")
    print()
    if cong == 0 and tot["InProtoErrors"] == 0:
        print("    Zero congestion notifications and zero protocol errors. On a run that")
        print("    completes, the fabric is not congested and is not erroring, which points")
        print("    the instability at the software stack rather than at the network.")
    else:
        print("    Non-zero congestion or errors. Compare against a failing run before")
        print("    attributing the instability to the software stack.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

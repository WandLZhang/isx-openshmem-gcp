# Upstream issues

Both are filed.

- **libfabric** — [ofiwg/libfabric#12673](https://github.com/ofiwg/libfabric/issues/12673).
  Two findings. `verbs;ofi_rxd` fails `fi_av_insert` at the default GID index on a RoCE v2
  NIC whose GID 0 is link-local, which `fi_pingpong` reproduces without any application
  code. And `verbs;ofi_rxm` connection establishment stalls at about 8,000 connections per
  node, with the scaling table, the warmup experiment, the `FI_OFI_RXM_CQ_EQ_FAIRNESS`
  mitigation and six excluded causes.

- **Sandia OpenSHMEM** — [Sandia-OpenSHMEM/SOS#1239](https://github.com/Sandia-OpenSHMEM/SOS/issues/1239).
  The same two blockers as SOS startup failures, traced through `transport_ofi.c`, plus two
  requests: retry `fi_getinfo` with `FI_PROGRESS_MANUAL` when `FI_PROGRESS_AUTO` returns
  `ENODATA`, and print the returned count in the `populate_av` failure message.

The reproducer they both reference is `livelock_repro.c` in this directory. The full
diagnosis is in `results/ROOTCAUSE_connection_establishment.md`.

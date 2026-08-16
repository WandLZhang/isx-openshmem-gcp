# Scale-out recipe

Everything here is arithmetic against measured per-node numbers. Nothing in it needs new
code beyond the memory work noted in step 2. The purpose is that whoever gets the capacity
can execute without re-deriving any of it.

## The headline: storage dominates, endpoints come free

The two scale requirements are not independent, and the endpoint one is the easy half.

- **4,096 endpoints** at the measured 32 PEs/node needs **128 nodes**.
- **1 PB in memory** needs **about 800 nodes**, because each node holds 1,488 GB.

800 nodes at 32 PEs/node is **25,600 endpoints**, which overshoots the endpoint
requirement by 6x. So size the cluster for the petabyte and the endpoint count is
satisfied automatically. There is no configuration where endpoints are the binding
constraint.

## Step 1 — pick the node count from the memory footprint

An endpoint is one OpenSHMEM PE. Per node, on `h4d-highmem-192` (192 vCPU, 1,488 GB):

| ISx64 footprint | keys per node | nodes for 1 PB | endpoints at 32 PEs/node |
|---|---:|---:|---:|
| 2.30x, as the code stands today | 626 GB | 1,597 | 51,104 |
| 2.02x, after the cheap in-place fixes | 713 GB | 1,403 | 44,896 |
| **1.15x, after the streamed exchange** | **1,252 GB** | **799** | **25,568** |

The footprint multiple is peak resident memory over the key array. It is derived from the
allocations in `src/isx64/isx64_win.c` and explained in `TASKS.md` under Goal 2.

**Do the memory work first.** It sets the node count at 799 instead of 1,597. A single
zone has held at most 870 H4D machines, so 799 is near that limit and 1,597 is beyond it.

## Step 2 — the memory work, in priority order

All three are local changes in `src/isx64/isx64_win.c`, no distributed logic touched.

1. **Receive slack 1.3 → 1.02** (line 252, `NUM_KEYS_PER_PE * 1.3`). One constant. The
   1.3 is a safety margin on statistical imbalance; at 25,600 PEs the law of large numbers
   makes 1.02 ample. Saves 0.28x.
2. **In-place bucketize** (lines 242-248). Today `keys` and `send` both exist. Permute
   `keys` in place by destination with a cycle-following permutation and drop `send`
   entirely. Saves the `keys` + `send` overlap.
3. **Streamed exchange.** Release `send` incrementally as `exchange_windowed` drains it
   while `recv` fills, so the two never both hold a full copy. This is the only one that
   needs real design, and it is what takes 2.02x down to about 1.15x.

## Step 3 — size the symmetric heap, which grows with PE count

The windowed heap is `NUM_PES × WINDOW_KEYS_PER_PEER × 8` **per PE**, so it grows
linearly with total PEs:

| total PEs | `WINDOW_KEYS_PER_PEER` | heap per PE | heap per node at 32 PEs |
|---:|---:|---:|---:|
| 4,096 | 16,384 (current) | 537 MB | 17 GB |
| 25,600 | 16,384 (current) | 3.36 GB | **107 GB** |
| 25,600 | **4,096** | 839 MB | **27 GB** |

At 25,600 PEs the current window would consume 107 GB per node, which comes straight out
of the key budget. **Reduce `WINDOW_KEYS_PER_PEER` to 4096** (`src/isx64/isx64_win.c`
line 58) for runs above roughly 8,000 PEs. Throughput was flat across window sizes in the
windowed-exchange validation, so this costs nothing measurable.

**This step disappears entirely if the inverted schedule in
`docs/streamed_exchange_design.md` is implemented.** Because the exchange already rotates
destinations, every PE receives from exactly one sender per step, so the window needs one
slot rather than `NUM_PES` slots: 3.36 GB → 131 KB per PE at the target shape. That is
applicable on its own, ahead of the streaming work it came from.

## Step 4 — the numbers to actually run

Target: 1 PB, streamed footprint, 800 nodes, 32 PEs/node = 25,600 endpoints.

```
total keys      = 1e15 bytes / 8 bytes  = 1.25e14 keys
keys per PE     = 1.25e14 / 25,600      = 4.88e9
key bytes/PE    = 39.1 GB
at 1.15x        = 45 GB/PE  x 32 PEs    = 1,440 GB/node   (fits 1,488)
```

Blueprint (`infra/h4d/isx-slurm-h4d-e1.yaml`):

```yaml
h4d_cluster_size: 800        # was 2
zone: <single zone>          # Cloud RDMA cannot cross zones
```

Launch:

```bash
export SHMEM_OFI_PROVIDER="verbs;ofi_rxm"
# Both settings improve the standalone reproducer. Neither changed ISx64 results on
# 2 nodes. They cost nothing, so they are kept here. Re-measure at real node counts.
export FI_VERBS_GID_IDX=1              # routable GID, see results/ROOTCAUSE_*.md
export FI_OFI_RXM_CQ_EQ_FAIRNESS=1     # stops data traffic starving CM progress
export SHMEM_SYMMETRIC_SIZE=2G         # NUM_PES x WINDOW x 8, with WINDOW=4096

srun -N800 --ntasks-per-node=32 --mpi=pmi2 --export=ALL \
     ./isx64stream 4880000000 1 results/pb_run
```

`isx64stream` holds the symmetric window at `WINDOW * 8` bytes per PE instead of
`NUM_PES * WINDOW * 8`, which removes step 3 above. That part is structural and holds.

Its stability advantage does not. A five-run sample gave 4/5 against the windowed build's
1/5 at 64 PEs per node. Repeating at twenty runs gave **7/20**, so the 4/5 was a lucky
draw. The windowed build's twenty-run figure was still measuring when this was written;
check `results/` for it before choosing. Treat any completion rate here as provisional
until it is measured at real node counts.

## Step 5 — quota

| target | nodes | vCPU of H4D, one region |
|---|---:|---:|
| 4,096 endpoints only | 128 | 24,576 |
| **1 PB and 25,600 endpoints** | **800** | **153,600** |

Self-service is capped at 500 in every region, so both need an approved escalation via
`the internal capacity escalation path`. Quota grants permission. It does not reserve machines. Confirm the
zone holds the machines as a separate step. us-central1-a and us-central1-b were both
exhausted at a time when quota was sufficient.

## Re-validation once capacity lands

Do not assume the 2-node results transfer. In order:

1. **32 PEs/node on 4 nodes.** Confirms the instability is not specific to this node pair,
   which is the one hypothesis that could never be tested at 2 nodes.
2. **The stability settings at 128+ nodes.** `FI_VERBS_GID_IDX=1` and
   `FI_OFI_RXM_CQ_EQ_FAIRNESS=1` were measured on 2 nodes. Connections per node scale as
   `PEs_per_node × total_PEs`, so at 25,600 PEs and 32 PEs/node that is 819,200
   connections per node against 8,192 today. This is the largest single unknown in the
   whole plan, and step 1 above tests it.
3. **A 10-run reproducibility set** at whatever the largest stable shape turns out to be,
   before attempting the full petabyte.
4. **Then the petabyte run.**

Realistically, expect step 2 to fail at first and expect the connection-establishment work
in `results/ROOTCAUSE_connection_establishment.md` to need finishing. 819,200 connections
per node is two orders of magnitude beyond where it currently breaks.

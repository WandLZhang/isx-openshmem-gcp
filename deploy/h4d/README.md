# ISx64 weak-scaling ladder on H4D

## What this measures

ISx64 is a distributed bucket sort of 64-bit integers over OpenSHMEM one-sided RMA. Each PE
generates its own keys, puts every key into the symmetric heap of the PE that owns its value
range, sorts what arrives, and checks its boundary against its neighbour. The cost is an
irregular all-to-all rather than arithmetic, so this measures the fabric.

## Build

```bash
bash deploy/h4d/01_build_sos.sh     # libfabric with psm3, then SOS with one patch, to /opt/isx
source /opt/isx/env.sh              # puts oshcc on PATH. make fails without this
make -C src/cpu                     # writes bin/isx64_win at the repo root
```

`01_build_sos.sh` applies `sos-psm3-stx.patch`. PSM3 does not implement shared transmit
contexts and SOS binds one unconditionally, so without the patch `shmem_init()` aborts at
`transport_ofi.c:606`.

## Verify

```bash
make -C src/cpu provider-check      # psm3 offers FI_RMA and FI_ATOMIC
```

`make -C src/cpu check` runs `oshrun`, which needs a launcher SOS found at configure time.
If it reports `could not find a launcher`, skip it. `run_ladder.sh` pre-flights every rung
with a two-PE cross-node put before it spends the allocation, which covers the same ground.

## Run

```bash
bash deploy/h4d/run_ladder.sh                 # 4 8 16 22 32 143
bash deploy/h4d/run_ladder.sh 4 8 16          # a subset
```

## Results

| Nodes | PEs | Data | Baseline |
|------:|-------:|----------:|---|
| 2 | 384 | 1.40 TB | 275 s, measured, 20/20 |
| 4 | 768 | 2.80 TB | |
| 8 | 1,536 | 5.60 TB | |
| 16 | 3,072 | 11.20 TB | |
| 22 | 4,224 | 15.40 TB | |
| 32 | 6,144 | 22.40 TB | |
| 143 | 27,456 | 100.10 TB | |

Only the 2-node row is measured. Everything above it is arithmetic.

Keys per PE is fixed at 455,729,166 at every rung, which is 700 GB of keys per node and 98%
of usable memory. Because the per-PE work is constant, flat wall time is the ideal. Whether
it stays flat is the deliverable, so a climb at high endpoint counts is a result and not a
failure.

22 nodes is the customer's 4,096-endpoint requirement. 143 nodes is the 100 TB target.

**If 143 nodes cannot be obtained, the ladder through 32 nodes is a complete deliverable**
and it still closes the endpoint requirement at rung 22. H4D has returned
`ZONE_RESOURCE_POOL_EXHAUSTED` on us in a zone that read as mostly free, so we would rather
say this up front than have the request stall on the last rung.

## What to return

- Wall time per rung, and `$ISX64_OUT/ladder.csv`
- Where and how it broke, for any rung that fails. That is the result we want most
- Network utilisation and memory bandwidth telemetry if you can capture it. We found no RoCE
  byte counter on H4D and derived bandwidth from wall time, which is the weakest part of the
  study's telemetry deliverable
- If capacity allows, a 20-run set at the largest rung that passes

The binary under test is `isx64_win`, which self-verifies and prints `verification : PASSED`.
Take pass or fail from its log rather than from `isx64`.

## Constraints

All nodes in one zone. Cloud RDMA does not cross zones.

No MPI. The study excludes MPI-based implementations, and `01_build_sos.sh` asserts there is
no `libmpi` in `libsma.so`. Launch is `srun --mpi=pmi2`, which is PMI bootstrap only and puts
no MPI in the data path.

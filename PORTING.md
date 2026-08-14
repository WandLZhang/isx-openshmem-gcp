# Porting ISx from 32-bit to 64-bit

Every change from upstream [ISx v1.1](https://github.com/ParRes/ISx), why it was needed,
and what happens if you skip it. Changes are marked `ISX64` in the source.

The study requires a sort of 64-bit unsigned integers. Upstream sorts 32-bit signed
integers drawn from a 2^28 key space. The header comment says:

```c
// The data type used for the keys
// If you change this, you will have to change the SHMEM API calls used
typedef int KEY_TYPE;
```

The API calls are the smallest part.

## Why every counter also had to change

The target is more than 1 PB of keys. At 8 bytes per key that is 1.25e14 keys. Spread
across about 7,000 endpoints, each endpoint holds roughly **1.75e10 keys**.

A signed 32-bit integer holds 2.147e9. So every per-PE count in upstream overflows by
about **8x**. The overflow is not caught anywhere. Bucket sizes go negative, the prefix
scan wraps, keys are written to negative offsets, and the program still prints a
successful sort.

| upstream | ISx64 | why |
|---|---|---|
| `typedef int KEY_TYPE` | `uint64_t` | the study sorts 64-bit unsigned keys |
| `int * local_bucket_sizes` | `COUNT_TYPE *` (`int64_t`) | reaches keys-per-PE, 1.75e10 |
| `int * local_bucket_offsets`, `int ** send_offsets` | `COUNT_TYPE *` | running prefix sum over all local keys |
| `int temp` in the prefix scan | `COUNT_TYPE acc` | same |
| `uint32_t bucket_index`, `uint32_t index` | `uint64_t` | index into the local key array |
| `unsigned int total_keys_sent` | `uint64_t` | wraps at 4.29e9 keys sent |
| `int read_offset_from_self`, `int my_send_size` | `COUNT_TYPE` | per-bucket counts at scale |
| `int my_min_key = my_rank * BUCKET_WIDTH` | `uint64_t` | see below, this one is worse |
| `unsigned int key_index` | `uint64_t` | offset into the key space |
| `shmem_int_put` | `shmem_uint64_put` | the API change the comment warned about |
| `shmem_int_fadd` | `shmem_atomic_fetch_add` | the named-type form is deprecated in OpenSHMEM 1.4 |

### `my_min_key` deserves its own note

Upstream computes the bucket bounds as:

```c
const int my_min_key = my_rank * BUCKET_WIDTH;
const int my_max_key = (my_rank+1) * BUCKET_WIDTH - 1;
```

With `MAX_KEY` at 2^60 and a few thousand buckets, `BUCKET_WIDTH` is around 1.8e14. That
product overflows `int` for **every rank including rank 1**. The bounds become garbage,
and since these are the values `verify_results()` checks against, verification would pass
or fail for reasons unrelated to the sort. The bound check is the thing that proves the
sort worked, so a silently broken bound check is worse than no check.

## Two changes that are not type widening

### 1. The local sort: counting sort to radix sort

Upstream's `count_local_keys()`:

```c
int * my_local_key_counts = malloc(BUCKET_WIDTH * sizeof(int));
```

This is a counting sort over the key space. It is O(key space), not O(keys), and it is
only viable because upstream's key space is 2^28.

At `MAX_KEY = 2^60` over 7,000 buckets, `BUCKET_WIDTH` is about 1.8e14, so that
allocation is roughly **1.4 PB per endpoint**. Counting sort is a property of a small key
space, not of bucket sort.

ISx64 uses an LSD radix sort at 8 bits per pass, 8 passes over the 60-bit space. It is
O(n) in received keys and independent of the key space. This also matches what the study
specifies for the local phase, "bitonic or radix sort".

A consequence worth stating: upstream's verification checked a histogram sum, which is
meaningless once there is no histogram. ISx64 verifies ascending order directly, which is
a stronger check.

### 2. The symmetric receive buffer: fixed to dynamic

Upstream:

```c
#define KEY_BUFFER_SIZE (1uLL<<28uLL)
my_bucket_keys = shmem_malloc(KEY_BUFFER_SIZE * sizeof(KEY_TYPE));
```

A fixed 268 million keys. At scale a PE receives about 1.75e10, which is **65x** larger.

This is the most dangerous of the defects, because the write is one-sided. A PE computes
an offset into a peer's buffer and writes there without the peer participating. Nothing
on the target validates the offset. Overrunning it corrupts whatever the peer had in its
symmetric heap after the buffer, and the symptom appears later, somewhere else, on a
different rank.

ISx64 sizes the buffer at runtime from keys-per-PE times `KEY_BUFFER_SLACK` (1.2, to
absorb statistical imbalance in the uniform distribution), and bounds-checks every offset
before the put:

```c
if ((uint64_t)(write_off + nsend) > KEY_BUFFER_SIZE) { ... shmem_global_exit(2); }
```

If you see `RECV_OVERFLOW`, raise the slack. Seeing the error is the correct outcome;
upstream would have corrupted memory instead.

## Key generation

Upstream uses `pcg32_boundedrand_r`, which returns `uint32_t` and takes a `uint32_t`
bound. With a 64-bit key type it would still only ever produce keys below 2^32, leaving
the entire upper half of the key space untested while appearing to work. ISx64 composes
two PCG32 draws into a 64-bit value.

The seed also changed. Upstream mixes wall-clock time into the per-rank seed, which makes
a run unreproducible. The study requires "consistent results across multiple runs using
the same configuration", so the seed is derived from the rank alone.

## What did not change

The algorithm. Key generation, bucket-by-destination, one-sided exchange, local sort,
verify. The communication pattern that the study is measuring is untouched: an irregular
all-to-all of one-sided puts against remote symmetric heaps, with an atomic fetch-add to
claim the destination range.

`MAX_KEY = 2^60` rather than 2^64 leaves the top bits clear so the bucket index can be
carried in the most significant bits, which is how the study describes routing. Bucket
assignment itself remains upstream's `key / BUCKET_WIDTH`, which is equivalent for a
uniform distribution and keeps the diff reviewable against upstream.

## Verifying the port before spending money

```bash
gcc -O2 -std=c11 -Wall -Wextra -DNDEBUG -I tests -I src/isx64 \
    -o bin/isx64_stub src/isx64/isx64.c src/isx64/pcg_basic.c -lm
./bin/isx64_stub 4000000 2 /tmp/isx64.log
```

`tests/shmem_stub.h` implements the OpenSHMEM calls for a single PE. This exercises the
radix sort, the counter widths and the verification. It does not exercise the fabric, so
it cannot produce a meaningful performance number and it cannot catch a one-sided
semantics bug. Its job is to prove the arithmetic before a cluster exists.

/*
 * ISx64 parameters. Derived from ISx v1.1 (Copyright (c) 2015, Intel Corporation),
 * BSD 3-clause. See LICENSE-ISx. Upstream: https://github.com/ParRes/ISx
 *
 * Changes from upstream params.h are marked ISX64. The short version:
 * upstream ISx sorts 32-bit keys drawn from a 2^28 key space, and its local sort is a
 * counting sort over that space. A 64-bit key space breaks both of those, so this file
 * changes the key type and the sizing model together. Changing only the typedef gives
 * you a program that compiles, runs, and is wrong.
 */
#ifndef _PARAMS_H
#define _PARAMS_H

#include <stdint.h>

#define MAJOR_VERSION_NUMBER 1
#define MINOR_VERSION_NUMBER 1
#define ISX64_VERSION "0.1"

#define OPENSHMEM_COMPLIANT

/* ISX64: was `typedef int KEY_TYPE`.
 *
 * Upstream's comment said "If you change this, you will have to change the SHMEM API
 * calls used". That is true and it is the smallest of the consequences. See PORTING.md;
 * every bucket counter, offset and key-space index in the program was also `int`, and
 * at petabyte scale each of them overflows. */
typedef uint64_t KEY_TYPE;
#define KEY_FMT PRIu64
#define SHMEM_PUT_KEY shmem_uint64_put

/* ISX64: counters must hold keys-per-PE, not keys-per-bucket.
 *
 * At 1 PB across 7,136 PEs each PE holds about 1.75e10 keys. A signed 32-bit counter
 * tops out at 2.15e9, so upstream's `int` bucket counters overflow by about 8x. The
 * overflow is silent: sizes go negative, offsets wrap, and the sort reports success on
 * corrupted data. */
typedef int64_t COUNT_TYPE;
#define COUNT_FMT PRId64

#define STRONG 1
#define WEAK 2
#define WEAK_ISOBUCKET 3

#define ISO_BUCKET_WIDTH (8192uLL)

/* ISX64: was (1uLL<<28uLL), a 268-million key space.
 *
 * A 2^28 space cannot be called a 64-bit sort, and with 1.75e10 keys per PE every key
 * value would appear about 65 times. The study needs a key space large enough that
 * duplicates are rare, which means the sort has to be a real sort rather than a
 * histogram. 2^60 leaves room for the bucket index in the top bits without touching the
 * sign bit. */
#ifdef DEBUG
#define DEFAULT_MAX_KEY (32uLL)
#else
#define DEFAULT_MAX_KEY (1uLL<<60uLL)
#endif

/* ISX64: the local sort.
 *
 * Upstream count_local_keys() allocates BUCKET_WIDTH * sizeof(int) and histograms the
 * received keys. That is a counting sort, and it is only tractable because BUCKET_WIDTH
 * is small when MAX_KEY is 2^28. With MAX_KEY 2^60 and 7,136 buckets, BUCKET_WIDTH is
 * about 1.8e14, so the allocation alone would be 1.4 PB per PE.
 *
 * So the local sort is an LSD radix sort over the received keys. This matches what the
 * study actually asks for ("bitonic or radix sort") and makes the local phase O(n) in
 * keys rather than O(key space).
 *
 * RADIX_BITS=8 gives 256 buckets per pass and 8 passes for the 60-bit space. */
#define RADIX_BITS 8
#define RADIX_BUCKETS (1u << RADIX_BITS)

/* ISX64: symmetric receive buffer sizing.
 *
 * Upstream hardcodes KEY_BUFFER_SIZE (1uLL<<28uLL), a fixed 268M keys, which is 2 GB at
 * 8 bytes per key. At petabyte scale a PE receives about 1.75e10 keys, 65x more, and the
 * put lands outside the symmetric heap. There is no bounds check on the remote write, so
 * the failure is memory corruption on the target PE rather than an error.
 *
 * The buffer is now sized at runtime as keys_per_pe * KEY_BUFFER_SLACK. The slack covers
 * statistical imbalance in the uniform key distribution; keys are uniform so the
 * expected receive equals keys_per_pe, and 1.2 is far outside the observed spread at
 * these counts. Raise it if you see the RECV_OVERFLOW abort. */
#define KEY_BUFFER_SLACK 1.2

#define BURN_IN (1u)
#define BARRIER_ATA
#define PRINT_MAX 64

#endif

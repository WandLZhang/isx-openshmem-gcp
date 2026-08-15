/*
 * ISx64: 64-bit integer sort over OpenSHMEM one-sided RMA.
 *
 * Derived from ISx v1.1, Copyright (c) 2015 Intel Corporation, BSD 3-clause.
 * See LICENSE-ISx. Upstream: https://github.com/ParRes/ISx
 *
 * The algorithm is unchanged from upstream: generate uniform keys, bucket them by
 * destination PE, push each bucket into the destination's symmetric heap with one-sided
 * puts, sort locally, verify. What changed is every place where upstream assumed the key
 * and the counters fit in 32 bits. PORTING.md lists each one.
 *
 * Communication is shmem_uint64_put plus shmem_atomic_fetch_add. There is no MPI in this
 * program and no two-sided operation anywhere: a PE writes into a peer's memory without
 * the peer participating, which is the property the study is testing.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <time.h>
#include <shmem.h>

#include "params.h"
#include "isx64.h"
#include "pcg_basic.h"

#define ROOT_PE 0

/* Symmetric. Peers fetch-and-add into these to claim their slot in our buffer. */
long long int receive_offset = 0;
long long int my_bucket_size = 0;

static long long int llWrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
static long pSync[SHMEM_REDUCE_SYNC_SIZE];

KEY_TYPE *my_bucket_keys;      /* symmetric receive buffer */
uint64_t KEY_BUFFER_SIZE;      /* ISX64: runtime-sized, was a fixed 1uLL<<28 */

uint64_t NUM_PES, TOTAL_KEYS, NUM_KEYS_PER_PE, NUM_BUCKETS, BUCKET_WIDTH;
uint64_t MAX_KEY_VAL, NUM_ITERATIONS;

static double t_input, t_bcount, t_boffset, t_bucketize, t_ata, t_sort, t_total;

static double now(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ISX64: upstream used pcg32_boundedrand_r, which returns uint32_t and takes a uint32_t
 * bound. It cannot express a key above 2^32, so with a 64-bit KEY_TYPE every generated
 * key would still land in the bottom 32 bits and the top half of the key space would go
 * untested. Compose two draws instead. */
static inline uint64_t pcg64_bounded(pcg32_random_t *rng, uint64_t bound)
{
  uint64_t hi = (uint64_t)pcg32_random_r(rng);
  uint64_t lo = (uint64_t)pcg32_random_r(rng);
  return ((hi << 32) | lo) % bound;
}

static pcg32_random_t seed_my_rank(void)
{
  pcg32_random_t rng;
  /* Deterministic per rank. Upstream mixed in wall time, which makes a run
   * unreproducible; the study requires "consistent results across multiple runs using
   * the same configuration", so the seed is the rank alone. */
  pcg32_srandom_r(&rng, (uint64_t)shmem_my_pe(), (uint64_t)shmem_my_pe() * 2u + 1u);
  return rng;
}

static char *parse_params(const int argc, char **argv)
{
  NUM_PES = (uint64_t)shmem_n_pes();
  MAX_KEY_VAL = DEFAULT_MAX_KEY;
  NUM_BUCKETS = NUM_PES;
  BUCKET_WIDTH = (uint64_t)ceil((double)MAX_KEY_VAL / NUM_BUCKETS);
  char *log_file;

  if (argc == 3) { NUM_ITERATIONS = 1u; log_file = argv[2]; }
  else if (argc == 4) { NUM_ITERATIONS = strtoull(argv[2], NULL, 10); log_file = argv[3]; }
  else {
    if (shmem_my_pe() == 0)
      printf("Usage: %s <keys_per_pe> [iterations] <log_file>\n", argv[0]);
    shmem_finalize();
    exit(1);
  }

  NUM_KEYS_PER_PE = strtoull(argv[1], NULL, 10);
  TOTAL_KEYS = NUM_KEYS_PER_PE * NUM_PES;

  /* ISX64: size the symmetric buffer from the run, not from a #define. */
  KEY_BUFFER_SIZE = (uint64_t)(NUM_KEYS_PER_PE * KEY_BUFFER_SLACK);

  assert(MAX_KEY_VAL > NUM_PES);
  assert(NUM_KEYS_PER_PE > 0);

  if (shmem_my_pe() == ROOT_PE) {
    const double gib = (double)TOTAL_KEYS * sizeof(KEY_TYPE) / (1024.0*1024.0*1024.0);
    printf("ISx64 v%s  (ISx v%d.%d derivative)\n",
           ISX64_VERSION, MAJOR_VERSION_NUMBER, MINOR_VERSION_NUMBER);
    printf("  PEs                : %" PRIu64 "\n", NUM_PES);
    printf("  Keys per PE        : %" PRIu64 "\n", NUM_KEYS_PER_PE);
    printf("  Total keys         : %" PRIu64 "\n", TOTAL_KEYS);
    printf("  Total key bytes    : %.2f GiB (%.4f PB)\n", gib, gib*1024.0*1024.0*1024.0/1e15);
    printf("  Max key value      : %" PRIu64 " (2^%.0f)\n",
           MAX_KEY_VAL, log2((double)MAX_KEY_VAL));
    printf("  Bucket width       : %" PRIu64 "\n", BUCKET_WIDTH);
    printf("  Recv buffer per PE : %" PRIu64 " keys (%.2f GiB)\n",
           KEY_BUFFER_SIZE, (double)KEY_BUFFER_SIZE*sizeof(KEY_TYPE)/(1024.0*1024.0*1024.0));
    printf("  Iterations         : %" PRIu64 "\n", NUM_ITERATIONS);
    fflush(stdout);
  }
  return log_file;
}

static KEY_TYPE *make_input(void)
{
  const double t0 = now();
  KEY_TYPE *const keys = malloc(NUM_KEYS_PER_PE * sizeof(KEY_TYPE));
  if (!keys) { fprintf(stderr, "PE %d: make_input malloc failed\n", shmem_my_pe()); shmem_global_exit(1); }

  pcg32_random_t rng = seed_my_rank();
  for (uint64_t i = 0; i < NUM_KEYS_PER_PE; ++i)
    keys[i] = pcg64_bounded(&rng, MAX_KEY_VAL);

  t_input += now() - t0;
  return keys;
}

/* ISX64: returns COUNT_TYPE (int64) rather than int. */
static COUNT_TYPE *count_local_bucket_sizes(KEY_TYPE const *const keys)
{
  const double t0 = now();
  COUNT_TYPE *const sizes = calloc(NUM_BUCKETS, sizeof(COUNT_TYPE));
  if (!sizes) { fprintf(stderr, "PE %d: bucket size calloc failed\n", shmem_my_pe()); shmem_global_exit(1); }

  for (uint64_t i = 0; i < NUM_KEYS_PER_PE; ++i) {
    /* ISX64: bucket index was uint32_t. With a 2^60 key space the quotient still fits,
     * but the intermediate must not be truncated, so it is uint64 throughout. */
    const uint64_t b = (uint64_t)(keys[i] / BUCKET_WIDTH);
    sizes[b < NUM_BUCKETS ? b : NUM_BUCKETS - 1]++;
  }
  t_bcount += now() - t0;
  return sizes;
}

static COUNT_TYPE *compute_local_bucket_offsets(COUNT_TYPE const *const sizes,
                                                COUNT_TYPE **send_offsets)
{
  const double t0 = now();
  COUNT_TYPE *const offsets = malloc(NUM_BUCKETS * sizeof(COUNT_TYPE));
  *send_offsets = malloc(NUM_BUCKETS * sizeof(COUNT_TYPE));
  if (!offsets || !*send_offsets) { fprintf(stderr, "PE %d: offset malloc failed\n", shmem_my_pe()); shmem_global_exit(1); }

  offsets[0] = 0;
  (*send_offsets)[0] = 0;
  /* ISX64: `int temp` upstream. This is a running prefix sum over all keys on the PE,
   * so it reaches keys_per_pe and overflows int at 2.1e9. */
  COUNT_TYPE acc = 0;
  for (uint64_t i = 1; i < NUM_BUCKETS; ++i) {
    acc = offsets[i-1] + sizes[i-1];
    offsets[i] = acc;
    (*send_offsets)[i] = acc;
  }
  t_boffset += now() - t0;
  return offsets;
}

static KEY_TYPE *bucketize_local_keys(KEY_TYPE const *const keys, COUNT_TYPE *const offsets)
{
  const double t0 = now();
  KEY_TYPE *const out = malloc(NUM_KEYS_PER_PE * sizeof(KEY_TYPE));
  if (!out) { fprintf(stderr, "PE %d: bucketize malloc failed\n", shmem_my_pe()); shmem_global_exit(1); }

  for (uint64_t i = 0; i < NUM_KEYS_PER_PE; ++i) {
    const KEY_TYPE key = keys[i];
    const uint64_t b = (uint64_t)(key / BUCKET_WIDTH);
    /* ISX64: index was uint32_t and would wrap past 4.3e9 keys. */
    const COUNT_TYPE idx = offsets[b < NUM_BUCKETS ? b : NUM_BUCKETS - 1]++;
    out[idx] = key;
  }
  t_bucketize += now() - t0;
  return out;
}

/*
 * The phase the study exists to measure: an irregular all-to-all built from one-sided
 * puts. Each PE atomically claims a range in the target's receive buffer, then writes
 * straight into it. The target does not participate.
 */
static KEY_TYPE *exchange_keys(COUNT_TYPE const *const send_offsets,
                               COUNT_TYPE const *const sizes,
                               KEY_TYPE const *const bucketed)
{
  const double t0 = now();
  const int my_rank = shmem_my_pe();
  uint64_t total_sent = 0;  /* ISX64: was unsigned int, overflows at 4.3e9 keys */
  /* Operations in flight before a forced quiet. Tunable so the knee can be measured. */
  static long throttle = -1;
  long since_quiet = 0;
  if (throttle < 0) {
    const char *e = getenv("ISX64_THROTTLE");
    throttle = e ? atol(e) : 0;
  }

  const long long int self_off =
      shmem_atomic_fetch_add(&receive_offset, (long long int)sizes[my_rank], my_rank);
  if ((uint64_t)(self_off + sizes[my_rank]) > KEY_BUFFER_SIZE) {
    fprintf(stderr, "PE %d: RECV_OVERFLOW self %lld + %" COUNT_FMT " > %" PRIu64
                    " -- raise KEY_BUFFER_SLACK\n",
            my_rank, self_off, sizes[my_rank], KEY_BUFFER_SIZE);
    shmem_global_exit(2);
  }
  memcpy(&my_bucket_keys[self_off], &bucketed[send_offsets[my_rank]],
         (size_t)sizes[my_rank] * sizeof(KEY_TYPE));

  for (uint64_t i = 0; i < NUM_PES; ++i) {
    /* Rotate the start so all PEs do not hammer PE 0 first. Upstream does the same. */
    const int target = (int)((my_rank + i) % NUM_PES);
    if (target == my_rank) continue;

    const COUNT_TYPE read_off = send_offsets[target];
    const COUNT_TYPE nsend = sizes[target];
    if (nsend == 0) continue;

    const long long int write_off =
        shmem_atomic_fetch_add(&receive_offset, (long long int)nsend, target);

    /* ISX64: upstream had no bounds check here at all. The put is one-sided, so an
     * out-of-range offset silently corrupts the target PE's heap instead of failing.
     * At petabyte scale with upstream's fixed 268M-key buffer this always happens. */
    if ((uint64_t)(write_off + nsend) > KEY_BUFFER_SIZE) {
      fprintf(stderr, "PE %d: RECV_OVERFLOW target %d, %lld + %" COUNT_FMT " > %" PRIu64 "\n",
              my_rank, target, write_off, nsend, KEY_BUFFER_SIZE);
      shmem_global_exit(2);
    }

    /* ISX64: shmem_int_put -> shmem_uint64_put. */
    SHMEM_PUT_KEY(&my_bucket_keys[write_off], &bucketed[read_off], (size_t)nsend, target);
    total_sent += (uint64_t)nsend;

    /* ISX64: bound the number of operations in flight.
     *
     * Upstream issues NUM_PES puts back to back and only quiets at the end. The H4D
     * provider reports a transmit queue depth of 2048, and at 64 PEs per node sharing one
     * 200 Gbps NIC the aggregate outstanding count runs far past that. The symptom is
     * -FI_EAGAIN returned forever and SOS spinning in try_again until it exhausts its
     * 2^30 retry budget.
     *
     * A periodic quiet caps outstanding operations at ISX64_THROTTLE per PE. 0 disables
     * it and restores upstream behaviour. */
    if (throttle && (++since_quiet >= throttle)) { shmem_quiet(); since_quiet = 0; }
  }

#ifdef BARRIER_ATA
  shmem_barrier_all();
#endif
  t_ata += now() - t0;
  return my_bucket_keys;
}

/*
 * ISX64: replaces upstream count_local_keys().
 *
 * Upstream histogrammed into an array of BUCKET_WIDTH ints. At MAX_KEY 2^60 over 7,136
 * buckets, BUCKET_WIDTH is about 1.8e14 and that array would be 1.4 PB per PE. Counting
 * sort is a property of a small key space, not of this algorithm, so the local phase is
 * an LSD radix sort: O(n) in received keys and independent of the key space.
 */
static void radix_sort_local(KEY_TYPE *keys, const uint64_t n)
{
  const double t0 = now();
  if (n < 2) { t_sort += now() - t0; return; }

  KEY_TYPE *tmp = malloc(n * sizeof(KEY_TYPE));
  if (!tmp) { fprintf(stderr, "PE %d: radix tmp malloc failed\n", shmem_my_pe()); shmem_global_exit(1); }

  KEY_TYPE *src = keys, *dst = tmp;
  const int passes = (int)ceil(log2((double)MAX_KEY_VAL) / RADIX_BITS);

  for (int p = 0; p < passes; ++p) {
    uint64_t count[RADIX_BUCKETS] = {0};
    const int shift = p * RADIX_BITS;

    for (uint64_t i = 0; i < n; ++i)
      count[(src[i] >> shift) & (RADIX_BUCKETS - 1)]++;

    uint64_t sum = 0;
    for (unsigned b = 0; b < RADIX_BUCKETS; ++b) { const uint64_t c = count[b]; count[b] = sum; sum += c; }

    for (uint64_t i = 0; i < n; ++i)
      dst[count[(src[i] >> shift) & (RADIX_BUCKETS - 1)]++] = src[i];

    KEY_TYPE *swap = src; src = dst; dst = swap;
  }

  /* Odd pass count leaves the result in tmp. */
  if (src != keys) memcpy(keys, src, n * sizeof(KEY_TYPE));
  free(tmp);
  t_sort += now() - t0;
}

/*
 * Verification. Three independent checks, all required by the study.
 */
static int verify_results(KEY_TYPE const *const keys)
{
  shmem_barrier_all();
  int error = 0;
  const int my_rank = shmem_my_pe();

  /* ISX64: upstream computed these as `int`. my_rank * BUCKET_WIDTH with BUCKET_WIDTH
   * near 1.8e14 overflows int immediately, so every bound check was meaningless. */
  const uint64_t my_min = (uint64_t)my_rank * BUCKET_WIDTH;
  const uint64_t my_max = (my_rank == (int)NUM_PES - 1)
                            ? MAX_KEY_VAL
                            : ((uint64_t)(my_rank + 1) * BUCKET_WIDTH - 1);

  /* 1. every key I hold belongs in my bucket */
  for (long long int i = 0; i < my_bucket_size; ++i) {
    if (keys[i] < my_min || keys[i] > my_max) {
      fprintf(stderr, "PE %d: key %" PRIu64 " outside [%" PRIu64 ", %" PRIu64 "]\n",
              my_rank, keys[i], my_min, my_max);
      error = 1;
      break;
    }
  }

  /* 2. my keys are in ascending order after the local sort */
  for (long long int i = 1; i < my_bucket_size; ++i) {
    if (keys[i] < keys[i-1]) {
      fprintf(stderr, "PE %d: local sort out of order at %lld\n", my_rank, i);
      error = 1;
      break;
    }
  }

  /* 3. no key was lost or duplicated across the whole run */
  static long long int total = 0;
  shmem_longlong_sum_to_all(&total, &my_bucket_size, 1, 0, 0, (int)NUM_PES, llWrk, pSync);
  shmem_barrier_all();
  if (my_rank == ROOT_PE && total != (long long int)TOTAL_KEYS) {
    fprintf(stderr, "Verification failed: %lld keys survived, expected %" PRIu64 "\n",
            total, TOTAL_KEYS);
    error = 1;
  }

  /* Global order: max(PE i) <= min(PE i+1) holds by construction because bucket ranges
   * are disjoint and check 1 proved every key is inside its own range. */
  return error;
}

static int bucket_sort(void)
{
  int err = 0;
  for (uint64_t it = 0; it < NUM_ITERATIONS + BURN_IN; ++it) {
    if (it == BURN_IN) {
      t_input = t_bcount = t_boffset = t_bucketize = t_ata = t_sort = t_total = 0;
    }
    receive_offset = 0;
    shmem_barrier_all();
    const double t0 = now();

    KEY_TYPE *keys = make_input();
    COUNT_TYPE *sizes = count_local_bucket_sizes(keys);
    COUNT_TYPE *send_offsets;
    COUNT_TYPE *offsets = compute_local_bucket_offsets(sizes, &send_offsets);
    KEY_TYPE *bucketed = bucketize_local_keys(keys, offsets);

    KEY_TYPE *recv = exchange_keys(send_offsets, sizes, bucketed);
    my_bucket_size = receive_offset;

    radix_sort_local(recv, (uint64_t)my_bucket_size);

    shmem_barrier_all();
    t_total += now() - t0;

    if (it == NUM_ITERATIONS + BURN_IN - 1) err = verify_results(recv);

    free(keys); free(sizes); free(offsets); free(send_offsets); free(bucketed);
  }
  return err;
}

static void report(const char *log_file)
{
  const int my_rank = shmem_my_pe();
  const double iters = (double)NUM_ITERATIONS;

  /* Time to solution is the slowest PE, not the average. */
  static double t_max;
  static double t_in;
  t_in = t_total;
  shmem_double_max_to_all(&t_max, &t_in, 1, 0, 0, (int)NUM_PES, NULL, pSync);
  shmem_barrier_all();

  if (my_rank == ROOT_PE) {
    const double per_iter = t_max / iters;
    const double bytes = (double)TOTAL_KEYS * sizeof(KEY_TYPE);
    printf("\n=== ISx64 results ===\n");
    printf("  time to solution   : %.3f s  (slowest PE, per iteration)\n", per_iter);
    printf("  sorted             : %.4f PB\n", bytes / 1e15);
    printf("  aggregate rate     : %.2f GB/s\n", bytes / per_iter / 1e9);
    printf("  phase breakdown on PE 0 (s/iter):\n");
    printf("    input     %.3f\n    bcount    %.3f\n    boffset   %.3f\n",
           t_input/iters, t_bcount/iters, t_boffset/iters);
    printf("    bucketize %.3f\n    all2all   %.3f\n    radix     %.3f\n",
           t_bucketize/iters, t_ata/iters, t_sort/iters);
    fflush(stdout);

    if (log_file) {
      FILE *fp = fopen(log_file, "a");
      if (fp) {
        fprintf(fp, "%" PRIu64 ",%" PRIu64 ",%.6f,%.6f,%.6f,%.6f\n",
                NUM_PES, NUM_KEYS_PER_PE, per_iter, t_ata/iters, t_sort/iters,
                bytes / per_iter / 1e9);
        fclose(fp);
      }
    }
  }
}

int main(int argc, char **argv)
{
  shmem_init();

  for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; ++i) pSync[i] = SHMEM_SYNC_VALUE;
  shmem_barrier_all();

  char *log_file = parse_params(argc, argv);

  /* ISX64: allocated after parse_params because the size now depends on the run. */
  my_bucket_keys = (KEY_TYPE *)shmem_malloc(KEY_BUFFER_SIZE * sizeof(KEY_TYPE));
  if (!my_bucket_keys) {
    fprintf(stderr, "PE %d: shmem_malloc of %" PRIu64 " keys failed. The symmetric heap "
                    "is too small; raise SHMEM_SYMMETRIC_SIZE.\n",
            shmem_my_pe(), KEY_BUFFER_SIZE);
    shmem_global_exit(1);
  }
  shmem_barrier_all();

  const int err = bucket_sort();
  report(log_file);

  if (shmem_my_pe() == ROOT_PE)
    printf("  verification       : %s\n", err ? "FAILED" : "PASSED");

  shmem_free(my_bucket_keys);
  shmem_finalize();
  return err;
}

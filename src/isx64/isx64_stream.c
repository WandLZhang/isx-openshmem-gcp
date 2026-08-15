/*
 * ISx64-windowed: decouples dataset size from the OpenSHMEM symmetric heap.
 *
 * Derived from ISx v1.1, Copyright (c) 2015 Intel Corporation, BSD 3-clause.
 * See LICENSE-ISx.
 *
 * WHY THIS EXISTS
 *
 * On H4D Cloud RDMA the symmetric heap will not grow past about 32 GB per node
 * (results/BLOCKER_symmetric_heap_20260814.md). The provider rejects scalable memory
 * registration, so SOS must use basic mode, which pins and registers the entire heap at
 * shmem_init() before any data moves. Stock ISx sizes its receive buffer to the whole
 * per-PE dataset, so dataset size runs straight into that ceiling: 32 GB/node over 870
 * nodes is 27 TB, against a 1 PB target.
 *
 * The fix is to stop putting the dataset in the symmetric heap.
 *
 * Only the landing zone has to be symmetric. Here the symmetric allocation is a fixed
 * window of WINDOW_KEYS_PER_PEER keys per (sender, receiver) pair. The exchange runs in
 * rounds: every PE puts a chunk into each peer's window, a barrier, every PE drains its
 * window into ordinary malloc'd memory, repeat. Ordinary memory needs no registration, so
 * the dataset can be as large as the node's RAM.
 *
 * The communication stays one-sided and OpenSHMEM-compliant. Each chunk is a
 * shmem_uint64_put straight into a peer's symmetric memory, and the count that describes
 * it is a shmem_longlong_p into the same peer. The receiver participates in neither.
 * What changes is that the transfer is chunked with flow control, not that it becomes
 * two-sided.
 *
 * Cost: one extra memcpy per key on the receive side, which is memory bandwidth rather
 * than network, plus two barriers per round.
 *
 *   oshcc -O3 -march=native -std=c11 -DNDEBUG -o isx64win isx64_win.c pcg_basic.c -lm
 *   srun -N2 --ntasks-per-node=16 --mpi=pmi2 ./isx64win <keys_per_pe> <iters> <log>
 *
 * The point to demonstrate: sort far more data than SHMEM_SYMMETRIC_SIZE would allow.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include <stdint.h>
#include <time.h>
#include <shmem.h>

#include "params.h"
#include "pcg_basic.h"

#define ROOT_PE 0

/* Keys each sender may stage in each receiver per round. The symmetric footprint is
 * NUM_PES * WINDOW_KEYS_PER_PEER * 8 bytes and does not depend on the dataset at all.
 * 16384 keys = 128 KB per peer, which is a reasonable RDMA write and keeps the window
 * at 128 MB even with 1,000 PEs. */
#ifndef WINDOW_KEYS_PER_PEER
#define WINDOW_KEYS_PER_PEER (16384u)
#endif

static long pSync[SHMEM_REDUCE_SYNC_SIZE];
static long long llWrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];

static KEY_TYPE   *win;        /* symmetric: NUM_PES * WINDOW_KEYS_PER_PEER */
static long long  *win_count;  /* symmetric: NUM_PES, keys staged by each sender */

static uint64_t NUM_PES, NUM_KEYS_PER_PE, TOTAL_KEYS, BUCKET_WIDTH, MAX_KEY_VAL, NUM_ITERATIONS;
static double t_input, t_bucket, t_exch, t_sort, t_total;
static uint64_t g_rounds;

static double now(void)
{
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static inline uint64_t pcg64_bounded(pcg32_random_t *rng, uint64_t bound)
{
  uint64_t hi = (uint64_t)pcg32_random_r(rng), lo = (uint64_t)pcg32_random_r(rng);
  return ((hi << 32) | lo) % bound;
}

/* In-place LSD radix over 8-bit digits, using one scratch buffer.
 * Kept out of the peak by freeing the send array before this runs. */
static void radix_sort(KEY_TYPE *keys, uint64_t n)
{
  if (n < 2) return;
  KEY_TYPE *tmp = malloc(n * sizeof(KEY_TYPE));
  if (!tmp) { fprintf(stderr, "PE %d: radix tmp alloc failed\n", shmem_my_pe()); shmem_global_exit(1); }
  KEY_TYPE *src = keys, *dst = tmp;
  const int passes = (int)ceil(log2((double)MAX_KEY_VAL) / RADIX_BITS);
  for (int p = 0; p < passes; ++p) {
    uint64_t cnt[RADIX_BUCKETS] = {0};
    const int sh = p * RADIX_BITS;
    for (uint64_t i = 0; i < n; ++i) cnt[(src[i] >> sh) & (RADIX_BUCKETS - 1)]++;
    uint64_t s = 0;
    for (unsigned b = 0; b < RADIX_BUCKETS; ++b) { uint64_t c = cnt[b]; cnt[b] = s; s += c; }
    for (uint64_t i = 0; i < n; ++i) dst[cnt[(src[i] >> sh) & (RADIX_BUCKETS - 1)]++] = src[i];
    KEY_TYPE *sw = src; src = dst; dst = sw;
  }
  if (src != keys) memcpy(keys, src, n * sizeof(KEY_TYPE));
  free(tmp);
}

/*
 * The streamed all-to-all: destinations outside, windows inside.
 *
 * The windowed version put rounds outside and destinations inside, so every PE wrote to
 * every peer in each round and each PE's window needed one slot per sender:
 * NUM_PES * WINDOW * 8 bytes per PE, which is 3.36 GB per PE at 25,600 PEs.
 *
 * Inverting the nesting exploits the rotation that was already there. At step i, PE p
 * sends to (p + i) % n, so every PE targets a distinct destination and therefore
 * receives from exactly one sender per step. The window collapses to a SINGLE slot:
 * WINDOW * 8 bytes per PE, independent of PE count. At 25,600 PEs that is 131 KB
 * instead of 3.36 GB.
 *
 * The tradeoff is real and is the reason this is a separate binary rather than a
 * replacement: the original overlaps every destination within a round, and this
 * serialises the exchange into n steps of one put each. That trades injection
 * concurrency for memory. Measure both before choosing.
 *
 * send[]      keys already grouped by destination, in ordinary memory
 * soff[d]     where destination d's run starts
 * scnt[d]     how many keys destination d gets
 * recv        ordinary memory, sized by the caller
 * returns     how many keys arrived
 */
static uint64_t exchange_streamed(const KEY_TYPE *send, const uint64_t *soff,
                                  const uint64_t *scnt, KEY_TYPE *recv,
                                  uint64_t recv_cap)
{
  const int me = shmem_my_pe();
  const uint64_t n = NUM_PES;
  uint64_t recv_n = 0;
  g_rounds = 0;

  /* Every PE must run the same number of inner iterations so the barriers line up.
   * Take one global max of "most keys owed to any single destination" rather than a
   * reduction per step. Keys are uniform over destinations, so this is close to the
   * mean and the padding is small. */
  long long mymax = 0;
  for (uint64_t d = 0; d < n; ++d)
    if ((long long)scnt[d] > mymax) mymax = (long long)scnt[d];
  static long long gmax, lmax;
  lmax = mymax;
  shmem_longlong_max_to_all(&gmax, &lmax, 1, 0, 0, (int)n, llWrk, pSync);
  shmem_barrier_all();
  const uint64_t wins = ((uint64_t)gmax + WINDOW_KEYS_PER_PEER - 1) / WINDOW_KEYS_PER_PEER;

  for (uint64_t i = 0; i < n; ++i) {
    const int dst = (int)((me + i) % n);
    uint64_t sent = 0;
    for (uint64_t w = 0; w < wins; ++w) {
      win_count[0] = 0;
      shmem_barrier_all();

      const uint64_t left = scnt[dst] - sent;
      if (left) {
        const uint64_t k = left < WINDOW_KEYS_PER_PEER ? left : WINDOW_KEYS_PER_PEER;
        /* One-sided into the single window slot, then tell the target how many.
         * Exactly one PE writes to any given target in this step. */
        shmem_uint64_put(win, &send[soff[dst] + sent], k, dst);
        shmem_longlong_p(&win_count[0], (long long)k, dst);
        sent += k;
      }
      shmem_quiet();
      shmem_barrier_all();

      const uint64_t got = (uint64_t)win_count[0];
      if (got) {
        if (recv_n + got > recv_cap) {
          fprintf(stderr, "PE %d: recv overflow %" PRIu64 " + %" PRIu64 " > %" PRIu64 "\n",
                  me, recv_n, got, recv_cap);
          shmem_global_exit(2);
        }
        memcpy(recv + recv_n, win, got * sizeof(KEY_TYPE));
        recv_n += got;
      }
      g_rounds++;
    }
  }
  return recv_n;
}

int main(int argc, char **argv)
{
  shmem_init();
  for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; ++i) pSync[i] = SHMEM_SYNC_VALUE;
  shmem_barrier_all();

  if (argc < 3) {
    if (shmem_my_pe() == 0) printf("usage: %s <keys_per_pe> [iters] <log>\n", argv[0]);
    shmem_finalize(); return 1;
  }
  NUM_PES = (uint64_t)shmem_n_pes();
  NUM_KEYS_PER_PE = strtoull(argv[1], NULL, 10);
  NUM_ITERATIONS = (argc >= 4) ? strtoull(argv[2], NULL, 10) : 1;
  TOTAL_KEYS = NUM_KEYS_PER_PE * NUM_PES;
  MAX_KEY_VAL = DEFAULT_MAX_KEY;
  BUCKET_WIDTH = (uint64_t)ceil((double)MAX_KEY_VAL / NUM_PES);

  const uint64_t win_keys = WINDOW_KEYS_PER_PEER;   /* ONE slot, not NUM_PES slots */
  win       = shmem_malloc(win_keys * sizeof(KEY_TYPE));
  win_count = shmem_malloc(1 * sizeof(long long));
  if (!win || !win_count) {
    fprintf(stderr, "PE %d: symmetric window alloc failed\n", shmem_my_pe());
    shmem_global_exit(1);
  }

  if (shmem_my_pe() == ROOT_PE) {
    printf("ISx64-windowed  (dataset in ordinary memory, symmetric window only)\n");
    printf("  PEs                 : %" PRIu64 "\n", NUM_PES);
    printf("  keys/PE             : %" PRIu64 "\n", NUM_KEYS_PER_PE);
    printf("  total keys          : %" PRIu64 "  (%.3f GB)\n",
           TOTAL_KEYS, TOTAL_KEYS * 8.0 / 1e9);
    printf("  SYMMETRIC per PE    : %.1f MB   <-- fixed, independent of dataset\n",
           (win_keys * 8.0 + NUM_PES * 8.0) / 1e6);
    printf("  dataset per PE      : %.3f GB  <-- ordinary memory, unregistered\n",
           NUM_KEYS_PER_PE * 8.0 / 1e9);
    printf("  ratio dataset/heap  : %.1fx\n",
           (NUM_KEYS_PER_PE * 8.0) / (win_keys * 8.0));
    fflush(stdout);
  }
  shmem_barrier_all();

  int err = 0;
  for (uint64_t it = 0; it < NUM_ITERATIONS + BURN_IN; ++it) {
    if (it == BURN_IN) t_input = t_bucket = t_exch = t_sort = t_total = 0;
    shmem_barrier_all();
    const double tt0 = now();

    /* generate */
    double t0 = now();
    KEY_TYPE *keys = malloc(NUM_KEYS_PER_PE * sizeof(KEY_TYPE));
    if (!keys) { fprintf(stderr, "PE %d: keys alloc failed\n", shmem_my_pe()); shmem_global_exit(1); }
    pcg32_random_t rng; pcg32_srandom_r(&rng, (uint64_t)shmem_my_pe(), 1u);
    for (uint64_t i = 0; i < NUM_KEYS_PER_PE; ++i) keys[i] = pcg64_bounded(&rng, MAX_KEY_VAL);
    t_input += now() - t0;

    /* group by destination, in place into a second ordinary buffer */
    t0 = now();
    uint64_t *cnt = calloc(NUM_PES, sizeof(uint64_t));
    uint64_t *off = malloc(NUM_PES * sizeof(uint64_t));
    for (uint64_t i = 0; i < NUM_KEYS_PER_PE; ++i) {
      uint64_t b = keys[i] / BUCKET_WIDTH; if (b >= NUM_PES) b = NUM_PES - 1;
      cnt[b]++;
    }
    off[0] = 0;
    for (uint64_t d = 1; d < NUM_PES; ++d) off[d] = off[d-1] + cnt[d-1];
    uint64_t *cur = malloc(NUM_PES * sizeof(uint64_t));
    memcpy(cur, off, NUM_PES * sizeof(uint64_t));

    /* Group by destination IN PLACE, so there is no second full copy of the key array.
     * American-flag permutation: for each bucket, while its region is not yet filled,
     * take the element sitting at its cursor and either keep it (already home) or swap
     * it to its own bucket's cursor. Each swap places one element permanently, so this
     * is O(n) with O(NUM_PES) scratch instead of O(n).
     *
     * Peak resident was max(keys+send, send+recv) = max(2.00x, 2.30x). Dropping `send`
     * removes the first term; the recv slack below removes most of the second. */
    for (uint64_t b = 0; b < NUM_PES; ++b) {
      const uint64_t end = off[b] + cnt[b];
      while (cur[b] < end) {
        uint64_t d = keys[cur[b]] / BUCKET_WIDTH; if (d >= NUM_PES) d = NUM_PES - 1;
        if (d == b) { cur[b]++; }
        else { KEY_TYPE t = keys[cur[b]]; keys[cur[b]] = keys[cur[d]]; keys[cur[d]] = t; cur[d]++; }
      }
    }
    KEY_TYPE *send = keys;            /* same buffer, now in destination order */
    free(cur);
    t_bucket += now() - t0;

    /* Exchange through the fixed window.
     * Slack was 1.3, a margin on how unevenly keys land. Keys are uniform over
     * NUM_PES buckets, so the relative standard deviation of a PE's receive count is
     * 1/sqrt(keys per bucket); at production sizes that is far under a percent, and 1.02
     * is still many sigma. This is the single largest memory saving in the file. */
    const uint64_t cap = (uint64_t)(NUM_KEYS_PER_PE * 1.02) + 4096;
    KEY_TYPE *recv = malloc(cap * sizeof(KEY_TYPE));
    if (!recv) { fprintf(stderr, "PE %d: recv alloc failed\n", shmem_my_pe()); shmem_global_exit(1); }
    t0 = now();
    const uint64_t n = exchange_streamed(send, off, cnt, recv, cap);
    t_exch += now() - t0;
    free(send); free(off); free(cnt);  /* frees `keys` too, they are the same buffer, so
                                        * the radix scratch below reuses this memory */

    /* local sort */
    t0 = now();
    radix_sort(recv, n);
    t_sort += now() - t0;

    shmem_barrier_all();
    t_total += now() - tt0;

    /* verify on the last iteration */
    if (it == NUM_ITERATIONS + BURN_IN - 1) {
      const int me = shmem_my_pe();
      const uint64_t lo = (uint64_t)me * BUCKET_WIDTH;
      const uint64_t hi = (me == (int)NUM_PES - 1) ? MAX_KEY_VAL : (uint64_t)(me+1) * BUCKET_WIDTH - 1;
      for (uint64_t i = 0; i < n; ++i)
        if (recv[i] < lo || recv[i] > hi) { fprintf(stderr, "PE %d: key out of bucket\n", me); err = 1; break; }
      for (uint64_t i = 1; i < n; ++i)
        if (recv[i] < recv[i-1]) { fprintf(stderr, "PE %d: not sorted at %" PRIu64 "\n", me, i); err = 1; break; }
      static long long total, mine_ll; mine_ll = (long long)n;
      shmem_longlong_sum_to_all(&total, &mine_ll, 1, 0, 0, (int)NUM_PES, llWrk, pSync);
      shmem_barrier_all();
      if (me == ROOT_PE && total != (long long)TOTAL_KEYS) {
        fprintf(stderr, "lost keys: %lld of %" PRIu64 "\n", total, TOTAL_KEYS); err = 1;
      }
    }
    free(recv);
  }

  if (shmem_my_pe() == ROOT_PE) {
    const double it = (double)NUM_ITERATIONS;
    printf("\n=== results ===\n");
    printf("  exchange rounds     : %" PRIu64 "\n", g_rounds);
    printf("  time to solution    : %.3f s\n", t_total / it);
    printf("    generate  %.3f\n    bucket    %.3f\n    exchange  %.3f\n    radix     %.3f\n",
           t_input/it, t_bucket/it, t_exch/it, t_sort/it);
    printf("  aggregate rate      : %.2f GB/s\n", TOTAL_KEYS * 8.0 / (t_total/it) / 1e9);
    printf("  verification        : %s\n", err ? "FAILED" : "PASSED");
  }
  shmem_free(win); shmem_free(win_count);
  shmem_finalize();
  return err;
}

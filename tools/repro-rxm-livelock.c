/*
 * Minimal reproducer: SOS livelocks in try_again on verbs;ofi_rxm over Intel irdma
 * when many PEs per node each issue an unquiesced all-to-all of one-sided puts.
 *
 * Stripped from ISx64 to the smallest thing that still fails. No sort, no key
 * generation, no verification. Every PE puts a fixed block into every peer's symmetric
 * buffer, then barriers. That is all.
 *
 * OBSERVED (2 x h4d-highmem-192, us-east1-b, libfabric 2.6.0, SOS 1.5.3).
 * Completions out of attempts, ROUNDS=4, 256 KB puts:
 *
 *   PEs/node  QUIET_EVERY  USE_ATOMIC   result
 *   16        0            0            2/3
 *   16        1            0            3/3
 *   64        0            0            2/3, then 1/4 on a later job = 3/7
 *   64        1            0            0/3
 *   64        0            1            0/4
 *
 * On failure every PE spins at ~98% CPU, state R, in try_again until it exhausts the
 * 2^30 retry budget, then:
 *     ERROR: transport_ofi.h:596: try_again
 *            Operation retry limit exceeded (1073741824)
 *
 * Three things worth noting for anyone triaging this:
 *
 *  1. Puts alone are sufficient. There is no sort, no atomic, and no collective here
 *     beyond a barrier, and it still fails. Whatever this is, it is below OpenSHMEM.
 *
 *  2. shmem_quiet() after every put INVERTS with scale. At 16 PEs/node it helps
 *     (3/3 vs 2/3). At 64 PEs/node it hurts (0/3 vs 3/7). A per-operation quiet
 *     serialises the injection but multiplies the number of completion waits, so if
 *     the stall is in waiting for a completion rather than in queueing, quiescing more
 *     often makes it strictly more likely to hit.
 *
 *  3. The remote atomic is not the trigger, though it may be an aggravator. Adding
 *     shmem_atomic_fetch_add against one 8-byte counter per target gives 0/4 against
 *     3/7 without it. Directionally worse and consistent with full ISx64 never passing
 *     at this size, but 0/4 vs 3/7 is not a significant difference on its own.
 *
 * ROOT CAUSE (2026-08-15): connection establishment, not the data path. Round 0 costs
 * 47x a steady-state round and scales as connections-per-node = PEs_per_node *
 * total_PEs (512 -> 8192 is 16.0x; 0.368 -> 6.193 s is 16.8x). Failing runs print no
 * round at all, so they die inside round 0. WARMUP=1 opens the connections one at a
 * time first: the 6.2 s moves into the warmup and round 0 drops to 0.098 s, which
 * confirms the attribution, but completion only goes 0/5 -> 1/5 and the hang moves into
 * the warmup. So it is not concurrency of connection setup, it is that establishing
 * ~8k connections per node on this provider is unreliable. See
 * results/rxm-connection-limit.md.
 *
 * NOT the cause, each tested and ruled out:
 *   - QP count limits. irdma0 reports max_qp = 899,068.
 *   - shared receive contexts. FI_OFI_RXM_USE_SRX=1 gives 0/5, and 0/5 with warmup.
 *   - fabric congestion. irdma0 hw_counters show cnpSent/cnpHandled/cnpIgnored all 0,
 *     InProtoErrors 0, CRC_errors 0, and both nodes pass the vendor health check.
 *   - transmit queue depth. Raising FI_OFI_RXM_TX_SIZE, FI_OFI_RXM_RX_SIZE,
 *     FI_VERBS_TX_SIZE, FI_VERBS_RX_SIZE and the RXM_MSG variants to 16384 does not
 *     help at 64 PEs/node and makes 32 PEs/node worse (2/10 -> 0/10).
 *   - shared transmit contexts. SHMEM_OFI_STX_AUTO=1 and SHMEM_OFI_STX_MAX=8: no change.
 *   - completion semantics. Requesting FI_TRANSMIT_COMPLETE instead of
 *     FI_DELIVERY_COMPLETE (see libfabric#5601) does not change stability.
 *   - (WITHDRAWN) "SOS manual progress is a regression, 0/3". oshcc embeds an RPATH
 *     that overrides LD_LIBRARY_PATH, so that test was almost certainly running the
 *     auto-progress libsma. Manual progress on rxm has not been cleanly measured.
 *     Force the runtime with LD_PRELOAD and verify with ldd.
 *   - bounce buffering being disabled. Provider mode is 0x0, no FI_CONTEXT, so
 *     ctx->bounce_buffers is non-NULL and the EAGAIN path does call
 *     shmem_transport_ofi_drain_cq().
 *
 * BUILD (SOS must be configured --enable-ofi-mr=basic --enable-hard-polling; the
 * provider rejects scalable MR, and it does not support FI_RMA_EVENT which SOS requests
 * unless hard polling is on):
 *
 *   oshcc -O2 -o livelock_repro livelock_repro.c
 *
 * RUN:
 *
 *   export SHMEM_OFI_PROVIDER="verbs;ofi_rxm" SHMEM_SYMMETRIC_SIZE=1G
 *   srun -N2 --ntasks-per-node=64 --mpi=pmi2 --export=ALL ./livelock_repro
 *
 *   QUIET_EVERY=1 srun ... ./livelock_repro     # helps at 16 PEs/node, hurts at 64
 *   WARMUP=1      srun ... ./livelock_repro     # moves connection cost out of round 0
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <shmem.h>

#define BLOCK_WORDS 32768          /* 256 KB per put */
#define ROUNDS_DEFAULT 4           /* override with ROUNDS=n; ISx64 does ~128 */

/* ISx claims space in the target's receive buffer with a fetch-and-add against a single
 * 8-byte counter on that target, once per destination. Every PE therefore hits the same
 * address on every peer. USE_ATOMIC=1 reproduces that; USE_ATOMIC=0 does puts only. */
long long receive_offset = 0;

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void)
{
    shmem_init();
    const int me = shmem_my_pe(), n = shmem_n_pes();

    /* Symmetric landing zone: one BLOCK_WORDS slot per peer. */
    uint64_t *recv = shmem_malloc((size_t)n * BLOCK_WORDS * sizeof(uint64_t));
    uint64_t *send = malloc(BLOCK_WORDS * sizeof(uint64_t));
    if (!recv || !send) { fprintf(stderr, "PE %d: alloc failed\n", me); shmem_global_exit(1); }
    for (int i = 0; i < BLOCK_WORDS; ++i) send[i] = (uint64_t)me * BLOCK_WORDS + i;

    const char *q = getenv("QUIET_EVERY");
    const long quiet_every = q ? atol(q) : 0;
    const char *a = getenv("USE_ATOMIC");
    const int use_atomic = a ? atoi(a) : 0;
    const char *w = getenv("WARMUP");
    const int warmup = w ? atoi(w) : 0;
    /* ISx64 at 64 PEs/node issues ~16k puts per PE (128 windowed rounds x 128 peers)
     * against this reproducer's 512. Operation count is the largest single difference
     * between them, so make it a variable. */
    const char *rr = getenv("ROUNDS");
    const int rounds = rr ? atoi(rr) : ROUNDS_DEFAULT;

    if (me == 0) {
        printf("PEs=%d  block=%d KB  rounds=%d  QUIET_EVERY=%ld  USE_ATOMIC=%d\n",
               n, BLOCK_WORDS * 8 / 1024, rounds, quiet_every, use_atomic);
        printf("symmetric per PE = %.1f MB\n",
               (double)n * BLOCK_WORDS * 8 / 1e6);
        fflush(stdout);
    }
    shmem_barrier_all();

    /* Round 0 costs 47x a steady-state round and scales as PEs_per_node * total_PEs,
     * which is the connection count per node, not the byte count. ofi_rxm is
     * connection-oriented over verbs RC and establishes lazily on first message, so an
     * all-to-all makes every PE open every connection at once. WARMUP=1 opens them one
     * at a time instead: an 8-byte put to each peer, quiesced, in rotated order. */
    if (warmup) {
        const double w0 = now();
        static uint64_t probe = 0;
        for (int i = 0; i < n; ++i) {
            const int dst = (me + i) % n;
            shmem_uint64_put(&recv[(size_t)me * BLOCK_WORDS], &probe, 1, dst);
            shmem_quiet();
        }
        shmem_barrier_all();
        if (me == 0) { printf("  warmup %.3f s\n", now() - w0); fflush(stdout); }
    }

    const double t0 = now();
    for (int r = 0; r < rounds; ++r) {
        long since = 0;
        /* Rotate the start so all PEs do not target rank 0 first. Same as ISx. */
        for (int i = 0; i < n; ++i) {
            const int dst = (me + i) % n;
            /* The contended part: a blocking fetch-add against one address on dst. */
            if (use_atomic) (void)shmem_atomic_fetch_add(&receive_offset, 1LL, dst);
            shmem_uint64_put(&recv[(size_t)me * BLOCK_WORDS], send, BLOCK_WORDS, dst);
            if (quiet_every && ++since >= quiet_every) { shmem_quiet(); since = 0; }
        }
        shmem_quiet();
        shmem_barrier_all();
        if (me == 0 && (r < 3 || r % 32 == 0))
            { printf("  round %d done (%.3f s)\n", r, now() - t0); fflush(stdout); }
    }

    if (me == 0) printf("COMPLETED in %.3f s\n", now() - t0);
    shmem_free(recv);
    free(send);
    shmem_finalize();
    return 0;
}

/*
 * ISx64 on NVSHMEM. One endpoint per GPU.
 *
 * The requirements name `nvshmem_put64_nblock` alongside `shmem_put64`, so NVSHMEM is a
 * responsive programming model for this study. It is also the reason to prefer GPUs here:
 * inside an NVLink domain a remote write is a memory operation, so none of the libfabric
 * connection-establishment machinery documented in
 * results/ROOTCAUSE_connection_establishment.md exists to fail.
 *
 * The three ISx phases are unchanged from src/isx64/isx64_win.c:
 *   1. each PE generates its own keys with a PRNG, locally, nothing on the wire
 *   2. one-sided put of each destination's keys into that peer's symmetric heap
 *   3. local sort, then a boundary check against the neighbouring PE
 *
 * SCHEDULE. Destinations on the outside, windows inside, matching isx64_stream.c. At step
 * i, PE p sends to (p + i) % n, so every PE targets a distinct destination and receives
 * from exactly one sender per step. The symmetric window is therefore ONE slot rather
 * than one per peer: WINDOW * 8 bytes per PE instead of NUM_PES * WINDOW * 8. At 4,096
 * endpoints that is 128 KB instead of 512 MB. See docs/streamed_exchange_design.md.
 *
 * NVLINK DOMAIN SIZE. NVLink tops out at 72 GPUs. Above that, NVSHMEM falls back to RoCE
 * between domains and the network re-enters the picture. At 4,096 endpoints that is 57
 * NVL72 domains, so a flat all-to-all still crosses the network for 56/57 of its traffic.
 * Making the bucket assignment rack-aware, so most keys stay in the domain that generated
 * them, is the change that matters at scale. Not implemented here; see the README.
 *
 * BUILD:
 *   nvcc -O3 -std=c++17 -arch=sm_80 \
 *        -I$NVSHMEM_HOME/include -I../isx64 \
 *        isx_nvshmem.cu ../isx64/pcg_basic.c \
 *        -L$NVSHMEM_HOME/lib -lnvshmem_host -lnvshmem_device -lcudart -o isx_nvshmem
 *
 * RUN (one process per GPU):
 *   nvshmrun -n 2 ./isx_nvshmem <keys_per_pe> [iters]
 *   # or under Slurm: srun -n 2 ./isx_nvshmem <keys_per_pe> [iters]
 *
 * STATUS: validated on 2 x A100 with NVLink. Never run on GB300. The algorithm and the
 * NVSHMEM calls are exercised; nothing here says anything about GB300 performance.
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cinttypes>
#include <cmath>
#include <ctime>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <nvshmem.h>
#include <nvshmemx.h>

extern "C" {
#include "params.h"
}

#ifndef WINDOW_KEYS
#define WINDOW_KEYS (1u << 20)      /* 8 MB per window, one slot per PE */
#endif

#define CUDA_OK(call)                                                              \
  do {                                                                             \
    cudaError_t _e = (call);                                                       \
    if (_e != cudaSuccess) {                                                       \
      fprintf(stderr, "PE %d: %s:%d %s\n", nvshmem_my_pe(), __FILE__, __LINE__,    \
              cudaGetErrorString(_e));                                             \
      nvshmem_global_exit(1);                                                      \
    }                                                                              \
  } while (0)

static double now_s(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------------------------------------------------------------------------------
 * Phase 1. Each PE generates its own keys.
 *
 * Seeded from the PE id so runs are reproducible with the same configuration, which the
 * study requires. Generating centrally and scattering would not be ISx phase 1, and it
 * would cap the dataset at one device: the same defect that was found and fixed in the
 * JAX implementation.
 * --------------------------------------------------------------------------------- */
__global__ void gen_keys(KEY_TYPE *keys, uint64_t n, uint64_t max_key, uint64_t seed)
{
  const uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  curandStatePhilox4_32_10_t st;
  curand_init(seed, tid, 0, &st);
  for (uint64_t i = tid; i < n; i += stride) {
    /* Two 32-bit draws make a 64-bit value. curand has no native 64-bit uniform. */
    const uint64_t hi = curand(&st), lo = curand(&st);
    keys[i] = ((hi << 32) | lo) % max_key;
  }
}

__global__ void compute_dest(const KEY_TYPE *keys, uint32_t *dest, uint64_t n,
                             uint64_t bucket_width, uint32_t n_pes)
{
  const uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  for (uint64_t i = tid; i < n; i += stride) {
    uint64_t b = keys[i] / bucket_width;
    dest[i] = (uint32_t)(b >= n_pes ? n_pes - 1 : b);
  }
}

/* Counts per destination from the sorted dest array, by finding run boundaries. */
__global__ void count_dests(const uint32_t *dest_sorted, uint64_t n,
                            uint64_t *counts, uint32_t n_pes)
{
  const uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  for (uint64_t i = tid; i < n; i += stride)
    atomicAdd((unsigned long long *)&counts[dest_sorted[i]], 1ULL);
}

__global__ void check_bucket(const KEY_TYPE *keys, uint64_t n,
                             uint64_t lo, uint64_t hi, int *err)
{
  const uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  for (uint64_t i = tid; i < n; i += stride) {
    if (keys[i] < lo || keys[i] > hi) atomicExch(err, 1);
    if (i > 0 && keys[i] < keys[i - 1]) atomicExch(err, 2);
  }
}

int main(int argc, char **argv)
{
  nvshmem_init();
  const int me = nvshmem_my_pe();
  const int n_pes = nvshmem_n_pes();
  CUDA_OK(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));

  if (argc < 2) {
    if (me == 0) printf("usage: %s <keys_per_pe> [iters]\n", argv[0]);
    nvshmem_finalize();
    return 1;
  }
  const uint64_t keys_per_pe = strtoull(argv[1], NULL, 10);
  const int iters = (argc > 2) ? atoi(argv[2]) : 1;

  const uint64_t max_key = DEFAULT_MAX_KEY;
  const uint64_t bucket_width = (uint64_t)ceil((double)max_key / n_pes);
  const uint64_t total_keys = keys_per_pe * (uint64_t)n_pes;
  const uint64_t recv_cap = (uint64_t)(keys_per_pe * 1.02) + 4096;

  if (me == 0) {
    printf("ISx-NVSHMEM  %d PEs (1 per GPU)\n", n_pes);
    printf("  keys/PE       : %" PRIu64 "\n", keys_per_pe);
    printf("  total keys    : %" PRIu64 "  (%.2f GB)\n", total_keys, total_keys * 8.0 / 1e9);
    printf("  max key       : 2^%.0f\n", log2((double)max_key));
    printf("  window        : %u keys, %.1f MB per PE (one slot, not one per peer)\n",
           WINDOW_KEYS, WINDOW_KEYS * 8.0 / 1e6);
    fflush(stdout);
  }

  /* Symmetric: one window slot and one count. Independent of PE count. */
  KEY_TYPE *win = (KEY_TYPE *)nvshmem_malloc(WINDOW_KEYS * sizeof(KEY_TYPE));
  long long *win_count = (long long *)nvshmem_malloc(sizeof(long long));
  long long *red_src = (long long *)nvshmem_malloc(sizeof(long long));
  long long *red_dst = (long long *)nvshmem_malloc(sizeof(long long));
  if (!win || !win_count || !red_src || !red_dst) {
    fprintf(stderr, "PE %d: nvshmem_malloc failed\n", me);
    nvshmem_global_exit(2);
  }

  /* Ordinary device memory. Not symmetric, so it costs nothing on peers. */
  KEY_TYPE *keys = NULL, *keys_sorted = NULL, *recv = NULL;
  uint32_t *dest = NULL, *dest_sorted = NULL;
  uint64_t *d_counts = NULL;
  CUDA_OK(cudaMalloc(&keys, keys_per_pe * sizeof(KEY_TYPE)));
  CUDA_OK(cudaMalloc(&keys_sorted, keys_per_pe * sizeof(KEY_TYPE)));
  CUDA_OK(cudaMalloc(&dest, keys_per_pe * sizeof(uint32_t)));
  CUDA_OK(cudaMalloc(&dest_sorted, keys_per_pe * sizeof(uint32_t)));
  CUDA_OK(cudaMalloc(&recv, recv_cap * sizeof(KEY_TYPE)));
  CUDA_OK(cudaMalloc(&d_counts, (size_t)n_pes * sizeof(uint64_t)));

  void *cub_tmp = NULL;
  size_t cub_bytes = 0, cub_bytes_b = 0;
  cub::DeviceRadixSort::SortPairs(NULL, cub_bytes, dest, dest_sorted,
                                  keys, keys_sorted, keys_per_pe);
  cub::DeviceRadixSort::SortKeys(NULL, cub_bytes_b, recv, recv, recv_cap);
  if (cub_bytes_b > cub_bytes) cub_bytes = cub_bytes_b;
  CUDA_OK(cudaMalloc(&cub_tmp, cub_bytes));

  uint64_t *h_counts = (uint64_t *)malloc((size_t)n_pes * sizeof(uint64_t));
  uint64_t *h_off = (uint64_t *)malloc((size_t)n_pes * sizeof(uint64_t));

  const int BLK = 256, GRID = 1024;
  double t_gen = 0, t_bucket = 0, t_exch = 0, t_sort = 0, t_total = 0;
  uint64_t recv_n = 0;
  int err = 0;

  for (int it = 0; it < iters + (int)BURN_IN; ++it) {
    if (it == (int)BURN_IN) t_gen = t_bucket = t_exch = t_sort = t_total = 0;
    nvshmem_barrier_all();
    const double tt0 = now_s();

    /* ---- phase 1: generate, locally ---- */
    double t0 = now_s();
    gen_keys<<<GRID, BLK>>>(keys, keys_per_pe, max_key, (uint64_t)me);
    CUDA_OK(cudaDeviceSynchronize());
    t_gen += now_s() - t0;

    /* ---- phase 2a: group by destination ----
     * Sorting the (dest, key) pairs by dest groups the keys by destination, which is what
     * the exchange needs. CUB does this in one pass; the CPU version hand-rolls an
     * American-flag permutation for the same result. */
    t0 = now_s();
    compute_dest<<<GRID, BLK>>>(keys, dest, keys_per_pe, bucket_width, (uint32_t)n_pes);
    cub::DeviceRadixSort::SortPairs(cub_tmp, cub_bytes, dest, dest_sorted,
                                    keys, keys_sorted, keys_per_pe);
    CUDA_OK(cudaMemset(d_counts, 0, (size_t)n_pes * sizeof(uint64_t)));
    count_dests<<<GRID, BLK>>>(dest_sorted, keys_per_pe, d_counts, (uint32_t)n_pes);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(h_counts, d_counts, (size_t)n_pes * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
    h_off[0] = 0;
    for (int d = 1; d < n_pes; ++d) h_off[d] = h_off[d - 1] + h_counts[d - 1];
    t_bucket += now_s() - t0;

    /* ---- phase 2b: the exchange ----
     * Destination outside, windows inside. Every PE receives from exactly one sender per
     * step, so the window needs a single slot. */
    t0 = now_s();
    recv_n = 0;

    uint64_t my_max = 0;
    for (int d = 0; d < n_pes; ++d) if (h_counts[d] > my_max) my_max = h_counts[d];
    long long my_max_ll = (long long)my_max, g_max_ll = 0;
    CUDA_OK(cudaMemcpy(red_src, &my_max_ll, sizeof(long long), cudaMemcpyHostToDevice));
    nvshmem_longlong_max_reduce(NVSHMEM_TEAM_WORLD, red_dst, red_src, 1);
    CUDA_OK(cudaMemcpy(&g_max_ll, red_dst, sizeof(long long), cudaMemcpyDeviceToHost));
    const uint64_t wins = ((uint64_t)g_max_ll + WINDOW_KEYS - 1) / WINDOW_KEYS;

    for (int i = 0; i < n_pes; ++i) {
      const int dst = (me + i) % n_pes;
      uint64_t sent = 0;
      for (uint64_t w = 0; w < wins; ++w) {
        const long long zero = 0;
        CUDA_OK(cudaMemcpy(win_count, &zero, sizeof(long long), cudaMemcpyHostToDevice));
        nvshmem_barrier_all();

        const uint64_t left = h_counts[dst] - sent;
        if (left) {
          const uint64_t k = left < WINDOW_KEYS ? left : WINDOW_KEYS;
          /* One-sided. The target does not participate. */
          nvshmem_uint64_put_nbi((uint64_t *)win,
                                 (const uint64_t *)(keys_sorted + h_off[dst] + sent),
                                 k, dst);
          const long long kk = (long long)k;
          nvshmem_longlong_p(win_count, kk, dst);
          sent += k;
        }
        nvshmem_quiet();
        nvshmem_barrier_all();

        long long got = 0;
        CUDA_OK(cudaMemcpy(&got, win_count, sizeof(long long), cudaMemcpyDeviceToHost));
        if (got) {
          if (recv_n + (uint64_t)got > recv_cap) {
            fprintf(stderr, "PE %d: recv overflow %" PRIu64 " + %lld > %" PRIu64 "\n",
                    me, recv_n, got, recv_cap);
            nvshmem_global_exit(3);
          }
          CUDA_OK(cudaMemcpy(recv + recv_n, win, (size_t)got * sizeof(KEY_TYPE),
                             cudaMemcpyDeviceToDevice));
          recv_n += (uint64_t)got;
        }
      }
    }
    t_exch += now_s() - t0;

    /* ---- phase 3: local sort ----
     * Radix sort is comparison-free and bandwidth-bound, so the absence of a 64-bit
     * integer datapath on Blackwell does not matter here. Digit extraction is a shift
     * and a mask. */
    t0 = now_s();
    if (recv_n > 1)
      cub::DeviceRadixSort::SortKeys(cub_tmp, cub_bytes, recv, recv, recv_n);
    CUDA_OK(cudaDeviceSynchronize());
    t_sort += now_s() - t0;

    nvshmem_barrier_all();
    t_total += now_s() - tt0;
  }

  /* ---- verification, same checks as the CPU version ---- */
  {
    const uint64_t lo = (uint64_t)me * bucket_width;
    const uint64_t hi = (me == n_pes - 1) ? max_key : (uint64_t)(me + 1) * bucket_width - 1;
    int *d_err = NULL;
    CUDA_OK(cudaMalloc(&d_err, sizeof(int)));
    CUDA_OK(cudaMemset(d_err, 0, sizeof(int)));
    check_bucket<<<GRID, BLK>>>(recv, recv_n, lo, hi, d_err);
    CUDA_OK(cudaDeviceSynchronize());
    int h_err = 0;
    CUDA_OK(cudaMemcpy(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_err == 1) { fprintf(stderr, "PE %d: key out of bucket\n", me); err = 1; }
    if (h_err == 2) { fprintf(stderr, "PE %d: not sorted\n", me); err = 1; }
    CUDA_OK(cudaFree(d_err));

    long long mine = (long long)recv_n, total = 0;
    CUDA_OK(cudaMemcpy(red_src, &mine, sizeof(long long), cudaMemcpyHostToDevice));
    nvshmem_longlong_sum_reduce(NVSHMEM_TEAM_WORLD, red_dst, red_src, 1);
    CUDA_OK(cudaMemcpy(&total, red_dst, sizeof(long long), cudaMemcpyDeviceToHost));
    if (me == 0 && total != (long long)total_keys) {
      fprintf(stderr, "lost keys: %lld of %" PRIu64 "\n", total, total_keys);
      err = 1;
    }
  }

  if (me == 0) {
    const double n = (double)iters;
    printf("\n=== results ===\n");
    printf("  time to solution    : %.3f s\n", t_total / n);
    printf("    generate  %.3f\n    bucket    %.3f\n    exchange  %.3f\n    radix     %.3f\n",
           t_gen / n, t_bucket / n, t_exch / n, t_sort / n);
    printf("  aggregate rate      : %.2f GB/s\n", total_keys * 8.0 / (t_total / n) / 1e9);
    printf("  verification        : %s\n", err ? "FAILED" : "PASSED");
  }

  nvshmem_free(win);
  nvshmem_free(win_count);
  nvshmem_free(red_src);
  nvshmem_free(red_dst);
  cudaFree(keys); cudaFree(keys_sorted); cudaFree(dest); cudaFree(dest_sorted);
  cudaFree(recv); cudaFree(d_counts); cudaFree(cub_tmp);
  free(h_counts); free(h_off);
  nvshmem_finalize();
  return err;
}

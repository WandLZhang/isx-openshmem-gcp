/*
 * Single-PE OpenSHMEM shim, for correctness testing only.
 *
 * This exists so the ISx64 algorithm can be proven correct on a laptop before anyone
 * pays for a cluster. It implements the handful of OpenSHMEM calls ISx64 uses, for
 * NUM_PES == 1, where every put is a local memcpy and every collective is the identity.
 *
 * What this DOES prove: the radix sort sorts, the bucket routing is correct, the
 * counters do not overflow, verification catches what it should.
 *
 * What this CANNOT prove: anything about the fabric. One-sided semantics, RDMA,
 * atomics against a remote PE and the all-to-all are all no-ops at one PE. Never quote
 * a performance number from a stub run.
 */
#ifndef _SHMEM_STUB_H
#define _SHMEM_STUB_H

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define SHMEM_REDUCE_MIN_WRKDATA_SIZE 16
#define SHMEM_REDUCE_SYNC_SIZE 16
#define SHMEM_SYNC_VALUE 0

static inline void shmem_init(void) {}
static inline void shmem_finalize(void) {}
static inline int  shmem_my_pe(void) { return 0; }
static inline int  shmem_n_pes(void) { return 1; }
static inline void *shmem_malloc(size_t n) { return malloc(n); }
static inline void shmem_free(void *p) { free(p); }
static inline void shmem_barrier_all(void) {}
static inline void shmem_quiet(void) {}
static inline void shmem_global_exit(int c) { exit(c); }

static inline long long shmem_atomic_fetch_add(long long *t, long long v, int pe)
{ (void)pe; const long long old = *t; *t += v; return old; }

static inline void shmem_uint64_put(uint64_t *dst, const uint64_t *src, size_t n, int pe)
{ (void)pe; memcpy(dst, src, n * sizeof(uint64_t)); }

static inline void shmem_longlong_sum_to_all(long long *dst, long long *src, int n,
                                             int s, int l, int sz, long long *w, long *y)
{ (void)s;(void)l;(void)sz;(void)w;(void)y; for (int i=0;i<n;++i) dst[i]=src[i]; }

static inline void shmem_double_max_to_all(double *dst, double *src, int n,
                                           int s, int l, int sz, double *w, long *y)
{ (void)s;(void)l;(void)sz;(void)w;(void)y; for (int i=0;i<n;++i) dst[i]=src[i]; }

#endif

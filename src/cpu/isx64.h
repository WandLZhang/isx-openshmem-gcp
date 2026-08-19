/*
 * ISx64 declarations. Derived from ISx v1.1 isx.h,
 * Copyright (c) 2015 Intel Corporation, BSD 3-clause. See LICENSE-ISx.
 *
 * Signatures differ from upstream wherever a count crossed the 32-bit boundary:
 * upstream returned and consumed `int *` for bucket sizes and offsets, which caps a PE
 * at 2.1e9 keys. They are COUNT_TYPE (int64_t) here.
 */
#ifndef _ISX64_H
#define _ISX64_H

#include <stdint.h>
#include "params.h"

extern uint64_t NUM_PES, TOTAL_KEYS, NUM_KEYS_PER_PE, NUM_BUCKETS, BUCKET_WIDTH;
extern uint64_t MAX_KEY_VAL, NUM_ITERATIONS, KEY_BUFFER_SIZE;

extern KEY_TYPE *my_bucket_keys;
extern long long int receive_offset;
extern long long int my_bucket_size;

#endif

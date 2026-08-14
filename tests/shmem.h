/* Resolves `#include <shmem.h>` to the single-PE shim when building the local
 * correctness test with -I tests. A real OpenSHMEM install provides its own shmem.h
 * earlier on the include path, so this file is never used on a cluster. */
#include "shmem_stub.h"

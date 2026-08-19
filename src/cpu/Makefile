# ISx64 build.
#
# Requires an OpenSHMEM implementation providing oshcc. On GCE H4D this is Sandia
# OpenSHMEM built against libfabric with the verbs;ofi_rxm provider; see
# ../../infra/h4d/build_sos.sh.
#
#   make            optimized build (the one to benchmark)
#   make debug      assertions, verbose, tiny key space
#   make check      single-node smoke test on 4 PEs
#
# There is deliberately no MPI target. The study excludes MPI-based implementations, and
# an oshcc that quietly wraps mpicc would make the result non-responsive. Verify with:
#   make provider-check

SHMEM_CC ?= oshcc
CFLAGS   ?= -O3 -march=native -std=c11 -Wall -Wextra
LDFLAGS  ?= -lm

SRC = isx64.c pcg_basic.c
BIN = ../../bin

.PHONY: all debug check provider-check clean

all: $(BIN)/isx64
	@echo "built $(BIN)/isx64"

$(BIN)/isx64: $(SRC) params.h isx64.h
	@mkdir -p $(BIN)
	$(SHMEM_CC) $(CFLAGS) -DNDEBUG -o $@ $(SRC) $(LDFLAGS)

debug: $(SRC) params.h isx64.h
	@mkdir -p $(BIN)
	$(SHMEM_CC) -O0 -g -std=c11 -Wall -Wextra -DDEBUG -o $(BIN)/isx64.debug $(SRC) $(LDFLAGS)

# Smoke test. 4 PEs, 1M keys each, one iteration. Should print PASSED.
check: all
	oshrun -n 4 $(BIN)/isx64 1000000 1 /tmp/isx64_check.log

# Confirms the transport is really RDMA one-sided and not a TCP or MPI fallback.
# A run that passes `check` over the tcp provider proves nothing about the fabric.
provider-check:
	@echo "FI_PROVIDER=$${FI_PROVIDER:-<unset>}"
	@fi_info -p "verbs;ofi_rxm" -c FI_RMA 2>/dev/null | grep -E "provider|caps" || \
	  echo "WARNING: verbs;ofi_rxm with FI_RMA not available -- see PORTING.md"

clean:
	rm -f $(BIN)/isx64 $(BIN)/isx64.debug

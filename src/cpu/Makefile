# ISx64 build.
#
# Requires an OpenSHMEM implementation providing oshcc. On GCE H4D this is Sandia
# OpenSHMEM built against libfabric with the psm3 provider; see
# ../../deploy/h4d/01_build_sos.sh.
#
#   make            all three exchange schedules
#   make debug      assertions, verbose, tiny key space
#   make check      single-node smoke test on 4 PEs
#
# isx64_win is the one every H4D measurement in results/ was taken with. isx64 is the
# direct port with the receive buffer in the symmetric heap, which caps the dataset at the
# heap ceiling. isx64_stream is the single-slot-window schedule that src/gpu mirrors.
#
# There is deliberately no MPI target. The study excludes MPI-based implementations, and
# an oshcc that quietly wraps mpicc would make the result non-responsive. Verify with:
#   make provider-check

SHMEM_CC ?= oshcc
CFLAGS   ?= -O3 -march=native -std=c11 -Wall -Wextra
LDFLAGS  ?= -lm

COMMON  = pcg_basic.c
VARIANTS = isx64 isx64_win isx64_stream
BIN = ../../bin

.PHONY: all debug check provider-check clean

all: $(addprefix $(BIN)/,$(VARIANTS))
	@echo "built: $(VARIANTS)"

$(BIN)/%: %.c $(COMMON) params.h isx64.h
	@mkdir -p $(BIN)
	$(SHMEM_CC) $(CFLAGS) -DNDEBUG -o $@ $< $(COMMON) $(LDFLAGS)

debug: isx64.c $(COMMON) params.h isx64.h
	@mkdir -p $(BIN)
	$(SHMEM_CC) -O0 -g -std=c11 -Wall -Wextra -DDEBUG -o $(BIN)/isx64.debug isx64.c $(COMMON) $(LDFLAGS)

# Smoke test. 4 PEs, 1M keys each, one iteration. Should print PASSED.
# oshrun needs a launcher; on a bare VM install mpich and use mpiexec.hydra instead.
check: all
	oshrun -n 4 $(BIN)/isx64_win 1000000 1 /tmp/isx64_check.log

# Confirms the transport is really RDMA one-sided and not a TCP or MPI fallback.
# A run that passes `check` over the tcp provider proves nothing about the fabric.
provider-check:
	@echo "FI_PROVIDER=$${FI_PROVIDER:-<unset>}"
	@fi_info -p psm3 -c FI_RMA 2>/dev/null | grep -E "provider|caps" || \
	  echo "WARNING: psm3 with FI_RMA not available -- see results/h4d-psm3.md"

clean:
	rm -f $(addprefix $(BIN)/,$(VARIANTS)) $(BIN)/isx64.debug

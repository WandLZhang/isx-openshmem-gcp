#!/usr/bin/env bash
# Weak-scaling ladder for ISx64 on H4D, then the 100 TB target.
#
#   bash deploy/h4d/run_ladder.sh              # 4 8 16 22 32 143
#   bash deploy/h4d/run_ladder.sh 4 8 16       # just those rungs
#
# Keys per PE is held constant at every rung on purpose. This is weak scaling, so the
# data grows with the node count and each PE does the same amount of work throughout.
# Flat wall time is the ideal. Whether it stays flat is the measurement, so a climb at
# high endpoint counts is a result rather than a failure.
#
# 455,729,166 keys per PE is 700 GB of keys per node, which is 1,414 GB resident at the
# measured 2.02x footprint against about 1,440 GB usable. That is the largest per-node
# dataset validated, 20/20 on two nodes.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNGS=("$@")
[ ${#RUNGS[@]} -eq 0 ] && RUNGS=(4 8 16 22 32 143)

PPN=${PPN:-192}
KEYS=${KEYS:-455729166}
# Capture a caller override once. Reading ISX64_TIMEOUT inside the loop does not work,
# because the first rung exports it and every later ${ISX64_TIMEOUT:-...} then finds it
# already set.
USER_TIMEOUT=${ISX64_TIMEOUT:-}

OUT=${ISX64_OUT:-$HOME/isx64_results}
mkdir -p "$OUT"
CSV="$OUT/ladder.csv"
[ -f "$CSV" ] || echo "nodes,pes,keys_per_pe,total_tb,seconds,result" > "$CSV"

printf '%-7s %-9s %-9s %s\n' nodes PEs data status
results=()

for N in "${RUNGS[@]}"; do
  PES=$((N * PPN))
  TB=$(awk -v n="$N" -v p="$PPN" -v k="$KEYS" 'BEGIN{printf "%.2f", n*p*k*8/1e12}')

  # The last rung holds the most nodes for the longest, so give it room to finish but
  # do not let a sick run retry five times across 143 nodes.
  if [ -n "$USER_TIMEOUT" ]; then
    T=$USER_TIMEOUT
  elif [ "$N" -ge 100 ]; then
    T=3600
  else
    T=1800
  fi
  export ISX64_TIMEOUT=$T

  if [ "$N" -ge 100 ]; then ATTEMPTS=2; else ATTEMPTS=3; fi

  T0=$(date +%s)
  if bash "$HERE/run_isx64.sh" "$N" "$PPN" "$KEYS" "$ATTEMPTS" > "$OUT/ladder.$N.log" 2>&1; then
    STATUS=PASS
  else
    STATUS=FAIL
  fi
  T1=$(date +%s)
  SECS=$((T1 - T0))

  printf '%-7s %-9s %-9s %s in %s s\n' "$N" "$PES" "${TB} TB" "$STATUS" "$SECS"
  echo "$N,$PES,$KEYS,$TB,$SECS,$STATUS" >> "$CSV"
  results+=("$N|$PES|$TB|$SECS|$STATUS")
done

echo
echo "=== summary ==="
printf '%-8s %-10s %-10s %-10s %s\n' nodes PEs data seconds result
for r in "${results[@]}"; do
  IFS='|' read -r n p tb s st <<< "$r"
  printf '%-8s %-10s %-10s %-10s %s\n' "$n" "$p" "${tb} TB" "$s" "$st"
done
echo
echo "csv:  $CSV"
echo "logs: $OUT/ladder.<nodes>.log"

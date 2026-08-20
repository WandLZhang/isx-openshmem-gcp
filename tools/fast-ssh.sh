#!/usr/bin/env bash
# Fast repeated SSH to cluster nodes.
#
# Measured 2026-08-14 against an h4d-highmem-192 in us-central1-b, per command:
#
#   gcloud compute ssh --tunnel-through-iap         3.5  s
#   raw ssh to an external IP, fresh connection     0.7  s   (one 11.9 s outlier in 3)
#   raw ssh + ControlMaster, external IP            0.11 s
#   raw ssh + ControlMaster, over IAP               0.11 s
#
# Two things follow.
#
# The win is connection reuse, not the external endpoint. Multiplexed IAP and multiplexed
# external are the same speed, so there is no reason to attach public IPs or relax
# compute.vmExternalIpAccess to get a fast shell. On a build that runs hundreds of small
# remote commands, 3.5 s versus 0.11 s is the difference between a coffee break and not.
#
# gcloud compute ssh is slow because it re-resolves the instance, re-checks OS Login and
# rebuilds the tunnel on every invocation. It is the right tool for the first connection
# and the wrong one for the next two hundred.
#
#   ./hssh.sh isx-probe0 'hostname'
#   ./hssh.sh isx-probe0 < script.sh
#   HOSTS="n1 n2 n3" ./hssh.sh --all 'uptime'
set -uo pipefail

PROJECT="${PROJECT:?set PROJECT}"
ZONE="${ZONE:-us-central1-b}"
SSH_USER="${SSH_USER:-$(gcloud config get-value account 2>/dev/null | tr '@.' '__')}"
KEY="${KEY:-$HOME/.ssh/google_compute_engine}"

# ControlPersist keeps the master open after the last session so the next command reuses
# it. 600 s covers a normal build loop; raise it for long sessions.
COMMON=(
  -i "${KEY}"
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
  -o ControlMaster=auto
  -o ControlPersist=600
)

hssh() {
  local host="$1"; shift
  ssh "${COMMON[@]}" \
    -o "ControlPath=/tmp/hssh-${PROJECT}-%r@%h:%p" \
    -o "ProxyCommand=gcloud compute start-iap-tunnel ${host} 22 --listen-on-stdin --project=${PROJECT} --zone=${ZONE} --verbosity=error" \
    "${SSH_USER}@${host}" "$@"
}

if [[ "${1:-}" == "--all" ]]; then
  shift
  : "${HOSTS:?set HOSTS to a space separated list}"
  for h in ${HOSTS}; do
    printf '=== %s ===\n' "${h}"
    hssh "${h}" "$@"
  done
else
  hssh "$@"
fi

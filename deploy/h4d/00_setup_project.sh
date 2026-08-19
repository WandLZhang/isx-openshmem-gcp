#!/usr/bin/env bash
# Prepare a clean project for H4D Cloud RDMA.
#
# The Government deliverable requires a recipe that reproduces the environment without
# intervention, so this script assumes an empty project and does everything from there.
#
# The org-policy section is the part that is easy to miss and expensive to debug. On a
# default-hardened org, one constraint stops H4D dead and the failure message does not
# mention it: compute.trustedImageProjects does not include cloud-hpc-image-public, which
# is where the HPC VM image lives. Cloud RDMA requires that image. Without the override
# the instance create fails on image access, which reads like an IAM problem.
set -euo pipefail

PROJECT="${PROJECT:?set PROJECT}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-us-central1-b}"
RDMA_PREFIX="${RDMA_PREFIX:-isx-rdma}"
RDMA_CIDR="${RDMA_CIDR:-10.20.0.0/16}"
VPC_CIDR="${VPC_CIDR:-10.10.0.0/16}"

echo "==> project ${PROJECT}, zone ${ZONE}"

echo "==> enabling APIs"
gcloud services enable \
  compute.googleapis.com \
  orgpolicy.googleapis.com \
  file.googleapis.com \
  secretmanager.googleapis.com \
  --project="${PROJECT}"

# ---------------------------------------------------------------------------------
# Org policy: return this project to Google defaults.
#
# A default-hardened sandbox org inherits constraints that were written for shared
# multi-tenant projects. Several of them block an HPC RDMA cluster, and their failure
# messages do not name the constraint, so the debugging cost is high and repeated.
# Rather than guess which ones bite, reset the whole set on this project.
#
# `gcloud org-policies reset` sets a project-level policy of `reset: true`, which
# restores the Google default and ignores inheritance. Scope is this project only; the
# org and every other project are untouched.
#
# Setting these needs orgpolicy.policy.set. In some orgs that permission cannot be
# granted through a custom role, so an org admin may have to run this section.
#
# The one that is non-negotiable is compute.trustedImageProjects. Cloud RDMA requires
# the HPC VM image from cloud-hpc-image-public, and if that project is not allowed the
# instance create fails on image access, which reads like an IAM problem.
# ---------------------------------------------------------------------------------
# Check the effective policy first and reset only if it actually blocks this build.
# Blanket-resetting everything would disable constraints that have nothing to do with
# H4D, and this script is a deliverable someone else re-runs in their own environment.
#
# DRY_RUN=1 reports what would change without changing it.
effective() {
  gcloud org-policies describe "$1" --project="${PROJECT}" --effective \
    --format="value(spec.rules)" 2>/dev/null
}

reset_if() {
  local constraint="$1" blocks="$2" why="$3"
  if [[ "${blocks}" != "yes" ]]; then
    printf '    ok     %-45s %s\n' "${constraint}" "not blocking"
    return
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '    WOULD  %-45s %s\n' "${constraint}" "${why}"
    return
  fi
  if gcloud org-policies reset "${constraint}" --project="${PROJECT}" >/dev/null 2>&1; then
    printf '    RESET  %-45s %s\n' "${constraint}" "${why}"
  else
    printf '    FAILED %-45s %s\n' "${constraint}" "needs orgpolicy.policy.set"
    BLOCKED_ON_ADMIN+=("${constraint}")
  fi
}

BLOCKED_ON_ADMIN=()
echo "==> org policy: checking what actually blocks an H4D RDMA cluster"

# 1. The hard blocker. Cloud RDMA requires the HPC VM image, which lives in
#    cloud-hpc-image-public. If the allowlist exists and omits that project, no H4D node
#    boots, and the error looks like an IAM failure rather than a policy one.
TIP=$(effective compute.trustedImageProjects)
if [[ -n "${TIP}" && "${TIP}" != *"cloud-hpc-image-public"* ]]; then
  reset_if compute.trustedImageProjects yes "allowlist omits cloud-hpc-image-public"
else
  reset_if compute.trustedImageProjects no ""
fi

# 2. Shielded VM. Only a problem if the HPC image does not publish shielded support.
#    Checked rather than assumed, because on most orgs this is fine.
if [[ "$(effective compute.requireShieldedVm)" == *"enforce': True"* ]]; then
  SHIELD_OK=$(gcloud compute images list --project=cloud-hpc-image-public \
    --filter="family~hpc-rocky-linux-8" --format="value(shieldedInstanceInitialState)" \
    --limit=1 2>/dev/null)
  if [[ -z "${SHIELD_OK}" ]]; then
    reset_if compute.requireShieldedVm yes "HPC image does not advertise shielded support"
  else
    reset_if compute.requireShieldedVm no ""
  fi
else
  reset_if compute.requireShieldedVm no ""
fi

# 3. External IPs. This build reaches the login node over IAP, so a denyAll here is
#    only a blocker if you intend to attach public IPs.
if [[ "$(effective compute.vmExternalIpAccess)" == *"denyAll': True"* && "${WANT_EXTERNAL_IP:-0}" == "1" ]]; then
  reset_if compute.vmExternalIpAccess yes "WANT_EXTERNAL_IP=1 but external IPs are denied"
else
  reset_if compute.vmExternalIpAccess no ""
fi

# Left alone deliberately, with the reason, so the next person does not re-litigate it:
#   compute.disableSerialPortAccess  makes fabric debugging harder but blocks nothing.
#                                    Set ALLOW_SERIAL=1 to reset it anyway.
#   compute.requireOsLogin           Cluster Toolkit and Slurm work with OS Login.
#   compute.vmCanIpForward           H4D uses multiple vNICs but does not forward
#                                    packets it did not originate, so denyAll is fine.
#   iam.* and storage.*              nothing in this build touches them.
if [[ "${ALLOW_SERIAL:-0}" == "1" ]]; then
  reset_if compute.disableSerialPortAccess yes "ALLOW_SERIAL=1, for fabric debugging"
fi

if [[ ${#BLOCKED_ON_ADMIN[@]} -gt 0 ]]; then
  echo
  echo "An org admin must reset these; orgpolicy.policy.set is not grantable via a" >&2
  echo "custom role in some orgs:" >&2
  for C in "${BLOCKED_ON_ADMIN[@]}"; do
    echo "  gcloud org-policies reset ${C} --project=${PROJECT}" >&2
  done
fi

# Custom constraints cannot be guessed; they are org-specific and named by whoever
# created them. List whatever is still in force so nothing blocks silently later.
echo "==> custom constraints still in force on this project"
gcloud org-policies list --project="${PROJECT}" --format="value(constraint)" 2>/dev/null \
  | grep -i "custom\." | sed 's/^/    /' || true
echo "    (a custom constraint that restricts firewall ranges will not stop this build;"
echo "     the RDMA VPC only needs internal CIDR rules. It will stop a public L4 LB.)"

# Verify the one that matters rather than trusting the reset.
echo "==> verifying the HPC image is reachable"
if gcloud compute images list --project=cloud-hpc-image-public \
     --filter="family~hpc-rocky-linux-8" --format="value(name)" --limit=1 2>/dev/null | grep -q .; then
  echo "    OK, cloud-hpc-image-public is readable"
else
  echo "" >&2
  echo "FATAL: cannot read cloud-hpc-image-public." >&2
  echo "H4D Cloud RDMA requires the HPC VM image and no node will boot without it." >&2
  echo "compute.trustedImageProjects is still restricting this project. An org admin" >&2
  echo "must run: gcloud org-policies reset compute.trustedImageProjects --project=${PROJECT}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------------
# Networking.
#
# H4D needs two networks. A normal VPC carries TCP/IP and SSH on a gVNIC. A second VPC
# built with the Falcon network profile carries Cloud RDMA on a single IRDMA vNIC. The
# RDMA VPC has no internet path and is zonal in effect: all nodes must sit in one zone
# because Cloud RDMA does not cross zones. That single fact caps the size of any
# OpenSHMEM job on GCP at whatever one zone can hold.
# ---------------------------------------------------------------------------------
echo "==> standard VPC"
gcloud compute networks create "${RDMA_PREFIX}-mgmt" \
  --project="${PROJECT}" --subnet-mode=custom || true
gcloud compute networks subnets create "${RDMA_PREFIX}-mgmt-sub" \
  --project="${PROJECT}" --network="${RDMA_PREFIX}-mgmt" \
  --region="${REGION}" --range="${VPC_CIDR}" || true

# Internal-only firewall. Deliberately not 0.0.0.0/0: some orgs enforce a custom
# constraint that rejects any rule that wide, and the RDMA fabric does not need it.
gcloud compute firewall-rules create "${RDMA_PREFIX}-mgmt-internal" \
  --project="${PROJECT}" --network="${RDMA_PREFIX}-mgmt" \
  --allow=tcp,udp,icmp --source-ranges="${VPC_CIDR}" || true
gcloud compute firewall-rules create "${RDMA_PREFIX}-mgmt-iap-ssh" \
  --project="${PROJECT}" --network="${RDMA_PREFIX}-mgmt" \
  --allow=tcp:22 --source-ranges=35.235.240.0/20 || true

echo "==> Falcon RDMA VPC (profile ${ZONE}-vpc-falcon)"
gcloud compute networks create "${RDMA_PREFIX}-net" \
  --project="${PROJECT}" \
  --network-profile="${ZONE}-vpc-falcon" \
  --subnet-mode=custom
gcloud compute networks subnets create "${RDMA_PREFIX}-sub-0" \
  --project="${PROJECT}" --network="${RDMA_PREFIX}-net" \
  --region="${REGION}" --range="${RDMA_CIDR}"

# ---------------------------------------------------------------------------------
# Placement. RDMA latency depends on physical locality, so the nodes must be packed.
# Placement policies have a maximum size; past it you need several and locality across
# them is not guaranteed. Finding where that starts to hurt is one of the study's
# stated goals ("identification of inflection points").
# ---------------------------------------------------------------------------------
echo "==> compact placement policy"
gcloud compute resource-policies create group-placement "${RDMA_PREFIX}-compact" \
  --project="${PROJECT}" --region="${REGION}" --collocation=COLLOCATED || true

echo "==> the quota that actually binds H4D (NOT the general CPUS quota)"
# CPUS_PER_VM_FAMILY defaults to 500 per region. h4d-highmem-192 is 192 vCPU, so that is
# 2 nodes, regardless of a CPUS quota in the thousands. Self-service override is capped at
# 500, so growing past 2 nodes needs a quota request through support.
gcloud alpha services quota list --service=compute.googleapis.com \
  --consumer=projects/${PROJECT} --filter="metric:cpus_per_vm_family" 2>/dev/null \
  | grep -iA2 "H4D" | head -6 || echo "    (could not read; check manually)"
echo "    500 / 192 vCPU = 2 nodes. For 128 nodes request 24576."

cat <<EOF

Done. Verify before building the cluster:

  gcloud compute networks describe ${RDMA_PREFIX}-net --project=${PROJECT} \\
    --format='value(name,networkProfile)'          # must show a falcon profile

  gcloud compute images list --project=cloud-hpc-image-public \\
    --filter='family~hpc-rocky-linux-8' --format='value(name,creationTimestamp)'
                                                    # need build 20250917 or later

Next: 01_cluster.sh
EOF

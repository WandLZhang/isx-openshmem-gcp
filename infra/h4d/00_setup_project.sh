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
# Org policy overrides.
#
# Setting these needs orgpolicy.policy.set. In some orgs that permission cannot be
# granted through a custom role, so this may have to be run by an org admin. Each
# override is scoped to this project alone.
# ---------------------------------------------------------------------------------
echo "==> org policy: allow the HPC VM image (the blocker)"
cat > /tmp/trusted_images.yaml <<EOF
name: projects/${PROJECT}/policies/compute.trustedImageProjects
spec:
  inheritFromParent: true
  rules:
    - values:
        allowedValues:
          - projects/cloud-hpc-image-public
EOF
gcloud org-policies set-policy /tmp/trusted_images.yaml --project="${PROJECT}" || {
  echo "FAILED to set trustedImageProjects." >&2
  echo "H4D cannot boot without projects/cloud-hpc-image-public. Ask an org admin." >&2
  exit 1
}

echo "==> org policy: serial console, for debugging the fabric"
gcloud org-policies reset compute.disableSerialPortAccess --project="${PROJECT}" || true

echo "==> org policy: IP forwarding, H4D uses 2-10 vNICs"
gcloud org-policies reset compute.vmCanIpForward --project="${PROJECT}" || true

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

cat <<EOF

Done. Verify before building the cluster:

  gcloud compute networks describe ${RDMA_PREFIX}-net --project=${PROJECT} \\
    --format='value(name,networkProfile)'          # must show a falcon profile

  gcloud compute images list --project=cloud-hpc-image-public \\
    --filter='family~hpc-rocky-linux-8' --format='value(name,creationTimestamp)'
                                                    # need build 20250917 or later

Next: 01_cluster.sh
EOF

#!/usr/bin/env bash
# One-time setup before deploying hpc-slurm.yaml:
#   1. Enables the required Google Cloud APIs
#   2. Grants the deployer (a person) the roles needed to create the cluster and SSH in
#   3. Grants the default Compute Engine service account the roles the Slurm
#      controller needs to create and delete compute VMs
#
# Run it as someone who can change IAM on the project (Owner or Project IAM Admin).
#
# Usage:
#   ./setup-permissions.sh <PROJECT_ID> <DEPLOYER_EMAIL>
# Example:
#   ./setup-permissions.sh my-gcp-project alice@example.com
#
# If the deployer is already a project Owner, set SKIP_USER_ROLES=1 to only
# enable APIs, add the SSH roles, and set up the service account.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <PROJECT_ID> <DEPLOYER_EMAIL>" >&2
  exit 1
fi

PROJECT_ID="$1"
DEPLOYER="user:$2"
SKIP_USER_ROLES="${SKIP_USER_ROLES:-0}"

grant() {  # grant <member> <role>
  echo "  + $2 -> $1"
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="$1" --role="$2" --condition=None --quiet >/dev/null
}

echo "== Project: $PROJECT_ID"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
VM_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

# 1. APIs ----------------------------------------------------------------
echo "== Enabling APIs (can take a minute)"
gcloud services enable --project "$PROJECT_ID" \
  compute.googleapis.com \
  file.googleapis.com \
  storage.googleapis.com \
  serviceusage.googleapis.com \
  iap.googleapis.com \
  oslogin.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com
# Add servicenetworking.googleapis.com if you add the private_service_access module.

# Enabling the Compute API creates the default service account; make sure it exists.
if ! gcloud iam service-accounts describe "$VM_SA" --project "$PROJECT_ID" >/dev/null 2>&1; then
  echo "ERROR: default compute service account $VM_SA not found." >&2
  echo "It may have been deleted or disabled. Create a dedicated service account and" >&2
  echo "set it in the blueprint instead (service_account_email on the controller)." >&2
  exit 1
fi

# 2. Deployer roles ------------------------------------------------------
echo "== Deployer roles for $DEPLOYER"
if [[ "$SKIP_USER_ROLES" != "1" ]]; then
  grant "$DEPLOYER" roles/compute.admin                     # VPC, firewall, NAT, VMs, templates
  grant "$DEPLOYER" roles/file.editor                       # Filestore /home share
  grant "$DEPLOYER" roles/storage.admin                     # cluster config bucket + its IAM
  grant "$DEPLOYER" roles/iam.serviceAccountUser            # attach the service account to VMs
  grant "$DEPLOYER" roles/serviceusage.serviceUsageConsumer # use the enabled APIs
else
  echo "  (skipped: SKIP_USER_ROLES=1)"
fi
# SSH access to the private login VM
grant "$DEPLOYER" roles/iap.tunnelResourceAccessor          # SSH through IAP
grant "$DEPLOYER" roles/compute.osAdminLogin                # OS Login with sudo

# 3. VM service account roles -------------------------------------------
echo "== Service account roles for $VM_SA"
VM_MEMBER="serviceAccount:$VM_SA"
grant "$VM_MEMBER" roles/compute.instanceAdmin.v1   # controller creates/deletes compute VMs
grant "$VM_MEMBER" roles/iam.serviceAccountUser     # attach this SA to the new compute VMs
grant "$VM_MEMBER" roles/storage.objectViewer       # read cluster config from the bucket
grant "$VM_MEMBER" roles/logging.logWriter          # send logs
grant "$VM_MEMBER" roles/monitoring.metricWriter    # send metrics

# 4. Show the result ----------------------------------------------------
echo "== Current roles"
for M in "$DEPLOYER" "$VM_MEMBER"; do
  echo "-- $M"
  gcloud projects get-iam-policy "$PROJECT_ID" \
    --flatten="bindings[].members" \
    --filter="bindings.members:$M" \
    --format="value(bindings.role)"
done

echo
echo "Done. IAM changes can take a minute or two to take effect."
echo "Next: check C2 vCPU and Filestore quotas, then run gcluster create/deploy."

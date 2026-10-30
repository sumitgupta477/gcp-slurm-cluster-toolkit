#!/usr/bin/env bash
# Build the serving container with Cloud Build and push it to Artifact Registry.
# Run once (and again whenever serving/ changes) from your laptop or Cloud Shell.
#
#   ./build_image.sh <PROJECT_ID> [REGION]
#
# Prints the image URI to use as SERVING_IMAGE for `train.py --register`.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <PROJECT_ID> [REGION]" >&2
  exit 1
fi
PROJECT_ID="$1"; REGION="${2:-us-central1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/serving/hf-sentiment-server:latest"

gcloud builds submit "$HERE/serving" --project "$PROJECT_ID" --region "$REGION" --tag "$IMAGE" >&2
echo "$IMAGE"

#!/usr/bin/env bash
# Build the serving image, register the trained model with Vertex AI, deploy it to an
# endpoint with one NVIDIA L4 GPU, and send a test prediction.
# Run from this folder on your laptop or Cloud Shell (not on the cluster).
#
#   ./deploy_vertex.sh <PROJECT_ID> <GCS_MODEL_URI> [REGION]
#   e.g. ./deploy_vertex.sh my-project gs://my-bucket/models/sentiment-v1
#
# COST: the endpoint keeps one g2-standard-8 + L4 running (and billing) until you run
#       ./cleanup_vertex.sh. Deploying usually takes 10-20 minutes.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <PROJECT_ID> <GCS_MODEL_URI> [REGION]" >&2
  exit 1
fi
PROJECT_ID="$1"; MODEL_URI="${2%/}"; REGION="${3:-us-central1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME="sentiment-$(date +%Y%m%d-%H%M%S)"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/serving/hf-sentiment-server:latest"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"   # 8 vCPUs, 32 GB RAM, 1x NVIDIA L4

gcloud config set project "$PROJECT_ID" >/dev/null

echo "== Checking the model exists at $MODEL_URI"
gcloud storage ls "$MODEL_URI/config.json" >/dev/null

echo "== 1/4 Building the serving image with Cloud Build: $IMAGE"
gcloud builds submit "$HERE/serving" --region "$REGION" --tag "$IMAGE"

echo "== 2/4 Registering the model with Vertex AI: $NAME"
gcloud ai models upload --region "$REGION" \
  --display-name "$NAME" \
  --artifact-uri "$MODEL_URI" \
  --container-image-uri "$IMAGE" \
  --container-ports 8080 \
  --container-health-route /health \
  --container-predict-route /predict
MODEL_ID="$(gcloud ai models list --region "$REGION" --filter="displayName=$NAME" \
  --format='value(name.basename())' --limit 1)"
echo "   model ID: $MODEL_ID"

echo "== 3/4 Creating an endpoint and deploying (slowest step)"
gcloud ai endpoints create --region "$REGION" --display-name "$NAME-endpoint"
ENDPOINT_ID="$(gcloud ai endpoints list --region "$REGION" --filter="displayName=$NAME-endpoint" \
  --format='value(name.basename())' --limit 1)"
echo "   endpoint ID: $ENDPOINT_ID"
gcloud ai endpoints deploy-model "$ENDPOINT_ID" --region "$REGION" \
  --model "$MODEL_ID" --display-name "$NAME" \
  --machine-type "$MACHINE_TYPE" \
  --accelerator type=nvidia-l4,count=1 \
  --min-replica-count 1 --max-replica-count 1 \
  --traffic-split 0=100

echo "== 4/4 Test prediction"
REQUEST="$(mktemp)"
cat > "$REQUEST" <<'JSON'
{"instances": ["An absolute masterpiece, I loved every minute.", "Boring, too long, and the ending made no sense."]}
JSON
gcloud ai endpoints predict "$ENDPOINT_ID" --region "$REGION" --json-request "$REQUEST"
rm -f "$REQUEST"

cat > "$HERE/.last_deploy" <<EOF
PROJECT_ID=$PROJECT_ID
REGION=$REGION
ENDPOINT_ID=$ENDPOINT_ID
MODEL_ID=$MODEL_ID
EOF
echo
echo "Endpoint $ENDPOINT_ID is LIVE and billing. When finished:  ./cleanup_vertex.sh"

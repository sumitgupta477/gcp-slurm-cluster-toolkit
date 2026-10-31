#!/usr/bin/env bash
# Serve a model from a Vertex AI endpoint with one NVIDIA L4 GPU, then send a test prediction.
# Run from this folder on your laptop or Cloud Shell (not on the cluster).
#
# A) Deploy a version already in the Model Registry (e.g. registered by `train.py --register`):
#      ./deploy_vertex.sh <PROJECT_ID> --model <DISPLAY_NAME> [VERSION]
#      e.g. ./deploy_vertex.sh my-project --model sentiment-classifier        # default (newest) version
#           ./deploy_vertex.sh my-project --model sentiment-classifier 3      # a specific version
#
# B) Build the image, register a model straight from Cloud Storage, and deploy it:
#      ./deploy_vertex.sh <PROJECT_ID> <GCS_MODEL_URI>
#
# Region: us-central1, or set REGION=... in the environment.
# Request-response logging: BigQuery dataset LOG_DATASET (default vertex_logging), rate LOG_RATE (default 1.0).
#
# COST: the endpoint keeps one g2-standard-8 + L4 running (and billing) until you run
#       ./cleanup_vertex.sh. Deploying usually takes 10-20 minutes.
set -euo pipefail

usage() {
  sed -n '5,12p' "$0" >&2
  exit 1
}
[[ $# -ge 2 ]] || usage
PROJECT_ID="$1"; shift
REGION="${REGION:-us-central1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"   # 8 vCPUs, 32 GB RAM, 1x NVIDIA L4
STAMP="$(date +%Y%m%d-%H%M%S)"
gcloud config set project "$PROJECT_ID" >/dev/null

if [[ "$1" == "--model" ]]; then
  # ---- A) registered model -------------------------------------------------
  [[ $# -ge 2 ]] || usage
  DISPLAY_NAME="$2"; VERSION="${3:-}"
  MODEL_ID="$(gcloud ai models list --region "$REGION" --filter="displayName=$DISPLAY_NAME" \
    --format='value(name.basename())' --limit 1)"
  [[ -n "$MODEL_ID" ]] || { echo "No registered model named '$DISPLAY_NAME' in $REGION" >&2; exit 1; }
  # Vertex AI addresses a specific version as MODEL_ID@VERSION (no suffix = default version).
  MODEL_REF="$MODEL_ID${VERSION:+@$VERSION}"
  NAME="${DISPLAY_NAME}${VERSION:+-v$VERSION}-$STAMP"
  echo "== Deploying registered model $DISPLAY_NAME (${VERSION:-default version}), ID $MODEL_REF"
else
  # ---- B) straight from Cloud Storage --------------------------------------
  MODEL_URI="${1%/}"
  NAME="sentiment-$STAMP"
  echo "== Checking the model exists at $MODEL_URI"
  gcloud storage ls "$MODEL_URI/config.json" >/dev/null
  echo "== Building the serving image"
  IMAGE="$("$HERE/build_image.sh" "$PROJECT_ID" "$REGION")"
  echo "== Registering the model with Vertex AI: $NAME"
  gcloud ai models upload --region "$REGION" \
    --display-name "$NAME" \
    --artifact-uri "$MODEL_URI" \
    --container-image-uri "$IMAGE" \
    --container-ports 8080 \
    --container-health-route /health \
    --container-predict-route /predict
  MODEL_ID="$(gcloud ai models list --region "$REGION" --filter="displayName=$NAME" \
    --format='value(name.basename())' --limit 1)"
  MODEL_REF="$MODEL_ID"
fi
echo "   model: $MODEL_REF"

echo "== Creating an endpoint (logging requests/responses to BigQuery) and deploying (slowest step)"
# Every request and response is logged to BigQuery for drift monitoring (monitor_drift.py).
# The dataset is created by setup_vertex.sh; Vertex AI creates the table.
LOG_TABLE="bq://${PROJECT_ID}.${LOG_DATASET:-vertex_logging}.${NAME//-/_}"
gcloud ai endpoints create --region "$REGION" --display-name "$NAME-endpoint" \
  --request-response-logging-rate "${LOG_RATE:-1.0}" \
  --request-response-logging-table "$LOG_TABLE"
echo "   request-response logs: $LOG_TABLE"
ENDPOINT_ID="$(gcloud ai endpoints list --region "$REGION" --filter="displayName=$NAME-endpoint" \
  --format='value(name.basename())' --limit 1)"
echo "   endpoint ID: $ENDPOINT_ID"
gcloud ai endpoints deploy-model "$ENDPOINT_ID" --region "$REGION" \
  --model "$MODEL_REF" --display-name "$NAME" \
  --machine-type "$MACHINE_TYPE" \
  --accelerator type=nvidia-l4,count=1 \
  --min-replica-count 1 --max-replica-count 1 \
  --traffic-split 0=100

echo "== Test prediction"
REQUEST="$(mktemp)"
cat > "$REQUEST" <<'JSON'
{"instances": ["An absolute masterpiece, I loved every minute.", "Boring, too long, and the ending made no sense."]}
JSON
gcloud ai endpoints predict "$ENDPOINT_ID" --region "$REGION" --json-request "$REQUEST"
rm -f "$REQUEST"

# Only option B creates a registry entry just for this deployment, so only it should be deleted on cleanup.
cat > "$HERE/.last_deploy" <<EOF
PROJECT_ID=$PROJECT_ID
REGION=$REGION
ENDPOINT_ID=$ENDPOINT_ID
MODEL_ID=$MODEL_ID
DELETE_MODEL=$([[ "${DISPLAY_NAME:-}" ]] && echo no || echo yes)
EOF
echo
echo "Endpoint $ENDPOINT_ID is LIVE and billing. When finished:  ./cleanup_vertex.sh"
echo "Check drift once it has traffic:  python monitor_drift.py --project $PROJECT_ID --region $REGION --endpoint-id $ENDPOINT_ID"

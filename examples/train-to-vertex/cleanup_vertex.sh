#!/usr/bin/env bash
# Stop billing: undeploy and delete the endpoint created by deploy_vertex.sh.
# Model Registry entries you deployed by name (--model) are KEPT, so version history survives;
# the one-off model created by deploying straight from Cloud Storage is deleted.
# Model files in Cloud Storage and the image in Artifact Registry are always kept.
#
#   ./cleanup_vertex.sh                                         # uses IDs saved by deploy_vertex.sh
#   ./cleanup_vertex.sh <PROJECT_ID> <ENDPOINT_ID> [MODEL_ID_TO_DELETE] [REGION]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
if [[ $# -ge 2 ]]; then
  PROJECT_ID="$1"; ENDPOINT_ID="$2"; MODEL_ID="${3:-}"; REGION="${4:-us-central1}"
  DELETE_MODEL=$([[ -n "$MODEL_ID" ]] && echo yes || echo no)
elif [[ -f "$HERE/.last_deploy" ]]; then
  # shellcheck disable=SC1091
  source "$HERE/.last_deploy"
else
  echo "Usage: $0 <PROJECT_ID> <ENDPOINT_ID> [MODEL_ID_TO_DELETE] [REGION]  (or run deploy_vertex.sh first)" >&2
  exit 1
fi

echo "== Undeploying everything from endpoint $ENDPOINT_ID"
for DEPLOYED_ID in $(gcloud ai endpoints describe "$ENDPOINT_ID" --project "$PROJECT_ID" \
    --region "$REGION" --format='value(deployedModels[].id)' | tr ';,' '  '); do
  gcloud ai endpoints undeploy-model "$ENDPOINT_ID" --project "$PROJECT_ID" \
    --region "$REGION" --deployed-model-id "$DEPLOYED_ID"
done

echo "== Deleting endpoint $ENDPOINT_ID"
gcloud ai endpoints delete "$ENDPOINT_ID" --project "$PROJECT_ID" --region "$REGION" --quiet

if [[ "${DELETE_MODEL:-no}" == "yes" && -n "${MODEL_ID:-}" ]]; then
  echo "== Deleting model $MODEL_ID"
  gcloud ai models delete "$MODEL_ID" --project "$PROJECT_ID" --region "$REGION" --quiet
else
  echo "== Keeping registered model ${MODEL_ID:-} (its versions stay in the Model Registry)"
fi
rm -f "$HERE/.last_deploy"
echo "Done. No Vertex AI endpoint is running for this example."

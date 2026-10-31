#!/usr/bin/env bash
# One-time setup for the train-to-Vertex example. Run from your laptop or Cloud Shell
# as a project Owner (or someone who can change IAM), AFTER ../../setup-permissions.sh.
#
#   ./setup_vertex.sh <PROJECT_ID> <BUCKET_NAME> [DEPLOYER_EMAIL] [REGION]
#
# It:
#   1. enables the Vertex AI, Artifact Registry and Cloud Build APIs
#   2. creates the model bucket (if missing) and an Artifact Registry repo for the server image
#   3. lets the cluster VMs' service account WRITE to that bucket only (to upload trained models)
#   4. lets that service account push images (new projects run Cloud Build as it)
#   5. optionally grants a deployer the roles to build, upload and deploy models
#   6. creates the BigQuery dataset for endpoint request-response logs (drift monitoring)
#      and lets the cluster query it, so monitor_drift.py can also run from the cluster
#   7. optionally lets the cluster read ONE BigQuery dataset for training data:
#        BQ_DATASET=my-project.reviews ./setup_vertex.sh <PROJECT_ID> <BUCKET_NAME> ...
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <PROJECT_ID> <BUCKET_NAME> [DEPLOYER_EMAIL] [REGION]" >&2
  exit 1
fi
PROJECT_ID="$1"; BUCKET="$2"; DEPLOYER="${3:-}"; REGION="${4:-us-central1}"
REPO="serving"

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
VM_SA="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo "== Enabling APIs"
gcloud services enable --project "$PROJECT_ID" \
  aiplatform.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

echo "== Bucket gs://$BUCKET"
if ! gcloud storage buckets describe "gs://$BUCKET" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://$BUCKET" --project "$PROJECT_ID" \
    --location "$REGION" --uniform-bucket-level-access
fi
# Cluster VMs upload trained models here (write to this bucket only, not project-wide).
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member="$VM_SA" --role="roles/storage.objectCreator" >/dev/null
echo "  + roles/storage.objectCreator on gs://$BUCKET -> $VM_SA"

echo "== Artifact Registry repo '$REPO' in $REGION"
if ! gcloud artifacts repositories describe "$REPO" --location "$REGION" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPO" --project "$PROJECT_ID" \
    --location "$REGION" --repository-format docker --description "Vertex AI serving images"
fi

grant() {  # grant <member> <role>
  echo "  + $2 -> $1"
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="$1" --role="$2" \
    --condition=None --quiet >/dev/null
}

echo "== Cloud Build permissions"
# Newer projects run Cloud Build as the default compute service account; it needs to push
# the image and read the uploaded source (it already has storage.objectViewer and
# logging.logWriter from setup-permissions.sh).
grant "$VM_SA" roles/artifactregistry.writer
# Lets training jobs on the cluster register new model versions (train.py --register).
grant "$VM_SA" roles/aiplatform.user
LEGACY_CB_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
if gcloud iam service-accounts describe "$LEGACY_CB_SA" --project "$PROJECT_ID" >/dev/null 2>&1; then
  grant "serviceAccount:$LEGACY_CB_SA" roles/artifactregistry.writer   # older projects
fi

SA_EMAIL="${VM_SA#serviceAccount:}"

grant_dataset_reader() {  # grant_dataset_reader <project:dataset>
  # Dataset-level READER (= BigQuery Data Viewer on this dataset only), using the
  # documented show -> edit access list -> update method.
  local access_json
  access_json="$(mktemp)"
  bq --project_id "$PROJECT_ID" show --format=prettyjson "$1" > "$access_json"
  python3 - "$access_json" "$SA_EMAIL" <<'PY'
import json, sys
path, email = sys.argv[1], sys.argv[2]
ds = json.load(open(path))
entry = {"role": "READER", "userByEmail": email}
if entry not in ds.setdefault("access", []):
    ds["access"].append(entry)
json.dump(ds, open(path, "w"))
PY
  bq --project_id "$PROJECT_ID" update --source "$access_json" "$1" >/dev/null
  rm -f "$access_json"
  echo "  + READER on $1 -> $SA_EMAIL"
}

echo "== BigQuery"
gcloud services enable --project "$PROJECT_ID" bigquery.googleapis.com bigquerystorage.googleapis.com
grant "$VM_SA" roles/bigquery.jobUser             # run queries (billed to $PROJECT_ID)
grant "$VM_SA" roles/bigquery.readSessionUser     # fast reads via the BigQuery Storage API

LOG_DATASET="${LOG_DATASET:-vertex_logging}"
echo "== Request-response log dataset ${PROJECT_ID}:${LOG_DATASET} (drift monitoring)"
if ! bq --project_id "$PROJECT_ID" show "${PROJECT_ID}:${LOG_DATASET}" >/dev/null 2>&1; then
  bq --project_id "$PROJECT_ID" mk --dataset --location "$REGION" \
    --description "Vertex AI endpoint request-response logs" "${PROJECT_ID}:${LOG_DATASET}"
fi
grant_dataset_reader "${PROJECT_ID}:${LOG_DATASET}"

if [[ -n "${BQ_DATASET:-}" ]]; then
  BQ_ID="${BQ_DATASET/./:}"                      # my-project.reviews -> my-project:reviews
  [[ "$BQ_ID" == *:* ]] || BQ_ID="${PROJECT_ID}:${BQ_ID}"
  echo "== BigQuery read access to training dataset $BQ_ID"
  grant_dataset_reader "$BQ_ID"
fi

if [[ -n "$DEPLOYER" ]]; then
  echo "== Deployer roles for $DEPLOYER"
  grant "user:$DEPLOYER" roles/aiplatform.user               # upload models, create/deploy endpoints
  grant "user:$DEPLOYER" roles/cloudbuild.builds.editor      # build the serving image
  grant "user:$DEPLOYER" roles/artifactregistry.writer       # push images
  grant "user:$DEPLOYER" roles/bigquery.jobUser              # run monitor_drift.py
  grant "user:$DEPLOYER" roles/bigquery.dataViewer           # read request-response logs
  grant "user:$DEPLOYER" roles/monitoring.metricWriter       # monitor_drift.py --publish-metrics
fi

echo
echo "Done. Next:"
echo "  On the cluster:  sbatch --export=ALL,GCS_MODEL_URI=gs://$BUCKET/models/sentiment-v1 train.sh"
echo "  Then here:       REGION=$REGION ./deploy_vertex.sh $PROJECT_ID gs://$BUCKET/models/sentiment-v1"
echo "  Or auto-register from training: ./build_image.sh $PROJECT_ID $REGION  (see README)"

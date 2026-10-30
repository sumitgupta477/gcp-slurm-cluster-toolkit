"""A small prediction server for Vertex AI custom containers.

Vertex AI tells the container where things are through environment variables:
  AIP_STORAGE_URI    gs:// folder holding the model files (from `--artifact-uri`)
  AIP_HTTP_PORT      port to listen on (default 8080)
  AIP_HEALTH_ROUTE   health-check path (Vertex waits for 200 here before sending traffic)
  AIP_PREDICT_ROUTE  prediction path

Request:  {"instances": ["great movie", {"text": "boring"}]}
Response: {"predictions": [{"label": "POSITIVE", "score": 0.98}, ...]}

For local testing without Vertex, set AIP_STORAGE_URI to a local folder instead of gs://.
"""
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")
STATE = {"pipeline": None}


def download_model(uri: str) -> str:
    """Return a local folder with the model files, downloading from GCS if needed."""
    if not uri.startswith("gs://"):
        return uri  # already a local path (local testing)
    from google.cloud import storage

    bucket_name, _, prefix = uri[len("gs://"):].partition("/")
    prefix = prefix.rstrip("/") + "/" if prefix else ""
    local_dir = Path(tempfile.mkdtemp(prefix="model-"))
    count = 0
    for blob in storage.Client().list_blobs(bucket_name, prefix=prefix):
        if blob.name.endswith("/"):
            continue  # folder placeholder objects
        target = local_dir / blob.name[len(prefix):]
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target))
        count += 1
    if count == 0:
        raise RuntimeError(f"No model files found under {uri}")
    print(f"Downloaded {count} files from {uri} to {local_dir}")
    return str(local_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import torch
    from transformers import pipeline

    uri = os.environ.get("AIP_STORAGE_URI") or os.environ.get("MODEL_DIR")
    if not uri:
        raise RuntimeError("Set AIP_STORAGE_URI (Vertex AI does this) or MODEL_DIR")
    model_dir = download_model(uri)
    device = 0 if torch.cuda.is_available() else -1
    STATE["pipeline"] = pipeline("text-classification", model=model_dir, tokenizer=model_dir, device=device)
    print(f"Model loaded on {'GPU ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")
    yield


app = FastAPI(lifespan=lifespan)


async def health():
    if STATE["pipeline"] is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


async def predict(request: Request):
    body = await request.json()
    instances = body.get("instances") if isinstance(body, dict) else None
    if not isinstance(instances, list) or not instances:
        raise HTTPException(status_code=400, detail='Expected {"instances": ["text", ...]}')
    texts = [i["text"] if isinstance(i, dict) else str(i) for i in instances]
    results = STATE["pipeline"](texts, truncation=True)
    return {"predictions": [{"label": r["label"], "score": float(r["score"])} for r in results]}


app.add_api_route(HEALTH_ROUTE, health, methods=["GET"])
app.add_api_route(PREDICT_ROUTE, predict, methods=["POST"])

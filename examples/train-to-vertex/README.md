# Train on Slurm, serve on Vertex AI

Fine-tune a Hugging Face model (DistilBERT, positive / negative movie reviews) on the cluster's **NVIDIA L4** with `sbatch`, save it to **Cloud Storage**, and serve it from a **Vertex AI endpoint**.

```mermaid
flowchart LR
    job["sbatch train.sh<br/>(L4 GPU node)"] --> home["/home/models/...<br/>(Filestore)"]
    job -->|upload| gcs[("gs://BUCKET/models/sentiment-v1")]
    gcs -->|AIP_STORAGE_URI| server["Serving container<br/>(FastAPI + Transformers)"]
    server --> ep["Vertex AI endpoint<br/>g2-standard-8 + L4"]
    client["Your app / gcloud"] -->|"{instances: [...]}"| ep
```

## Files

| File | Runs on | What it does |
|---|---|---|
| `setup_vertex.sh` | Laptop / Cloud Shell, once | Enables APIs, creates the bucket and image repo, grants permissions |
| `setup_env.sh` | Login VM, once | Creates the Python training environment in `~/venv-train` (shared via `/home`) |
| `train.sh` | Slurm job (`sbatch`) | Requests 1 GPU and runs `train.py` |
| `train.py` | GPU node | Fine-tunes the model, saves it to `/home`, uploads it to GCS |
| `serving/` | Vertex AI | Dockerfile + FastAPI server that loads the model from `AIP_STORAGE_URI` |
| `deploy_vertex.sh` | Laptop / Cloud Shell | Builds the image, registers the model, deploys an endpoint, tests it |
| `cleanup_vertex.sh` | Laptop / Cloud Shell | Undeploys and deletes the endpoint and model (stops billing) |

## Steps

**0. Prerequisites:** the cluster is deployed (see the main README) and `../../setup-permissions.sh` has been run.

**1. One-time Vertex AI setup** (laptop or Cloud Shell, as a project Owner):
```bash
./setup_vertex.sh <PROJECT_ID> <BUCKET_NAME> <your-email>
```
This also gives the cluster VMs **write access to that one bucket**, which the base setup doesn't include (it only grants read access).

**2. One-time training environment** (on the login VM):
```bash
git clone https://github.com/sumitgupta477/gcp-slurm-cluster-toolkit.git
cd gcp-slurm-cluster-toolkit/examples/train-to-vertex
bash setup_env.sh
```

**3. Train** (on the login VM, from this folder):
```bash
sbatch --export=ALL,GCS_MODEL_URI=gs://<BUCKET_NAME>/models/sentiment-v1 train.sh
squeue --me                            # PD -> CF (GPU node starting) -> R
tail -f train-to-vertex-<JOBID>.out    # watch training
```
The defaults (2,000 reviews, 1 epoch) take a few minutes on an L4. Pass extra options to `train.py` after the script name, e.g. `train.sh --epochs 2 --train_samples 10000`.

**4. Deploy to Vertex AI** (laptop or Cloud Shell):
```bash
./deploy_vertex.sh <PROJECT_ID> gs://<BUCKET_NAME>/models/sentiment-v1
```
It ends with a test prediction like:
```json
{"predictions": [{"label": "POSITIVE", "score": 0.98}, {"label": "NEGATIVE", "score": 0.97}]}
```

**5. Stop billing:**
```bash
./cleanup_vertex.sh
```

## How the serving container works

Vertex AI starts the container with these environment variables, and `server.py` uses them:

| Variable | Meaning |
|---|---|
| `AIP_STORAGE_URI` | GCS folder with the model (from `--artifact-uri`); downloaded at startup |
| `AIP_HTTP_PORT` | Port to listen on (8080) |
| `AIP_HEALTH_ROUTE` | `/health`: returns 503 while loading, 200 when ready |
| `AIP_PREDICT_ROUTE` | `/predict`: takes `{"instances": [...]}`, returns `{"predictions": [...]}` |

Test it locally without Vertex AI (needs `pip install -r serving/requirements.txt`):
```bash
cd serving && AIP_STORAGE_URI=~/models/<your-model-folder> uvicorn server:app --port 8080
curl -X POST localhost:8080/predict -H 'Content-Type: application/json' -d '{"instances": ["great film"]}'
```

## Costs

- **Training:** only while the job runs; the GPU node shuts down 5 minutes after it finishes.
- **Endpoint:** one `g2-standard-8` + L4 runs **24/7 until you run `cleanup_vertex.sh`**, even with no traffic. This is the main cost to watch.
- **Storage:** a few hundred MB per model in GCS, plus the serving image (a few GB) in Artifact Registry.

## Notes

- Training and serving pin the same versions (`torch 2.5.1`, `transformers 4.46.3`), so the saved model loads identically.
- Vertex AI also needs **L4 quota for prediction** in the region. It's separate from the Compute Engine quota the Slurm cluster uses, so check **IAM & Admin → Quotas**.
- In the same project, Vertex AI's service agent can normally read the bucket and pull the image. If deployment fails on permissions, check the deployment logs in **Vertex AI → Endpoints**.
- Tested locally on CPU (training on a toy model and dataset, the server with Vertex AI's environment variables, and GCS upload/download against a fake bucket), but **not yet run on the cluster or Vertex AI**.

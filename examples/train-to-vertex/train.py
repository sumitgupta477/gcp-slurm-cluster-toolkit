"""Fine-tune a Hugging Face model for sentiment analysis on the Slurm cluster's L4 GPU.

Saves the model + tokenizer to --output_dir (on the shared /home), and optionally
uploads that folder to Cloud Storage (--gcs_uri) so Vertex AI can serve it.

Training data comes from (pick one):
  * a Hugging Face Hub dataset (default: imdb), or a local save_to_disk() folder: --dataset
  * a BigQuery query returning `text` and `label` columns (+ optional `split`):  --bq_query

With --register, the uploaded model is also added to the Vertex AI Model Registry as a
NEW VERSION of one model (created on the first run), optionally only if accuracy is good
enough (--min_accuracy). Registering does not serve it; deploy a version later.

Every run also saves drift_baseline.json in the model folder: what normal inputs and
predictions look like for THIS model (held-out reviews + its own predictions). It travels
with the model to Cloud Storage and the registry; monitor_drift.py compares live traffic to it.

    python train.py --output_dir ~/models/sentiment-v1 --gcs_uri gs://BUCKET/models/sentiment-v1
    python train.py --output_dir ~/models/run-7 --gcs_uri gs://BUCKET/models/run-7 \\
        --register --serving_image REGION-docker.pkg.dev/PROJECT/serving/hf-sentiment-server:latest
    python train.py --output_dir ~/models/sentiment-bq \\
        --bq_query 'SELECT review AS text, sentiment AS label FROM `my-project.reviews.movie_reviews`'
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import drift  # same folder; shared with monitor_drift.py

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True, help="Where to save the trained model (e.g. on /home)")
    p.add_argument("--gcs_uri", default="", help="Optional gs://bucket/path to upload the model to")
    p.add_argument("--model_name", default="distilbert-base-uncased", help="Hub model ID or local path")
    p.add_argument("--dataset", default="imdb",
                   help="Hub dataset ID, or a local folder saved with datasets' save_to_disk()")
    p.add_argument("--bq_query", default="",
                   help="BigQuery SQL returning `text` and `label` (0/1 or negative/positive) columns, "
                        "plus an optional `split` column ('train'/'test'). Overrides --dataset.")
    p.add_argument("--bq_query_file", default="",
                   help="Path to a .sql file with the query (avoids shell-quoting backticks)")
    p.add_argument("--bq_project", default=None,
                   help="Project that runs (and pays for) the query; defaults to the VM's project")
    p.add_argument("--test_size", type=float, default=0.2,
                   help="Fraction held out for evaluation when the query has no `split` column")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--train_samples", type=int, default=2000)  # small subset: quick, cheap demo
    p.add_argument("--eval_samples", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--max_length", type=int, default=256)
    # Vertex AI Model Registry
    p.add_argument("--register", action="store_true",
                   help="Add the uploaded model to the Vertex AI Model Registry as a new version")
    p.add_argument("--model_display_name", default="sentiment-classifier",
                   help="Registry model that new versions are added to")
    p.add_argument("--serving_image", default=os.environ.get("SERVING_IMAGE", ""),
                   help="Serving container image (from build_image.sh); env SERVING_IMAGE")
    p.add_argument("--region", default=os.environ.get("REGION", "us-central1"))
    p.add_argument("--project", default=os.environ.get("PROJECT_ID") or None,
                   help="Defaults to the VM's project")
    p.add_argument("--min_accuracy", type=float, default=None,
                   help="Only register if eval accuracy is at least this (e.g. 0.85)")
    args = p.parse_args()
    if args.bq_query and args.bq_query_file:
        p.error("use --bq_query or --bq_query_file, not both")
    if args.register and not args.gcs_uri:
        p.error("--register needs --gcs_uri (Vertex AI reads the model from Cloud Storage)")
    if args.register and not args.serving_image:
        p.error("--register needs --serving_image or the SERVING_IMAGE env var (run build_image.sh first)")
    if args.bq_query_file:
        args.bq_query = Path(args.bq_query_file).expanduser().read_text()
    return args


LABEL_WORDS = {"0": 0, "neg": 0, "negative": 0, "1": 1, "pos": 1, "positive": 1}


def to_label_ids(values) -> list:
    """Map labels like 0/1, 'neg'/'pos' or 'Negative'/'Positive' to 0/1."""
    ids, bad = [], set()
    for v in values:
        key = str(int(v)) if isinstance(v, (bool, int, float, np.integer, np.floating)) else str(v).strip().lower()
        if key in LABEL_WORDS:
            ids.append(LABEL_WORDS[key])
        else:
            bad.add(v)
    if bad:
        raise ValueError(f"Unsupported label values {sorted(map(str, bad))[:5]}: use 0/1 or negative/positive")
    return ids


def load_bigquery(query: str, project, test_size: float, seed: int = 42) -> DatasetDict:
    """Run a BigQuery query and turn the result into train/test datasets."""
    from google.cloud import bigquery  # imported here so non-BigQuery runs don't need it

    df = bigquery.Client(project=project).query(query).to_dataframe()
    print(f"BigQuery returned {len(df)} rows, columns: {list(df.columns)}")
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Query must return columns `text` and `label` (missing: {sorted(missing)}). "
                         "Rename with AS, e.g. SELECT review AS text, sentiment AS label ...")
    df = df.dropna(subset=["text", "label"])
    df = df.assign(text=df["text"].astype(str), label=to_label_ids(df["label"]))

    if "split" in df.columns:
        split = df["split"].astype(str).str.strip().str.lower()
        parts = {name: df[split == name][["text", "label"]] for name in ("train", "test")}
        if parts["train"].empty or parts["test"].empty:
            raise ValueError("The `split` column must contain both 'train' and 'test' rows")
        return DatasetDict({name: Dataset.from_pandas(part, preserve_index=False) for name, part in parts.items()})

    full = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
    return full.train_test_split(test_size=test_size, seed=seed)


def upload_dir_to_gcs(local_dir: Path, gcs_uri: str) -> None:
    """Copy every file in local_dir to gs://bucket/prefix/ (keeps sub-folders)."""
    from google.cloud import storage  # imported here so local runs don't need it

    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"--gcs_uri must start with gs://, got {gcs_uri!r}")
    bucket_name, _, prefix = gcs_uri[len("gs://"):].partition("/")
    bucket = storage.Client().bucket(bucket_name)
    for path in sorted(local_dir.rglob("*")):
        if path.is_file():
            blob_name = f"{prefix.rstrip('/')}/{path.relative_to(local_dir).as_posix()}".lstrip("/")
            bucket.blob(blob_name).upload_from_filename(str(path))
            print(f"uploaded gs://{bucket_name}/{blob_name}")


def register_model(args, metrics: dict) -> dict:
    """Upload the GCS model to the Vertex AI Model Registry as a new default version."""
    from google.cloud import aiplatform  # imported here so local runs don't need it

    aiplatform.init(project=args.project, location=args.region)
    name = args.model_display_name
    existing = aiplatform.Model.list(filter=f'display_name="{name}"', order_by="create_time desc")
    parent = existing[0].resource_name if existing else None

    accuracy = metrics.get("eval_accuracy")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    labels = {"source": "slurm", "slurm_job": str(job_id).lower()[:63]}
    if accuracy is not None:
        labels["eval_accuracy"] = f"{accuracy:.4f}".replace(".", "_")

    model = aiplatform.Model.upload(
        display_name=name,
        artifact_uri=args.gcs_uri,
        serving_container_image_uri=args.serving_image,
        serving_container_ports=[8080],
        serving_container_health_route="/health",
        serving_container_predict_route="/predict",
        parent_model=parent,           # None on the first run -> creates the model (version 1)
        is_default_version=True,       # newest version becomes the default
        version_description=f"Slurm job {job_id}; eval_accuracy={accuracy}; data={args.bq_query and 'bigquery' or args.dataset}",
        labels=labels,
        sync=True,
    )
    info = {
        "model": model.resource_name,
        "display_name": name,
        "version_id": model.version_id,
        "new_model": parent is None,
        "artifact_uri": args.gcs_uri,
        "eval_accuracy": accuracy,
    }
    print(f"Registered {name} version {model.version_id} ({model.resource_name})")
    return info


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    print(f"CUDA available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    # 1. Data: BigQuery, a local dataset folder, or a Hub dataset (default: IMDB reviews)
    if args.bq_query:
        data = load_bigquery(args.bq_query, args.bq_project, args.test_size)
    elif os.path.isdir(args.dataset):
        data = load_from_disk(args.dataset)
    else:
        data = load_dataset(args.dataset)
    train_ds = data["train"].shuffle(seed=42).select(range(min(args.train_samples, len(data["train"]))))
    eval_ds = data["test"].shuffle(seed=42).select(range(min(args.eval_samples, len(data["test"]))))
    eval_texts = list(eval_ds["text"])

    # 2. Tokenize: text -> numbers
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)

    # 3. Model: pretrained model + new 2-label classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return {"accuracy": float((np.argmax(logits, axis=-1) == labels).mean())}

    # 4. Train
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/checkpoints",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=25,
            fp16=torch.cuda.is_available(),
            report_to="none",
        ),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(f"Final evaluation: {metrics}")

    # 5. Save model + tokenizer + metrics
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Model saved to {output_dir}")

    # 5b. Drift baseline: held-out reviews + this model's own predictions and confidence
    probs = torch.softmax(torch.tensor(trainer.predict(eval_ds).predictions), dim=-1).numpy()
    id2label = trainer.model.config.id2label
    rows = [drift.features(t, id2label[int(p.argmax())], float(p.max())) for t, p in zip(eval_texts, probs)]
    baseline = drift.build_baseline(rows, source=f"Slurm job {os.environ.get('SLURM_JOB_ID', 'local')}")
    baseline["eval_accuracy"] = metrics.get("eval_accuracy")
    (output_dir / "drift_baseline.json").write_text(json.dumps(baseline, indent=2))
    print(f"Drift baseline: {len(rows)} held-out predictions, "
          f"{baseline['features']['pred_positive']['mean']:.1%} predicted POSITIVE")

    # 6. Optional: upload to Cloud Storage for Vertex AI
    if args.gcs_uri:
        upload_dir_to_gcs(output_dir, args.gcs_uri)
        print(f"Model uploaded to {args.gcs_uri}")

    # 7. Optional: register in the Vertex AI Model Registry (does not deploy)
    if args.register:
        accuracy = metrics.get("eval_accuracy")
        if args.min_accuracy is not None and (accuracy is None or accuracy < args.min_accuracy):
            print(f"NOT registering: eval_accuracy={accuracy} is below --min_accuracy={args.min_accuracy}")
            return
        info = register_model(args, metrics)
        (output_dir / "registry.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

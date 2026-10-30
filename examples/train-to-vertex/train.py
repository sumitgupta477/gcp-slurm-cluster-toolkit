"""Fine-tune a Hugging Face model for sentiment analysis on the Slurm cluster's L4 GPU.

Saves the model + tokenizer to --output_dir (on the shared /home), and optionally
uploads that folder to Cloud Storage (--gcs_uri) so Vertex AI can serve it.

    python train.py --output_dir ~/models/sentiment-v1 --gcs_uri gs://BUCKET/models/sentiment-v1
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True, help="Where to save the trained model (e.g. on /home)")
    p.add_argument("--gcs_uri", default="", help="Optional gs://bucket/path to upload the model to")
    p.add_argument("--model_name", default="distilbert-base-uncased", help="Hub model ID or local path")
    p.add_argument("--dataset", default="imdb",
                   help="Hub dataset ID, or a local folder saved with datasets' save_to_disk()")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--train_samples", type=int, default=2000)  # small subset: quick, cheap demo
    p.add_argument("--eval_samples", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--max_length", type=int, default=256)
    return p.parse_args()


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


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    print(f"CUDA available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    # 1. Data: IMDB movie reviews (positive / negative), shuffled and trimmed
    data = load_from_disk(args.dataset) if os.path.isdir(args.dataset) else load_dataset(args.dataset)
    train_ds = data["train"].shuffle(seed=42).select(range(min(args.train_samples, len(data["train"]))))
    eval_ds = data["test"].shuffle(seed=42).select(range(min(args.eval_samples, len(data["test"]))))

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

    # 6. Optional: upload to Cloud Storage for Vertex AI
    if args.gcs_uri:
        upload_dir_to_gcs(output_dir, args.gcs_uri)
        print(f"Model uploaded to {args.gcs_uri}")


if __name__ == "__main__":
    main()

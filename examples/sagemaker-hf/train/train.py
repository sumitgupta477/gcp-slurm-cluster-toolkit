"""Fine-tune a Hugging Face model for sentiment analysis (positive / negative movie reviews).

SageMaker runs this script inside the Hugging Face training container on a GPU machine.
Hyperparameters from the launcher arrive as command-line flags, e.g. `--epochs 1`.
Whatever is saved to SM_MODEL_DIR (/opt/ml/model) is uploaded to S3 as model.tar.gz
when the job finishes, and that is what the endpoint later loads.

It also builds the DRIFT BASELINE: the held-out reviews plus the model's own predictions
and confidence, summarised by drift.py. It is saved to SM_OUTPUT_DATA_DIR and, with
--drift_baseline_s3, uploaded to S3 for monitor_drift.py to compare live traffic against.
"""
import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset, load_from_disk

import drift
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="distilbert-base-uncased")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--train_samples", type=int, default=2000)  # small subset = quick, cheap demo
    p.add_argument("--eval_samples", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--dataset", default="imdb", help="Hub dataset ID, or a local save_to_disk() folder")
    p.add_argument("--drift_baseline_s3", default="", help="s3://bucket/key.json to upload the drift baseline to")
    return p.parse_args()


def prediction_rows(trainer, eval_ds, texts) -> list:
    """Drift features for each held-out text, using the model's prediction and confidence."""
    logits = trainer.predict(eval_ds).predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    labels = trainer.model.config.id2label
    return [drift.features(text, labels[int(p.argmax())], float(p.max())) for text, p in zip(texts, probs)]


def upload_to_s3(path: str, s3_uri: str) -> None:
    import boto3  # available in the SageMaker training containers

    bucket, _, key = s3_uri[len("s3://"):].partition("/")
    boto3.client("s3").upload_file(path, bucket, key)
    print(f"Uploaded drift baseline to {s3_uri}")


def main():
    args = parse_args()
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")

    # 1. Data: IMDB movie reviews from the Hugging Face Hub (or a local copy), shuffled and trimmed
    data = load_from_disk(args.dataset) if os.path.isdir(args.dataset) else load_dataset(args.dataset)
    train_ds = data["train"].shuffle(seed=42).select(range(min(args.train_samples, len(data["train"]))))
    eval_ds = data["test"].shuffle(seed=42).select(range(min(args.eval_samples, len(data["test"]))))
    eval_texts = list(eval_ds["text"])

    # 2. Tokenize: turn text into numbers the model understands
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)

    # 3. Model: a pretrained model plus a new 2-label classification head
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
    training_args = TrainingArguments(
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
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(f"Final evaluation: {metrics}")

    # 5. Save the model + tokenizer; SageMaker uploads this folder to S3
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 6. Drift baseline: what "normal" inputs and predictions look like for this model
    rows = prediction_rows(trainer, eval_ds, eval_texts)
    baseline = drift.build_baseline(rows, source=f"training job {os.environ.get('TRAINING_JOB_NAME', 'local')}")
    baseline["eval_accuracy"] = metrics.get("eval_accuracy")
    baseline_path = os.path.join(output_dir, "drift_baseline.json")
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"Drift baseline: {len(rows)} held-out predictions, "
          f"{baseline['features']['pred_positive']['mean']:.1%} predicted POSITIVE")
    if args.drift_baseline_s3:
        upload_to_s3(baseline_path, args.drift_baseline_s3)


if __name__ == "__main__":
    main()

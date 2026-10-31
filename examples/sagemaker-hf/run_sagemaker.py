"""Train a Hugging Face model on SageMaker, deploy it to an endpoint, and get a prediction.

Uses the SageMaker Python SDK v3 (ModelTrainer for training, ModelBuilder for serving).

Usage:
    pip install "sagemaker>=3"
    python run_sagemaker.py --role arn:aws:iam::<ACCOUNT_ID>:role/<SageMakerRole> --region us-east-1

    # Inside SageMaker Studio you can omit --role (it uses the notebook's role).
    # --keep-endpoint   leave the endpoint running (it is billed until you delete it!)
    #                   needed for drift monitoring: then run monitor_drift.py
    # --skip-deploy     train only

Drift monitoring: training uploads a drift baseline to
s3://<default-bucket>/hf-sentiment/monitoring/baseline.json, and the endpoint captures
every request/response under s3://<default-bucket>/hf-sentiment/datacapture/.
"""
import argparse
import json
import os
import time
from pathlib import Path

# Training and serving machines: ml.g6.xlarge = 4 vCPUs, 16 GB RAM, 1x NVIDIA L4 GPU (24 GB)
DEFAULT_INSTANCE = "ml.g6.xlarge"

# Hugging Face container versions known to the SDK (sagemaker-core image_uri_config/huggingface.json)
TRAINING_IMAGE = dict(version="4.56.2", base_framework_version="pytorch2.8.0", py_version="py312")
INFERENCE_IMAGE = dict(version="4.51.3", base_framework_version="pytorch2.6.0", py_version="py312")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--role", help="SageMaker execution role ARN (optional inside SageMaker Studio)")
    p.add_argument("--region", default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1")
    p.add_argument("--train-instance-type", default=DEFAULT_INSTANCE)
    p.add_argument("--deploy-instance-type", default=DEFAULT_INSTANCE)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--train-samples", type=int, default=2000)
    p.add_argument("--keep-endpoint", action="store_true", help="Do not delete the endpoint at the end")
    p.add_argument("--skip-deploy", action="store_true", help="Only run the training job")
    p.add_argument("--bucket", help="S3 bucket for the drift baseline and data capture (default: SageMaker default bucket)")
    p.add_argument("--prefix", default="hf-sentiment", help="S3 prefix for monitoring files")
    p.add_argument("--capture-percent", type=int, default=100, help="Share of endpoint requests captured (1-100)")
    return p.parse_args()


def main():
    args = parse_args()

    # The SDK creates AWS clients at import time, so the region must be set BEFORE importing it.
    os.environ["AWS_DEFAULT_REGION"] = args.region

    import boto3
    from sagemaker.core import image_uris
    from sagemaker.core.helper.session_helper import Session, get_execution_role
    from sagemaker.core.model_monitor import DataCaptureConfig
    from sagemaker.core.resources import Endpoint, EndpointConfig
    from sagemaker.core.training.configs import Compute, SourceCode, StoppingCondition
    from sagemaker.serve.builder.schema_builder import SchemaBuilder
    from sagemaker.serve.model_builder import ModelBuilder
    from sagemaker.serve.spec.inference_spec import InferenceSpec
    from sagemaker.serve.utils.types import ModelServer
    from sagemaker.train.model_trainer import ModelTrainer

    session = Session(boto_session=boto3.Session(region_name=args.region))
    role = args.role or get_execution_role()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bucket = args.bucket or session.default_bucket()
    baseline_s3 = f"s3://{bucket}/{args.prefix}/monitoring/baseline.json"
    capture_s3 = f"s3://{bucket}/{args.prefix}/datacapture"

    # ---------------------------------------------------------------- 1. TRAIN
    training_image = image_uris.retrieve(
        framework="huggingface", region=args.region, image_scope="training",
        instance_type=args.train_instance_type, **TRAINING_IMAGE,
    )
    print(f"Training image: {training_image}")

    trainer = ModelTrainer(
        sagemaker_session=session,
        role=role,
        training_image=training_image,
        source_code=SourceCode(
            source_dir=str(Path(__file__).parent / "train"),
            entry_script="train.py",
            requirements="requirements.txt",
        ),
        compute=Compute(instance_type=args.train_instance_type, instance_count=1, volume_size_in_gb=50),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=3600),  # safety cap: 1 hour
        hyperparameters={  # passed to train.py as --epochs 1 --train_samples 2000 ...
            "model_name": "distilbert-base-uncased",
            "epochs": args.epochs,
            "train_samples": args.train_samples,
            "drift_baseline_s3": baseline_s3,  # train.py uploads the drift baseline here
        },
        base_job_name="hf-distilbert-imdb",
    )
    print("Starting training job (machine starts, trains, then shuts down)...")
    trainer.train(wait=True)
    print("Training finished. Model saved to S3 as model.tar.gz.")
    print(f"Drift baseline: {baseline_s3}")

    if args.skip_deploy:
        return

    # ---------------------------------------------------------------- 2. DEPLOY
    class SentimentSpec(InferenceSpec):
        """How the endpoint loads the model and answers a request."""

        def load(self, model_dir: str):
            import torch
            from transformers import pipeline

            return pipeline(
                "text-classification", model=model_dir, tokenizer=model_dir,
                device=0 if torch.cuda.is_available() else -1,
            )

        def invoke(self, input_object, model):
            text = input_object["inputs"] if isinstance(input_object, dict) else input_object
            return model(text)

    inference_image = image_uris.retrieve(
        framework="huggingface", region=args.region, image_scope="inference",
        instance_type=args.deploy_instance_type, **INFERENCE_IMAGE,
    )
    print(f"Inference image: {inference_image}")

    builder = ModelBuilder(
        model=trainer,  # use the artifacts from the training job above
        inference_spec=SentimentSpec(),
        schema_builder=SchemaBuilder(
            {"inputs": "What a great movie!"},
            [{"label": "POSITIVE", "score": 0.99}],
        ),
        model_server=ModelServer.MMS,
        image_uri=inference_image,
        role_arn=role,
        sagemaker_session=session,
        instance_type=args.deploy_instance_type,
        dependencies={"auto": False},  # the container already has transformers + torch
    )
    model = builder.build(model_name=f"hf-sentiment-{stamp}", region=args.region)
    print("Deploying endpoint (usually several minutes)...")
    endpoint = builder.deploy(
        endpoint_name=f"hf-sentiment-{stamp}",
        initial_instance_count=1,
        instance_type=args.deploy_instance_type,
        # Save requests/responses to S3 as plain JSON for drift monitoring (monitor_drift.py)
        data_capture_config=DataCaptureConfig(
            enable_capture=True,
            sampling_percentage=args.capture_percent,
            destination_s3_uri=capture_s3,
            json_content_types=["application/json"],
            sagemaker_session=session,
        ),
    )
    print(f"Endpoint ready: {endpoint.endpoint_name} (capturing traffic to {capture_s3})")

    # ---------------------------------------------------------------- 3. PREDICT
    try:
        for review in ["An absolute masterpiece, I loved every minute.",
                       "Boring, too long, and the ending made no sense."]:
            response = endpoint.invoke(
                body=json.dumps({"inputs": review}), content_type="application/json"
            )
            print(f"{review!r} -> {json.loads(response.body.read().decode('utf-8'))}")
    finally:
        # ------------------------------------------------------------ 4. CLEAN UP
        if args.keep_endpoint:
            print(f"Endpoint {endpoint.endpoint_name} is still running and billing. Delete it when done.")
            print(f"Check drift:  python monitor_drift.py --endpoint {endpoint.endpoint_name} "
                  f"--bucket {bucket} --prefix {args.prefix} --region {args.region}")
        else:
            ep = Endpoint.get(endpoint_name=endpoint.endpoint_name)
            config = EndpointConfig.get(endpoint_config_name=ep.endpoint_config_name)
            endpoint.delete()
            config.delete()
            model.delete()
            print("Endpoint, endpoint config and model deleted (training artifacts stay in S3).")


if __name__ == "__main__":
    main()

"""Check a SageMaker endpoint for drift against the baseline built during training.

Reads the endpoint's captured traffic from the last --hours hours
(s3://<bucket>/<prefix>/datacapture/<endpoint>/AllTraffic/YYYY/MM/DD/HH/*.jsonl),
computes the same features as training (train/drift.py), and scores each feature with
the Population Stability Index (PSI): < 0.1 none, 0.1-0.2 moderate, >= 0.2 significant.

    python monitor_drift.py --endpoint hf-sentiment-20261030-120000
    python monitor_drift.py --endpoint ... --publish-metrics --save-report
    python monitor_drift.py --endpoint ... --create-alarm --sns-topic-arn arn:aws:sns:...:alerts

Exit code: 0 = no/moderate drift or not enough data, 2 = significant drift (handy for cron/CI).
Uses only boto3, so it also fits in a scheduled Lambda (see README).
"""
import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "train"))
import drift  # noqa: E402

NAMESPACE = "HFSentiment/ModelDrift"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--bucket", help="Default: the SageMaker default bucket, sagemaker-<region>-<account>")
    p.add_argument("--prefix", default="hf-sentiment")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--variant", default="AllTraffic")
    p.add_argument("--hours", type=int, default=24, help="How much recent traffic to analyse")
    p.add_argument("--min-samples", type=int, default=30, help="Minimum captured predictions to score drift")
    p.add_argument("--publish-metrics", action="store_true", help=f"Publish MaxPSI/SampleCount/PSI to CloudWatch ({NAMESPACE})")
    p.add_argument("--save-report", action="store_true", help="Write the JSON report to s3://<bucket>/<prefix>/monitoring/reports/")
    p.add_argument("--create-alarm", action="store_true", help="Create/update a CloudWatch alarm on MaxPSI")
    p.add_argument("--sns-topic-arn", help="SNS topic the alarm notifies (with --create-alarm)")
    p.add_argument("--threshold", type=float, default=0.2, help="Alarm threshold on MaxPSI")
    args = p.parse_args(argv)
    if args.create_alarm and not args.sns_topic_arn:
        p.error("--create-alarm needs --sns-topic-arn")
    return args


def load_json(s3, bucket, key):
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except s3.exceptions.NoSuchKey:
        return None


def read_rows(s3, bucket, prefix, endpoint, variant, now, hours, limit=100000):
    rows, paginator = [], s3.get_paginator("list_objects_v2")
    for h in range(hours):
        hour = (now - timedelta(hours=h)).strftime("%Y/%m/%d/%H/")
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/datacapture/{endpoint}/{variant}/{hour}"):
            for obj in page.get("Contents", []):
                body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read().decode("utf-8", errors="replace")
                for line in body.splitlines():
                    for text, label, score in drift.parse_capture_line(line) if line.strip() else []:
                        rows.append(drift.features(text, label, score))
                        if len(rows) >= limit:
                            return rows
    return rows


def check(args, s3, cloudwatch=None, now=None) -> dict:
    now = now or datetime.now(timezone.utc)
    baseline_key = f"{args.prefix}/monitoring/baseline.json"
    baseline = load_json(s3, args.bucket, baseline_key)
    if baseline is None:
        return {"status": "no_baseline", "detail": f"s3://{args.bucket}/{baseline_key} not found (run training first)"}

    rows = read_rows(s3, args.bucket, args.prefix, args.endpoint, args.variant, now, args.hours)
    report = {"checked_at": now.isoformat(), "endpoint": args.endpoint, "window_hours": args.hours,
              "samples": len(rows), "baseline": {k: baseline.get(k) for k in ("created", "source", "count")}}
    dim = [{"Name": "Endpoint", "Value": args.endpoint}]
    metrics = [{"MetricName": "SampleCount", "Dimensions": dim, "Value": len(rows), "Unit": "Count"}]

    if len(rows) < args.min_samples:
        report["status"] = "insufficient_data"
    else:
        feats = drift.compare(baseline, rows)
        worst = max(feats, key=lambda f: feats[f]["psi"])
        report.update(status=drift.severity(feats[worst]["psi"]), max_psi=feats[worst]["psi"],
                      max_psi_feature=worst, features=feats)
        metrics.append({"MetricName": "MaxPSI", "Dimensions": dim, "Value": feats[worst]["psi"], "Unit": "None"})
        metrics.extend({"MetricName": "PSI", "Dimensions": dim + [{"Name": "Feature", "Value": n}],
                        "Value": s["psi"], "Unit": "None"} for n, s in feats.items())

    if args.publish_metrics and cloudwatch is not None:
        cloudwatch.put_metric_data(Namespace=NAMESPACE, MetricData=metrics)
    if args.save_report:
        key = f"{args.prefix}/monitoring/reports/{args.endpoint}/{now:%Y/%m/%d/%H}.json"
        s3.put_object(Bucket=args.bucket, Key=key, Body=json.dumps(report, indent=2).encode(),
                      ContentType="application/json")
        report["report"] = f"s3://{args.bucket}/{key}"
    return report


def create_alarm(cloudwatch, args):
    cloudwatch.put_metric_alarm(
        AlarmName=f"{args.endpoint}-model-drift",
        AlarmDescription=f"Live traffic on {args.endpoint} drifted from the training baseline (max PSI).",
        Namespace=NAMESPACE, MetricName="MaxPSI",
        Dimensions=[{"Name": "Endpoint", "Value": args.endpoint}],
        Statistic="Maximum", Period=3600, EvaluationPeriods=1,
        Threshold=args.threshold, ComparisonOperator="GreaterThanOrEqualToThreshold",
        TreatMissingData="notBreaching", AlarmActions=[args.sns_topic_arn],
    )


def print_report(report):
    print(f"Endpoint {report.get('endpoint')}: {report['status']} "
          f"({report.get('samples', 0)} samples, last {report.get('window_hours')}h)")
    if "features" in report:
        print(f"{'feature':24} {'PSI':>8} {'baseline mean':>14} {'live mean':>10}")
        for name, s in sorted(report["features"].items(), key=lambda kv: -kv[1]["psi"]):
            flag = {"significant": "  <-- drift", "moderate": "  <-- watch"}.get(drift.severity(s["psi"]), "")
            print(f"{name:24} {s['psi']:8.3f} {s['baseline_mean']:14.3f} {s['live_mean']:10.3f}{flag}")
    if report.get("detail"):
        print(report["detail"])
    if report.get("report"):
        print(f"Report saved: {report['report']}")


def main(argv=None):
    args = parse_args(argv)
    import boto3

    session = boto3.Session(region_name=args.region)
    if not args.bucket:
        account = session.client("sts").get_caller_identity()["Account"]
        args.bucket = f"sagemaker-{args.region}-{account}"
    s3, cloudwatch = session.client("s3"), session.client("cloudwatch")
    report = check(args, s3, cloudwatch)
    if args.create_alarm:
        create_alarm(cloudwatch, args)
        print(f"Alarm {args.endpoint}-model-drift -> {args.sns_topic_arn}")
    print_report(report)
    return 2 if report["status"] == "significant" else 0


if __name__ == "__main__":
    sys.exit(main())

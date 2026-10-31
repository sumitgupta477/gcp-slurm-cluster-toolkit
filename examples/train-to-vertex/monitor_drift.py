"""Check a Vertex AI endpoint for drift against the baseline saved with the deployed model.

Vertex AI logs each request and response to BigQuery (request-response logging, turned on
by deploy_vertex.sh). This script reads the last --hours of logs, computes the same features
as training (drift.py) and scores each feature with the Population Stability Index (PSI):
< 0.1 none, 0.1-0.2 moderate, >= 0.2 significant. A feature only counts if its shift is also
statistically significant (chi-square test, Bonferroni-corrected); the drift score is the
highest PSI among those features (0 if none).

By default it finds everything from the endpoint ID:
  * the BigQuery log table, from the endpoint's logging configuration
  * the baseline, gs://<deployed model version's folder>/drift_baseline.json (written by train.py)

    python monitor_drift.py --project my-project --endpoint-id 1234567890
    python monitor_drift.py ... --publish-metrics --save-report gs://BUCKET/monitoring/reports
    python monitor_drift.py ... --print-alert-policy CHANNEL_ID > policy.json
        && gcloud monitoring policies create --policy-from-file policy.json

Exit code: 0 = no/moderate drift or not enough data, 2 = significant drift.
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drift  # noqa: E402

METRIC_PREFIX = "custom.googleapis.com/model_drift"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", required=True)
    p.add_argument("--region", default="us-central1")
    p.add_argument("--endpoint-id", required=True)
    p.add_argument("--table", help="BigQuery log table (project.dataset.table or bq://...); default: from the endpoint")
    p.add_argument("--baseline", help="gs://.../drift_baseline.json or a local file; default: from the deployed model")
    p.add_argument("--hours", type=int, default=24, help="How much recent traffic to analyse")
    p.add_argument("--min-samples", type=int, default=30, help="Minimum logged predictions to score drift")
    p.add_argument("--limit", type=int, default=100000, help="Maximum log rows to read")
    p.add_argument("--time-column", default="logging_time")
    p.add_argument("--request-column", default="request_payload")
    p.add_argument("--response-column", default="response_payload")
    p.add_argument("--publish-metrics", action="store_true", help=f"Write {METRIC_PREFIX}/* to Cloud Monitoring")
    p.add_argument("--save-report", metavar="GS_PREFIX", help="Write the JSON report under this gs:// prefix")
    p.add_argument("--print-alert-policy", metavar="NOTIFICATION_CHANNEL_ID",
                   help="Print a Cloud Monitoring alert policy (JSON) on the drift score and exit")
    p.add_argument("--threshold", type=float, default=0.2, help="Alert threshold on the drift score")
    args = p.parse_args(argv)
    for col in (args.time_column, args.request_column, args.response_column):
        if not _IDENTIFIER.match(col):
            p.error(f"invalid column name: {col!r}")
    return args


# --------------------------------------------------------------------------- lookups
def resolve_from_endpoint(project, region, endpoint_id):
    """(log table, baseline gs:// URI) from the endpoint's config and its deployed model."""
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)
    endpoint = aiplatform.Endpoint(endpoint_id)
    cfg = endpoint.gca_resource.predict_request_response_logging_config
    table = cfg.bigquery_destination.output_uri if cfg and cfg.enabled else None
    deployed = endpoint.gca_resource.deployed_models
    baseline = None
    if deployed:
        dm = deployed[0]
        model = aiplatform.Model(dm.model, version=getattr(dm, "model_version_id", None) or None)
        baseline = f"{model.uri.rstrip('/')}/drift_baseline.json"
    return table, baseline


def normalize_table(table: str) -> str:
    table = table[len("bq://"):] if table.startswith("bq://") else table
    if not re.match(r"^[A-Za-z0-9_\-]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_\-]+$", table):
        raise ValueError(f"Expected project.dataset.table, got {table!r}")
    return table


def load_baseline(uri: str, storage_client=None) -> dict:
    if uri.startswith("gs://"):
        bucket, _, key = uri[len("gs://"):].partition("/")
        if storage_client is None:
            from google.cloud import storage
            storage_client = storage.Client()
        return json.loads(storage_client.bucket(bucket).blob(key).download_as_text())
    return json.loads(Path(uri).read_text())


def fetch_rows(bq_client, table, args) -> list:
    from google.cloud import bigquery

    sql = (f"SELECT {args.request_column} AS req, {args.response_column} AS res "
           f"FROM `{table}` "
           f"WHERE {args.time_column} >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @hours HOUR) "
           f"LIMIT @limit")
    config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("hours", "INT64", args.hours),
        bigquery.ScalarQueryParameter("limit", "INT64", args.limit),
    ])
    rows = []
    for record in bq_client.query(sql, job_config=config).result():
        for text, label, score in drift.parse_logged_row(record["req"], record["res"]):
            rows.append(drift.features(text, label, score))
    return rows


# --------------------------------------------------------------------------- scoring + outputs
def score(baseline: dict, rows: list, args, now=None) -> dict:
    now = now or datetime.now(timezone.utc)
    report = {"checked_at": now.isoformat(), "endpoint_id": args.endpoint_id, "window_hours": args.hours,
              "samples": len(rows), "baseline": {k: baseline.get(k) for k in ("created", "source", "count")}}
    if len(rows) < args.min_samples:
        report["status"] = "insufficient_data"
        return report
    feats = drift.compare(baseline, rows)
    drift_score, feature = drift.drift_score(feats)
    report.update(status=drift.severity(drift_score), drift_score=drift_score, drift_feature=feature, features=feats)
    return report


def time_series(project, report, now=None) -> list:
    """Cloud Monitoring time series for the report (drift score, sample count, PSI per feature)."""
    from google.cloud import monitoring_v3

    now = now or datetime.now(timezone.utc)
    seconds = int(now.timestamp())

    def series(name, value, labels):
        ts = monitoring_v3.TimeSeries()
        ts.metric.type = f"{METRIC_PREFIX}/{name}"
        ts.metric.labels.update(labels)
        ts.resource.type = "global"
        ts.resource.labels["project_id"] = project
        ts.points = [monitoring_v3.Point({"interval": {"end_time": {"seconds": seconds}},
                                          "value": {"double_value": float(value)}})]
        return ts

    ep = {"endpoint_id": str(report["endpoint_id"])}
    out = [series("sample_count", report["samples"], ep)]
    if "drift_score" in report:
        out.append(series("drift_score", report["drift_score"], ep))
        out.extend(series("psi", s["psi"], {**ep, "feature": name}) for name, s in report["features"].items())
    return out


def alert_policy(args, channel_id: str) -> dict:
    return {
        "displayName": f"Model drift on Vertex AI endpoint {args.endpoint_id}",
        "combiner": "OR",
        "conditions": [{
            "displayName": f"Drift score above {args.threshold}",
            "conditionThreshold": {
                "filter": (f'metric.type="{METRIC_PREFIX}/drift_score" AND resource.type="global" '
                           f'AND metric.label.endpoint_id="{args.endpoint_id}"'),
                "comparison": "COMPARISON_GT",
                "thresholdValue": args.threshold,
                "duration": "0s",
                "aggregations": [{"alignmentPeriod": "3600s", "perSeriesAligner": "ALIGN_MAX"}],
            },
        }],
        "notificationChannels": [f"projects/{args.project}/notificationChannels/{channel_id}"],
        "documentation": {"content": "Live traffic drifted from the training baseline. "
                                     "Run monitor_drift.py to see which feature moved.", "mimeType": "text/markdown"},
    }


def save_report(report, gs_prefix, storage_client=None) -> str:
    if storage_client is None:
        from google.cloud import storage
        storage_client = storage.Client()
    bucket, _, prefix = gs_prefix[len("gs://"):].partition("/")
    key = f"{prefix.rstrip('/')}/{report['endpoint_id']}/{report['checked_at'][:13].replace('-', '/').replace('T', '/')}.json".lstrip("/")
    storage_client.bucket(bucket).blob(key).upload_from_string(json.dumps(report, indent=2), content_type="application/json")
    return f"gs://{bucket}/{key}"


def print_report(report):
    print(f"Endpoint {report.get('endpoint') or report.get('endpoint_id')}: {report['status']} "
          f"({report.get('samples', 0)} samples, last {report.get('window_hours')}h)")
    if "features" in report:
        print(f"{'feature':24} {'PSI':>7} {'p-value':>9} {'baseline mean':>14} {'live mean':>10}")
        for name, s in sorted(report["features"].items(), key=lambda kv: -kv[1]["psi"]):
            level = drift.severity(s["psi"]) if s["significant"] else "none"
            flag = {"significant": "  <-- drift", "moderate": "  <-- watch"}.get(level, "")
            print(f"{name:24} {s['psi']:7.3f} {s['p_value']:9.2g} {s['baseline_mean']:14.3f} {s['live_mean']:10.3f}{flag}")
    if report.get("detail"):
        print(report["detail"])
    if report.get("report"):
        print(f"Report saved: {report['report']}")


def main(argv=None, clients=None):
    args = parse_args(argv)
    if args.print_alert_policy:
        print(json.dumps(alert_policy(args, args.print_alert_policy), indent=2))
        return 0
    clients = clients or {}

    table, baseline_uri = args.table, args.baseline
    if not table or not baseline_uri:
        found_table, found_baseline = clients.get("resolve", resolve_from_endpoint)(args.project, args.region, args.endpoint_id)
        table, baseline_uri = table or found_table, baseline_uri or found_baseline
    if not table:
        print("Request-response logging is not enabled on this endpoint (see deploy_vertex.sh), or pass --table.")
        return 1
    if not baseline_uri:
        print("No deployed model found to locate drift_baseline.json; pass --baseline.")
        return 1

    try:
        baseline = load_baseline(baseline_uri, clients.get("storage"))
    except Exception as err:  # noqa: BLE001 - report any lookup failure plainly
        print(f"Could not read baseline {baseline_uri}: {err}")
        return 1

    if "bigquery" not in clients:
        from google.cloud import bigquery
        clients["bigquery"] = bigquery.Client(project=args.project)
    rows = fetch_rows(clients["bigquery"], normalize_table(table), args)
    report = score(baseline, rows, args)

    if args.publish_metrics:
        if "monitoring" not in clients:
            from google.cloud import monitoring_v3
            clients["monitoring"] = monitoring_v3.MetricServiceClient()
        clients["monitoring"].create_time_series(name=f"projects/{args.project}", time_series=time_series(args.project, report))
    if args.save_report:
        report["report"] = save_report(report, args.save_report, clients.get("storage"))
    print_report(report)
    return 2 if report["status"] == "significant" else 0


if __name__ == "__main__":
    sys.exit(main())

"""Tests for drift monitoring in the SageMaker example.

    pip install pytest boto3                                  # drift + monitor tests
    pip install torch transformers datasets accelerate        # + the real training test
    pytest examples/sagemaker-hf/tests

The training test fine-tunes a tiny randomly initialised DistilBERT on a toy dataset on CPU
(no downloads), so it checks train.py's baseline code path end to end in seconds.
"""
import base64
import io
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "train"))
sys.path.insert(0, str(HERE))
import drift  # noqa: E402
import monitor_drift  # noqa: E402

POS = ["great", "loved", "brilliant", "wonderful", "fun"]
NEG = ["awful", "hated", "boring", "terrible", "bad"]


# --------------------------------------------------------------------------- fakes + helpers
class NoSuchKey(Exception):
    pass


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.exceptions = SimpleNamespace(NoSuchKey=NoSuchKey)

    def put_object(self, Bucket, Key, Body, **_):
        self.objects[(Bucket, Key)] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise NoSuchKey(Key)
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def get_paginator(self, _):
        s3 = self

        class P:
            def paginate(self, Bucket, Prefix):
                keys = sorted(k for b, k in s3.objects if b == Bucket and k.startswith(Prefix))
                yield {"Contents": [{"Key": k} for k in keys]} if keys else {}
        return P()


class FakeCloudWatch:
    def __init__(self):
        self.metrics, self.alarms = [], []

    def put_metric_data(self, Namespace, MetricData):
        self.metrics.append((Namespace, MetricData))

    def put_metric_alarm(self, **kw):
        self.alarms.append(kw)


def review(rnd, positive):
    w = rnd.choice(POS if positive else NEG)
    return f"the movie was {w} and the acting was {w}"


def capture_line(texts, preds, b64=False):
    req = json.dumps({"inputs": texts if len(texts) > 1 else texts[0]})
    res = json.dumps(preds)
    def part(data, mode):
        if b64:
            return {"observedContentType": "application/json", "mode": mode, "encoding": "BASE64",
                    "data": base64.b64encode(data.encode()).decode()}
        return {"observedContentType": "application/json", "mode": mode, "encoding": "JSON", "data": data}
    return json.dumps({"captureData": {"endpointInput": part(req, "INPUT"), "endpointOutput": part(res, "OUTPUT")},
                       "eventMetadata": {"eventId": "e", "inferenceTime": "2026-10-30T12:00:00Z"}, "eventVersion": "0"})


def args_for(**kw):
    base = dict(endpoint="hf-ep", bucket="b", prefix="hf-sentiment", region="us-east-1", variant="AllTraffic",
                hours=3, min_samples=30, publish_metrics=True, save_report=True, create_alarm=False,
                sns_topic_arn=None, threshold=0.2)
    base.update(kw)
    return SimpleNamespace(**base)


def put_capture(s3, lines, hours_ago=0, endpoint="hf-ep"):
    t = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    key = f"hf-sentiment/datacapture/{endpoint}/AllTraffic/{t:%Y/%m/%d/%H}/{t:%M-%S}-{random.random():.6f}.jsonl"
    s3.put_object(Bucket="b", Key=key, Body="\n".join(lines) + "\n")


def baseline_rows(n=400, seed=0):
    rnd = random.Random(seed)
    rows = []
    for _ in range(n):
        pos = rnd.random() < 0.5
        rows.append(drift.features(review(rnd, pos), "POSITIVE" if pos else "NEGATIVE", rnd.uniform(0.8, 0.99)))
    return rows


# --------------------------------------------------------------------------- drift library
def test_features():
    f = drift.features("Is this the best film ever?!", "POSITIVE", 0.93)
    assert f["pred_positive"] == 1 and f["confidence"] == 0.93 and f["input_exclamations"] == 1
    assert f["input_is_question"] == 0  # ends with '!'
    assert drift.features("meh", "LABEL_0", 0.5)["pred_positive"] == 0
    assert set(f) == set(drift.FEATURES)


@pytest.mark.parametrize("b64", [False, True])
def test_parse_capture_single_and_batch(b64):
    one = capture_line(["good film"], [{"label": "POSITIVE", "score": 0.9}], b64)
    assert drift.parse_capture_line(one) == [("good film", "POSITIVE", 0.9)]
    two = capture_line(["a", "b"], [{"label": "POSITIVE", "score": 0.9}, {"label": "NEGATIVE", "score": 0.8}], b64)
    assert [t for t, _, _ in drift.parse_capture_line(two)] == ["a", "b"]
    nested = capture_line(["x"], [[{"label": "NEGATIVE", "score": 0.7}]], b64)
    assert drift.parse_capture_line(nested) == [("x", "NEGATIVE", 0.7)]
    assert drift.parse_capture_line("garbage") == []


def test_values_outside_training_range_count_as_drift():
    # Regression: with a constant baseline, larger/smaller live values used to share the
    # baseline's bin and score PSI 0.
    rows = [drift.features("the movie was great and the acting was great", "POSITIVE", 0.9)] * 100
    base = drift.build_baseline(rows)
    same = drift.compare(base, rows)["input_words"]["psi"]
    longer = drift.compare(base, [drift.features("word " * 30, "POSITIVE", 0.9)] * 50)["input_words"]["psi"]
    shorter = drift.compare(base, [drift.features("ok", "POSITIVE", 0.9)] * 50)["input_words"]["psi"]
    assert same == pytest.approx(0) and longer >= 0.2 and shorter >= 0.2


# --------------------------------------------------------------------------- monitor_drift.py
def setup_s3():
    s3 = FakeS3()
    s3.put_object(Bucket="b", Key="hf-sentiment/monitoring/baseline.json",
                  Body=json.dumps(drift.build_baseline(baseline_rows(), source="test")))
    return s3


def test_monitor_no_baseline():
    assert monitor_drift.check(args_for(), FakeS3(), FakeCloudWatch())["status"] == "no_baseline"


def test_monitor_insufficient_data():
    s3, cw = setup_s3(), FakeCloudWatch()
    put_capture(s3, [capture_line(["good film"], [{"label": "POSITIVE", "score": 0.9}])] * 5)
    rep = monitor_drift.check(args_for(), s3, cw)
    assert rep["status"] == "insufficient_data" and rep["samples"] == 5
    assert [m["MetricName"] for m in cw.metrics[0][1]] == ["SampleCount"]


def test_monitor_similar_traffic_no_drift():
    s3, cw, rnd = setup_s3(), FakeCloudWatch(), random.Random(3)
    lines = []
    for i in range(150):
        pos = rnd.random() < 0.5
        lines.append(capture_line([review(rnd, pos)], [{"label": "POSITIVE" if pos else "NEGATIVE",
                                                        "score": rnd.uniform(0.8, 0.99)}], b64=i % 3 == 0))
    put_capture(s3, lines[:75], 0)
    put_capture(s3, lines[75:], 2)
    put_capture(s3, lines, 6)                        # outside the 3h window
    put_capture(s3, lines, 0, endpoint="other-ep")   # another endpoint
    rep = monitor_drift.check(args_for(), s3, cw)
    assert rep["samples"] == 150 and rep["status"] == "none", rep
    assert cw.metrics[0][0] == "HFSentiment/ModelDrift"
    assert json.loads(s3.objects[("b", rep["report"].split("s3://b/")[1])])["status"] == "none"


def test_monitor_detects_prediction_and_input_drift():
    s3, cw = setup_s3(), FakeCloudWatch()
    # New traffic: long non-English support emails; the model says NEGATIVE with low confidence.
    text = "Ich möchte mein Geld zurück, Bestellung 48213 kam kaputt an!! " * 3
    put_capture(s3, [capture_line([text], [{"label": "NEGATIVE", "score": 0.55}])] * 60)
    rep = monitor_drift.check(args_for(), s3, cw)
    assert rep["status"] == "significant" and rep["max_psi"] >= 0.2
    for f in ("pred_positive", "confidence", "input_words", "input_non_ascii_ratio"):
        assert rep["features"][f]["psi"] >= 0.2, f


def test_monitor_alarm_and_exit_code(monkeypatch, capsys):
    cw = FakeCloudWatch()
    monitor_drift.create_alarm(cw, args_for(create_alarm=True, sns_topic_arn="arn:aws:sns:us-east-1:1:alerts"))
    alarm = cw.alarms[0]
    assert alarm["MetricName"] == "MaxPSI" and alarm["Threshold"] == 0.2
    assert alarm["AlarmActions"] == ["arn:aws:sns:us-east-1:1:alerts"] and alarm["TreatMissingData"] == "notBreaching"

    s3 = setup_s3()
    text = "Ich möchte mein Geld zurück, Bestellung 48213 kam kaputt an!! " * 3
    put_capture(s3, [capture_line([text], [{"label": "NEGATIVE", "score": 0.55}])] * 60)

    class FakeSession:
        def __init__(self, region_name):
            pass

        def client(self, name):
            return {"s3": s3, "cloudwatch": cw, "sts": SimpleNamespace(get_caller_identity=lambda: {"Account": "1"})}[name]

    import boto3
    monkeypatch.setattr(boto3, "Session", FakeSession)
    code = monitor_drift.main(["--endpoint", "hf-ep", "--bucket", "b", "--hours", "3"])
    out = capsys.readouterr().out
    assert code == 2 and "significant" in out and "<-- drift" in out
    with pytest.raises(SystemExit):
        monitor_drift.parse_args(["--endpoint", "x", "--create-alarm"])  # needs --sns-topic-arn


# --------------------------------------------------------------------------- train.py (real, tiny)
def test_train_builds_baseline(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    datasets = pytest.importorskip("datasets")
    from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizerFast

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "the", "movie", "was", "and", "acting"] + POS + NEG
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    tok = DistilBertTokenizerFast(str(tmp_path / "vocab.txt"), do_lower_case=True)
    cfg = DistilBertConfig(vocab_size=len(vocab), dim=32, n_layers=1, n_heads=2, hidden_dim=64, max_position_embeddings=64)
    DistilBertForSequenceClassification(cfg).save_pretrained(tmp_path / "base")
    tok.save_pretrained(tmp_path / "base")
    rnd = random.Random(0)
    make = lambda n: [{"text": review(rnd, l), "label": l} for l in (rnd.randint(0, 1) for _ in range(n))]
    datasets.DatasetDict(train=datasets.Dataset.from_list(make(200)),
                         test=datasets.Dataset.from_list(make(60))).save_to_disk(str(tmp_path / "data"))

    env = dict(os.environ, SM_MODEL_DIR=str(tmp_path / "model"), SM_OUTPUT_DATA_DIR=str(tmp_path / "out"),
               TRAINING_JOB_NAME="hf-test-job")
    subprocess.run([sys.executable, str(HERE / "train" / "train.py"), "--model_name", str(tmp_path / "base"),
                    "--dataset", str(tmp_path / "data"), "--epochs", "2", "--learning_rate", "5e-3",
                    "--max_length", "32", "--train_samples", "200", "--eval_samples", "60"],
                   check=True, env=env, cwd=HERE / "train", capture_output=True)
    base = json.loads((tmp_path / "out" / "drift_baseline.json").read_text())
    assert base["count"] == 60 and set(base["features"]) == set(drift.FEATURES)
    assert "hf-test-job" in base["source"] and base["eval_accuracy"] is not None
    assert 0.5 <= base["features"]["confidence"]["mean"] <= 1.0
    assert (tmp_path / "model" / "config.json").exists()

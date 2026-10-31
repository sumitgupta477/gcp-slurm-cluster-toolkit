"""Tests for drift monitoring in the train-to-Vertex example.

    pip install pytest google-cloud-bigquery google-cloud-monitoring    # monitor tests
    pip install torch transformers datasets accelerate                  # + the real training test
    pytest examples/train-to-vertex/tests

BigQuery, Cloud Storage and Cloud Monitoring are replaced by in-memory fakes; rows are
shaped like Vertex AI request-response logs (one JSON-encoded element per instance).
"""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import drift  # noqa: E402
import monitor_drift  # noqa: E402

POS = ["great", "loved", "brilliant", "wonderful", "fun"]
NEG = ["awful", "hated", "boring", "terrible", "bad"]


def review(rnd, positive):
    w = rnd.choice(POS if positive else NEG)
    return f"the movie was {w} and the acting was {w}"


# --------------------------------------------------------------------------- fakes
class FakeBlob:
    def __init__(self, store, key):
        self.store, self.key = store, key

    def download_as_text(self):
        return self.store[self.key]

    def upload_from_string(self, data, content_type=None):
        self.store[self.key] = data


class FakeStorage:
    def __init__(self):
        self.store = {}

    def bucket(self, name):
        store = self.store
        return type("B", (), {"blob": lambda _self, key: FakeBlob(store, f"{name}/{key}")})()


class FakeBigQuery:
    def __init__(self, rows):
        self.rows, self.queries = rows, []

    def query(self, sql, job_config=None):
        self.queries.append((sql, job_config))
        rows = self.rows
        return type("J", (), {"result": lambda _self: rows})()


class FakeMonitoring:
    def __init__(self):
        self.calls = []

    def create_time_series(self, name, time_series):
        self.calls.append((name, time_series))


def logged_row(texts, preds):
    """A row as Vertex AI logs it: arrays of JSON-encoded instances and predictions."""
    return {"req": [json.dumps(t) for t in texts], "res": [json.dumps(p) for p in preds]}


def baseline_doc(n=400, seed=0):
    rnd = random.Random(seed)
    rows = []
    for _ in range(n):
        pos = rnd.random() < 0.5
        rows.append(drift.features(review(rnd, pos), "POSITIVE" if pos else "NEGATIVE", rnd.uniform(0.8, 0.99)))
    return drift.build_baseline(rows, source="test")


def run_main(bq_rows, extra_args=(), baseline=None):
    storage, mon, bq = FakeStorage(), FakeMonitoring(), FakeBigQuery(bq_rows)
    storage.store["models-bucket/run-1/drift_baseline.json"] = json.dumps(baseline or baseline_doc())
    resolve = lambda project, region, ep: ("bq://proj.vertex_logging.sentiment_x", "gs://models-bucket/run-1/drift_baseline.json")
    code = monitor_drift.main(["--project", "proj", "--endpoint-id", "123", "--hours", "6", *extra_args],
                              clients={"resolve": resolve, "storage": storage, "bigquery": bq, "monitoring": mon})
    return code, storage, mon, bq


# --------------------------------------------------------------------------- drift library
def test_parse_logged_row_formats():
    assert drift.parse_logged_row(['"good film"'], ['{"label": "POSITIVE", "score": 0.9}']) == [("good film", "POSITIVE", 0.9)]
    assert drift.parse_logged_row([json.dumps({"text": "meh"})], [{"label": "NEGATIVE", "score": 0.6}]) == [("meh", "NEGATIVE", 0.6)]
    whole = drift.parse_logged_row(json.dumps({"instances": ["a", "b"]}),
                                   json.dumps({"predictions": [{"label": "POSITIVE", "score": 0.9},
                                                               {"label": "NEGATIVE", "score": 0.8}]}))
    assert [t for t, _, _ in whole] == ["a", "b"]
    assert drift.parse_logged_row(["not json"], ["also not json"]) == []


def test_values_outside_training_range_count_as_drift():
    rows = [drift.features("the movie was great and the acting was great", "POSITIVE", 0.9)] * 100
    base = drift.build_baseline(rows)
    assert drift.compare(base, rows)["input_words"]["psi"] == pytest.approx(0)
    assert drift.compare(base, [drift.features("word " * 30, "POSITIVE", 0.9)] * 50)["input_words"]["psi"] >= 0.2


def test_chi2_sf_matches_known_critical_values():
    assert drift.chi2_sf(3.841, 1) == pytest.approx(0.05, abs=1e-3)
    assert drift.chi2_sf(11.070, 5) == pytest.approx(0.05, abs=1e-3)
    assert drift.chi2_sf(23.209, 10) == pytest.approx(0.01, abs=1e-3)


def _sample(rnd, n, shift=False):
    rows = []
    for _ in range(n):
        pos = rnd.random() < (0.1 if shift else 0.5)
        words = rnd.randint(80, 200) if shift else rnd.randint(5, 60)
        conf = rnd.uniform(0.5, 0.75) if shift else min(0.999, rnd.betavariate(8, 1.2))
        rows.append(drift.features("word " * words, "POSITIVE" if pos else "NEGATIVE", conf))
    return rows


def test_low_false_alarm_rate_and_high_detection():
    # Unchanged traffic, small and large samples, small and large baselines: almost never flagged.
    alarms = 0
    for seed in range(60):
        rnd = random.Random(seed)
        base = drift.build_baseline(_sample(rnd, rnd.choice([60, 500])))
        score, _ = drift.drift_score(drift.compare(base, _sample(rnd, rnd.choice([30, 100, 200]))))
        alarms += score >= 0.2
    assert alarms <= 2, alarms
    # Shifted traffic (labels, length and confidence all move) is flagged even with 30 samples.
    for seed in range(20):
        rnd = random.Random(1000 + seed)
        base = drift.build_baseline(_sample(rnd, 60))
        score, feature = drift.drift_score(drift.compare(base, _sample(rnd, 30, shift=True)))
        assert score >= 0.2 and feature is not None


# --------------------------------------------------------------------------- monitor_drift.py
def test_no_drift_on_similar_traffic(capsys):
    rnd = random.Random(5)
    rows = []
    for _ in range(100):
        pos = rnd.random() < 0.5
        rows.append(logged_row([review(rnd, pos)], [{"label": "POSITIVE" if pos else "NEGATIVE",
                                                     "score": rnd.uniform(0.8, 0.99)}]))
    code, storage, mon, bq = run_main(rows, ["--publish-metrics", "--save-report", "gs://reports/drift"])
    assert code == 0
    out = capsys.readouterr().out
    assert "none (100 samples" in out
    sql, cfg = bq.queries[0]
    assert "FROM `proj.vertex_logging.sentiment_x`" in sql and "logging_time" in sql
    assert {p.name: p.value for p in cfg.query_parameters} == {"hours": 6, "limit": 100000}
    name, series = mon.calls[0]
    assert name == "projects/proj"
    types = {s.metric.type for s in series}
    assert types == {f"{monitor_drift.METRIC_PREFIX}/{m}" for m in ("sample_count", "drift_score", "psi")}
    assert all(s.resource.type == "global" and s.metric.labels["endpoint_id"] == "123" for s in series)
    assert any(k.startswith("reports/drift/123/") for k in storage.store)


def test_detects_drift_and_exit_code(capsys):
    text = "Ich möchte mein Geld zurück, Bestellung 48213 kam kaputt an!! " * 3
    rows = [logged_row([text], [{"label": "NEGATIVE", "score": 0.55}])] * 40
    code, *_ = run_main(rows)
    out = capsys.readouterr().out
    assert code == 2 and "significant" in out and "<-- drift" in out


def test_insufficient_data_and_missing_pieces(capsys):
    code, *_ = run_main([logged_row(["fine"], [{"label": "POSITIVE", "score": 0.9}])] * 3)
    assert code == 0 and "insufficient_data" in capsys.readouterr().out
    code = monitor_drift.main(["--project", "p", "--endpoint-id", "1"],
                              clients={"resolve": lambda *a: (None, "gs://x/y.json")})
    assert code == 1 and "not enabled" in capsys.readouterr().out


def test_alert_policy_and_validation(capsys):
    assert monitor_drift.main(["--project", "proj", "--endpoint-id", "123", "--print-alert-policy", "987"]) == 0
    policy = json.loads(capsys.readouterr().out)
    cond = policy["conditions"][0]["conditionThreshold"]
    assert 'metric.label.endpoint_id="123"' in cond["filter"] and cond["thresholdValue"] == 0.2
    assert "drift_score" in cond["filter"]
    assert policy["notificationChannels"] == ["projects/proj/notificationChannels/987"]
    with pytest.raises(SystemExit):
        monitor_drift.parse_args(["--project", "p", "--endpoint-id", "1", "--time-column", "x; DROP TABLE y"])
    assert monitor_drift.normalize_table("bq://p.d.t") == "p.d.t"
    with pytest.raises(ValueError):
        monitor_drift.normalize_table("p.d`; --")


# --------------------------------------------------------------------------- train.py (real, tiny)
def test_train_writes_baseline_into_model_folder(tmp_path):
    pytest.importorskip("torch")
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

    out = tmp_path / "model"
    subprocess.run([sys.executable, str(HERE / "train.py"), "--output_dir", str(out),
                    "--model_name", str(tmp_path / "base"), "--dataset", str(tmp_path / "data"),
                    "--epochs", "2", "--learning_rate", "5e-3", "--max_length", "32"],
                   check=True, env=dict(os.environ, SLURM_JOB_ID="777"), capture_output=True)
    base = json.loads((out / "drift_baseline.json").read_text())
    assert base["count"] == 60 and "Slurm job 777" in base["source"] and base["eval_accuracy"] is not None
    assert set(base["features"]) == set(drift.FEATURES)
    assert (out / "config.json").exists() and (out / "model.safetensors").exists()

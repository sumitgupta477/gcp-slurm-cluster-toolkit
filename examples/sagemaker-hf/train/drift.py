"""Drift detection for the sentiment classifier, in plain Python (no numpy).

Shared by train.py (builds the baseline from the model's predictions on held-out data)
and monitor_drift.py (scores live endpoint traffic against it), so both sides compute
features in exactly the same way.

Features per request/response:
  input text:  words, characters, non-ASCII share, digit share, '!' count, is-question
  prediction:  predicted POSITIVE (0/1) and the model's confidence

Drift per feature is the Population Stability Index (PSI):
  < 0.1 no meaningful change · 0.1-0.2 moderate · >= 0.2 significant
"""
import base64
import bisect
import json
import math
from datetime import datetime, timezone

FEATURES = {
    "input_words": "numeric",
    "input_chars": "numeric",
    "input_non_ascii_ratio": "numeric",
    "input_digit_ratio": "numeric",
    "input_exclamations": "numeric",
    "input_is_question": "binary",
    "pred_positive": "binary",
    "confidence": "numeric",
}
PSI_EPSILON = 1e-4


def _ratio(count, total):
    return count / total if total else 0.0


def features(text: str, label: str, score: float) -> dict:
    text = text or ""
    return {
        "input_words": len(text.split()),
        "input_chars": len(text),
        "input_non_ascii_ratio": _ratio(sum(not c.isascii() for c in text), len(text)),
        "input_digit_ratio": _ratio(sum(c.isdigit() for c in text), len(text)),
        "input_exclamations": text.count("!"),
        "input_is_question": int(text.rstrip().endswith("?")),
        "pred_positive": int(str(label).upper().startswith("POS") or str(label) == "LABEL_1"),
        "confidence": float(score),
    }


# --------------------------------------------------------------------------- capture parsing
def _decode(event_data: dict) -> str:
    data = event_data.get("data", "")
    if str(event_data.get("encoding", "")).upper() == "BASE64":
        data = base64.b64decode(data).decode("utf-8", errors="replace")
    return data


def parse_capture_line(line: str) -> list:
    """One SageMaker data-capture JSON line -> [(text, label, score), ...] (a request may hold several texts)."""
    try:
        capture = json.loads(line)["captureData"]
        request = json.loads(_decode(capture["endpointInput"]))
        response = json.loads(_decode(capture["endpointOutput"]))
    except (KeyError, TypeError, ValueError):
        return []
    texts = request.get("inputs") if isinstance(request, dict) else request
    texts = texts if isinstance(texts, list) else [texts]
    preds = response if isinstance(response, list) else [response]
    out = []
    for text, pred in zip(texts, preds):
        if isinstance(pred, list) and pred:  # some pipelines nest results per input
            pred = pred[0]
        if isinstance(text, str) and isinstance(pred, dict) and "label" in pred and "score" in pred:
            out.append((text, pred["label"], pred["score"]))
    return out


# --------------------------------------------------------------------------- baseline + PSI
def _quantile_edges(values: list, bins: int) -> list:
    """Bin edges for a numeric feature: quantile cuts, plus an edge at the baseline minimum
    and just above the baseline maximum. Values outside the range seen in training then
    land in their own (empty-in-baseline) bins and show up as drift, even when the
    baseline is narrow or constant."""
    ordered = sorted(values)
    n = len(ordered)
    lo, hi = ordered[0], ordered[-1]
    edges = [lo]
    for i in range(1, bins):
        cut = ordered[min(n - 1, int(i * n / bins))]
        if cut > edges[-1]:
            edges.append(cut)
    top = hi + max(abs(hi), 1.0) * 1e-9
    if top > edges[-1]:
        edges.append(top)
    return edges


def _proportions(values, kind, edges):
    if kind == "binary":
        counts = [0, 0]
        for v in values:
            counts[1 if v else 0] += 1
    else:
        counts = [0] * (len(edges) + 1)
        for v in values:
            counts[bisect.bisect_right(edges, v)] += 1
    total = sum(counts)
    return [c / total if total else 0.0 for c in counts]


def build_baseline(rows: list, bins: int = 10, source: str = "") -> dict:
    if not rows:
        raise ValueError("Cannot build a baseline from zero rows")
    feats = {}
    for name, kind in FEATURES.items():
        values = [r[name] for r in rows]
        edges = [] if kind == "binary" else _quantile_edges(values, bins)
        feats[name] = {"kind": kind, "edges": edges, "proportions": _proportions(values, kind, edges),
                       "mean": sum(values) / len(values)}
    return {"created": datetime.now(timezone.utc).isoformat(), "source": source, "count": len(rows), "features": feats}


def psi(expected, actual, eps=PSI_EPSILON):
    total = 0.0
    for e, a in zip(expected, actual):
        e, a = max(e, eps), max(a, eps)
        total += (a - e) * math.log(a / e)
    return total


def compare(baseline: dict, rows: list) -> dict:
    result = {}
    for name, spec in baseline["features"].items():
        values = [r[name] for r in rows]
        actual = _proportions(values, spec["kind"], spec["edges"])
        result[name] = {"psi": round(psi(spec["proportions"], actual), 6),
                        "baseline_mean": round(spec["mean"], 6),
                        "live_mean": round(sum(values) / len(values), 6) if values else None}
    return result


def severity(score: float) -> str:
    return "significant" if score >= 0.2 else "moderate" if score >= 0.1 else "none"

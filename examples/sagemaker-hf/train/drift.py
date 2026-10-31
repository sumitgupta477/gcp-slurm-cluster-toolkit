"""Drift detection for the sentiment classifier, in plain Python (no numpy).

Shared by train.py (builds the baseline from the model's predictions on held-out data)
and monitor_drift.py (scores live endpoint traffic against it), so both sides compute
features in exactly the same way.

Features per request/response:
  input text:  words, characters, non-ASCII share, digit share, '!' count, is-question
  prediction:  predicted POSITIVE (0/1) and the model's confidence

Drift per feature is the Population Stability Index (PSI):
  < 0.1 no meaningful change · 0.1-0.2 moderate · >= 0.2 significant
A feature only counts if the shift is also statistically significant (chi-square test,
Bonferroni-corrected across features), so small or noisy samples don't raise false alarms.
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
SMOOTHING = 0.5  # extra count per bin on both sides (keeps empty bins from dominating)
ALPHA = 0.01     # false-alarm rate for the significance test, shared across all features


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
    land in their own bins and show up as drift, even when the baseline is narrow or constant."""
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


def _counts(values: list, kind: str, edges: list) -> list:
    if kind == "binary":
        counts = [0, 0]
        for v in values:
            counts[1 if v else 0] += 1
    else:
        counts = [0] * (len(edges) + 1)
        for v in values:
            counts[bisect.bisect_right(edges, v)] += 1
    return counts


def build_baseline(rows: list, bins: int = 10, source: str = "") -> dict:
    """rows: list of feature dicts. Returns a JSON-serialisable baseline with per-feature bin counts."""
    if not rows:
        raise ValueError("Cannot build a baseline from zero rows")
    feats = {}
    for name, kind in FEATURES.items():
        values = [r[name] for r in rows]
        edges = [] if kind == "binary" else _quantile_edges(values, bins)
        counts = _counts(values, kind, edges)
        feats[name] = {
            "kind": kind,
            "edges": edges,
            "counts": counts,
            "proportions": [c / len(values) for c in counts],
            "mean": sum(values) / len(values),
        }
    return {"created": datetime.now(timezone.utc).isoformat(), "source": source, "count": len(rows), "features": feats}


def psi(expected_counts: list, actual_counts: list, smoothing: float = SMOOTHING):
    """Population Stability Index between two binned samples -> (psi, bins_used).

    Bins empty on both sides are dropped; every remaining bin gets `smoothing` extra counts
    on both sides, so a bin that is empty in one sample cannot blow the score up."""
    used = [i for i, (e, a) in enumerate(zip(expected_counts, actual_counts)) if e + a > 0] or [0]
    k = len(used)
    ne, na = sum(expected_counts), sum(actual_counts)
    total = 0.0
    for i in used:
        e = (expected_counts[i] + smoothing) / (ne + smoothing * k)
        a = (actual_counts[i] + smoothing) / (na + smoothing * k)
        total += (a - e) * math.log(a / e)
    return total, k


def chi2_sf(x: float, df: int) -> float:
    """P(Chi-square(df) >= x): regularized upper incomplete gamma Q(df/2, x/2)."""
    a, x = df / 2.0, x / 2.0
    if x <= 0:
        return 1.0
    log_prefix = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1:  # series for the lower tail P, then Q = 1 - P
        term = total = 1.0 / a
        ap = a
        for _ in range(1000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-12:
                break
        return max(0.0, 1.0 - total * math.exp(log_prefix))
    tiny = 1e-300  # continued fraction for Q (Lentz's method)
    b = x + 1 - a
    c, d = 1.0 / tiny, 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-12:
            break
    return min(1.0, h * math.exp(log_prefix))


def compare(baseline: dict, rows: list, alpha: float = ALPHA) -> dict:
    """Per feature: PSI, chi-square p-value, and whether the shift is statistically significant.

    For two samples of sizes n and m, n_eff * PSI (n_eff = 1 / (1/n + 1/m)) is approximately
    chi-square with (bins - 1) degrees of freedom when nothing changed. A feature counts as
    drifted only if this test is significant at alpha / number-of-features (Bonferroni), so
    small or noisy samples do not raise false alarms."""
    per_feature_alpha = alpha / len(baseline["features"])
    n_base = baseline["count"]
    result = {}
    for name, spec in baseline["features"].items():
        values = [r[name] for r in rows]
        base_counts = spec.get("counts") or [round(p * n_base) for p in spec["proportions"]]
        live_counts = _counts(values, spec["kind"], spec["edges"])
        score, k = psi(base_counts, live_counts)
        n_eff = 1.0 / (1.0 / max(len(values), 1) + 1.0 / max(n_base, 1))
        p_value = chi2_sf(score * n_eff, max(k - 1, 1))
        result[name] = {
            "psi": round(score, 6),
            "p_value": float(f"{p_value:.6g}"),
            "significant": p_value < per_feature_alpha,
            "baseline_mean": round(spec["mean"], 6),
            "live_mean": round(sum(values) / len(values), 6) if values else None,
        }
    return result


def drift_score(features: dict) -> tuple:
    """(highest PSI among statistically significant features, that feature); (0.0, None) if none."""
    sig = {n: s["psi"] for n, s in features.items() if s["significant"]}
    if not sig:
        return 0.0, None
    worst = max(sig, key=sig.get)
    return sig[worst], worst


def severity(score: float) -> str:
    """For a drift_score: >= 0.2 significant, >= 0.1 moderate, otherwise none."""
    return "significant" if score >= 0.2 else "moderate" if score >= 0.1 else "none"

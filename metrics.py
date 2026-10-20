"""Metrics shared by baselines.py and analyze.py."""

import numpy as np
from sklearn.metrics import f1_score

RISK = 0.05  # target error rate for coverage
N_BINS = 10


def f1(y, p):
    return f1_score(y, (np.asarray(p) >= 0.5).astype(int), zero_division=0)


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece(y, p, n_bins=N_BINS):
    """Expected calibration error with equal-mass bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    total = 0.0
    for idx in np.array_split(order, n_bins):
        if len(idx):
            total += len(idx) * abs(p[idx].mean() - y[idx].mean())
    return float(total / len(p))


def confidence(p):
    p = np.asarray(p, float)
    return np.maximum(p, 1 - p)


def coverage_threshold(p_pool, risk=RISK):
    """Lowest confidence threshold t such that, by the model's own probabilities, the expected error
    among messages with confidence >= t is at most `risk`.

    Chosen on (unlabeled) training-pool predictions, so it needs no test labels; a miscalibrated model
    shows up as a realized test risk above `risk`.
    """
    conf = np.sort(confidence(p_pool))[::-1]
    expected_err = np.cumsum(1 - conf) / np.arange(1, len(conf) + 1)
    ok = np.nonzero(expected_err <= risk)[0]
    return float(conf[ok[-1]]) if len(ok) else 1.0 + 1e-9  # nothing qualifies: automate nothing


def coverage_at(y, p, t):
    """Share of messages automated at threshold t and the realized error rate among them."""
    y, p = np.asarray(y), np.asarray(p, float)
    auto = confidence(p) >= t
    cov = float(auto.mean())
    err = float(((p[auto] >= 0.5).astype(int) != y[auto]).mean()) if auto.any() else 0.0
    return cov, err


def all_metrics(y, p, t):
    cov, err = coverage_at(y, p, t)
    return {"f1": f1(y, p), "brier": brier(y, p), "ece": ece(y, p), "coverage": cov, "auto_risk": err}

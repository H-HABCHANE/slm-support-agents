"""Mesures d'évaluation : classification, latence et mémoire."""

from __future__ import annotations

import os
import statistics
import time

import psutil
from sklearn.metrics import accuracy_score, f1_score


def classification_scores(y_true: list[str], y_pred: list[str]) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "n": len(y_true),
    }


def rss_mb() -> float:
    """Mémoire résidente du processus (Mo)."""
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


def timed_predict(classifier, texts: list[str]) -> tuple[list[tuple[str, float]], float]:
    """Prédit ticket par ticket et renvoie la latence médiane en millisecondes (conditions d'un service)."""
    preds, times = [], []
    for t in texts:
        t0 = time.perf_counter()
        preds += classifier.predict([t])
        times.append((time.perf_counter() - t0) * 1000)
    return preds, statistics.median(times)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[k]

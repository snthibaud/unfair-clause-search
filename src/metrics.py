"""Pure metric functions shared across training and evaluation modules."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef


def _active_categories(y_true: np.ndarray) -> list[int]:
    """Return indices of categories with at least one positive label."""
    return [i for i in range(y_true.shape[1]) if y_true[:, i].sum() > 0]


def macro_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged MCC across all categories that have at least one positive."""
    scores = [matthews_corrcoef(y_true[:, i], y_pred[:, i]) for i in _active_categories(y_true)]
    return float(np.mean(scores)) if scores else 0.0


def macro_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Macro-averaged Average Precision (threshold-agnostic ranking quality).

    AP is the area under the precision-recall curve — it summaries
    per-category ranking quality across all possible thresholds.
    Unlike MCC/F1, it does not depend on a single threshold choice.
    """
    scores = [average_precision_score(y_true[:, i], y_prob[:, i]) for i in _active_categories(y_true)]
    return float(np.mean(scores)) if scores else 0.0


def _find_best_threshold_1d(y_true_1d: np.ndarray, y_prob_1d: np.ndarray, metric: str = "mcc") -> float:
    """Find the probability threshold maximising the given metric for one category."""
    scorer = matthews_corrcoef if metric == "mcc" else lambda yt, yp: f1_score(yt, yp, zero_division=0)
    best_t, best_val = 0.5, -1.0
    for t in np.arange(0.1, 0.95, 0.05):
        pred = (y_prob_1d >= t).astype(int)
        val = scorer(y_true_1d, pred)
        if val > best_val:
            best_val, best_t = val, t
    return round(best_t, 2)


def find_thresholds(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "mcc") -> list[float]:
    """Find per-category probability thresholds that maximise the given metric.

    Supports 'mcc' (default) and 'f1'.
    """
    thresholds: list[float] = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() == 0:
            thresholds.append(0.5)
        else:
            thresholds.append(_find_best_threshold_1d(y_true[:, i], y_prob[:, i], metric))
    return thresholds


def threshold_predict(y_prob: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """Apply per-category thresholds to produce binary predictions."""
    return np.column_stack([
        (y_prob[:, i] >= thresholds[i]).astype(int)
        for i in range(y_prob.shape[1])
    ])

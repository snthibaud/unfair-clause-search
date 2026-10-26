"""Leave-one-document-out evaluation matching Lippi et al. (2019) protocol.

Trains and evaluates on a per-document hold-out basis (50 folds).
Reports macro-averaged F1, MCC, and AP for both detection (binary) and
categorization (8 independent classifiers, unfair-only).

Usage:
    python -m src.evaluate_loo
    python -m src.evaluate_loo --model legal-bert-embedding
    python -m src.evaluate_loo --model legal-bert --categorization
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

import faiss
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.multioutput import MultiOutputClassifier

from src.config import DEFAULT_MODEL, get_model_spec, index_dir
from src.data_loader import CATEGORIES, CATEGORY_NAMES
from src.metrics import find_thresholds

# ── document boundaries ──────────────────────────────────────────

def _doc_boundaries(data_dir: str = "data"):
    """Return list of (start, end) sentence indices for each document."""
    from pathlib import Path
    sentences_dir = Path(data_dir) / "Sentences"
    boundaries = []
    offset = 0
    for path in sorted(sentences_dir.glob("*.txt")):
        with open(path) as f:
            n = sum(1 for line in f if line.strip())
        boundaries.append((offset, offset + n))
        offset += n
    return boundaries


# ── LOO evaluation ───────────────────────────────────────────────

def _fit_multi_clf(X_train, y_train, C=50.0):
    return MultiOutputClassifier(
        LogisticRegression(C=C, max_iter=5000, class_weight="balanced", random_state=42)
    ).fit(X_train, y_train)


def _proba_pos(clf, X):
    return np.column_stack([p[:, 1] for p in clf.predict_proba(X)])


def _compute_detection_metrics(y_true, y_prob):
    """Compute per-category and overall binary metrics from predictions."""
    ps, rs, f1s, mccs, aps = [], [], [], [], []
    thresholds = find_thresholds(y_true, y_prob, "mcc")
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() == 0:
            continue
        pred = (y_prob[:, c] >= thresholds[c]).astype(int)
        ps.append(precision_score(y_true[:, c], pred, zero_division=0))
        rs.append(recall_score(y_true[:, c], pred, zero_division=0))
        f1s.append(f1_score(y_true[:, c], pred, zero_division=0))
        mccs.append(matthews_corrcoef(y_true[:, c], pred))
        aps.append(average_precision_score(y_true[:, c], y_prob[:, c]))

    # Overall detection (union of all categories)
    y_bin_true = (y_true.sum(axis=1) > 0).astype(int)
    y_bin_prob = y_prob.max(axis=1)
    best_t, best_mcc = 0.5, -1.0
    for t in np.arange(0.1, 0.95, 0.05):
        pred = (y_bin_prob >= t).astype(int)
        m = matthews_corrcoef(y_bin_true, pred)
        if m > best_mcc:
            best_mcc, best_t = m, t
    det_pred = (y_bin_prob >= best_t).astype(int)

    return {
        "detection": {
            "P": precision_score(y_bin_true, det_pred, zero_division=0),
            "R": recall_score(y_bin_true, det_pred, zero_division=0),
            "F1": f1_score(y_bin_true, det_pred, zero_division=0),
        },
        "per_category": {
            "macro_P": float(np.mean(ps)),
            "macro_R": float(np.mean(rs)),
            "macro_F1": float(np.mean(f1s)),
            "macro_MCC": float(np.mean(mccs)),
            "macro_AP": float(np.mean(aps)),
        },
    }


def evaluate_loo_detection(model_key: str, C: float = 50.0):
    """LOO evaluation for multi-output unfair clause detection.

    Uses a fixed C value (pre-tuned; the MCC plateau above C=10 makes
    per-fold re-tuning unnecessary).  Returns macro-averaged metrics.
    """
    spec = get_model_spec(model_key)
    idir = index_dir(spec)

    index = faiss.read_index(os.path.join(idir, "faiss.index"))
    with open(os.path.join(idir, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    X = index.reconstruct_n(0, index.ntotal).astype(np.float64)
    y = np.array(metadata["multi_labels"], dtype=np.uint8)

    boundaries = _doc_boundaries("data")
    assert len(boundaries) == 50

    all_y_true = []
    all_y_prob = []

    t0 = time.time()
    for fold, (start, end) in enumerate(boundaries):
        test_idx = np.arange(start, end)
        train_idx = np.setdiff1d(np.arange(len(X)), test_idx)

        clf = _fit_multi_clf(X[train_idx], y[train_idx], C)
        y_prob = _proba_pos(clf, X[test_idx])

        all_y_true.append(y[test_idx])
        all_y_prob.append(y_prob)

        if (fold + 1) % 10 == 0:
            print(f"  fold {fold+1}/50 ({time.time()-t0:.0f}s)", flush=True)

    y_true = np.vstack(all_y_true)
    y_prob = np.vstack(all_y_prob)
    return _compute_detection_metrics(y_true, y_prob)


def _fit_one_fold_cat(X, y, train_idx, test_idx, c, C):
    """Train one binary LR for category c on unfair-only training data."""
    ufa_train = y[train_idx].sum(axis=1) > 0
    y_tr_c = y[train_idx][ufa_train, c]
    if y_tr_c.sum() == 0:
        return None, None
    clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced", random_state=42)
    clf.fit(X[train_idx][ufa_train], y_tr_c)
    return y[test_idx, c], clf.predict_proba(X[test_idx])[:, 1]


def _run_loo_cat_folds(X, y, boundaries, C):
    """Run 50-fold LOO for categorization, return raw per-category predictions."""
    all_yt = {c: [] for c in range(len(CATEGORIES))}
    all_yp = {c: [] for c in range(len(CATEGORIES))}

    t0 = time.time()
    for fold, (start, end) in enumerate(boundaries):
        test_idx = np.arange(start, end)
        train_idx = np.setdiff1d(np.arange(len(X)), test_idx)

        for c in range(len(CATEGORIES)):
            yt, yp = _fit_one_fold_cat(X, y, train_idx, test_idx, c, C)
            if yt is not None:
                all_yt[c].append(yt)
                all_yp[c].append(yp)

        if (fold + 1) % 10 == 0:
            print(f"  fold {fold+1}/50 ({time.time()-t0:.0f}s)", flush=True)

    return all_yt, all_yp


def _eval_one_category(yt, yp):
    """Compute F1 and AP for one category on the given subset."""
    if len(yt) == 0:
        return 0.0, 0.0
    best_t, best_mcc = 0.5, -1.0
    for t in np.arange(0.1, 0.95, 0.05):
        pred = (yp >= t).astype(int)
        m = matthews_corrcoef(yt, pred)
        if m > best_mcc:
            best_mcc, best_t = m, t
    pred = (yp >= best_t).astype(int)
    return f1_score(yt, pred, zero_division=0), average_precision_score(yt, yp)


def _make_unfair_mask(yt_cat):
    """Build boolean mask for sentences where ANY category is positive."""
    mask = np.zeros(len(yt_cat[0]), dtype=bool)
    for c in range(len(CATEGORIES)):
        mask = mask | (yt_cat[c] == 1)
    return mask


def _eval_cat_on_unfair(all_yt, all_yp):
    """Evaluate categorization on the unfair-only subset (Lippi protocol)."""
    yt_cat = {c: np.concatenate(all_yt[c]) if all_yt[c] else np.array([]) for c in range(len(CATEGORIES))}
    yp_cat = {c: np.concatenate(all_yp[c]) if all_yp[c] else np.array([]) for c in range(len(CATEGORIES))}
    unfair_mask = _make_unfair_mask(yt_cat)

    per_cat, f1s, aps = {}, [], []
    for c in range(len(CATEGORIES)):
        f1, ap = _eval_one_category(yt_cat[c][unfair_mask], yp_cat[c][unfair_mask])
        per_cat[CATEGORIES[c]] = {"F1": f1, "AP": ap}
        f1s.append(f1)
        aps.append(ap)

    return per_cat, float(np.mean(f1s)), float(np.mean(aps))


def evaluate_loo_categorization(model_key: str, C: float = 50.0):
    """LOO 8 independent LRs trained on unfair-only sentences (Lippi setup).

    Returns per-category and macro F1/AP.
    """
    spec = get_model_spec(model_key)
    idir = index_dir(spec)

    index = faiss.read_index(os.path.join(idir, "faiss.index"))
    with open(os.path.join(idir, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    X = index.reconstruct_n(0, index.ntotal).astype(np.float64)
    y = np.array(metadata["multi_labels"], dtype=np.uint8)
    boundaries = _doc_boundaries("data")

    all_yt, all_yp = _run_loo_cat_folds(X, y, boundaries, C)
    per_cat, macro_f1, macro_ap = _eval_cat_on_unfair(all_yt, all_yp)

    return {"per_category": per_cat, "macro_F1": macro_f1, "macro_AP": macro_ap}


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Leave-one-document-out evaluation")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--categorization", action="store_true",
                        help="Also run 8-LR categorization (slower)")
    parser.add_argument("--C", type=float, default=50.0,
                        help="Regularisation strength (default: 50)")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"C = {args.C}")
    print()

    print("=== Detection (multi-output LR, LOO) ===")
    det = evaluate_loo_detection(args.model, args.C)
    d = det["detection"]
    print(f"  Binary detection:  P={d['P']:.3f}  R={d['R']:.3f}  F1={d['F1']:.3f}")
    pc = det["per_category"]
    print(f"  Per-category macro: P={pc['macro_P']:.3f}  R={pc['macro_R']:.3f}  "
          f"F1={pc['macro_F1']:.3f}  MCC={pc['macro_MCC']:.3f}  AP={pc['macro_AP']:.3f}")

    if args.categorization:
        print()
        print("=== Categorization (8 independent LRs, unfair-only, LOO) ===")
        cat = evaluate_loo_categorization(args.model, args.C)
        print(f"  Macro F1: {cat['macro_F1']:.3f}  Macro AP: {cat['macro_AP']:.3f}")
        print()
        lippi_f1 = {"A": 0.823, "CH": 0.823, "CR": 0.745, "J": 0.970,
                    "LAW": 0.932, "LTD": 0.932, "TER": 0.853, "USE": 0.953}
        print(f"  {'Category':<30} {'Our F1':>8} {'Lippi F1':>8} {'Our AP':>8}")
        print(f"  {'-'*54}")
        for cat_key in CATEGORIES:
            name = CATEGORY_NAMES[cat_key]
            our_f1 = cat["per_category"][cat_key]["F1"]
            our_ap = cat["per_category"][cat_key]["AP"]
            lippi = lippi_f1.get(cat_key, 0)
            print(f"  {cat_key} ({name}):{' '*(28-len(cat_key)-len(name)-3)}"
                  f"{our_f1:>8.3f} {lippi:>8.3f} {our_ap:>8.3f}")


if __name__ == "__main__":
    main()

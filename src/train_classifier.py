"""Train an embedding-agnostic Logistic Regression classifier on frozen embeddings.

Usage:
    python -m src.train_classifier
    python -m src.train_classifier --model all-minilm
"""

from __future__ import annotations

import argparse
import os
import pickle

import faiss
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, matthews_corrcoef
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputClassifier

from src.config import DEFAULT_MODEL, get_model_spec, index_dir, model_dir
from src.data_loader import CATEGORIES, CATEGORY_NAMES, load_dataset
from src.metrics import find_thresholds, macro_ap, macro_mcc, threshold_predict

C_VALUES = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]


def _fit_classifier(X_train, y_train, c: float):
    return MultiOutputClassifier(
        LogisticRegression(C=c, max_iter=5000, class_weight="balanced", random_state=42)
    ).fit(X_train, y_train)


def _proba_pos(clf, X) -> np.ndarray:
    return np.column_stack([p[:, 1] for p in clf.predict_proba(X)])


def _tune_c(X_train, y_train, c_values: list[float]) -> float:
    best_c, best_mcc = c_values[0], -1.0
    for c in c_values:
        clf = MultiOutputClassifier(
            LogisticRegression(C=c, max_iter=5000, class_weight="balanced", random_state=42)
        )
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring=_cv_mcc_scorer)
        mcc = float(np.mean(scores))
        print(f"  C={c:>6.1f}  MCC={mcc:.4f}", flush=True)
        if mcc > best_mcc:
            best_mcc, best_c = mcc, c
    return best_c


def _cv_mcc_scorer(estimator, X, y):
    return macro_mcc(y, estimator.predict(X))


def _load_data(spec, data_dir):
    """Load FAISS index, labels, and train/val split."""
    idir = index_dir(spec)
    index = faiss.read_index(os.path.join(idir, "faiss.index"))
    with open(os.path.join(idir, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    X = index.reconstruct_n(0, index.ntotal).astype(np.float64)
    y = np.array(metadata["multi_labels"], dtype=np.uint8)
    dataset = load_dataset(data_dir)
    return X, y, dataset


def train(model_key: str = DEFAULT_MODEL, data_dir: str = "data", c: float | None = None) -> None:
    spec = get_model_spec(model_key)
    mdir = model_dir(spec)
    os.makedirs(mdir, exist_ok=True)
    print(f"Model: {spec.key} ({spec.name})")

    X, y, dataset = _load_data(spec, data_dir)
    X_train, y_train = X[dataset["train_idx"]], y[dataset["train_idx"]]
    X_val, y_val = X[dataset["val_idx"]], y[dataset["val_idx"]]
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}", flush=True)
    print(f"  Pos counts (train): {y_train.sum(axis=0).tolist()}", flush=True)

    print("\nTuning C via 5-fold CV...", flush=True)
    best_c = _tune_c(X_train, y_train, C_VALUES) if c is None else c
    print(f"\nC={best_c}" + (" (from CV)" if c is None else " (fixed)"), flush=True)

    clf = _fit_classifier(X_train, y_train, best_c)

    y_prob_pos = _proba_pos(clf, X_val)
    thresholds = find_thresholds(y_val, y_prob_pos)
    y_pred = threshold_predict(y_prob_pos, thresholds)

    print(f"\nValidation macro MCC: {macro_mcc(y_val, y_pred):.4f}", flush=True)
    print(f"Validation macro AP:  {macro_ap(y_val, y_prob_pos):.4f}", flush=True)
    print("Thresholds:", flush=True)
    for cat, t in zip(CATEGORIES, thresholds):
        print(f"  {cat} ({CATEGORY_NAMES[cat]}): {t}", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print("Per-category report", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(
        classification_report(
            y_val, y_pred,
            target_names=[f"{c} ({CATEGORY_NAMES[c]})" for c in CATEGORIES],
            zero_division=0,
        ),
        flush=True,
    )
    print("Per-category MCC:", flush=True)
    for i, cat in enumerate(CATEGORIES):
        print(f"  {cat} ({CATEGORY_NAMES[cat]}): {matthews_corrcoef(y_val[:, i], y_pred[:, i]):.4f}", flush=True)

    save_path = os.path.join(mdir, "classifier.pkl")
    _save_classifier(save_path, clf, thresholds)
    print(f"\nSaved to {save_path}", flush=True)


def _save_classifier(path, clf, thresholds):
    """Persist classifier and metadata to disk."""
    with open(path, "wb") as f:
        pickle.dump(
            {
                "estimator": clf,
                "thresholds": thresholds,
                "categories": CATEGORIES,
                "category_names": CATEGORY_NAMES,
            },
            f,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Logistic Regression on frozen embeddings")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--c", type=float, default=None, help="Skip CV and use this C value")
    args = parser.parse_args()
    train(args.model, args.data_dir, args.c)
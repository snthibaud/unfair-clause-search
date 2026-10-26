"""Diagnose per-category performance differences.

Compares MCC-tuned vs F1-tuned thresholds, and prints confusion matrices
for each category to help understand where and why the classifier fails.

Usage:
    python -m src.diagnose
    python -m src.diagnose --model legal-bert-embedding
"""

import argparse
import os
import pickle

import faiss
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from src.config import DEFAULT_MODEL, get_model_spec, index_dir
from src.data_loader import CATEGORIES, CATEGORY_NAMES, load_dataset
from src.metrics import find_thresholds


def load_predictions(model_key=DEFAULT_MODEL):
    spec = get_model_spec(model_key)
    mdir = model_dir(spec)
    idir = index_dir(spec)

    index = faiss.read_index(os.path.join(idir, "faiss.index"))
    with open(os.path.join(idir, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    X = index.reconstruct_n(0, index.ntotal).astype(np.float64)
    y = np.array(metadata["multi_labels"], dtype=np.uint8)
    ds = load_dataset("data")
    X_val, y_val = X[ds["val_idx"]], y[ds["val_idx"]]

    with open(os.path.join(mdir, "classifier.pkl"), "rb") as f:
        ckpt = pickle.load(f)
    clf = ckpt["estimator"]

    y_prob = clf.predict_proba(X_val)
    y_prob_pos = np.column_stack([p[:, 1] for p in y_prob])
    return y_val, y_prob_pos


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose per-category classifier performance"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    y_val, y_prob = load_predictions(args.model)

    print("=== Threshold comparison: MCC vs F1 ===\n")
    print(f"{'Category':<30} {'MCC-thr':>8} {'F1-thr':>8} {'Δ':>6} "
          f"{'MCC-F1':>8} {'F1-F1':>8} {'Δ F1':>8} "
          f"{'MCC-MCC':>8} {'F1-MCC':>8}")
    print("-" * 100)

    for i, cat in enumerate(CATEGORIES):
        mcc_thresholds = find_thresholds(y_val, y_prob, "mcc")
        f1_thresholds = find_thresholds(y_val, y_prob, "f1")

        mcc_pred = (y_prob[:, i] >= mcc_thresholds[i]).astype(int)
        f1_pred = (y_prob[:, i] >= f1_thresholds[i]).astype(int)

        mcc_f1 = f1_score(y_val[:, i], mcc_pred, zero_division=0)
        f1_f1 = f1_score(y_val[:, i], f1_pred, zero_division=0)
        mcc_mcc = matthews_corrcoef(y_val[:, i], mcc_pred)
        f1_mcc = matthews_corrcoef(y_val[:, i], f1_pred)

        name = f"{cat} ({CATEGORY_NAMES[cat]})"
        print(
            f"{name:<30} {mcc_thresholds[i]:>8.2f} {f1_thresholds[i]:>8.2f} "
            f"{f1_thresholds[i] - mcc_thresholds[i]:>+6.2f} "
            f"{mcc_f1:>8.3f} {f1_f1:>8.3f} {f1_f1 - mcc_f1:>+8.3f} "
            f"{mcc_mcc:>8.3f} {f1_mcc:>8.3f}"
        )

    print("\n\n=== Confusion matrices (MCC thresholds) ===\n")
    mcc_thresholds = find_thresholds(y_val, y_prob, "mcc")
    for i, cat in enumerate(CATEGORIES):
        pred = (y_prob[:, i] >= mcc_thresholds[i]).astype(int)
        cm = confusion_matrix(y_val[:, i], pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        p = precision_score(y_val[:, i], pred, zero_division=0)
        r = recall_score(y_val[:, i], pred, zero_division=0)
        f1 = f1_score(y_val[:, i], pred, zero_division=0)
        mcc = matthews_corrcoef(y_val[:, i], pred)
        n_pos = y_val[:, i].sum()
        print(f"{cat} ({CATEGORY_NAMES[cat]})  pos={n_pos}")
        print(f"  TN={tn:>5}  FP={fp:>4}  FN={fn:>3}  TP={tp:>3}  "
              f"P={p:.2f}  R={r:.2f}  F1={f1:.2f}  MCC={mcc:.2f}")
        print()


if __name__ == "__main__":
    main()

"""Bootstrap confidence intervals for macro AP across all three models.

Usage:
    python -m src.bootstrap_ap
    python -m src.bootstrap_ap --n-bootstrap 5000
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

import faiss
import numpy as np
from src.config import get_model_spec, index_dir, model_dir
from src.data_loader import load_dataset

MODEL_KEYS = ["legal-bert", "legal-bert-embedding", "all-minilm"]


def load_model(model_key: str):
    spec = get_model_spec(model_key)
    mdir = model_dir(spec)
    idir = index_dir(spec)

    idx = faiss.read_index(os.path.join(idir, "faiss.index"))
    with open(os.path.join(idir, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    X = idx.reconstruct_n(0, idx.ntotal).astype(np.float64)
    y = np.array(metadata["multi_labels"], dtype=np.uint8)

    with open(os.path.join(mdir, "classifier.pkl"), "rb") as f:
        ckpt = pickle.load(f)

    ds = load_dataset("data")
    X_val, y_val = X[ds["val_idx"]], y[ds["val_idx"]]
    y_prob = ckpt["estimator"].predict_proba(X_val)
    y_prob_pos = np.column_stack([p[:, 1] for p in y_prob])

    return y_val, y_prob_pos


def macro_ap_fast(y_true, y_prob):
    """Macro AP using inline computation (no sklearn overhead)."""
    active = [c for c in range(y_true.shape[1]) if y_true[:, c].sum() > 0]
    if not active:
        return 0.0
    return float(np.mean([_ap_fast(y_true[:, c], y_prob[:, c]) for c in active]))


def _ap_fast(y_true, y_prob):
    """Average precision without sklearn overhead."""
    order = np.argsort(-y_prob, kind="mergesort")
    yt = y_true[order]
    n_pos = yt.sum()
    if n_pos == 0:
        return 0.0
    ranks = np.where(yt == 1)[0] + 1.0
    cum = np.cumsum(yt)
    return float(np.mean(cum[ranks.astype(int) - 1] / ranks))


def _bootstrap_ap_vectorised(y_true, y_prob, all_idx):
    """Bootstrap AP using inline AP computation (no sklearn per-iteration)."""
    B, n = all_idx.shape
    active_cats = [c for c in range(y_true.shape[1]) if y_true[:, c].sum() > 0]
    aps = np.zeros(B)

    for c in active_cats:
        yt_c = y_true[:, c]
        yp_c = y_prob[:, c]
        for i in range(B):
            ii = all_idx[i]
            aps[i] += _ap_fast(yt_c[ii], yp_c[ii])

    return aps / len(active_cats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    B = args.n_bootstrap
    rng = np.random.RandomState(args.seed)

    # ── load all models once ──────────────────────────────────────
    print("Loading models...", end=" ", flush=True)
    data: dict[str, dict] = {}
    for key in MODEL_KEYS:
        y_val, y_prob = load_model(key)
        data[key] = {"y_val": y_val, "y_prob": y_prob}
    n_val = len(data[MODEL_KEYS[0]]["y_val"])
    print("done.", flush=True)

    # ── pre-generate all bootstrap indices ────────────────────────
    print(f"Generating {B} bootstrap index sets...", end=" ", flush=True)
    idx_val = rng.randint(0, n_val, size=(B, n_val))
    print("done.", flush=True)

    # ── bootstrap each model ──────────────────────────────────────
    results: dict[str, dict] = {}
    for key in MODEL_KEYS:
        print(f"  {key}...", end=" ", flush=True)
        d = data[key]

        ap_multi = _bootstrap_ap_vectorised(d["y_val"], d["y_prob"], idx_val)

        results[key] = {
            "multi_ap_mean": float(np.mean(ap_multi)),
            "multi_ap_std":  float(np.std(ap_multi)),
            "multi_ap_ci95": (float(np.percentile(ap_multi, 2.5)),
                              float(np.percentile(ap_multi, 97.5))),
        }
        print(f"AP {results[key]['multi_ap_mean']:.4f}", flush=True)

    # ── pairwise differences (paired bootstrap) ───────────────────
    print("\nPairwise Δ AP (paired bootstrap)...", flush=True)
    t0 = time.time()
    pairs = [
        ("legal-bert-embedding", "legal-bert"),
        ("legal-bert-embedding", "all-minilm"),
    ]
    for a, b in pairs:
        ya, ypa = data[a]["y_val"], data[a]["y_prob"]
        yb, ypb = data[b]["y_val"], data[b]["y_prob"]
        diffs = np.empty(B)
        for i in range(B):
            ii = idx_val[i]
            diffs[i] = macro_ap_fast(ya[ii], ypa[ii]) - macro_ap_fast(yb[ii], ypb[ii])
        mean_d = float(np.mean(diffs))
        std_d = float(np.std(diffs))
        lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
        sig = "YES" if lo > 0 else ("YES" if hi < 0 else "no")
        print(f"  {a} − {b}: Δ = {mean_d:+.4f} ± {std_d:.4f}  [{lo:+.4f}, {hi:+.4f}]  p<0.05: {sig}",
              flush=True)
    print(f"  (pairwise took {time.time()-t0:.1f}s)", flush=True)

    # ── Results ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Bootstrap AP ({B} iterations, 95% CI) — Multi-LR (full val, n={n_val})")
    print(f"{'='*70}")
    print(f"{'Model':<28} {'AP':>6} {'±':>5} {'CI low':>6} {'CI high':>6}")
    print("-" * 56)
    for key in MODEL_KEYS:
        r = results[key]
        print(f"{key:<28} {r['multi_ap_mean']:.4f} {r['multi_ap_std']:.4f}  "
              f"{r['multi_ap_ci95'][0]:.4f}  {r['multi_ap_ci95'][1]:.4f}")


if __name__ == "__main__":
    main()

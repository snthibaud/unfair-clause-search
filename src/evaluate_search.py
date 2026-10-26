"""
Evaluate semantic similarity search quality using label overlap as weak supervision.

Two sentences are "relevant" only when they share at least one positive
unfairness label.  Only queries with ≥1 positive label are evaluated, since
all-zero (fair) queries have no label signal to measure against.

This is a lower-bound proxy — sentences can be semantically similar without
sharing an unfairness category — but it provides a clean quantitative signal
without additional annotation.

Metrics:
    Precision@k  — fraction of top-k results sharing ≥1 label with the query
    MRR          — mean reciprocal rank of the first relevant result
    Random Prec@k — same metrics on randomly chosen candidates (baseline)
    Gain         — Precision@k / Random Precision@k

Usage:
    python -m src.evaluate_search
    python -m src.evaluate_search --model legal-bert-embedding
    python -m src.evaluate_search --top-k 10
    python -m src.evaluate_search --json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from functools import partial
from typing import Sequence

import faiss
import numpy as np

from src.config import DEFAULT_MODEL, get_model_spec, index_dir


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Aggregate evaluation results for semantic search quality."""

    n_queries: int
    top_k: int
    precision_at_k: float
    mrr: float
    random_precision_at_k: float
    precision_gain: float

    def to_dict(self) -> dict:
        return {
            "n_queries": self.n_queries,
            "top_k": self.top_k,
            "precision_at_k": round(self.precision_at_k, 4),
            "mrr": round(self.mrr, 4),
            "random_precision_at_k": round(self.random_precision_at_k, 4),
            "precision_gain": round(self.precision_gain, 2),
        }


# ---------------------------------------------------------------------------
# Label helpers (pure)
# ---------------------------------------------------------------------------


def _label_set(row: np.ndarray) -> frozenset[int]:
    return frozenset(np.where(row == 1)[0])


def _is_relevant(q_labs: frozenset[int], r_labs: frozenset[int]) -> bool:
    """Two sentences are relevant if they share at least one positive label.

    All-zero (fair) sentences are never relevant to anything under this
    definition, since they carry no positive label to overlap with.
    """
    return bool(q_labs & r_labs)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_artifacts(model_key: str) -> tuple[faiss.Index, list[str], np.ndarray]:
    """Load the FAISS index, sentence list, and label matrix."""
    idir = index_dir(get_model_spec(model_key))
    index = faiss.read_index(os.path.join(idir, "faiss.index"))
    with open(os.path.join(idir, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    labels = np.array(metadata["multi_labels"], dtype=np.uint8)
    return index, metadata["sentences"], labels


# ---------------------------------------------------------------------------
# Single-query evaluation
# ---------------------------------------------------------------------------


def _compute_mrr(query_labs, result_labs):
    """Mean reciprocal rank: 1/rank of first relevant result, or 0 if none."""
    for rank, rl in enumerate(result_labs, 1):
        if _is_relevant(query_labs, rl):
            return 1.0 / rank
    return 0.0


def _evaluate_one(
    query_id: int,
    vectors: np.ndarray,
    labels: np.ndarray,
    index: faiss.Index,
    top_k: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return (precision@k, mrr, random_precision@k) for one query."""
    q_vec = vectors[query_id : query_id + 1].copy()
    q_labs = _label_set(labels[query_id])

    scores, indices = index.search(q_vec, top_k + 1)
    mask = indices[0] != query_id
    result_ids = indices[0][mask][:top_k]
    result_labs = [_label_set(labels[rid]) for rid in result_ids]

    precision = np.mean([_is_relevant(q_labs, rl) for rl in result_labs]).item()
    mrr = _compute_mrr(q_labs, result_labs)
    rng = np.random.RandomState(seed + query_id)
    candidates = np.setdiff1d(np.arange(len(labels)), [query_id])
    rand_ids = rng.choice(candidates, top_k, replace=False)
    rand_labs = [_label_set(labels[rid]) for rid in rand_ids]
    rand_precision = np.mean([_is_relevant(q_labs, rl) for rl in rand_labs]).item()

    return precision, mrr, rand_precision


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def aggregate(triples: Sequence[tuple[float, float, float]], top_k: int) -> EvalResult:
    """Reduce per-query (precision, mrr, rand_precision) triples to an EvalResult."""
    precisions = [t[0] for t in triples]
    mrrs = [t[1] for t in triples]
    rand_precisions = [t[2] for t in triples]

    avg_precision = _mean(precisions)
    avg_rand = _mean(rand_precisions)

    return EvalResult(
        n_queries=len(triples),
        top_k=top_k,
        precision_at_k=avg_precision,
        mrr=_mean(mrrs),
        random_precision_at_k=avg_rand,
        precision_gain=avg_precision / max(avg_rand, 1e-9),
    )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def evaluate(
    model_key: str = DEFAULT_MODEL,
    top_k: int = 10,
    seed: int = 42,
) -> EvalResult:
    """Run the full evaluation and return aggregate results.

    Only queries with ≥1 positive label are evaluated; all-zero queries
    have no label signal to measure against and are skipped.
    """
    index, _sentences, labels = load_artifacts(model_key)
    n = index.ntotal

    labelled_mask = labels.sum(axis=1) > 0
    query_ids = np.where(labelled_mask)[0]

    vectors = index.reconstruct_n(0, n)

    eval_fn = partial(
        _evaluate_one,
        vectors=vectors,
        labels=labels,
        index=index,
        top_k=top_k,
        seed=seed,
    )

    triples = [eval_fn(int(qid)) for qid in query_ids]
    return aggregate(triples, top_k)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _tabulate(result: EvalResult) -> str:
    """Return a markdown table representation of an EvalResult."""
    d = result.to_dict()
    header = "| Queries | k | Prec@k | MRR | Rand-Prec | Gain |"
    sep = "|---------|---|--------|-----|-----------|------|"
    row = (
        f"| {d['n_queries']} | {d['top_k']} "
        f"| {d['precision_at_k']} | {d['mrr']} "
        f"| {d['random_precision_at_k']} | {d['precision_gain']}× |"
    )
    return f"{header}\n{sep}\n{row}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate FAISS search quality via label-overlap metrics"
    )
    parser.add_argument("--index-dir", default="models/index")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    result = evaluate(args.model, args.top_k, args.seed)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(_tabulate(result))
        print()
        print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
"""Shared FAISS index building utilities."""

import faiss
import numpy as np


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """L2-normalise embeddings and build an IndexFlatIP for cosine similarity."""
    faiss.normalize_L2(embeddings)
    idx = faiss.IndexFlatIP(embeddings.shape[1])
    idx.add(embeddings)
    return idx
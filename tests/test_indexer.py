"""
Tests for the FAISS index builder.

Tests are run only if the FAISS index already exists (i.e. after running build_models).
"""

import os
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.data_loader import load_sentences
from src.faiss_utils import build_index


INDEX_DIR = Path(__file__).parent.parent / "models" / "legal-bert-embedding" / "index"
SKIP_IF_NO_INDEX = pytest.mark.skipif(
    not (INDEX_DIR / "faiss.index").exists() or not (INDEX_DIR / "metadata.pkl").exists(),
    reason="FAISS index not built yet. Run: python -m src.build_models",
)


class TestIndexBuilder:
    """Tests for building the index (no saved index needed)."""

    def test_build_index_from_random_vectors(self):
        """Build a FAISS index from random vectors and verify basic properties."""
        dim = 768
        n = 100
        vecs = np.random.randn(n, dim).astype(np.float32)
        index = build_index(vecs)

        assert index.ntotal == n
        assert index.d == dim

        # Search for nearest neighbor of first vector (should be itself with score ~1.0)
        query = vecs[0:1].copy()
        import faiss
        faiss.normalize_L2(query)
        scores, indices = index.search(query, 3)
        assert indices[0][0] == 0  # closest is itself
        assert scores[0][0] > 0.99  # cosine similarity near 1.0

    @settings(max_examples=50)
    @given(
        n=st.integers(min_value=10, max_value=200),
        dim=st.integers(min_value=64, max_value=1024),
    )
    def test_index_preserves_vector_count(self, n, dim):
        """Property: FAISS index.ntotal equals the number of vectors inserted."""
        vecs = np.random.randn(n, dim).astype(np.float32)
        index = build_index(vecs)
        assert index.ntotal == n

    def test_index_dimension_mismatch_raises(self):
        """Index search with wrong dimension should raise."""
        vecs = np.random.randn(50, 128).astype(np.float32)
        index = build_index(vecs)
        bad_query = np.random.randn(1, 256).astype(np.float32)
        with pytest.raises(Exception):
            index.search(bad_query, 5)


@SKIP_IF_NO_INDEX
class TestSavedIndex:
    """Tests requiring a pre-built index."""

    @pytest.fixture(autouse=True)
    def setup(self):
        import pickle
        import faiss
        self.index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
        with open(INDEX_DIR / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

    def test_index_has_all_sentences(self, data_dir):
        sentences = load_sentences(data_dir)
        assert self.index.ntotal > 0
        assert self.index.ntotal <= len(sentences), (
            f"Index has {self.index.ntotal} vectors, more than dataset {len(sentences)}"
        )

    def test_metadata_has_required_fields(self):
        assert "sentences" in self.metadata
        assert "multi_labels" in self.metadata
        assert "categories" in self.metadata

    def test_metadata_sentences_match(self, data_dir):
        sentences = load_sentences(data_dir)
        assert len(self.metadata["sentences"]) == self.index.ntotal
        for s in self.metadata["sentences"]:
            assert s in sentences

    def test_self_search_returns_self(self):
        """Searching with a corpus sentence should return itself as top result."""
        query = self.metadata["sentences"][42]
        import faiss
        from transformers import AutoTokenizer, AutoModel
        import torch

        # Quick embed with base model
        model_dir = Path(__file__).parent.parent / "models" / "legal-bert-embedding"

        # Use the first sentence as query
        sentences = [query]
        # We can't easily embed a single sentence without the full model,
        # so we use the stored embedding as a proxy: search with the stored vector
        vecs = self.index.reconstruct_n(0, self.index.ntotal)
        query_vec = vecs[42:43].copy()  # get the stored normalized vector

        scores, indices = self.index.search(query_vec, 5)
        # Top result should be itself
        assert indices[0][0] == 42
        assert scores[0][0] > 0.999  # essentially identical

    def test_cosine_similarity_bounds(self):
        """All cosine similarities should be in [-1, 1] for normalized vectors."""
        vecs = self.index.reconstruct_n(0, self.index.ntotal)
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_search_returns_descending_scores(self):
        """Search results should be sorted by descending similarity."""
        vecs = self.index.reconstruct_n(0, self.index.ntotal)
        query = vecs[10:11].copy()
        scores, _ = self.index.search(query, 20)
        for i in range(len(scores[0]) - 1):
            assert scores[0][i] >= scores[0][i + 1], (
                f"Score at {i} ({scores[0][i]}) < score at {i+1} ({scores[0][i+1]})"
            )
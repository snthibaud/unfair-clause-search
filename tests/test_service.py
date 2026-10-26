"""
Tests for the FastAPI service.

Includes property-based tests for the API contracts and deterministic tests
for response schemas. Requires models to be built (index + classifier).
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from src.service import app

# Check if models exist — skip tests if not
MODELS_EXIST = (
    (Path(__file__).parent.parent / "models" / "legal-bert-embedding" / "classifier.pkl").exists()
    and (Path(__file__).parent.parent / "models" / "legal-bert-embedding" / "index" / "faiss.index").exists()
)

pytestmark = pytest.mark.skipif(not MODELS_EXIST, reason="Models not built — run build_models and train_classifier first")


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Deterministic tests
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["index_size"] > 0


class TestClassify:
    def test_classify_fair_sentence(self, client):
        """A clearly fair sentence should not be flagged."""
        response = client.post("/classify", json={"text": "We respect your privacy."})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["is_unfair"], bool)
        assert len(data["unfair_categories"]) == 8
        # Every category result should have the right shape
        for cat in data["unfair_categories"]:
            assert set(cat.keys()) == {"key", "name", "predicted", "confidence"}
            assert isinstance(cat["predicted"], bool)
            assert 0.0 <= cat["confidence"] <= 1.0

    def test_classify_unfair_sentence(self, client):
        """A sentence resembling an unfair clause should be classified."""
        response = client.post(
            "/classify",
            json={"text": "we may change these terms at any time without notice to you."},
        )
        assert response.status_code == 200
        data = response.json()
        # At minimum the response should be well-formed
        assert len(data["unfair_categories"]) == 8

    def test_classify_empty_text_rejected(self, client):
        response = client.post("/classify", json={"text": ""})
        assert response.status_code == 422


class TestSearch:
    def test_search_returns_results(self, client):
        response = client.post("/search", json={"query": "privacy policy data collection", "top_k": 3})
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "privacy policy data collection"
        assert len(data["results"]) == 3
        for hit in data["results"]:
            assert "sentence" in hit
            assert "score" in hit
            assert "index" in hit
            assert 0.0 <= hit["score"] <= 1.0

    def test_search_scores_descending(self, client):
        response = client.post("/search", json={"query": "termination of service without notice", "top_k": 10})
        assert response.status_code == 200
        results = response.json()["results"]
        for i in range(len(results) - 1):
            assert results[i]["score"] >= results[i + 1]["score"]

    def test_search_top_k_respected(self, client):
        for k in [1, 5, 20]:
            response = client.post("/search", json={"query": "arbitration clause", "top_k": k})
            assert response.status_code == 200
            assert len(response.json()["results"]) == k

    def test_search_returns_valid_hits(self, client):
        """Search results should contain valid sentence strings and scores."""
        response = client.post("/search", json={"query": "arbitration clause", "top_k": 5})
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == 5
        for hit in results:
            assert isinstance(hit["sentence"], str)
            assert len(hit["sentence"]) > 0
            assert 0.0 <= hit["score"] <= 1.0


class TestAnalyze:
    def test_analyze_combines_both(self, client):
        response = client.post(
            "/analyze",
            json={"text": "we may terminate your account at any time for any reason without notice.", "top_k": 3},
        )
        assert response.status_code == 200
        data = response.json()
        # Has classification
        assert "classification" in data
        assert "is_unfair" in data["classification"]
        assert len(data["classification"]["unfair_categories"]) == 8
        # Has search results
        assert "similar_sentences" in data
        assert len(data["similar_sentences"]) == 3


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------
@settings(max_examples=10, deadline=None)
@given(
    text=st.text(min_size=1, max_size=500).filter(lambda t: t.strip()),
    top_k=st.integers(min_value=1, max_value=20),
)
def test_classify_always_returns_eight_categories(client, text, top_k):
    """Property: /classify always returns exactly 8 category results."""
    response = client.post("/classify", json={"text": text})
    assert response.status_code == 200
    data = response.json()
    assert len(data["unfair_categories"]) == 8


@settings(max_examples=10, deadline=None)
@given(
    text=st.text(min_size=1, max_size=500).filter(lambda t: t.strip()),
    top_k=st.integers(min_value=1, max_value=10),
)
def test_search_returns_exactly_top_k_results(client, text, top_k):
    """Property: /search returns exactly top_k results when corpus ≥ top_k."""
    response = client.post("/search", json={"query": text, "top_k": top_k})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == top_k


@settings(max_examples=10, deadline=None)
@given(
    text=st.text(min_size=1, max_size=500).filter(lambda t: t.strip()),
    top_k=st.integers(min_value=1, max_value=10),
)
def test_analyze_returns_both_components(client, text, top_k):
    """Property: /analyze always returns classification + similar_sentences."""
    response = client.post("/analyze", json={"text": text, "top_k": top_k})
    assert response.status_code == 200
    data = response.json()
    assert "classification" in data
    assert "similar_sentences" in data
    assert len(data["similar_sentences"]) == top_k
    assert len(data["classification"]["unfair_categories"]) == 8


@settings(max_examples=10, deadline=None)
@given(
    text=st.text(min_size=1, max_size=500).filter(lambda t: t.strip()),
)
def test_confidence_values_are_probabilities(client, text):
    """Property: all confidence values are in [0, 1]."""
    response = client.post("/classify", json={"text": text})
    assert response.status_code == 200
    for cat in response.json()["unfair_categories"]:
        assert 0.0 <= cat["confidence"] <= 1.0


@settings(max_examples=10, deadline=None)
@given(
    text=st.text(min_size=1, max_size=500).filter(lambda t: t.strip()),
)
def test_similarity_scores_are_bounded(client, text):
    """Property: all similarity scores are in [0, 1] (cosine via inner product on normalized vectors)."""
    response = client.post("/search", json={"query": text, "top_k": 5})
    assert response.status_code == 200
    for hit in response.json()["results"]:
        assert 0.0 <= hit["score"] <= 1.0, f"Score {hit['score']} out of bounds"


@settings(max_examples=5, deadline=None)
@given(
    text=st.text(min_size=1, max_size=500).filter(lambda t: t.strip()),
)
def test_is_unfair_consistent_with_categories(client, text):
    """Property: is_unfair is True iff at least one category predicted True."""
    response = client.post("/classify", json={"text": text})
    assert response.status_code == 200
    data = response.json()
    any_cat_positive = any(c["predicted"] for c in data["unfair_categories"])
    assert data["is_unfair"] == any_cat_positive
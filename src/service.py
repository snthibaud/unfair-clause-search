"""FastAPI service exposing semantic search and unfair-clause classification.

Endpoints:
    POST /search     – semantic similarity search
    POST /classify   – multi-label unfair clause detection
    POST /analyze    – combined: classify + find similar sentences
    GET  /health     – health check

Set TOS_MODEL env var to switch embedding models (default: legal-bert-embedding).
"""

import os
import pickle
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config import get_model_spec, index_dir, model_dir
from src.data_loader import CATEGORIES, CATEGORY_NAMES
from src.embed import embed_batch, load_encoder_local

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_SPEC = get_model_spec()
_MODEL_DIR = model_dir(_SPEC)
_INDEX_DIR = index_dir(_SPEC)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Immutable application state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppState:
    encoder: Any
    tokenizer: Any
    estimator: Any
    thresholds: tuple[float, ...]
    index: faiss.Index
    sentences: tuple[str, ...]

    @property
    def spec(self):
        return _SPEC


_state: AppState | None = None


def _load_state() -> AppState:
    classifier_path = os.path.join(_MODEL_DIR, "classifier.pkl")
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(
            f"Classifier not found at {classifier_path}. "
            "Train it first with: python -m src.train_classifier"
        )

    encoder, tokenizer = load_encoder_local(_MODEL_DIR, _DEVICE)

    with open(classifier_path, "rb") as f:
        ckpt = pickle.load(f)

    idx = faiss.read_index(os.path.join(_INDEX_DIR, "faiss.index"))
    with open(os.path.join(_INDEX_DIR, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    print(f"Loaded classifier ({_SPEC.key}), index — {idx.ntotal} vectors")

    return AppState(
        encoder=encoder,
        tokenizer=tokenizer,
        estimator=ckpt["estimator"],
        thresholds=tuple(ckpt["thresholds"]),
        index=idx,
        sentences=tuple(metadata["sentences"]),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _state
    _state = _load_state()
    yield


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)


class CategoryResult(BaseModel):
    key: str
    name: str
    predicted: bool
    confidence: float


class ClassifyResponse(BaseModel):
    text: str
    is_unfair: bool
    unfair_categories: list[CategoryResult]


class SearchHit(BaseModel):
    sentence: str
    score: float
    index: int


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]


class AnalyzeResponse(BaseModel):
    text: str
    classification: ClassifyResponse
    similar_sentences: list[SearchHit]


class HealthResponse(BaseModel):
    status: str
    model: str
    index_size: int


# ---------------------------------------------------------------------------
# Helpers (pure, operating on AppState)
# ---------------------------------------------------------------------------

def _embed(state: AppState, texts: list[str]) -> np.ndarray:
    return embed_batch(texts, state.encoder, state.tokenizer, state.spec.max_length, _DEVICE)


def _classify(state: AppState, embedding: np.ndarray) -> ClassifyResponse:
    prob_pos = np.column_stack(
        [p[:, 1] for p in state.estimator.predict_proba(embedding)]
    )[0]
    results = [
        CategoryResult(
            key=cat,
            name=CATEGORY_NAMES[cat],
            predicted=bool(prob_pos[i] >= state.thresholds[i]),
            confidence=round(float(prob_pos[i]), 4),
        )
        for i, cat in enumerate(CATEGORIES)
    ]
    return ClassifyResponse(
        text="",
        is_unfair=any(r.predicted for r in results),
        unfair_categories=results,
    )


def _search(state: AppState, embedding: np.ndarray, top_k: int) -> list[SearchHit]:
    faiss.normalize_L2(embedding)
    scores, indices = state.index.search(embedding, top_k)
    return [
        SearchHit(
            sentence=state.sentences[int(idx)],
            score=round(float(score), 4),
            index=int(idx),
        )
        for score, idx in zip(scores[0], indices[0])
        if 0 <= idx < len(state.sentences)
    ]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="ToS Fairness Analyzer", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model=_SPEC.key,
        index_size=_state.index.ntotal if _state else 0,
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    emb = _embed(_state, [req.text])
    result = _classify(_state, emb)
    result.text = req.text
    return result


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    emb = _embed(_state, [req.query])
    return SearchResponse(query=req.query, results=_search(_state, emb, req.top_k))


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    emb = _embed(_state, [req.text])
    classification = _classify(_state, emb)
    classification.text = req.text
    return AnalyzeResponse(
        text=req.text,
        classification=classification,
        similar_sentences=_search(_state, emb, req.top_k),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.service:app", host="0.0.0.0", port=8000, reload=True)
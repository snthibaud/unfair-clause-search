"""
Central configuration and model registry.

Defines available embedding models and their parameters.  To add a new
model, register it in MODELS and rebuild with `python -m src.build_models --model <key>`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelSpec:
    """Specification for an embedding model."""

    key: str
    name: str                          # HuggingFace model ID (SentenceTransformer compatible)
    dim: int                           # embedding dimension
    max_length: int                    # tokenizer max length
    batch_size: int = 64               # embedding batch size

    @property
    def dir_name(self) -> str:
        return self.key


MODELS: dict[str, ModelSpec] = {
    "legal-bert": ModelSpec(
        key="legal-bert",
        name="nlpaueb/legal-bert-base-uncased",
        dim=768,
        max_length=256,
    ),
    "legal-bert-embedding": ModelSpec(
        key="legal-bert-embedding",
        name="maticzav/legal-bert-embedding",
        dim=768,
        max_length=256,
    ),
    "all-minilm": ModelSpec(
        key="all-minilm",
        name="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
        max_length=256,
    ),
}

DEFAULT_MODEL = "legal-bert-embedding"


def get_model_spec(key: Optional[str] = None) -> ModelSpec:
    """Resolve a model key (or env var) to a ModelSpec."""
    key = key or os.environ.get("TOS_MODEL", DEFAULT_MODEL)
    if key not in MODELS:
        raise KeyError(f"Unknown model '{key}'. Available: {list(MODELS)}")
    return MODELS[key]


def model_dir(spec: ModelSpec) -> str:
    return os.path.join("models", spec.dir_name)


def index_dir(spec: ModelSpec) -> str:
    return os.path.join(model_dir(spec), "index")
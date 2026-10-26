# Semantic Similarity Search — Unfair Clause Detection

Semantic search and classification of potentially unfair clauses in online Terms of Service, on the CLAUDETTE corpus.
Full report: docs/report.pdf (6 pages, experiments, design decisions, trade-offs).

## Data

The corpus is not included. Download it from the CLAUDETTE project (Lippi et al., 2019) and unpack it so that `data/` holds `Sentences/`, `Labels/` and the per-category `Labels_*` directories:

    curl -LO http://claudette.eui.eu/ToS.zip
    unzip ToS.zip && mv ToS data

## Quick start

    docker compose up --build

API on port 8000, Streamlit UI on port 8501.

For a faster build at lower accuracy (~1.5 min, MCC 0.65 vs 0.76):

    docker compose build --build-arg MODEL=all-minilm && docker compose up

## Development

    uv sync --extra dev
    uv run python -m src.build_models             # default: legal-bert-embedding
    uv run python -m src.train_classifier
    uv run python -m src.service                  # API on :8000
    uv run python -m src.ui                       # Streamlit on :8501

## Key files

    docs/report.pdf          Technical report
    src/service.py           FastAPI application
    src/evaluate_loo.py      Leave-one-document-out evaluation
    src/bootstrap_ap.py      Bootstrap significance testing
    src/config.py            Model registry (3 encoders)

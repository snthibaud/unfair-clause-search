ARG MODEL=legal-bert-embedding

FROM python:3.11-slim

ARG MODEL

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libopenblas0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync

COPY src/ ./src/
COPY data/ ./data/

RUN uv run python -m src.build_models --model ${MODEL} \
    && if [ "${MODEL}" = "legal-bert" ]; then C=50; else C=100; fi \
    && uv run python -m src.train_classifier --model ${MODEL} --c ${C}

ENV TOS_MODEL=${MODEL}

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "src.service"]
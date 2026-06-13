# syntax=docker/dockerfile:1
# uv-managed image. The vector index is baked in at build time (no API key
# needed), so `docker run` serves a populated /healthz immediately; /ask only
# needs ANTHROPIC_API_KEY passed at runtime.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# git is required by the ingestion step (sparse clone of the LangGraph docs).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Install dependencies first for layer caching (source changes don't re-resolve).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy the source and install the project itself.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Build the Chroma index (downloads docs + embedding model; no API key).
RUN uv run python -m rag_agent.ingest

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "rag_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]

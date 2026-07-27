# ---- build stage -------------------------------------------------------
FROM python:3.12-slim AS build

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir --prefix=/install ".[embed-local]"

# ---- runtime stage ------------------------------------------------------
FROM python:3.12-slim

# run as non-root
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=build /install /usr/local
COPY scripts/ scripts/
COPY data/golden_set.jsonl data/golden_set.jsonl

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "evalgate_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]

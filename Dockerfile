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

# Bake the embedding model into the image.
#
# fastembed otherwise downloads BAAI/bge-small-en-v1.5 (~130MB) from HuggingFace
# the first time TextEmbedding is constructed -- i.e. on every cold start of
# every container. That makes boot depend on a third party being up, requires
# outbound internet from the task, and takes far longer than the healthcheck's
# start period. Baking it in makes startup offline and immediate, and pins the
# exact model bytes into the image alongside the pinned onnxruntime, which is
# what the eval baseline's stability actually depends on.
ENV EMBEDDING__CACHE_DIR=/opt/fastembed
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/opt/fastembed')" \
    && chown -R appuser:appuser /opt/fastembed

COPY scripts/ scripts/
COPY data/golden_set.jsonl data/golden_set.jsonl
# The corpus ships in the image so ingest can run from the container itself
# (`docker compose run --rm api python scripts/ingest.py`) rather than needing a
# checkout on the host. It is 618K of committed, versioned text -- the same
# artifact the eval baseline is measured against.
COPY data/corpus/ data/corpus/

# One ONNX thread: the deployment target is a 2-vCPU burstable instance shared
# with Postgres, where ONNX's default thread-per-core pool costs memory and
# contends with the database for CPU credits without speeding up single queries.
ENV OMP_NUM_THREADS=1

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "evalgate_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]

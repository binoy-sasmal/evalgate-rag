"""Integration tests against a real pgvector instance.

Skipped unless RUN_PG_TESTS=1 (set by the CI integration job, or locally
after `docker compose up -d db`).
"""

import os

import pytest

from evalgate_rag.chunking import Chunk
from evalgate_rag.embeddings import HashEmbedder

pytestmark = pytest.mark.integration

requires_pg = pytest.mark.skipif(os.environ.get("RUN_PG_TESTS") != "1", reason="RUN_PG_TESTS != 1")


@requires_pg
def test_pgvector_upsert_and_search():
    from evalgate_rag.config import get_settings
    from evalgate_rag.store import PgVectorStore

    embedder = HashEmbedder()
    store = PgVectorStore(get_settings().db.dsn, dimension=embedder.dimension)

    chunks = [
        Chunk(text="Fines of up to 35 million EUR apply.", doc_id="it-Article 99", seq=0),
        Chunk(text="Human oversight requires a stop button.", doc_id="it-Article 14", seq=0),
    ]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))

    query_vec = embedder.embed(["What fines apply for violations?"])[0]
    results = store.dense_search(query_vec, top_k=1)
    assert results[0].chunk.doc_id == "it-Article 99"

    # upsert is idempotent on (doc_id, seq)
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    all_it = [c for c in store.all_chunks() if c.doc_id.startswith("it-")]
    assert len(all_it) == 2

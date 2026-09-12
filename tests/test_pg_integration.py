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


@requires_pg
def test_store_recovers_from_server_side_disconnect():
    """A dropped server connection must cost one request, not the process.

    This is the regression test for the pool. The previous store opened a
    single connection in __init__ and held it for the lifetime of the app, so
    anything that closed it server-side -- a restart, a failover, an idle
    timeout on a NAT path -- left the process running and permanently unable to
    reach the database. pg_terminate_backend reproduces exactly that, without
    needing to bounce the container.
    """
    import psycopg

    from evalgate_rag.config import get_settings
    from evalgate_rag.store import PgVectorStore

    dsn = get_settings().db.dsn
    store = PgVectorStore(dsn, dimension=HashEmbedder().dimension)
    assert store.count() >= 0  # prime the pool so a connection is actually open

    # Terminate every backend on this database except the one doing the killing.
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(
            """
            SELECT pg_terminate_backend(pid) FROM pg_stat_activity
            WHERE datname = current_database() AND pid <> pg_backend_pid()
            """
        )

    # The pool checks connections on checkout, so this transparently replaces
    # the dead one instead of raising OperationalError forever.
    assert store.count() >= 0
    store.close()

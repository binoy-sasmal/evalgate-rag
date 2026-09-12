"""Vector + document store.

PgVectorStore  — PostgreSQL with the pgvector extension (production path)
InMemoryStore  — same interface, pure Python (unit tests, chunking benchmark)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .chunking import Chunk

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id        BIGSERIAL PRIMARY KEY,
    doc_id    TEXT NOT NULL,
    seq       INT  NOT NULL,
    text      TEXT NOT NULL,
    embedding vector({dim}) NOT NULL,
    UNIQUE (doc_id, seq)
);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class Store(Protocol):
    def upsert(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None: ...
    def dense_search(self, query_vec: np.ndarray, top_k: int) -> list[ScoredChunk]: ...
    def all_chunks(self) -> list[Chunk]: ...
    def count(self) -> int: ...


class PgVectorStore:
    """pgvector-backed store over a connection *pool*.

    A single long-lived connection (the previous design) is fine against a
    local container but breaks permanently against a managed/remote Postgres:
    a restart, failover, maintenance window or an idle NAT timeout kills the
    socket, and nothing ever reopens it -- the process keeps serving, and every
    query fails, until someone restarts it. The pool checks a connection on
    checkout (`check_connection`) and transparently replaces dead ones, so a
    database bounce costs one failed request instead of the whole task.

    `register_vector` is applied per-connection via the pool's configure hook;
    it resolves the `vector` type OID, so the extension must already exist --
    hence the one-shot schema connection below, which runs before the pool opens.
    """

    def __init__(
        self,
        dsn: str,
        dimension: int,
        *,
        min_size: int = 1,
        max_size: int = 4,
        timeout_s: float = 10.0,
    ) -> None:
        import psycopg
        from pgvector.psycopg import register_vector
        from psycopg_pool import ConnectionPool

        # Schema first, on a throwaway connection: CREATE EXTENSION has to have
        # run before register_vector can look the type up on pooled connections.
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(SCHEMA_SQL.format(dim=dimension))

        def _configure(conn: Any) -> None:
            conn.autocommit = True
            register_vector(conn)

        self._pool = ConnectionPool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            configure=_configure,
            check=ConnectionPool.check_connection,
            timeout=timeout_s,
            open=True,
        )

    def upsert(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        with self._pool.connection() as conn, conn.cursor() as cur:
            for chunk, vec in zip(chunks, embeddings, strict=True):
                cur.execute(
                    """
                    INSERT INTO chunks (doc_id, seq, text, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (doc_id, seq)
                    DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding
                    """,
                    (chunk.doc_id, chunk.seq, chunk.text, vec),
                )

    def dense_search(self, query_vec: np.ndarray, top_k: int) -> list[ScoredChunk]:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT doc_id, seq, text, 1 - (embedding <=> %s) AS score
                FROM chunks ORDER BY embedding <=> %s LIMIT %s
                """,
                (query_vec, query_vec, top_k),
            )
            rows = cur.fetchall()
        return [ScoredChunk(Chunk(text=r[2], doc_id=r[0], seq=r[1]), float(r[3])) for r in rows]

    def all_chunks(self) -> list[Chunk]:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT doc_id, seq, text FROM chunks ORDER BY doc_id, seq")
            return [Chunk(text=r[2], doc_id=r[0], seq=r[1]) for r in cur.fetchall()]

    def count(self) -> int:
        """Number of ingested chunks. Used by the readiness probe to tell a
        started-but-unfed deployment (ingest never ran) apart from a healthy
        one -- an empty store retrieves nothing and answers every question with
        "I cannot answer this from the provided context", which otherwise looks
        like a model problem rather than a deployment problem."""
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks")
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._pool.close()


class InMemoryStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vecs: np.ndarray | None = None

    def upsert(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        self._chunks.extend(chunks)
        self._vecs = embeddings if self._vecs is None else np.vstack([self._vecs, embeddings])

    def dense_search(self, query_vec: np.ndarray, top_k: int) -> list[ScoredChunk]:
        if self._vecs is None or not len(self._chunks):
            return []
        vecs = self._vecs / (np.linalg.norm(self._vecs, axis=1, keepdims=True) + 1e-9)
        q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        scores = vecs @ q
        order = np.argsort(-scores)[:top_k]
        return [ScoredChunk(self._chunks[i], float(scores[i])) for i in order]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def count(self) -> int:
        return len(self._chunks)

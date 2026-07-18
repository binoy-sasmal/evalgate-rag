"""Vector + document store.

PgVectorStore  — PostgreSQL with the pgvector extension (production path)
InMemoryStore  — same interface, pure Python (unit tests, chunking benchmark)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

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


class PgVectorStore:
    def __init__(self, dsn: str, dimension: int) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute(SCHEMA_SQL.format(dim=dimension))
        register_vector(self._conn)

    def upsert(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        with self._conn.cursor() as cur:
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
        with self._conn.cursor() as cur:
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
        with self._conn.cursor() as cur:
            cur.execute("SELECT doc_id, seq, text FROM chunks ORDER BY doc_id, seq")
            return [Chunk(text=r[2], doc_id=r[0], seq=r[1]) for r in cur.fetchall()]


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

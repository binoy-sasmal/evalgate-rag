"""Hybrid retrieval: BM25 (lexical) + pgvector (dense) fused with
Reciprocal Rank Fusion. Same design as a LangChain EnsembleRetriever,
implemented directly for transparency and testability.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from .chunking import Chunk
from .embeddings import Embedder
from .store import ScoredChunk, Store

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenise(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _key(c: Chunk) -> tuple[str, int]:
    return (c.doc_id, c.seq)


class HybridRetriever:
    def __init__(self, store: Store, embedder: Embedder, rrf_k: int = 60) -> None:
        self._store = store
        self._embedder = embedder
        self._rrf_k = rrf_k
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[Chunk] = []
        self._bm25_built = False

    def refresh_bm25(self) -> None:
        """(Re)build the in-process BM25 index from the store. Call after ingest."""
        self._bm25_chunks = self._store.all_chunks()
        # An empty store means no BM25 results, not a fake single-doc index --
        # BM25Okapi's score array must line up 1:1 with _bm25_chunks below.
        self._bm25 = (
            BM25Okapi([_tokenise(c.text) for c in self._bm25_chunks]) if self._bm25_chunks else None
        )
        self._bm25_built = True

    def _bm25_search(self, query: str, top_k: int) -> list[ScoredChunk]:
        if not self._bm25_built:
            self.refresh_bm25()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenise(query))
        ranked = sorted(zip(self._bm25_chunks, scores, strict=True), key=lambda t: -t[1])[:top_k]
        return [ScoredChunk(c, float(s)) for c, s in ranked if s > 0]

    def retrieve(self, query: str, top_k: int = 4) -> list[ScoredChunk]:
        """RRF over BM25 and dense rankings; each leg fetches 2*top_k candidates."""
        candidates = max(top_k * 2, top_k)
        dense = self._store.dense_search(self._embedder.embed([query])[0], top_k=candidates)
        sparse = self._bm25_search(query, top_k=candidates)

        fused: dict[tuple[str, int], float] = {}
        chunks: dict[tuple[str, int], Chunk] = {}
        for ranking in (dense, sparse):
            for rank, sc in enumerate(ranking):
                k = _key(sc.chunk)
                fused[k] = fused.get(k, 0.0) + 1.0 / (self._rrf_k + rank + 1)
                chunks[k] = sc.chunk
        best = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        return [ScoredChunk(chunks[k], score) for k, score in best]

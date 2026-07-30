"""Chunk the corpus and load it into pgvector.

Usage:
    python scripts/ingest.py --strategy recursive
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from evalgate_rag.chunking import Chunk, chunk_fixed, chunk_recursive, chunk_semantic
from evalgate_rag.config import get_settings
from evalgate_rag.embeddings import Embedder, make_embedder
from evalgate_rag.store import Store

CORPUS_DIR = pathlib.Path("data/corpus")


def _chunk_doc(text: str, doc_id: str, strategy: str, embedder: Embedder) -> list[Chunk]:
    if strategy == "fixed":
        return chunk_fixed(text, doc_id)
    if strategy == "recursive":
        return chunk_recursive(text, doc_id)
    return chunk_semantic(text, doc_id, embed_fn=embedder.embed)


def ingest_corpus(corpus_dir: pathlib.Path, strategy: str, embedder: Embedder, store: Store) -> int:
    """Chunk every doc under corpus_dir and upsert into store. Returns the
    number of chunks written. Pure w.r.t. its store/embedder args so it can be
    exercised offline with an InMemoryStore + HashEmbedder (see tests)."""
    total = 0
    for path in sorted(corpus_dir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        chunks = _chunk_doc(doc["text"], doc["doc_id"], strategy, embedder)
        if not chunks:
            continue
        vecs = embedder.embed([c.text for c in chunks])
        store.upsert(chunks, vecs)
        total += len(chunks)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy", choices=["fixed", "recursive", "semantic"], default="recursive"
    )
    args = parser.parse_args()

    from evalgate_rag.store import PgVectorStore

    cfg = get_settings()
    embedder = make_embedder(cfg.embedding)
    store = PgVectorStore(cfg.db.dsn, dimension=embedder.dimension)

    total = ingest_corpus(CORPUS_DIR, args.strategy, embedder, store)
    print(f"Ingested {total} chunks with strategy={args.strategy}")

    # An ingest that wrote nothing means the corpus is missing or unreadable
    # (e.g. an empty/blocked fetch). Fail loudly here rather than let an empty
    # store slip through to retrieval -- which degrades silently to no context
    # (see HybridRetriever) and collapses every eval metric at the very last
    # gate instead of at its true cause.
    if total == 0:
        print(
            f"ERROR: ingested 0 chunks from {CORPUS_DIR}/ -- corpus is missing or empty. "
            "Ensure data/corpus/*.json exists (it is committed to the repo; refresh with "
            "scripts/fetch_corpus.py).",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

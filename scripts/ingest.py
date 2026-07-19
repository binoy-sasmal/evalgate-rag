"""Chunk the corpus and load it into pgvector.

Usage:
    python scripts/ingest.py --strategy recursive
"""

from __future__ import annotations

import argparse
import json
import pathlib

from evalgate_rag.chunking import chunk_fixed, chunk_recursive, chunk_semantic
from evalgate_rag.config import get_settings
from evalgate_rag.embeddings import make_embedder
from evalgate_rag.store import PgVectorStore

CORPUS_DIR = pathlib.Path("data/corpus")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy", choices=["fixed", "recursive", "semantic"], default="recursive"
    )
    args = parser.parse_args()

    cfg = get_settings()
    embedder = make_embedder(cfg.embedding)
    store = PgVectorStore(cfg.db.dsn, dimension=embedder.dimension)

    total = 0
    for path in sorted(CORPUS_DIR.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if args.strategy == "fixed":
            chunks = chunk_fixed(doc["text"], doc["doc_id"])
        elif args.strategy == "recursive":
            chunks = chunk_recursive(doc["text"], doc["doc_id"])
        else:
            chunks = chunk_semantic(doc["text"], doc["doc_id"], embed_fn=embedder.embed)
        if not chunks:
            continue
        vecs = embedder.embed([c.text for c in chunks])
        store.upsert(chunks, vecs)
        total += len(chunks)
    print(f"Ingested {total} chunks with strategy={args.strategy}")


if __name__ == "__main__":
    main()

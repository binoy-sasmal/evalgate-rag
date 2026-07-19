"""Benchmark the three chunking strategies on retrieval quality.

For each strategy, ingest the corpus into an in-memory store, run every
golden-set question through the hybrid retriever, and report:

  hit@4  — fraction of questions whose gold article appears in the top-4
  MRR    — mean reciprocal rank of the gold article
  chunks — resulting index size

Usage:
    python scripts/benchmark_chunking.py            # uses configured embedder
    python scripts/benchmark_chunking.py --hash     # offline smoke run

Prints a Markdown table ready to paste into the README.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from evalgate_rag.chunking import chunk_fixed, chunk_recursive, chunk_semantic
from evalgate_rag.config import get_settings
from evalgate_rag.embeddings import HashEmbedder, make_embedder
from evalgate_rag.retrieval import HybridRetriever
from evalgate_rag.store import InMemoryStore

CORPUS_DIR = pathlib.Path("data/corpus")
GOLDEN_SET = pathlib.Path("data/golden_set.jsonl")
TOP_K = 4


def load_golden() -> list[dict]:
    lines = GOLDEN_SET.read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def build_index(strategy: str, embedder) -> HybridRetriever:
    store = InMemoryStore()
    n = 0
    for path in sorted(CORPUS_DIR.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if strategy == "fixed":
            chunks = chunk_fixed(doc["text"], doc["doc_id"])
        elif strategy == "recursive":
            chunks = chunk_recursive(doc["text"], doc["doc_id"])
        else:
            chunks = chunk_semantic(doc["text"], doc["doc_id"], embed_fn=embedder.embed)
        if chunks:
            store.upsert(chunks, embedder.embed([c.text for c in chunks]))
            n += len(chunks)
    retriever = HybridRetriever(store, embedder)
    retriever.refresh_bm25()
    retriever.n_chunks = n  # type: ignore[attr-defined]
    return retriever


def evaluate(retriever: HybridRetriever, golden: list[dict]) -> tuple[float, float]:
    hits, rr_sum = 0, 0.0
    for item in golden:
        results = retriever.retrieve(item["question"], top_k=TOP_K)
        doc_ids = [r.chunk.doc_id for r in results]
        gold = item["gold_doc_id"]
        if gold in doc_ids:
            hits += 1
            rr_sum += 1.0 / (doc_ids.index(gold) + 1)
    n = len(golden)
    return hits / n, rr_sum / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash", action="store_true", help="offline hash embedder")
    args = parser.parse_args()

    embedder = HashEmbedder() if args.hash else make_embedder(get_settings().embedding)
    golden = load_golden()

    print(f"| Strategy | hit@{TOP_K} | MRR | chunks |")
    print("|---|---|---|---|")
    for strategy in ("fixed", "recursive", "semantic"):
        retriever = build_index(strategy, embedder)
        hit, mrr = evaluate(retriever, golden)
        print(f"| {strategy} | {hit:.2f} | {mrr:.2f} | {retriever.n_chunks} |")  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()

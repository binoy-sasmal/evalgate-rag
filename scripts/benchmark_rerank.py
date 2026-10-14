"""Benchmark cross-encoder reranking against plain hybrid retrieval.

Same method as benchmark_chunking.py: build the recursive-chunk index in
memory, run every golden-set question through the retriever, and score where
the gold article lands. Costs CPU time only, no tokens.

  hit@1, hit@4 -- fraction of questions whose gold article is in the top 1 / 4
  MRR@4        -- mean reciprocal rank of the gold article within the top 4
  ms/query     -- mean retrieval time, reranking included

With 20 questions, one question moving is 0.05 on hit@k. Read differences
smaller than that as noise.

Usage:
    python scripts/benchmark_rerank.py
    python scripts/benchmark_rerank.py --models Xenova/ms-marco-MiniLM-L-6-v2 --candidates 10 20
    python scripts/benchmark_rerank.py --per-question
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

from evalgate_rag.chunking import chunk_recursive
from evalgate_rag.config import get_settings
from evalgate_rag.embeddings import make_embedder
from evalgate_rag.rerank import CrossEncoderReranker
from evalgate_rag.retrieval import HybridRetriever
from evalgate_rag.store import InMemoryStore

CORPUS_DIR = pathlib.Path("data/corpus")
GOLDEN_SET = pathlib.Path("data/golden_set.jsonl")
TOP_K = 4
DEFAULT_MODELS = ["Xenova/ms-marco-MiniLM-L-6-v2", "Xenova/ms-marco-MiniLM-L-12-v2"]


def build_store(embedder) -> InMemoryStore:
    store = InMemoryStore()
    for path in sorted(CORPUS_DIR.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        chunks = chunk_recursive(doc["text"], doc["doc_id"])
        if chunks:
            store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    return store


def gold_ranks(retriever: HybridRetriever, golden: list[dict]) -> tuple[list[int | None], float]:
    ranks: list[int | None] = []
    timings: list[float] = []
    for item in golden:
        started = time.perf_counter()
        results = retriever.retrieve(item["question"], top_k=TOP_K)
        timings.append(time.perf_counter() - started)
        doc_ids = [r.chunk.doc_id for r in results]
        gold = item["gold_doc_id"]
        ranks.append(doc_ids.index(gold) + 1 if gold in doc_ids else None)
    return ranks, statistics.fmean(timings) * 1000


def summarise(ranks: list[int | None]) -> tuple[float, float, float]:
    n = len(ranks)
    hit1 = sum(r == 1 for r in ranks) / n
    hit4 = sum(r is not None for r in ranks) / n
    mrr = sum(1.0 / r for r in ranks if r is not None) / n
    return hit1, hit4, mrr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--candidates", nargs="+", type=int, default=[20])
    parser.add_argument("--per-question", action="store_true")
    args = parser.parse_args()

    cfg = get_settings()
    embedder = make_embedder(cfg.embedding)
    golden = [
        json.loads(ln) for ln in GOLDEN_SET.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    store = build_store(embedder)

    arms: list[tuple[str, HybridRetriever]] = [
        ("hybrid (RRF only)", HybridRetriever(store, embedder, rrf_k=cfg.rrf_k))
    ]
    for model in args.models:
        reranker = CrossEncoderReranker(model, cache_dir=cfg.rerank.cache_dir)
        for n in args.candidates:
            arms.append(
                (
                    f"+ {model.split('/')[-1]}, pool {n}",
                    HybridRetriever(
                        store, embedder, rrf_k=cfg.rrf_k, reranker=reranker, rerank_candidates=n
                    ),
                )
            )

    per_arm: dict[str, list[int | None]] = {}
    print(f"| Retrieval | hit@1 | hit@{TOP_K} | MRR@{TOP_K} | ms/query |")
    print("|---|---|---|---|---|")
    for name, retriever in arms:
        retriever.refresh_bm25()
        retriever.retrieve(golden[0]["question"], top_k=TOP_K)  # warm-up: model load, caches
        ranks, ms = gold_ranks(retriever, golden)
        per_arm[name] = ranks
        hit1, hit4, mrr = summarise(ranks)
        print(f"| {name} | {hit1:.2f} | {hit4:.2f} | {mrr:.2f} | {ms:.0f} |")

    if args.per_question:
        print("\nGold-article rank per question (- = not in top 4)")
        names = list(per_arm)
        print("id  " + "  ".join(f"{i:>4}" for i in range(len(names))))
        for idx, item in enumerate(golden):
            cells = "  ".join(f"{per_arm[n][idx] or '-':>4}" for n in names)
            print(f"{item['id']:>2}  {cells}  {item['gold_doc_id']}")
        for i, n in enumerate(names):
            print(f"  {i}: {n}")


if __name__ == "__main__":
    main()

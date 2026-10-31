"""Choose the semantic-cache similarity threshold from labelled question pairs.

Each pair in data/cache_eval_pairs.jsonl is (cached question, incoming
question, should_hit). For every threshold in the sweep, a fresh cache stores
the cached question and looks up the incoming one. Two rates matter:

- hit rate        -- share of should_hit pairs served from the cache (saved calls)
- false-hit rate  -- share of should-not-hit pairs served from the cache (wrong answers)

A false hit returns a confident answer to a different question, so the
threshold is the lowest one with zero false hits on this set, and the report
shows how many of those zero are owed to the anchor and negation guards
rather than to the embedding alone (the "embedding only" columns).

Local and free: uses the deployed embedder (fastembed BGE-small, ONNX), no
LLM calls and no tokens. The pairs are authored, single-author and
UNREVIEWED until a human promotes them, so the chosen threshold is a
starting point for this corpus, not a validated guarantee.

Usage:
    python scripts/benchmark_semantic_cache.py            # fastembed (deployed embedder)
    python scripts/benchmark_semantic_cache.py --hash     # offline smoke run, not meaningful
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evalgate_rag.chunking import Chunk
from evalgate_rag.config import EmbeddingSettings
from evalgate_rag.embeddings import HashEmbedder, make_embedder
from evalgate_rag.semantic_cache import SemanticCache, anchors, is_negated
from evalgate_rag.store import ScoredChunk

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ROOT / "data" / "cache_eval_pairs.jsonl"
REPORT_JSON = ROOT / "eval" / "semantic_cache_report.json"
REPORT_MD = ROOT / "eval" / "semantic_cache_report.md"
THRESHOLDS = [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]
_CTX = [ScoredChunk(chunk=Chunk(text="x", doc_id="Article 1", seq=0), score=1.0)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hash", action="store_true", help="offline hash embedder (smoke only)")
    args = ap.parse_args()

    pairs = [json.loads(line) for line in PAIRS.read_text(encoding="utf-8").splitlines() if line]
    embedder = HashEmbedder() if args.hash else make_embedder(EmbeddingSettings())
    embedder_name = "hash" if args.hash else "fastembed BAAI/bge-small-en-v1.5"

    # Raw similarity per pair, once: guards are evaluated separately below.
    sims: dict[str, float] = {}
    for p in pairs:
        vecs = embedder.embed([p["cached"], p["incoming"]])
        sims[p["id"]] = float(vecs[0] @ vecs[1])

    rows = []
    for threshold in THRESHOLDS:
        guarded = Counter[str]()
        raw = Counter[str]()
        for p in pairs:
            cache = SemanticCache(embedder, threshold=threshold)
            cache.store(p["cached"], "ns", "answer", _CTX)
            hit = cache.lookup(p["incoming"], "ns") is not None
            label = "tp" if p["should_hit"] else "fp"
            if hit:
                guarded[label] += 1
            if sims[p["id"]] >= threshold:
                raw[label] += 1
        n_pos = sum(p["should_hit"] for p in pairs)
        n_neg = len(pairs) - n_pos
        rows.append(
            {
                "threshold": threshold,
                "hit_rate": guarded["tp"] / n_pos,
                "false_hit_rate": guarded["fp"] / n_neg,
                "false_hits": guarded["fp"],
                "embedding_only_hit_rate": raw["tp"] / n_pos,
                "embedding_only_false_hits": raw["fp"],
            }
        )

    safe = [r for r in rows if r["false_hits"] == 0]
    chosen = min(safe, key=lambda r: r["threshold"]) if safe else None
    blocked_by_guard = [
        {
            "id": p["id"],
            "kind": p["kind"],
            "similarity": round(sims[p["id"]], 4),
            "anchors_differ": anchors(p["cached"]) != anchors(p["incoming"]),
            "negation_differs": is_negated(p["cached"]) != is_negated(p["incoming"]),
        }
        for p in pairs
        if not p["should_hit"]
        and (
            anchors(p["cached"]) != anchors(p["incoming"])
            or is_negated(p["cached"]) != is_negated(p["incoming"])
        )
    ]
    report = {
        "embedder": embedder_name,
        "pairs": len(pairs),
        "should_hit": sum(p["should_hit"] for p in pairs),
        "should_not_hit": sum(not p["should_hit"] for p in pairs),
        "review_status": sorted({p["review_status"] for p in pairs}),
        "sweep": rows,
        "chosen_threshold": chosen["threshold"] if chosen else None,
        "rule": "lowest threshold with zero false hits on this set",
        "similarities": {p["id"]: round(sims[p["id"]], 4) for p in pairs},
        "blocked_by_guard": blocked_by_guard,
    }
    if args.hash:
        print(json.dumps(report["sweep"], indent=2))
        print("hash embedder: smoke run only, report not written")
        return
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    REPORT_MD.write_text(_markdown(report), encoding="utf-8")
    print(_markdown(report))


def _markdown(report: dict) -> str:
    lines = [
        "# Semantic cache threshold sweep",
        "",
        f"Embedder: {report['embedder']}. Pairs: {report['pairs']} "
        f"({report['should_hit']} should hit, {report['should_not_hit']} should not). "
        f"Review status: {', '.join(report['review_status'])}.",
        "",
        "| Threshold | Hit rate | False hits "
        "| Embedding-only hit rate | Embedding-only false hits |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in report["sweep"]:
        lines.append(
            f"| {r['threshold']:.2f} | {r['hit_rate']:.0%} | {r['false_hits']} "
            f"| {r['embedding_only_hit_rate']:.0%} | {r['embedding_only_false_hits']} |"
        )
    lines += [
        "",
        f"Chosen threshold ({report['rule']}): **{report['chosen_threshold']}**.",
        "",
        "Generated by `scripts/benchmark_semantic_cache.py`. The pairs are authored by one",
        "person and unreviewed; treat the threshold as a starting point for this corpus.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()

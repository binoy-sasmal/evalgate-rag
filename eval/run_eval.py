"""Run the Ragas evaluation suite over the golden set and write metrics JSON.

Metrics: faithfulness, answer_relevancy, context_precision.
Every question is answered by the live pipeline (so retrieval AND generation
are both under test), and each run is traced to Langfuse when enabled.

The Ragas judge LLM points at the same OpenAI-compatible endpoint as the
pipeline (e.g. Groq's free tier), but judge *embeddings* always run locally
via fastembed — Groq has no embeddings endpoint, and this keeps Ragas from
ever calling a remote embeddings API.

Resumable: generated answers and judge scores are cached to eval/.cache/ as
they're produced, keyed by golden-set id (scores additionally by metric). A
rerun skips anything already cached, so a run that dies partway through (a
real risk on Groq's free tier — see pipeline.py and the RunConfig below)
doesn't have to redo work that already succeeded. The cache is invalidated
automatically if the prompt, model, retrieval top_k, embedding provider, or
golden_set.jsonl content changes; pass --fresh to force a clean run outright
(needed after a corpus re-ingest or chunking-strategy change, since those
aren't visible to this script).

Usage:
    python eval/run_eval.py --out eval/results.json [--limit 10] [--fresh]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import statistics
import sys

from evalgate_rag.config import Settings, get_settings
from evalgate_rag.embeddings import FastEmbedEmbedder, make_embedder
from evalgate_rag.pipeline import PROMPT_TEMPLATE, SYSTEM_PROMPT, LLMClient, RAGPipeline, Tracer
from evalgate_rag.retrieval import HybridRetriever
from evalgate_rag.store import PgVectorStore

GOLDEN_SET = pathlib.Path("data/golden_set.jsonl")
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision")

CACHE_DIR = pathlib.Path("eval/.cache")
ANSWERS_CACHE = CACHE_DIR / "answers.json"
SCORES_CACHE = CACHE_DIR / "judge_scores.json"

# Ragas fires judge calls with this many concurrent workers; keep it at 1 so
# it doesn't trip Groq free-tier RPM/TPM caps (LLMClient's own retry/throttle
# handle the rest).
JUDGE_MAX_WORKERS = 1

# Ragas' own per-call timeout and retry budget for judge calls (separate from
# LLMClient's 429 handling, which only covers pipeline generation — Ragas
# talks to the judge LLM directly). Groq free-tier calls occasionally stall;
# a smoke-test run against Groq hit a bare TimeoutError under the 180s
# default, so give it more headroom, and log retries so stalls are visible
# instead of just going quiet for minutes.
JUDGE_TIMEOUT_S = 300

# Kept deliberately small. A prior full-eval run hit Groq's *daily* token
# quota (100K TPD) partway through judging — a hard cap that doesn't clear on
# retry — and each RateLimitError's suggested wait (13-33 min) got retried by
# BOTH the OpenAI SDK's own client-level retries and Ragas' tenacity wrapper
# on top, stacking into a run that was still retrying after 59+ hours.
# ChatOpenAI(max_retries=0) below removes the SDK-level layer entirely, so
# these two numbers are the only retry budget — worst case a few minutes per
# job, not hours, if the same wall gets hit again.
JUDGE_MAX_RETRIES = 3
JUDGE_MAX_WAIT_S = 30


def build_pipeline() -> RAGPipeline:
    cfg = get_settings()
    embedder = make_embedder(cfg.embedding)
    store = PgVectorStore(cfg.db.dsn, dimension=embedder.dimension)
    retriever = HybridRetriever(store, embedder, rrf_k=cfg.rrf_k)
    retriever.refresh_bm25()
    return RAGPipeline(retriever, LLMClient(cfg.llm), Tracer(cfg.langfuse), cfg.retrieval_top_k)


def compute_fingerprint(cfg: Settings, golden_raw: str) -> str:
    """Fingerprint of the run's observable configuration, so the cache is
    reused only across retries of the same run — not silently across a
    prompt, model, top_k, embedding-provider, or golden-set change. Cannot
    see a corpus re-ingest or chunking-strategy change (that state lives in
    Postgres) — use --fresh after either of those.
    """
    payload = json.dumps(
        {
            "system_prompt": SYSTEM_PROMPT,
            "prompt_template": PROMPT_TEMPLATE,
            "model": cfg.llm.model,
            "base_url": cfg.llm.base_url,
            "top_k": cfg.retrieval_top_k,
            "embedding_provider": cfg.embedding.provider,
            "golden_set": golden_raw,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_cache(path: pathlib.Path, fingerprint: str, fresh: bool) -> dict:
    if fresh or not path.exists():
        return {"_fingerprint": fingerprint}
    cache = json.loads(path.read_text())
    if cache.get("_fingerprint") != fingerprint:
        print(
            f"[eval] cache at {path} is stale (config changed) -- starting clean", file=sys.stderr
        )
        return {"_fingerprint": fingerprint}
    return cache


def _save_cache(path: pathlib.Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="eval/results.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--fresh", action="store_true", help="ignore eval/.cache/ and recompute everything"
    )
    args = parser.parse_args()

    from datasets import Dataset
    from langchain_core.embeddings import Embeddings
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness
    from ragas.run_config import RunConfig

    class RagasFastEmbedEmbeddings(Embeddings):
        """LangChain `Embeddings` adapter over our local FastEmbedEmbedder."""

        def __init__(self, embedder: FastEmbedEmbedder) -> None:
            self._embedder = embedder

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._embedder.embed(texts).tolist()

        def embed_query(self, text: str) -> list[float]:
            return self._embedder.embed([text])[0].tolist()

    golden_raw = GOLDEN_SET.read_text()
    golden = [json.loads(ln) for ln in golden_raw.splitlines() if ln.strip()]
    if args.limit:
        golden = golden[: args.limit]

    cfg = get_settings()
    fingerprint = compute_fingerprint(cfg, golden_raw)
    answers_cache = _load_cache(ANSWERS_CACHE, fingerprint, args.fresh)
    judge_cache = _load_cache(SCORES_CACHE, fingerprint, args.fresh)

    pipeline = build_pipeline()
    rows = []
    for item in golden:
        key = str(item["id"])
        cached = answers_cache.get(key)
        if cached is not None:
            print(f"cached   {item['id']}: {item['question'][:60]}...")
        else:
            result = pipeline.answer(item["question"])
            cached = {
                "response": result.answer,
                "retrieved_contexts": [c.chunk.text for c in result.contexts],
            }
            answers_cache[key] = cached
            _save_cache(ANSWERS_CACHE, answers_cache)
            print(f"answered {item['id']}: {item['question'][:60]}...")
        rows.append(
            {
                "id": key,
                "user_input": item["question"],
                "response": cached["response"],
                "retrieved_contexts": cached["retrieved_contexts"],
                "reference": item["ground_truth"],
            }
        )

    judge_llm = ChatOpenAI(
        model=cfg.llm.model,
        api_key=cfg.llm.api_key,
        base_url=cfg.llm.base_url,
        temperature=0,
        max_retries=0,  # let Ragas' RunConfig be the only retry layer, not this too
    )
    judge_emb = RagasFastEmbedEmbeddings(FastEmbedEmbedder())
    run_config = RunConfig(
        max_workers=JUDGE_MAX_WORKERS,
        timeout=JUDGE_TIMEOUT_S,
        max_retries=JUDGE_MAX_RETRIES,
        max_wait=JUDGE_MAX_WAIT_S,
        log_tenacity=True,
    )

    # One metric at a time, and only for rows still missing that metric's
    # score, so a batch that partially fails (e.g. answer_relevancy exhausts
    # its retries under TPM pressure) doesn't cost the metrics that already
    # succeeded — each is scored, cached, and persisted independently.
    metric_objs = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
    }
    for metric_name, metric_obj in metric_objs.items():
        pending = [r for r in rows if metric_name not in judge_cache.get(r["id"], {})]
        if not pending:
            print(f"[eval] {metric_name}: all {len(rows)} rows already cached")
            continue
        print(f"[eval] {metric_name}: scoring {len(pending)}/{len(rows)} rows")
        ds = Dataset.from_list([{k: v for k, v in r.items() if k != "id"} for r in pending])
        report = evaluate(
            ds, metrics=[metric_obj], llm=judge_llm, embeddings=judge_emb, run_config=run_config
        )
        scores = report.to_pandas()[metric_name]
        for row, score in zip(pending, scores, strict=True):
            if not math.isnan(score):
                judge_cache.setdefault(row["id"], {})[metric_name] = float(score)
        _save_cache(SCORES_CACHE, judge_cache)

    metrics: dict[str, float | int | None] = {}
    for metric_name in METRIC_NAMES:
        scored = [judge_cache[r["id"]] for r in rows if metric_name in judge_cache.get(r["id"], {})]
        values = [s[metric_name] for s in scored]
        if values:
            metrics[metric_name] = round(statistics.fmean(values), 4)
        else:
            metrics[metric_name] = None
            print(
                f"[eval] WARNING: no successful samples for {metric_name} -- null", file=sys.stderr
            )
    metrics["n_questions"] = len(rows)

    pathlib.Path(args.out).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

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
import asyncio
import hashlib
import json
import math
import pathlib
import statistics
import sys
import time

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

# A prior full-eval run hit Groq's *daily* token quota (100K TPD) partway
# through judging — a hard cap that doesn't clear on retry — and each
# RateLimitError's suggested wait (13-33 min) got retried by BOTH the OpenAI
# SDK's own client-level retries and Ragas' tenacity wrapper on top, stacking
# into a run that was still retrying after 59+ hours. ChatOpenAI(max_retries=0)
# below removes the SDK-level layer entirely, so these two numbers are the
# only retry budget for a single call.
#
# They're not the main defense against 429s, though: Ragas' tenacity retry
# uses its own exponential+jitter backoff and never reads Groq's suggested
# `retry-after` wait, and max_workers=1 only limits *concurrency*, not
# *rate* — Ragas fires the next job the instant the last one returns, so it
# hammers the endpoint back-to-back and pins TPM near the ceiling for the
# whole judging phase. A run against the 20-question golden set saw usage
# sit at 10-11.7K/12K TPM continuously, and most jobs 429'd on their first
# attempt with too few retries left to land in a freed-up window — silently
# dropping ~75-90% of (question, metric) pairs as NaN. JUDGE_MIN_INTERVAL_S
# (applied per physical LLM call via ThrottledChatOpenAI below) is the real
# fix; these two are a secondary safety net for the occasional blip.
JUDGE_MAX_RETRIES = 6
JUDGE_MAX_WAIT_S = 45

# Minimum seconds between judge LLM calls (see comment above) — chosen so
# that even the larger observed requests (~2-2.6K tokens) stay comfortably
# under Groq's 12K TPM cap: 12000 / 8s-per-call * 60s ≈ 1 request every 8s
# supports ~1.5K tokens/request sustained, close to but under the ceiling.
JUDGE_MIN_INTERVAL_S = 8.0


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
    from pydantic import PrivateAttr
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

    class ThrottledChatOpenAI(ChatOpenAI):
        """Enforces a minimum delay between calls at the actual LLM-call
        boundary, so Ragas can't hammer the judge endpoint back-to-back and
        pin TPM at the ceiling regardless of how many metrics/sub-calls fan
        out per row (see JUDGE_MIN_INTERVAL_S comment)."""

        _min_interval_s: float = PrivateAttr(default=0.0)
        _last_call_at: float | None = PrivateAttr(default=None)

        def __init__(self, *, min_interval_s: float = 0.0, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self._min_interval_s = min_interval_s

        async def _agenerate(self, *args: object, **kwargs: object):  # type: ignore[override]
            if self._min_interval_s > 0:
                if self._last_call_at is not None:
                    remaining = self._min_interval_s - (time.monotonic() - self._last_call_at)
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                self._last_call_at = time.monotonic()
            return await super()._agenerate(*args, **kwargs)

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

    judge_llm = ThrottledChatOpenAI(
        model=cfg.llm.model,
        api_key=cfg.llm.api_key,
        base_url=cfg.llm.base_url,
        temperature=0,
        max_retries=0,  # let Ragas' RunConfig be the only retry layer, not this too
        min_interval_s=JUDGE_MIN_INTERVAL_S,
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
        # Sample count alongside each metric: n_questions alone can't tell you
        # whether an average reflects all 20 questions or a handful of
        # survivors after most judge calls got dropped as NaN under TPM
        # pressure -- a gap that made a 3-sample fluke look like a real score.
        metrics[f"{metric_name}_n"] = len(values)
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

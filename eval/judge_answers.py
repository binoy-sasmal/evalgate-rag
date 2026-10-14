"""Grade the live pipeline's answers with the calibrated correctness judge.

Answers every golden-set question through the real pipeline (pgvector store,
hybrid retrieval, optional reranking, the configured LLM), then asks the judge
(evalgate_rag.judge) whether each answer states what the reference says.
Reports the pass rate with the failing questions listed.

This is reported, not gated: eval/check_regression.py and eval/baseline.json
are unchanged. A pass rate becomes a gate only after a human has reviewed the
calibration labels and promoted a baseline for it.

Answers and verdicts are cached per arm in eval/.cache/ and invalidated when
the prompt, model, top_k, reranking or judge configuration changes.

Usage:
    python eval/judge_answers.py --arm baseline
    RERANK__ENABLED=true python eval/judge_answers.py --arm rerank
    python eval/judge_answers.py --compare baseline rerank
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

from evalgate_rag.config import get_settings, judge_llm_settings
from evalgate_rag.embeddings import make_embedder
from evalgate_rag.judge import JUDGE_SYSTEM_PROMPT, AnswerJudge, JudgeParseError
from evalgate_rag.pipeline import SYSTEM_PROMPT, LLMClient, RAGPipeline, Tracer
from evalgate_rag.rerank import make_reranker
from evalgate_rag.retrieval import HybridRetriever
from evalgate_rag.store import PgVectorStore

GOLDEN_SET = pathlib.Path("data/golden_set.jsonl")
CACHE_DIR = pathlib.Path("eval/.cache")
RESULTS_DIR = pathlib.Path("eval/results")


def run(arm: str) -> None:
    cfg = get_settings()
    judge_cfg = judge_llm_settings(cfg)
    golden = [
        json.loads(ln) for ln in GOLDEN_SET.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    fingerprint = hashlib.sha256(
        json.dumps(
            [
                SYSTEM_PROMPT,
                cfg.llm.model,
                cfg.retrieval_top_k,
                cfg.rerank.model_dump(),
                JUDGE_SYSTEM_PROMPT,
                judge_cfg.model,
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    cache_path = CACHE_DIR / f"judged_answers_{arm}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    if cache.get("_fingerprint") != fingerprint:
        cache = {"_fingerprint": fingerprint}

    embedder = make_embedder(cfg.embedding)
    store = PgVectorStore(cfg.db.dsn, dimension=embedder.dimension)
    retriever = HybridRetriever(
        store,
        embedder,
        rrf_k=cfg.rrf_k,
        reranker=make_reranker(cfg.rerank),
        rerank_candidates=cfg.rerank.candidates,
    )
    retriever.refresh_bm25()
    pipeline = RAGPipeline(retriever, LLMClient(cfg.llm), Tracer(cfg.langfuse), cfg.retrieval_top_k)
    judge = AnswerJudge(LLMClient(judge_cfg))

    for item in golden:
        key = str(item["id"])
        entry = cache.setdefault(key, {})
        if "answer" not in entry:
            result = pipeline.answer(item["question"])
            entry["answer"] = result.answer
            entry["retrieved"] = [c.chunk.doc_id for c in result.contexts]
        if "passed" not in entry and "unparsed" not in entry:
            try:
                v = judge.judge(item["question"], item["ground_truth"], entry["answer"])
                entry.update(passed=v.passed, reason=v.reason)
            except JudgeParseError as exc:
                entry["unparsed"] = str(exc)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        verdict = "UNPARSED" if "unparsed" in entry else ("PASS" if entry["passed"] else "FAIL")
        print(f"{key:>2} {verdict:<8} {item['question'][:60]}", file=sys.stderr)
    store.close()

    judged = [str(g["id"]) for g in golden if "passed" in cache[str(g["id"])]]
    passed = [k for k in judged if cache[k]["passed"]]
    summary = {
        "arm": arm,
        "answer_model": cfg.llm.model,
        "judge_model": judge_cfg.model,
        "rerank": cfg.rerank.model if cfg.rerank.enabled else None,
        "judged": len(judged),
        "pass_rate": round(len(passed) / len(judged), 4) if judged else None,
        "failed": {
            k: {"gold": next(g["gold_doc_id"] for g in golden if str(g["id"]) == k)}
            | {"retrieved": cache[k]["retrieved"], "reason": cache[k]["reason"]}
            for k in judged
            if not cache[k]["passed"]
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"judged_{arm}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("arm", "judged", "pass_rate")}))


def compare(a: str, b: str) -> None:
    ra, rb = (json.loads((RESULTS_DIR / f"judged_{x}.json").read_text("utf-8")) for x in (a, b))
    print(f"{a}: {ra['pass_rate']}  {b}: {rb['pass_rate']}  (n={ra['judged']}/{rb['judged']})")
    print(f"failed only under {a}: {sorted(set(ra['failed']) - set(rb['failed']), key=int)}")
    print(f"failed only under {b}: {sorted(set(rb['failed']) - set(ra['failed']), key=int)}")
    print(f"failed under both: {sorted(set(ra['failed']) & set(rb['failed']), key=int)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--arm")
    group.add_argument("--compare", nargs=2, metavar=("ARM_A", "ARM_B"))
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args.arm)


if __name__ == "__main__":
    main()

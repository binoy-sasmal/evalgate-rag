# CLAUDE.md — agent instructions for this repository

Context file for Claude Code (and other coding agents) working in this repo.

## What this project is

An eval-gated hybrid RAG service over the EU AI Act. FastAPI + PostgreSQL/pgvector
+ Langfuse tracing + Ragas evaluation, with a CI gate that fails the build when
eval metrics regress. The golden set is the product; the code serves it.

## Commands

```bash
pip install -e .[dev]                          # install
pytest -q                                      # offline unit tests (< 1 s)
RUN_PG_TESTS=1 pytest -q -m integration        # needs: docker compose up -d db
ruff check . && ruff format --check .          # lint
mypy src/                                      # types
python scripts/benchmark_chunking.py --hash    # offline retrieval benchmark
python eval/run_eval.py && python eval/check_regression.py   # full eval gate (costs tokens)
```

## Architecture in one paragraph

`api.build_app()` wires config → embedder → store → retriever → pipeline.
Everything is injected, nothing is global: tests pass a `FakePipeline` to
`build_app()`, an `InMemoryStore` to `HybridRetriever`, and the `HashEmbedder`
anywhere embeddings are needed. Retrieval is BM25 + dense fused with RRF
(`retrieval.py`); generation goes through one OpenAI-compatible client
(`pipeline.LLMClient`) so the LLM is swappable via env vars alone.

## Rules for agents

1. **Never edit `eval/baseline.json` to make the gate pass.** The baseline
   only changes via `python eval/check_regression.py --promote` after a
   human-reviewed improvement.
2. **Never add unreviewed pairs to `data/golden_set.jsonl`.** Generated
   candidates go to `golden_set_candidates.jsonl` with
   `"review_status": "UNREVIEWED"` until a human promotes them.
3. **Offline by default.** Unit tests must not require network, API keys, a
   database, or model downloads. Use `HashEmbedder`, `InMemoryStore`, and fake
   LLM clients. Anything needing Postgres is `@pytest.mark.integration`.
4. **One LLM client.** Don't add provider-specific SDKs; the OpenAI-compatible
   client + env vars covers OpenAI, Azure (via gateway), Groq, Ollama, vLLM.
5. **Write the test first** when changing chunking, retrieval, or fusion
   behaviour — these are the components the eval gate exists to protect.
6. Keep `ruff check` and `mypy src/` clean; line length 100.

## Files agents most often need

| Path | What it is |
|---|---|
| `src/evalgate_rag/retrieval.py` | Hybrid BM25 + dense + RRF |
| `src/evalgate_rag/chunking.py` | fixed / recursive / semantic strategies |
| `src/evalgate_rag/pipeline.py` | prompt, LLM client, Langfuse tracing |
| `eval/run_eval.py` | Ragas suite over the golden set |
| `eval/check_regression.py` | the CI gate itself |
| `.github/workflows/eval-gate.yml` | when/how the gate runs in CI |

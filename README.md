# evalgate-rag

**A RAG service that cannot silently get worse.**

Hybrid-retrieval question answering over the **EU AI Act** (Regulation (EU) 2024/1689),
with an evaluation gate wired into CI: every change is scored with
[Ragas](https://docs.ragas.io) against a curated golden set, and **the build fails
if faithfulness, answer relevancy, or context precision regress** beyond tolerance.

| | |
|---|---|
| Serving | FastAPI, Docker, non-root multi-stage image |
| Retrieval | BM25 + pgvector dense search, Reciprocal Rank Fusion |
| Store | PostgreSQL 16 + pgvector (HNSW index) |
| Observability | Langfuse tracing on every query (self-hosted via compose profile) |
| Evaluation | Ragas: faithfulness · answer relevancy · context precision |
| CI/CD | GitHub Actions: lint → types → unit tests → pgvector integration tests → **eval gate** → Docker build → push to GHCR |
| LLM | Any OpenAI-compatible endpoint (default: **Groq free tier**, `llama-3.3-70b-versatile`) |
| Embeddings | Local, offline via **fastembed** (BAAI/bge-small-en-v1.5, 384-dim) — Groq has no embeddings API |

## Why the eval gate matters

RAG systems degrade invisibly: a prompt tweak improves one answer and quietly
breaks five others; a chunking change shifts retrieval in ways no unit test sees.
Treating eval metrics like a test suite — with a baseline, a tolerance, and a
red build on regression — is the difference between a demo and a system.

```
PR opened ──▶ lint/type/unit ──▶ pgvector integration ──▶ [label: run-eval]
                                                              │
                                              Ragas over 50-question golden set
                                                              │
                                    delta vs eval/baseline.json > −0.03 ? ──▶ ❌ build fails
                                                              │
main ──▶ all of the above ──▶ Docker build ──▶ push ghcr.io/…:sha
```

## Quickstart

```bash
cp .env.example .env                      # add your Groq API key (console.groq.com/keys)
docker compose up -d db                   # Postgres + pgvector
pip install -e .[dev,embed-local]         # embed-local pulls fastembed (local, offline)

python scripts/fetch_corpus.py            # EU AI Act from EUR-Lex → data/corpus/
python scripts/ingest.py --strategy recursive
uvicorn evalgate_rag.api:app --reload

curl -s localhost:8000/query -X POST -H 'content-type: application/json' \
  -d '{"question": "What are the maximum fines for prohibited AI practices?"}' | jq
```

With tracing: `docker compose --profile tracing up -d`, open Langfuse at
`localhost:3000`, create a project, put the keys in `.env`, set `LANGFUSE__ENABLED=true`.
Every query then produces a trace with a retrieval span (chunk IDs + scores) and
a generation span.

## LLM

Default configuration runs generation and Ragas judging on **Groq's free tier**
(`llama-3.3-70b-versatile`, OpenAI-compatible) and embeddings **locally via
fastembed** — Groq has no embeddings endpoint, so nothing embedding-related
ever leaves the machine. Swapping to any other OpenAI-compatible provider
(OpenAI, Azure via gateway, Ollama, vLLM) is still just an env-var change.

Groq's free tier is rate-limited on `llama-3.3-70b-versatile` (30 RPM / ~12K
TPM / 1K RPD), plus — easy to miss — a **100K tokens/day (TPD)** cap. TPD is
the one that actually bites: the golden set's questions × (1 generation call
+ 3 judge-metric calls each, all resending the full retrieved context) is
enough to threaten it well before the RPM/TPM numbers become a problem.

- `LLMClient` retries HTTP 429s automatically, honouring the response's
  `retry-after` header and falling back to exponential backoff — but a 429
  whose suggested wait is long (a hard quota like TPD, not a short RPM/TPM
  blip) is raised immediately instead of retried; waiting once more inside
  the same process won't clear a daily cap.
- `LLM__MIN_INTERVAL_S` (default `0`) adds a minimum delay between requests so
  a full eval run stays under the TPM cap — set it to a few seconds for eval
  runs; leave it at `0` for interactive `/query` traffic.

If you switch `EMBEDDING__PROVIDER` (e.g. `fastembed` ↔ `openai`), the vector
dimension changes (384 vs. 1536), so **re-ingest the corpus** after switching:
`python scripts/ingest.py --strategy recursive` recreates the pgvector column
at the new embedder's dimension.

Run the evaluation locally:

```bash
pip install -e .[eval,embed-local]
python eval/run_eval.py --limit 3          # throttled smoke test — a few questions
python eval/run_eval.py                    # answers all golden-set questions, scores with Ragas
python eval/check_regression.py            # the same gate CI runs
```

The Ragas judge LLM reuses `LLM__*` settings (so it also runs on Groq) but
Ragas is configured with `RunConfig(max_workers=1)` so judge calls run
sequentially — parallel judge calls would blow through the RPM/TPM caps.

**The run is resumable.** Generated answers and judge scores are cached to
`eval/.cache/` as they're produced (per question, and per question+metric for
judge scores), so a run that dies partway through — a real risk against a
100K/day budget — doesn't redo work that already succeeded on rerun. The
cache auto-invalidates if the prompt, model, `retrieval_top_k`, embedding
provider, or `golden_set.jsonl` changes; pass `--fresh` to force a clean run
outright (needed after a corpus re-ingest or chunking-strategy change, since
that state lives in Postgres and isn't visible to this script).

## Chunking benchmark

Three strategies over the same corpus, scored on whether the gold article is
retrieved (hybrid retrieval, top-4). Reproduce with
`python scripts/benchmark_chunking.py`.

| Strategy | hit@4 | MRR | chunks |
|---|---|---|---|
| fixed (1000/200) | _run to fill_ | _…_ | _…_ |
| recursive | _…_ | _…_ | _…_ |
| semantic (25th pct) | _…_ | _…_ | _…_ |

<sub>Table is generated by the script — paste its output here after the first
real-embedding run. `--hash` gives an offline smoke run with meaningless scores.</sub>

## Golden set methodology

`data/golden_set.jsonl` holds 20 hand-curated question/ground-truth pairs across
the Act's core obligations (prohibitions, penalties, high-risk classification,
GPAI, transparency). `scripts/generate_golden_set.py` drafts further candidates
with an LLM into a separate file marked `UNREVIEWED`; pairs are only promoted
after human review. CI never gates on unreviewed ground truth.

## How I work with coding agents

This repository is built agent-first, and the workflow is part of the design:

- **`CLAUDE.md`** gives any coding agent the commands, the architecture summary,
  and — most importantly — the *rules*: never edit the eval baseline to make the
  gate pass, never promote unreviewed golden-set pairs, keep unit tests offline.
- **The eval gate is the agent's guardrail as much as mine.** Agents iterate
  quickly on prompts, chunking, and retrieval; the Ragas gate catches the
  regressions that neither the agent nor I would spot by reading a diff.
- **Dependency injection everywhere** (`build_app(pipeline=…)`, `HashEmbedder`,
  `InMemoryStore`) exists so an agent can run the entire test suite in under a
  second, offline, on every edit — tight loops make agent output verifiable.
- Typical loop: describe the change → agent writes the failing test → agent
  implements → `pytest` + `ruff` locally → PR → CI runs the eval gate with the
  `run-eval` label before merge.

## Project layout

```
src/evalgate_rag/     config · chunking · embeddings · store · retrieval · pipeline · api
scripts/              fetch_corpus · ingest · benchmark_chunking · generate_golden_set
eval/                 run_eval (Ragas) · check_regression (the gate) · baseline.json
data/                 golden_set.jsonl · corpus/ (generated)
tests/                offline unit tests + pgvector integration tests
.github/workflows/    ci.yml · eval-gate.yml
```

## Notes

- Corpus: the consolidated EU AI Act text from EUR-Lex (public). ~110 articles +
  annexes ≈ 200 documents after splitting.
- The Docker image runs as a non-root user with a healthcheck; compose ships a
  `tracing` profile with single-container Langfuse v2 (v3 needs ClickHouse —
  see their official compose).
- Swapping the LLM is an env-var change; the eval gate makes model swaps *measurable*.

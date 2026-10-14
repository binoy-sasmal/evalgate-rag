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
| Deployment | Terraform → single EC2 instance, ~$13/month (free on a legacy Free Tier account) ([docs/deploy-aws.md](docs/deploy-aws.md)) |
| CI/CD | GitHub Actions — `ci.yml` (lint · types · unit · pgvector integration · Docker build → GHCR) plus a separate **`eval-gate.yml`** (the Ragas gate) |
| LLM | Any OpenAI-compatible endpoint (default: **Groq free tier**, `qwen/qwen3.8-27b`) |
| Embeddings | Local, offline via **fastembed** (BAAI/bge-small-en-v1.5, 384-dim) — Groq has no embeddings API |

## Why the eval gate matters

RAG systems degrade invisibly: a prompt tweak improves one answer and quietly
breaks five others; a chunking change shifts retrieval in ways no unit test sees.
Treating eval metrics like a test suite — with a baseline, a tolerance, and a
red build on regression — is the difference between a demo and a system.

```
PR opened ──▶ lint/type/unit ──▶ pgvector integration ──▶ [label: run-eval]
                                                              │
                                              Ragas over 20-question golden set
                                                              │
                                    delta vs eval/baseline.json > −0.03 ? ──▶ ❌ build fails
                                                              │
main ──▶ ci.yml: lint/type/unit/integration ──▶ Docker build ──▶ push ghcr.io/…:sha
```

### The committed baseline

`eval/baseline.json` is the standard every run is measured against. A metric may
drift down by at most **0.03** before the build goes red:

| Metric | Baseline | Fails below |
|---|---|---|
| `faithfulness` | **0.8923** | 0.8623 |
| `answer_relevancy` | **0.8090** | 0.7790 |
| `context_precision` | **0.7892** | 0.7592 |

Measured over the 20-question golden set. The file changes **only** via
`python eval/check_regression.py --promote` after a human-reviewed improvement —
never by hand to turn a red build green, which is the failure mode that converts
a quality gate into decoration. Promotion itself refuses to run on a partial
sample or on results with missing metrics, so a half-finished run can't become
the new standard either. Promotions are visible in git history.

> **Re-baselining in progress.** Groq decommissioned `llama-3.3-70b-versatile`,
> the model `eval/baseline.json` was measured on. The default is now
> `qwen/qwen3.8-27b`; scores are not comparable across a model swap, so the
> gate's push-to-main trigger is temporarily disabled while a full-sample run
> on the new model is produced and promoted. The gate still runs on demand and
> on any PR labelled `run-eval`. No threshold was relaxed and the committed
> baseline was not edited — see `.github/workflows/eval-gate.yml`.

![The eval gate in GitHub Actions: each Ragas metric compared to eval/baseline.json with its sample count. This run passes; any metric regressing beyond the 0.03 tolerance turns the build red.](assets/eval-gate.png)

<sub>The eval gate on a real run — `faithfulness`, `answer_relevancy`, and `context_precision` each vs the committed baseline, with sample counts. The build fails on any regression beyond tolerance.</sub>

## Quickstart

```bash
cp .env.example .env                      # add your Groq API key (console.groq.com/keys)
docker compose up -d db                   # Postgres + pgvector
pip install -e .[dev,embed-local]         # embed-local pulls fastembed (local, offline)

# the EU AI Act corpus ships in the repo (data/corpus/) — no fetch needed.
# (to refresh it from EUR-Lex later: python scripts/fetch_corpus.py)
python scripts/ingest.py --strategy recursive
uvicorn evalgate_rag.api:app --reload

curl -s localhost:8000/query -X POST -H 'content-type: application/json' \
  -d '{"question": "What are the maximum fines for prohibited AI practices?"}' | jq
```

With tracing: `docker compose --profile tracing up -d`, open Langfuse at
`localhost:3000`, create a project, put the keys in `.env`, set `LANGFUSE__ENABLED=true`.
Every query then produces a trace with a retrieval span (chunk IDs + scores) and
a generation span.

![A Langfuse trace for one query: the retrieval span (input question, output chunks with doc_id and score) and the generation span, shown in the trace tree.](assets/langfuse-trace.png)

## LLM

Default configuration runs generation and Ragas judging on **Groq's free tier**
(`qwen/qwen3.8-27b`, OpenAI-compatible) and embeddings **locally via
fastembed** — Groq has no embeddings endpoint, so nothing embedding-related
ever leaves the machine. Swapping to any other OpenAI-compatible provider
(OpenAI, Azure via gateway, Ollama, vLLM) is still just an env-var change.

**Hosted models get retired, and that breaks the gate, not just the app.** The
default was `llama-3.3-70b-versatile` until Groq decommissioned it; the next
eval run died in four seconds on `404 model_not_found`, and — worse — orphaned
a baseline measured on a model that no longer exists. A dependency can be
pinned; a hosted model cannot. List what is actually offered before assuming a
model id is still live:

```bash
curl -H "Authorization: Bearer $LLM__API_KEY" \
     https://api.groq.com/openai/v1/models | jq '.data[].id'
```

`qwen/qwen3.8-27b` was chosen over `openai/gpt-oss-*` because the gpt-oss
models emit reasoning tokens — ~139 completion tokens versus qwen's 22 on the
same grounded prompt — and the Ragas judge runs on this same model, against a
daily cap a full run already strains.

Groq's free tier is rate-limited (30 RPM / ~12K
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
100K/day budget — doesn't redo work that already succeeded on rerun. In CI the
cache is persisted across runs (saved even when the gate fails on a partial),
so successive triggers accumulate toward a full sample across the daily budget.
The cache auto-invalidates if the prompt, model, `retrieval_top_k`, embedding
provider, or `golden_set.jsonl` changes; pass `--fresh` to force a clean run
outright (needed after a corpus re-ingest or chunking-strategy change, since
that state lives in Postgres and isn't visible to this script).

## Deploying to AWS

[`terraform/`](terraform/) stands the whole service up on a single EC2 instance —
API and pgvector side by side, ingest run automatically on first boot.

```
        allowed_cidr only          ┌──────────────────────────┐
   (no auth on /query, so this ───▶│  EC2 t3.micro            │
    is the access control)    :80  │  ┌────────┐  ┌────────┐  │
                                   │  │  api   │  │  pg16  │  │  docker compose
                                   │  │ :8000  │  │pgvector│  │
                                   │  └────────┘  └────────┘  │
                                   └──────────────────────────┘
                                        │ Internet Gateway (no NAT)
                                        ▼
                                   Groq · GHCR · SSM Parameter Store
```

**~$13/month**, and the omissions are the point: no load balancer (~$17), no NAT
gateway (~$33), no RDS (~$13), no Secrets Manager. Those four are most of what
turns a small AWS deployment into a $70 bill, and none are needed for one box —
a public subnet with a closed security group gives identical egress for $0, and
the database is reproducible from the image rather than precious.

The Groq key lives in SSM Parameter Store and is read at boot, so it never enters
Terraform state (a `sensitive = true` variable is still plaintext in
`terraform.tfstate`). Shell access is SSM Session Manager — no port 22, no key
pair, no bastion. IMDSv2 is required with a one-hop limit, so a compromised
container can't reach the instance role. A $1 budget alarm tracks cost *before*
credits are applied, because Budgets otherwise reports $0 on a credits-based
account while the balance quietly drains.

```bash
aws ssm put-parameter --name /evalgate-rag/llm-api-key   --type SecureString --value "gsk_..." --region eu-north-1

cd terraform && cp terraform.tfvars.example terraform.tfvars   # set allowed_cidr
terraform init && terraform apply

curl -s "$(terraform output -raw api_url)/ready"   # {"status":"ready","chunks":452}
```

Full runbook — prerequisites, cost table, teardown, and what to harden before this
is more than a demo — in [docs/deploy-aws.md](docs/deploy-aws.md).

![The deployed service on its EC2 instance: the host resolves to ip-10-20-1-83.eu-north-1.compute.internal, both the API and pgvector containers report healthy, /ready confirms 452 ingested chunks, and a live query returns a grounded answer citing Article 99 and Article 100 with its retrieved contexts and fusion scores.](assets/aws-deployment.png)

<sub>Running on the deployed instance. The retrieval scores are RRF sums: Article 5 and
Article 99 sit near 0.03 because <em>both</em> the BM25 and dense legs surfaced them, while
a single-leg hit scores about 1/61 — the hybrid retrieval doing visible work.</sub>

**`/ready` is the endpoint that matters.** `/health` is liveness only and does no
I/O, so a database blip never gets the container restarted. `/ready` checks the
store is reachable *and* non-empty — because an un-ingested deployment doesn't
crash, it retrieves nothing and answers every question with "I cannot answer this
from the provided context". That looks like a broken model. It's a missing step,
and failing readiness makes it say so.

### What deploying it actually surfaced

Running this somewhere real found bugs that local development structurally cannot,
which is the honest argument for deploying a side project at all:

- **A single Postgres connection**, opened at startup and never replaced. Fine
  against a loopback; against a remote database any failover or idle timeout wedges
  the process permanently while `/health` keeps returning 200. Now a pooled
  connection checked on checkout — with a test that kills every backend via
  `pg_terminate_backend` and asserts recovery.
- **A 130MB model download on every cold start.** `fastembed` fetched BGE-small
  from HuggingFace on first construction, making boot depend on a third party and
  overrunning the healthcheck's start period. Now baked into the image; verified by
  ingesting with `HF_HUB_OFFLINE=1`.
- **A per-request value written onto a shared object.** `/query` assigned `top_k`
  to the pipeline instance, which FastAPI shares across threadpool requests — so one
  caller's override leaked into the next. Now a parameter.
- **No `max_tokens` on the LLM request.** Providers reserve per-minute output budget
  against a request's *expected* output; with no ceiling declared, Groq reserved the
  model default of 2048 against a 1000 output-tokens-per-minute cap and rejected
  every call. Worse, the 429 retry path backed off and retried — but a 429 about
  request *size* never succeeds on retry, and each attempt reserved more of the
  budget it was waiting to free.

## Deploying to Azure

[`terraform/azure/`](terraform/azure/) is a second, independent deployment:
Azure Container Apps for the API, Azure OpenAI for the model, PostgreSQL
Flexible Server with pgvector, and Application Insights receiving the
OpenTelemetry stream described below.

**Azure deployment is deferred.** The Terraform configuration is written and
statically validated but has not been deployed to an Azure subscription. The
verified deployment is on AWS. `terraform
validate` passes and the pieces that can be tested offline are, but no resource
has been created in a real subscription. [`docs/deploy-azure.md`](docs/deploy-azure.md)
is the runbook and says the same thing at the top.

Two things in it are worth reading even without deploying:

- **No model API key exists.** Key access is disabled on the Azure OpenAI
  resource; the API authenticates with a short-lived token from its managed
  identity. That is a one-setting switch (`LLM__AUTH=azure_managed_identity`)
  on the same OpenAI-compatible client, with no Azure SDK added.
- **The database password never reaches Terraform state.** It is generated by
  an ephemeral resource and passed through write-only arguments to PostgreSQL
  and Key Vault.

## Observability

Langfuse shows what the model saw and said. The operations questions (how long
did a question take, where did the time go, what did it cost in tokens, how
often did it fail) go through OpenTelemetry instead, so they land in whatever
stack already watches the rest of a system.

```bash
TELEMETRY__ENABLED=true docker compose --profile observability up -d
# Grafana   http://localhost:3001   (dashboard "evalgate-rag", no login)
# Prometheus http://localhost:9090  (alert rules under /alerts)
```

```
API ──OTLP/HTTP──▶ OpenTelemetry Collector ──▶ Tempo        (traces)  ─┐
                                           └─▶ Prometheus   (metrics) ─┴▶ Grafana
```

**Traces.** Every question is one trace: a `rag.query` span with a
`rag.retrieval` child and a `chat <model>` child. The LLM span follows the
OpenTelemetry GenAI semantic conventions (`gen_ai.request.model`,
`gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`), each rate-limit
retry is an event on it, and a failure sets `error.type`. When Langfuse is on,
the root span carries `langfuse.trace_id`, so a slow span leads to its prompt.

**Metrics.**

| Metric | Type | What it answers |
|---|---|---|
| `rag.query.duration` | histogram, s | Time to answer, and the failure ratio via `error.type` |
| `rag.retrieval.duration` | histogram, s | Embed + dense + BM25 + fusion |
| `gen_ai.client.operation.duration` | histogram, s | The LLM call, retries and backoff included |
| `gen_ai.client.token.usage` | histogram, tokens | Spend by `gen_ai.token.type` (input / output) |
| `rag.llm.retries` | counter | Requests retried after an HTTP 429 |

**Alerts.** [`observability/alerts.yml`](observability/alerts.yml) defines three
Prometheus rules: p95 time to answer above 10 s, more than 5% of questions
failing, and sustained rate limiting. The thresholds are starting values chosen
from the service's known shape, not tuned against production traffic, and no
Alertmanager is wired up.

Design choices worth knowing:

- **No question, context or answer text on a span.** A trace backend has a wider
  audience than a Langfuse project; it gets timings and token counts only. A
  test asserts none of the three leak.
- **Off by default, and injected.** `Telemetry` takes its providers as arguments
  instead of using OpenTelemetry's process globals, so the unit suite asserts on
  real spans and metrics through in-memory exporters, offline.
- **One protocol out.** The API only speaks OTLP to one address. With
  `TELEMETRY__OTLP_ENDPOINT` unset it honours the standard
  `OTEL_EXPORTER_OTLP_*` variables, so a managed collector needs no code change.
- **A dead collector cannot stop the API.** Export is batched in the background.

**A live run, and what it showed.** On 2026-10-14 the 20 golden-set questions
went through the containerised API against Groq's free tier
(`qwen/qwen3.8-27b`), with this stack recording. All 20 were answered.

![Grafana dashboard during the 20-question run: answered rate, p50/p95 time to answer, retrieval vs LLM p95, tokens per minute split input/output, zero failures, and rate-limit retries peaking at five a minute.](assets/grafana-dashboard.png)

- **Retrieval is not the bottleneck.** It averaged 93 ms. The model call had a
  median of about 0.9 s when not rate-limited.
- **The slow tail is rate limiting, not the model.** Four of the 20 questions
  took 13 to 29 s. The traces put all of it in the model call, and the 10
  retries recorded were all against Groq's free-tier cap of 1,000 output tokens
  per minute.
- **Answers are much longer than this repo assumed.** The 20 answers used
  15,419 input and 3,940 output tokens, about 197 output tokens each. The
  comments that size `max_tokens` assume about 30, which was true of the
  earlier model. With 512 tokens reserved per request and about 200 actually
  generated, two or three questions a minute exhaust the free-tier output
  budget, which is the slow tail above.

![Tempo trace of the slowest question: rag.query at 28.99 s, of which rag.retrieval took 85.5 ms and the chat call 28.9 s, with retry events marked on it.](assets/tempo-trace.png)

What has not been exercised: any deployed instance, and traffic beyond this
one 20-question run. The alert thresholds have not been tuned against it.

## UI

A minimal Streamlit client for interacting with the API — a question box, a
`top_k` slider, the answer, and the retrieved contexts (each chunk's `doc_id`,
`seq`, `score`, and text, expandable) so hybrid retrieval and citations are
visible. It hits `/health` on load and degrades gracefully when the API is
unreachable or the LLM backend is rate-limited (a 502 from `/query`).

![The Streamlit UI answering a question about GPAI systemic risk: the grounded answer cites Article 51 (the 10^25 FLOP threshold), with the retrieved contexts listed below and an "API reachable" status.](assets/streamlit-ui.png)

```bash
pip install -e .[ui]                       # streamlit; reuses core httpx
uvicorn evalgate_rag.api:app --reload      # the API, in one terminal
streamlit run ui/app.py                    # this UI, in another
```

The base URL defaults to `http://localhost:8000`; override it with the
`EVALGATE_API_URL` env var or the sidebar field.

## Chunking benchmark

Three strategies over the same corpus, scored on whether the gold article is
retrieved (hybrid retrieval, top-4). Reproduce with
`python scripts/benchmark_chunking.py`.

| Strategy | hit@4 | MRR | chunks |
|---|---|---|---|
| fixed (1000/200) | 0.85 | 0.77 | 470 |
| recursive | 0.85 | 0.80 | 452 |
| semantic (25th pct) | 0.75 | 0.72 | 787 |

<sub>Table is generated by the script — reproduce with `python scripts/benchmark_chunking.py`
(local fastembed embeddings, no API calls). `--hash` gives an offline smoke run with
meaningless scores. Recursive wins on MRR with fewer chunks than semantic, which is why
it's the default ingest strategy.</sub>

## Reranking benchmark

A local cross-encoder (`RERANK__ENABLED=true`, fastembed's ONNX runtime, no
API calls) can reorder a 20-chunk pool from hybrid retrieval before the top-4
cut. Reproduce with `python scripts/benchmark_rerank.py --candidates 10 20`.

| Retrieval | hit@1 | hit@4 | MRR@4 | ms/query |
|---|---|---|---|---|
| hybrid (RRF only) | 0.75 | 0.85 | 0.80 | 24 |
| + ms-marco-MiniLM-L-6-v2, pool 10 | 0.80 | 0.85 | 0.82 | 206 |
| + ms-marco-MiniLM-L-6-v2, pool 20 | 0.80 | 0.85 | 0.82 | 422 |
| + ms-marco-MiniLM-L-12-v2, pool 10 | 0.80 | 0.85 | 0.82 | 388 |
| + ms-marco-MiniLM-L-12-v2, pool 20 | 0.80 | 0.85 | 0.82 | 795 |

<sub>Golden set, recursive chunks, local fastembed embeddings, measured 2026-10-14 on a laptop CPU.</sub>

**Reranking is off by default, because it does not earn its cost here.** It
moved two questions up to rank 1 and one down to rank 2: a net change of one
question in twenty, which is the noise floor of this golden set, for 10 to 30
times the retrieval latency. Neither model nor pool size changed that.

The interesting result is why. The three questions every variant misses are
**recall** failures, which no reranker can fix: for "When does an AI system
count as high-risk?" Article 6 is not in the fused top 50 at all, and Annex
III sits at rank 32, outside any pool tried. Only the deepfake question had
its gold article (Article 50, rank 15) within reach, and the cross-encoder
did not lift it. The next retrieval improvement is first-stage recall, not
reordering.

## Answer-correctness judge

**Current status:** the prompt now uses `reference-supported-v2`, matching the
documented strict review policy: additional factual claims must be supported by
the reference. The figures below are **historical, from the previous prompt**,
which allowed extra correct context and conflicted with that policy. They do
not validate the corrected judge. No v2 live calibration has been run yet.
Labels and old results are preserved; see [rubric and validation plan](docs/judge-rubric.md).
The default calibration output is now `eval/judge_calibration_report_v2.json`.

Ragas' faithfulness asks whether an answer is supported by what was
retrieved, which a confidently wrong answer built on the wrong chunk passes.
[`judge.py`](src/evalgate_rag/judge.py) asks whether the answer states what
the golden set's reference says, without contradicting it, and returns PASS
or FAIL with the deciding fact.

The judge is a different model family from the answering model
(`openai/gpt-oss-120b` against `qwen/qwen3.8-27b`), so it is not grading its
own family's answers.

**Calibrated before use.** [`eval/judge_calibration.jsonl`](eval/judge_calibration.jsonl)
holds 60 candidate answers whose correct verdict is known by construction:
for each golden question, a paraphrase of the reference (PASS), the same
answer with one central fact changed (FAIL, the planted error is recorded),
and a correct answer to a different question (FAIL).
`python eval/calibrate_judge.py` scores the judge against them.

| Run (2026-10-14) | Accuracy | Cohen's kappa | False passes | False fails |
|---|---|---|---|---|
| 1 | 0.967 | 0.923 | 0 of 40 | 2 of 20 |
| 2 | 0.983 | 0.962 | 0 of 40 | 1 of 20 |
| 3 | 0.983 | 0.962 | 0 of 40 | 1 of 20 |

It caught every planted error, including 10^23 for 10^25 FLOPs and EUR 15
million for 35 million, in all three runs. Every false fail was a paraphrase
that shortened an enumerated list, and the one verdict that changed between
runs was one of those: the judge is strict about complete lists, and
borderline there. Those labels are unreviewed, so they may be what is wrong.

**And then it met real answers.** Run over the pipeline's 20 real answers
(`python eval/judge_answers.py --arm baseline`), it passed 6. It failed all
three questions whose gold article retrieval never finds (see the reranking
benchmark), an independent cross-check, and it failed answers that state the
reference's facts and then add claims the reference does not contain, such
as the Article 100 fine for EU institutions next to the correct EUR 35
million or 7 %. No constructed case covered that shape of answer.

A human reviewed all 20 in [`eval/judge_real_answers.jsonl`](eval/judge_real_answers.jsonl)
and upheld every verdict, including that strictness: under this rubric, an
answer that goes beyond the reference fails. Labelled real answers join the
calibration as kind `real`. Judged afresh against those labels:

| Kind | Cases | Agreement |
|---|---|---|
| paraphrase | 20 | 0.95 |
| perturbed (planted error) | 20 | 1.00 |
| swapped (wrong question) | 20 | 1.00 |
| real pipeline answers | 20 | 0.90 |
| **all** | **80** | **0.963, Cohen's kappa 0.915** |

Read the real-answer figure for what it is. The labels confirm the judge's
own first-pass verdicts rather than being made blind to them, so 18 of 20 is
the judge agreeing with an endorsed version of itself, not with an
independent rater. The two it reversed, the systemic-risk threshold answer
and the logging answer, are the borderline ones: each omits or adds detail
relative to the reference without stating anything the reference
contradicts. The judge is consistent on clear cases and unstable on those,
and a blind second labelling would be the next step towards an independent
agreement figure.

**Reported, not gated.** `eval/check_regression.py` and `eval/baseline.json`
are unchanged. The pass rate becomes a gate only after the labels are
reviewed and a person promotes a baseline.

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
src/evalgate_rag/     config · chunking · embeddings · store · retrieval · pipeline · rerank · judge · auth · telemetry · api
scripts/              fetch_corpus · ingest · benchmark_chunking · benchmark_rerank · generate_golden_set
eval/                 run_eval (Ragas) · check_regression (the gate) · baseline.json
                      calibrate_judge · judge_answers · judge label sets
data/                 golden_set.jsonl · corpus/ (EU AI Act, committed)
tests/                offline unit tests + pgvector integration tests
terraform/            single-instance AWS deployment (Free Tier)
terraform/azure/      Container Apps + Azure OpenAI deployment (validated, not yet applied)
observability/        collector, Prometheus + alert rules, Tempo, Grafana dashboard
docs/                 deploy-aws.md · deploy-azure.md · future-work-critic-loop.md
.github/workflows/    ci.yml · eval-gate.yml
```

## Notes

- Corpus: the consolidated EU AI Act text from EUR-Lex (public), split into 125
  article/annex documents. It's **committed to the repo** (`data/corpus/`), not
  fetched at build time — the eval baseline is measured against it, so it has to
  be a stable, versioned artifact (and EUR-Lex's WAF blocks the live fetch from
  CI runners anyway). Refresh it with `scripts/fetch_corpus.py`.
- The Docker image runs as a non-root user with a healthcheck; compose ships a
  `tracing` profile with single-container Langfuse v2 (v3 needs ClickHouse —
  see their official compose).
- Swapping the LLM is an env-var change; the eval gate makes model swaps *measurable*.

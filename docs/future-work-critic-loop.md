# Future work: LangGraph critic-abstain loop

> **Status: not implemented.** This is a design document, kept so the reasoning
> (and the measurement traps found while writing it) isn't re-derived later.
> Written 2026-08-31 against commit `43c6145`.

## Goal

Re-model the pipeline as a stateful graph — `retrieve → generate → critique →
(accept | abstain)` — where a separate critic model checks whether the draft
answer is grounded in the retrieved context, and the system abstains rather than
emitting an ungrounded best-effort draft.

The deliverable is a defensible before/after number, not the feature:

> "Added a LangGraph critic loop that **abstains** when an answer isn't grounded in
> retrieved context, cutting ungrounded answers from X% to Y% at Z× median latency."

Note "abstains", not "re-retrieves or abstains" — query reformulation and
re-retrieval are deliberately out of scope.

### Scope

**In:** the four-node graph; a critic node using a separate model call with a
strict JSON contract; abstention on reject; config-driven threshold / model /
on-off; a `critique` span in the existing Langfuse instrumentation.

**Out:** query reformulation and re-retrieval; Phoenix / LangSmith; any change to
chunking, retrieval, or the corpus.

**Hard constraint:** the existing linear `retrieve → generate` path keeps working
unchanged and stays selectable at runtime. A baseline reconstructed after a
refactor is worthless for comparison.

---

## Three findings that shape the design

### 1. The eval gate is structurally hostile to abstention

`eval/.cache/judge_scores.json` holds per-question scores from the run behind the
current baseline. Questions 3 and 4 **already abstain** ("I cannot answer this from
the provided context") and both scored `answer_relevancy: 0.0` — Ragas penalises
noncommittal answers by design.

So the 0.809 baseline is eighteen ~0.899s plus two zeros. Each *additional*
abstention costs ≈0.045 on the mean — larger than the 0.03 tolerance in
[`eval/check_regression.py`](../eval/check_regression.py). Separately, abstentions
extract zero claims → faithfulness `NaN` → dropped by the `math.isnan` filter in
[`eval/run_eval.py`](../eval/run_eval.py), so more than four abstentions also trips
`MIN_SAMPLE_FRACTION`.

**Conclusion: the critic arm can never be gated on the existing Ragas metrics.**
The gate stays green because the gated configuration runs byte-identical code —
not because a threshold was relaxed.

### 2. X and Y are not trustworthy on the golden set alone

Faithfulness < 0.8 hits exactly **2 of 20** questions:

- q4 (0.6) — which is *already an abstention*
- q10 (0.333) — whose answer, "10^25 FLOPs [Article 51]", is *correct*. That is
  judge noise, not a hallucination.

Real headroom is roughly one question. A percentage computed from that, using a
noisy LLM judge on a rate-limited free tier, would not survive an interview
follow-up. Hence the probe set below: it moves the primary number onto a
deterministic, judge-free, binary measurement with a real effect size.

### 3. The baseline measurement is unversioned

`eval/results.json` and `eval/.cache/` are gitignored. The only complete baseline
run — including every per-question score — exists on one machine and is one
`--fresh` away from deletion. **Freezing it is checkpoint 0 and costs nothing.**

---

## Decisions

| Decision | Choice |
|---|---|
| Primary metric substrate | New `data/probe_set.jsonl` (~30 reviewed questions); golden set secondary |
| Critic model | Different model, configurable; default `llama-3.1-8b-instant`; warn if equal to the generator's |
| Run budget | k=3 for deterministic metrics (latency / tokens / abstention), k=1 for Ragas-judged metrics |
| Default pipeline mode | `linear` — the critic path is opt-in at every level |

---

## 1. Graph state schema

`src/evalgate_rag/graph.py`, a `TypedDict` (LangGraph's default reducer: last write wins).

```python
class TokenUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int

class CriticVerdict(BaseModel):          # pydantic, extra="forbid"
    grounded: bool
    unsupported_claims: list[str]
    confidence: float                     # Field(ge=0.0, le=1.0)

class CriticState(TypedDict, total=False):
    question: str                         # input
    top_k: int                            # input
    contexts: list[ScoredChunk]           # retrieve →
    context_block: str                    # retrieve →
    draft_answer: str                     # generate →
    generate_usage: TokenUsage | None     # generate →
    verdict: CriticVerdict | None         # critique →
    critic_raw: str                       # critique →  (raw text, for audit)
    critic_usage: TokenUsage | None       # critique →
    answer: str                           # accept | abstain →
    accepted: bool                        # accept | abstain →
    abstained: bool                       # accept | abstain →
    timings_ms: dict[str, float]          # every node appends its own key
```

| Node | Reads | Writes |
|---|---|---|
| `retrieve` | `question`, `top_k` | `contexts`, `context_block`, `timings_ms["retrieve"]` |
| `generate` | `question`, `context_block` | `draft_answer`, `generate_usage`, `timings_ms["generate"]` |
| `critique` | `question`, `context_block`, `draft_answer` | `verdict`, `critic_raw`, `critic_usage`, `timings_ms["critique"]` |
| `accept` | `draft_answer` | `answer`, `accepted=True`, `abstained=False` |
| `abstain` | `verdict` | `answer=ABSTAIN_MESSAGE`, `accepted=False`, `abstained=True` |

Conditional edge after `critique`:

```python
def route(state) -> Literal["accept", "abstain"]:
    v = state["verdict"]
    return "accept" if (v.grounded and v.confidence >= threshold) else "abstain"
```

**`confidence` is defined as "the critic's probability that *every* claim in the
answer is entailed by the context"** — a grounding score, not confidence-in-its-own-verdict.
Stated explicitly in the critic prompt so the model and the router agree; the
ambiguous reading is what silently makes a threshold meaningless.

`CriticGraphPipeline` wraps the compiled graph and exposes
`answer(question) -> RAGResult`, the same duck type as `RAGPipeline`, so
`api.build_app` and the eval harness accept either.

---

## 2. Files to add / touch

### Add

| File | Why |
|---|---|
| `src/evalgate_rag/graph.py` | State, nodes, router, `CriticGraphPipeline`. All new behaviour lives here. |
| `src/evalgate_rag/critic.py` | `CriticVerdict`, `CRITIC_SYSTEM_PROMPT`, `build_critic_prompt(...)`, `parse_verdict(...)`, `CriticParseError`. Isolated from `pipeline.py` **by construction** — see §3. |
| `data/probe_set.jsonl` | ~30 human-reviewed probe questions. Separate file; never read by `run_eval.py`; never gates CI. |
| `eval/ab_harness.py` | Runs one arm (`--arm linear\|critic`) over golden and/or probe set; records answer, contexts, abstention, verdict, per-node latency, per-call token usage. Resumable + fingerprinted like `run_eval.py`. |
| `eval/score_ab.py` | Ragas scoring of an arm's cached answers (the token-heavy half, run separately). |
| `eval/ab/` | Committed outputs: `{arm}_runs.jsonl`, `{arm}_summary.json`, `manifest.json`, `report.md`. |
| `eval/baseline_frozen/` | Checkpoint 0: committed copy of today's `results.json` + `.cache/*.json`. |
| `tests/test_graph.py` | The test list in §5. |

### Touch

| File | Change | Risk control |
|---|---|---|
| [`src/evalgate_rag/config.py`](../src/evalgate_rag/config.py) | Add `CriticSettings` (`model`, `base_url`, `api_key`, `temperature`, `timeout_s`, `confidence_threshold=0.7`, `parse_retries=0`) and `pipeline_mode: str = "linear"`. | Purely additive; defaults reproduce today's behaviour. |
| [`src/evalgate_rag/pipeline.py`](../src/evalgate_rag/pipeline.py) | Two additive changes only: `LLMClient.chat(..., *, response_format: dict \| None = None)` keyword-only, and `self.last_usage` set from the response's `usage` block. `RAGPipeline` and both prompt constants **untouched**. | `RAGPipeline.answer()` is the baseline arm — it must not change. Existing `FakeLLM` duck types keep working (new param is keyword-only with a default). |
| [`src/evalgate_rag/api.py`](../src/evalgate_rag/api.py) | `build_app` picks pipeline by `cfg.pipeline_mode`; builds both when langgraph is installed; `/query` accepts an optional `mode` override. `QueryResponse` gains nullable `abstained`, `critic_confidence`, `unsupported_claims`. | Existing keys and values unchanged in linear mode; the response only gains `null`s. |
| [`pyproject.toml`](../pyproject.toml) | New extra `graph = ["langgraph>=0.6,<0.7"]`. | See §7 — kept out of core deps deliberately. |
| [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | `quality` + `unit-tests` jobs install `.[dev,graph]`. | `eval-gate.yml` **not touched**. |
| [`Dockerfile`](../Dockerfile) | Install `[graph]` so the demo can serve critic mode. | |
| `README.md`, `CLAUDE.md` | Document the mode flag, the critic contract, and the probe set's status (never gates CI). | |

**[`eval/run_eval.py`](../eval/run_eval.py), [`eval/check_regression.py`](../eval/check_regression.py),
`eval/baseline.json` and [`.github/workflows/eval-gate.yml`](../.github/workflows/eval-gate.yml)
are not modified at all.** `ab_harness.py` duplicates ~40 lines of Ragas judge setup rather than
refactoring it out of `run_eval.py` — deliberate duplication, because `run_eval.py` is the gated
artifact and a refactor there is the one change that could silently move the baseline.
Consolidate only *after* the numbers are captured.

---

## 3. Critic contract

### Exact JSON schema

```json
{
  "grounded": true,
  "unsupported_claims": ["Fines may reach EUR 40 million."],
  "confidence": 0.82
}
```

`parse_verdict(raw)`:

1. Strip a leading/trailing ```` ```json ```` fence if present; otherwise use the
   string as-is. No regex scavenging for "the first `{...}`" — that is how
   malformed output silently passes.
2. `json.loads` → `CriticVerdict.model_validate` with `extra="forbid"` and
   `confidence` bounded to `[0, 1]`.
3. Consistency rule: `grounded is False` with an empty `unsupported_claims` is a
   parse error — a rejection must name what it rejected.
4. Any failure raises `CriticParseError`. **Never** falls back to accept — and
   never falls back to abstain either, since a silent abstain is equally a lie
   about what happened.
5. `CRITIC__PARSE_RETRIES` defaults to **0** (one strict re-ask available if
   wanted). Every parse failure is counted and reported as its own rate.

Propagation: `CriticParseError` is not caught in the graph. `/query` surfaces it via
the existing 502 path in `api.py`; the harness records the question as
`critic_error`, excludes it from the delivered set, and reports the error rate
alongside X and Y.

Requests set `response_format={"type": "json_object"}` (Groq supports it; the word
"json" appears in the prompt, as Groq requires).

### Prompt strategy (outline)

**System** — a strict grounding auditor, not a reviewer:

- You receive a QUESTION, numbered CONTEXT passages, and a CANDIDATE ANSWER
  **produced by an unrelated system**.
- Decompose the candidate answer into atomic factual claims. For each, decide
  whether it is *entailed* by the context passages.
- Use no outside knowledge, even for claims you believe are true.
- Do not judge style, completeness, tone, or helpfulness — only claim support.
- A citation like `[Article 5]` must match a passage's `doc_id`; a citation to a
  passage not present is an unsupported claim.
- `confidence` = your probability that **every** claim is entailed by the context.
- Output ONLY the JSON object; schema inline.

**User** — exactly three labelled blocks: `QUESTION`, `CONTEXT PASSAGES`,
`CANDIDATE ANSWER`. Temperature 0. Zero-shot initially; few-shot examples are the
first tuning lever if the critic under- or over-fires (they cost tokens against the
TPD cap, so measure first).

### How isolation is guaranteed

A critic that sees the generator's reasoning rubber-stamps its own output. Four
layers, in descending order of strength:

1. **Structural — there is no shared context to leak.** `LLMClient.chat()` is
   stateless: it builds a fresh `[system, user]` message list on every call and
   keeps no conversation object. The generator's turn cannot carry into the
   critic's turn because nothing persists between calls.
2. **By construction.** `build_critic_prompt(question, context_block, draft_answer)`
   is a pure function of exactly those three strings, living in `critic.py`, which
   does not import `SYSTEM_PROMPT` or `PROMPT_TEMPLATE`. The generator's
   instructions are not reachable from the critic's message.
3. **Separate instances.** `CriticSettings` builds its own `LLMClient` → own
   `httpx.Client`, own connection, own API key, own model string, own temperature.
   Default critic model `llama-3.1-8b-instant` differs from the generator's
   `llama-3.3-70b-versatile`; startup logs a warning when critic model *and*
   base_url both equal the generator's.
4. **Enforced by test** (§5): capture the critic's outbound HTTP body via
   `httpx.MockTransport` and assert (a) `SYSTEM_PROMPT` / `PROMPT_TEMPLATE`
   substrings are absent, (b) the body contains only question/context/draft,
   (c) `model == cfg.critic.model`, (d) `graph._critic_llm is not graph._gen_llm`.

Honest limit: a smaller critic is a noisier judge. Independence and judging power
trade off, and the probe set is what tells us which side we landed on.

---

## 4. Eval design

### Capture order (non-negotiable)

Baseline numbers are captured and committed **before any source file changes**.

- **Stage A (zero tokens, first commit):** copy `eval/results.json` and
  `eval/.cache/*.json` into `eval/baseline_frozen/` and commit. Preserves the
  existing full run (faithfulness 0.8923 n=20, answer_relevancy 0.809 n=20,
  context_precision 0.7892 n=17) plus every per-question score. Currently
  gitignored and unrecoverable.
- **Stage B (~30K tokens per run, still before the refactor):** run
  `eval/ab_harness.py --arm linear` k=3 over golden + probe sets to capture what
  the existing artifacts lack — per-question latency, token usage, abstention.
  Runs against **unmodified** `RAGPipeline`, so the baseline is a measurement, not
  a post-refactor reconstruction.

### Metrics

| Metric | How | Judge needed? |
|---|---|---|
| **Ungrounded rate on probe set (headline X/Y)** | Fraction of unanswerable / false-premise probes where the system **asserted** instead of abstaining | No — binary, deterministic |
| **Wrong-abstention rate** | Fraction of *answerable* probes where the system abstained | No |
| Abstention rate | abstained / n, per arm, per subset | No |
| Critic parse-error rate | `CriticParseError` count / n | No |
| Ungrounded rate on golden set (secondary) | Delivered answers with per-question faithfulness < τ. τ=0.8 fixed **a priori**; sensitivity reported at 0.7 / 0.8 / 0.9 so no one can allege threshold-shopping | Yes (Ragas) |
| Ragas trio per arm | faithfulness / answer_relevancy / context_precision + sample counts | Yes |
| **Z — latency** | median and p95 of end-to-end `answer()`, plus per-node breakdown; Z = median_critic / median_linear | No |
| Token cost delta | prompt + completion per query, mean and median, per arm, plus ratio | No |

Wrong-abstention rate is mandatory in the report. Without it the headline is
meaningless — you can drive ungrounded answers to zero by abstaining on everything.

Latency hygiene: measured with `LLM__MIN_INTERVAL_S=0`; throttle sleeps and 429
retry sleeps are recorded separately and excluded from timings; questions that hit
a 429 retry are flagged, excluded from the median, and the exclusion count stated.

### Where results go

`eval/ab/` — committed: `linear_runs.jsonl`, `critic_runs.jsonl` (raw per-question
records), `{arm}_summary.json`, `manifest.json`, `report.md` (the table the bullet
is read off). `eval/results.json` and `eval/baseline.json` untouched.

### Reproducibility

`eval/ab/manifest.json` records, per arm-run: git commit SHA, sha256 of
`data/corpus/` (sorted), chunking strategy, embedding provider + model + dimension,
`retrieval_top_k`, `rrf_k`, generator model, critic model, temperature,
`confidence_threshold`, sha256 of `golden_set.jsonl` and `probe_set.jsonl`, pinned
versions of ragas / langchain-openai / fastembed / onnxruntime / langgraph, and a
UTC timestamp.

**Temperature 0 is not determinism** on a served endpoint. Hence k=3 for the cheap
deterministic metrics — report median-of-runs *and the spread*, so Z is a range,
not a point estimate — and k=1 for the Ragas-judged metrics, whose cache
accumulates across days exactly as CI already does against the 100K TPD cap.

---

## 5. Tests (`tests/test_graph.py`, all offline)

`pytest.importorskip("langgraph")` at module top, so a bare `.[dev]` install still passes.

1. **Grounded answer accepted** — fake critic returns `{"grounded": true,
   "unsupported_claims": [], "confidence": 0.9}`; result equals the draft,
   `abstained is False`.
2. **Hallucinated answer rejected** — `grounded: false` with a claim listed; result
   is `ABSTAIN_MESSAGE`, `abstained is True`, and **the draft text appears nowhere
   in the result**.
3. **Confidence below threshold abstains** — `grounded: true, confidence: 0.4` with
   threshold 0.7 → abstain. Locks in the "grounded AND confident" router semantics.
4. **Malformed critic JSON raises** — plain prose, truncated JSON, and
   `{"grounded": true}` (missing keys) each raise `CriticParseError`; parametrised.
   Explicitly asserts it does *not* return the draft.
5. **`grounded: false` with empty `unsupported_claims` raises** — the consistency rule.
6. **`extra="forbid"`** — an extra key raises rather than being ignored.
7. **Fenced JSON parses** — ```` ```json {...} ``` ```` is accepted.
8. **Feature flag off restores exact baseline behaviour** — with default settings,
   `build_app()` wires a `RAGPipeline` (not the graph), asserted by type and by the
   graph module not being imported; plus a direct assertion that
   `Settings().pipeline_mode == "linear"`, so a future default flip cannot sneak
   past review.
9. **Critic isolation** (§3 layer 4) — MockTransport captures the critic request;
   assert generator prompt constants absent, `model == cfg.critic.model`, distinct
   client instances.
10. **`/query` in linear mode returns the pre-change payload** — existing keys and
    values identical; new fields `null`.
11. **`/query` with `mode=critic` routes to the graph**, `mode=linear` to `RAGPipeline`.
12. **`LLMClient.last_usage` populated** from a mocked response's `usage` block, and
    `None` when the provider omits it.
13. **Per-node timings recorded** — `timings_ms` has `retrieve`, `generate`, `critique`.
14. **502 on critic parse failure** via `TestClient`.

Existing tests in [`tests/test_core.py`](../tests/test_core.py) must pass unmodified —
that is the regression signal for the hard constraint.

---

## 6. How the CI eval gate stays green

- `pipeline_mode` defaults to `"linear"`. `eval-gate.yml` sets no new env var, so
  `run_eval.build_pipeline()` constructs the identical `RAGPipeline` from unmodified
  `pipeline.py` code. **The gated path executes zero new code.**
- `run_eval.py`'s cache fingerprint hashes `SYSTEM_PROMPT`, `PROMPT_TEMPLATE`,
  model, base_url, top_k, embedding provider and the golden set. None of those
  change, so the accumulated `eval/.cache/` stays valid and the quota-paced run
  keeps its progress. (Verify the fingerprint is unchanged before pushing — a
  cheap, offline check.)
- Test #8 makes a default flip visible in review.
- **The critic arm is measured, reported, and never gated.** Not a workaround —
  Ragas `answer_relevancy` scores noncommittal answers 0 by design, which makes it
  the wrong instrument for a system whose correct behaviour includes abstaining. No
  tolerance is relaxed, no metric dropped, no abstention-aware fudging added to
  `check_regression.py`.
- `ci.yml` unit/quality jobs gain `[graph]`; all new tests are offline (fakes and
  `MockTransport`), preserving the sub-second offline suite.

---

## 7. Risks

### What could make the critic useless

1. **The headroom problem, restated.** Even with a probe set, if the linear pipeline
   already abstains reasonably often — it does, 2/20 on the golden set, and its
   system prompt explicitly instructs abstention — the critic's marginal
   contribution may be small. The probe set is designed to expose exactly this: if
   the baseline abstains on most unanswerable probes, the honest finding is "the
   prompt already did this" and the bullet does not exist. **That is a real possible
   outcome** and should be reported rather than engineered around.
2. **Over-abstention.** A critic tuned to catch the 1–2 bad answers will likely also
   reject good ones. If wrong-abstention rate exceeds the ungrounded-rate reduction,
   the feature is net-negative — hence its mandatory place in the report.
3. **Self-preference.** Mitigated structurally (§3) and by defaulting to a different
   model, not eliminated. Same-family models still share training-data priors.
4. **Critic reliability on a free tier.** JSON adherence from `llama-3.1-8b-instant`
   is decent but unmeasured here; a high parse-error rate would dominate the result.
   It is measured and reported as its own rate.
5. **Judge noise contaminates the secondary metric.** q10's *correct* answer scored
   0.333. Ragas faithfulness as the definition of "ungrounded" imports that noise
   into the golden-set X and Y. This is why the probe set is primary.
6. **Token budget.** The critic roughly doubles generation-phase tokens, on a quota
   that already cannot finish a judged run in one day.
7. **Dependency footprint.** `langgraph` pulls `langchain-core` and, transitively,
   `langsmith` — the ecosystem this repo deliberately kept out of the serving path,
   and LangSmith is explicitly out of scope (inert unless `LANGCHAIN_TRACING_V2=1`,
   but present). Confining it to a `[graph]` extra with lazy import keeps
   `pip install -e .[dev]` and the linear path at **zero new dependencies**. Stated
   plainly: for a four-node graph, LangGraph is structure and vocabulary, not
   load-bearing correctness — a plain function would work. It earns its place as the
   extension point for the re-retrieval loop that is out of scope today.

### What this plan does not cover

Query reformulation and re-retrieval (so the bullet must say "abstains", not
"re-retrieves or abstains"); Phoenix / LangSmith; any change to chunking, retrieval,
or corpus; streaming; concurrency behaviour of the graph under load; the critic's
calibration curve (we measure a threshold, not a reliability diagram); multi-turn;
and whether abstention actually improves *user* outcomes — unmeasured, and
unmeasurable from this harness.

---

## 8. Ordering of work

| # | Checkpoint | Cost | Gate to proceed |
|---|---|---|---|
| **0** | **Freeze the existing baseline.** Commit `eval/results.json` + `eval/.cache/*.json` to `eval/baseline_frozen/`. | 0 tokens | Committed. Nothing else starts before this. |
| **1** | Curate `data/probe_set.jsonl` (~30 questions: answerable / unanswerable / false-premise), human-reviewed, with an explicit `expected_behaviour` field. | 0 tokens | Human-reviewed. |
| **2** | `eval/ab_harness.py` + `eval/score_ab.py`, plus the two additive `LLMClient` changes (`response_format`, `last_usage`). No graph yet. | 0 tokens | `pytest -q` green, `ruff` + `mypy` clean. |
| **3** | **Baseline numbers captured and committed.** `--arm linear` k=3 over golden + probe sets; `score_ab.py` k=1; write `eval/ab/linear_*` and `manifest.json`. | ~90K tokens + one judged pass | X, latency baseline, token baseline committed and reproducible. **The first checkpoint that matters.** |
| **4** | `critic.py` — verdict model, prompt, parser. Tests 4–7 written first (CLAUDE.md rule 5). | 0 tokens | Parser tests green. |
| **5** | `graph.py` + config + api wiring + `[graph]` extra. Tests 1–3, 8–14. | 0 tokens | Full offline suite green; `test_core.py` unmodified and passing. |
| **6** | `--arm critic` k=3 + judged pass. Write `eval/ab/critic_*` and `report.md` with X, Y, Z, abstention, wrong-abstention, parse-error, token delta. | ~90K tokens + one judged pass | Numbers in hand. |
| **7** | Confirm the gate: verify `run_eval.py`'s fingerprint is unchanged, push, watch `eval-gate.yml` go green off the accumulated cache. | Quota-paced | Gate green with `baseline.json` untouched. |
| **8** | README section: the graph, the contract, the probe-set methodology, and the results table — caveats from §7 stated, not buried. | 0 tokens | |

Checkpoints 4–6 only run if checkpoint 3 shows headroom. If the baseline already
abstains on most unanswerable probes, stop and report that instead — an honest null
result is worth more than a bullet the numbers don't support.

---

## Verification (when this is eventually built)

- `pytest -q` — full offline suite, sub-second, no network/DB/keys.
  `tests/test_core.py` unmodified throughout; that is the hard-constraint regression
  signal.
- `ruff check . && ruff format --check . && mypy src/` — clean, line length 100.
- `RUN_PG_TESTS=1 pytest -q -m integration` with `docker compose up -d db`.
- End-to-end, both modes from one process:

  ```bash
  uvicorn evalgate_rag.api:app --reload
  curl -s localhost:8000/query -X POST -H 'content-type: application/json' \
    -d '{"question":"What are the maximum fines for prohibited AI practices?","mode":"linear"}' | jq
  curl -s localhost:8000/query -X POST -H 'content-type: application/json' \
    -d '{"question":"What does the AI Act say about mandatory carbon offsets for data centres?","mode":"critic"}' | jq
  ```

  The second should abstain with a non-null `critic_confidence` and populated
  `unsupported_claims`; the first should be byte-identical to today's response
  modulo the new null fields.
- Langfuse: `docker compose --profile tracing up -d`, `LANGFUSE__ENABLED=true`,
  confirm the trace tree shows `retrieval` → `generation` → **`critique`** as its own
  span with the verdict as span output.
- `python eval/check_regression.py` against a linear-arm run — must reproduce today's
  pass with `eval/baseline.json` unmodified.

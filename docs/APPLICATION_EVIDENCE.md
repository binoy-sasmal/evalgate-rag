# Application Evidence: evalgate-rag

This is a routing document for job-application work. The repository remains the source of truth.

## Status

Review correction: the judge now uses `reference-supported-v2`; its previous
prompt allowed extra correct context despite stricter human labels. Existing
calibration figures are historical and must not be presented as validation of
the corrected prompt. A fresh full calibration and independent blind labeling
remain pending. See `docs/judge-rubric.md`. Telemetry failure paths now suppress
exception messages/stack traces and label malformed model responses as errors.

Complete portfolio project, extended in October 2026 with observability, an answer-correctness
judge, and reranking. Use `README.md` for the current implementation, baseline, evaluation
results, and limitations. The Azure deployment is written and validated but **has never been
applied**. Azure deployment is deferred; the verified deployment is on AWS.

## What it demonstrates

- Hybrid BM25 and dense retrieval with Reciprocal Rank Fusion.
- RAG evaluation used as a CI regression gate.
- FastAPI service design, offline unit testing, and tracing boundaries.
- Containerized deployment and infrastructure as code on AWS.
- Cost-aware, resumable evaluation under provider limits.
- OpenTelemetry traces and metrics (GenAI semantic conventions) feeding a local Collector,
  Prometheus, Tempo and Grafana stack with alert rules, exercised with live LLM traffic.
- Diagnosing latency from telemetry: the slow tail was provider rate limiting, not retrieval.
- An LLM-as-judge implementation with historical exploratory calibration and measured limits.
  The corrected reference-supported-v2 rubric awaits fresh calibration; see docs/judge-rubric.md.
- A cross-encoder reranking experiment that was benchmarked, rejected on evidence, and traced to a
  first-stage recall bottleneck.
- Design of a keyless Azure deployment: managed identity instead of a model API key, and a
  database password kept out of Terraform state. Design only.

## Evidence map

| Claim area | Verify here |
|---|---|
| Current architecture, results, and limitations | `README.md` |
| AWS deployment procedure and infrastructure scope | `docs/deploy-aws.md` |
| Telemetry design, live-run findings, dashboard and trace screenshots | `README.md` "Observability"; `src/evalgate_rag/telemetry.py`; `tests/test_telemetry.py`; `observability/`; `assets/grafana-dashboard.png`, `assets/tempo-trace.png` |
| Reranking benchmark and the recall diagnosis | `README.md` "Reranking benchmark"; `scripts/benchmark_rerank.py`; `src/evalgate_rag/rerank.py` |
| Judge design, historical calibration figures and current caveats | `README.md` "Answer-correctness judge"; `docs/judge-rubric.md`; `src/evalgate_rag/judge.py`; `eval/judge_calibration_report.json`; `eval/judge_calibration.jsonl`; `eval/judge_real_answers.jsonl`; `tests/test_judge.py` |
| Historical judge pass rate (old prompt, not current v2 validation) | `eval/results/judged_baseline.json` |
| Azure design and its unapplied status | `docs/deploy-azure.md`; `terraform/azure/`; `src/evalgate_rag/auth.py` |
| Deferred critic-loop work | `docs/future-work-critic-loop.md` |
| Dependencies and quality tooling | `pyproject.toml` |
| Current tests and CI behavior | `tests/` and `.github/workflows/` |

## Safe application framing

- Call it a production-shaped or deployed portfolio RAG service only to the extent supported by
  the current README. The deployed instance is the AWS one.
- Tie every evaluation number to the committed baseline and named metric.
- Distinguish implemented retrieval and evaluation from deferred critic-loop work.
- **Observability:** "instrumented with OpenTelemetry and verified with live LLM traffic in a local
  stack". The live run was a single 20-question run on a laptop.
- **Judge:** describe implementation and historical exploratory calibration methodology. Read
  `docs/judge-rubric.md` first. The original `eval/judge_calibration_report.json` is historical,
  not validation of the corrected prompt. Do not quote its agreement as a current result.
  A future `eval/judge_calibration_report_v2.json` must exist and be verified before use;
  independent blind human labeling remains a separate requirement.
- **Reranking:** frame it as an experiment that was measured and rejected, with the recall
  diagnosis as the finding. Do not frame it as an improvement shipped.
- **Azure:** "designed" only, in a project description. Never "deployed", and never Azure in a
  skills line.

## Unsupported claims

- Employer or customer production traffic.
- Continuous high-scale operation.
- Any deferred feature described in `docs/future-work-critic-loop.md` as implemented.
- Metrics not reproducible from the repository's current evaluation artifacts.
- Production monitoring, tuned alerting, or on-call operation. The alert thresholds are untuned,
  no Alertmanager is wired, and telemetry has never run against the deployed instance.
- Any Azure experience, deployment, cost, latency or reliability figure. Azure OpenAI,
  Application Insights, Key Vault and Container Apps were never used.
- The judge as a regression gate. Its pass rate is reported, not gated.
- Independent human agreement for the judge on real answers.
- Reranking as an improvement to answer quality.

## Verification rule

Before using a number, deployment claim, or tool name, check the current README and its referenced
artifact. Record the repository commit in the application analysis.

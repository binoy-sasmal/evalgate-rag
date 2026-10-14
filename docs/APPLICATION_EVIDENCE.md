# Application Evidence: evalgate-rag

This is a routing document for job-application work. The repository remains the source of truth.

## Status

Complete portfolio project, extended in October 2026 with observability, an answer-correctness
judge, and reranking. Use `README.md` for the current implementation, baseline, evaluation
results, and limitations. The Azure deployment is written and validated but **has never been
applied**.

## What it demonstrates

- Hybrid BM25 and dense retrieval with Reciprocal Rank Fusion.
- RAG evaluation used as a CI regression gate.
- FastAPI service design, offline unit testing, and tracing boundaries.
- Containerized deployment and infrastructure as code on AWS.
- Cost-aware, resumable evaluation under provider limits.
- OpenTelemetry traces and metrics (GenAI semantic conventions) feeding a local Collector,
  Prometheus, Tempo and Grafana stack with alert rules, exercised with live LLM traffic.
- Diagnosing latency from telemetry: the slow tail was provider rate limiting, not retrieval.
- An LLM-as-judge for answer correctness, calibrated against labelled cases before use, with its
  limits on real answers measured and stated.
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
| Judge design, calibration figures and caveats | `README.md` "Answer-correctness judge"; `src/evalgate_rag/judge.py`; `eval/judge_calibration_report.json`; `eval/judge_calibration.jsonl`; `eval/judge_real_answers.jsonl`; `tests/test_judge.py` |
| Judge pass rate on the pipeline's answers | `eval/results/judged_baseline.json` |
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
- **Judge:** "calibrated an LLM-as-judge on 80 labelled cases". Quote agreement from
  `eval/judge_calibration_report.json`. If citing the real-answer figure, say the human labels
  endorsed the judge's own first-pass verdicts and were not made blind to them. The 40
  hand-written constructed labels are unreviewed.
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

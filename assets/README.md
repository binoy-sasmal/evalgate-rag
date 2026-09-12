# assets

Images referenced by the top-level `README.md`. Expected files:

| File | What it shows | Referenced in |
|---|---|---|
| `eval-gate.png` | The eval gate in GitHub Actions — the Ragas metric table (baseline · current · delta · samples) with the pass/fail result | "Why the eval gate matters" |
| `streamlit-ui.png` | The Streamlit UI answering a question, with the grounded answer + cited article and the retrieved contexts | "UI" |
| `langfuse-trace.png` | *(optional)* A Langfuse trace showing the retrieval span (chunk IDs + scores) and the generation span | "UI" / "Notes" |
| `aws-deployment.png` | The service running on the deployed EC2 instance — hostname, both containers healthy, `/ready` reporting the ingested chunk count, and a live query's answer with its retrieved contexts | "Deploying to AWS" |

PNG preferred. Keep them reasonably sized (a few hundred KB); crop to the
relevant panel so they read well inline.

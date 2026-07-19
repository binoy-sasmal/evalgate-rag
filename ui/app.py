"""Minimal Streamlit UI for the evalgate-rag API.

A read-only client of the FastAPI service: it hits /health on load and POSTs
to /query on submit, rendering the answer plus the retrieved contexts so the
hybrid retrieval and citations are visible. Run with:

    uvicorn evalgate_rag.api:app --reload      # the API, in one terminal
    streamlit run ui/app.py                     # this UI, in another
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

DEFAULT_BASE_URL = os.getenv("EVALGATE_API_URL", "http://localhost:8000")
DEFAULT_TOP_K = 4  # matches Settings.retrieval_top_k
REQUEST_TIMEOUT_S = 60.0  # a live Groq call plus retries can be slow


def check_health(base_url: str) -> tuple[bool, str]:
    """Return (reachable, message) for the API's /health endpoint."""
    try:
        resp = httpx.get(f"{base_url}/health", timeout=5.0)
        resp.raise_for_status()
        if resp.json().get("status") == "ok":
            return True, "API reachable"
        return False, "API responded but reported an unexpected status"
    except httpx.HTTPError:
        return False, "API unreachable"


def query_api(base_url: str, question: str, top_k: int) -> dict:
    """POST to /query. Raises httpx.HTTPStatusError on non-2xx responses."""
    resp = httpx.post(
        f"{base_url}/query",
        json={"question": question, "top_k": top_k},
        timeout=REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()


def render_contexts(contexts: list[dict]) -> None:
    if not contexts:
        st.info("No contexts were returned for this answer.")
        return
    st.subheader(f"Retrieved contexts ({len(contexts)})")
    for i, ctx in enumerate(contexts, start=1):
        header = f"{i}. {ctx['doc_id']} · seq {ctx['seq']} · score {ctx['score']:.4f}"
        with st.expander(header):
            st.write(ctx["text"])


st.set_page_config(page_title="evalgate-rag", page_icon="⚖️", layout="wide")
st.title("⚖️ evalgate-rag")
st.caption("Ask questions about the EU AI Act — hybrid RAG over the golden corpus.")

with st.sidebar:
    st.header("Settings")
    base_url = st.text_input("API base URL", value=DEFAULT_BASE_URL).rstrip("/")
    top_k = st.slider("top_k (contexts retrieved)", min_value=1, max_value=20, value=DEFAULT_TOP_K)

    reachable, health_msg = check_health(base_url)
    if reachable:
        st.success(health_msg)
    else:
        st.error(f"{health_msg} — is the API running at {base_url}?")

question = st.text_area(
    "Question",
    placeholder="e.g. What are the maximum fines for prohibited AI practices?",
    height=100,
)
submit = st.button("Ask", type="primary")

if submit:
    trimmed = question.strip()
    if len(trimmed) < 3:
        st.warning("Please enter a question of at least 3 characters.")
    else:
        with st.spinner("Querying the RAG service…"):
            try:
                result = query_api(base_url, trimmed, top_k)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 502:
                    st.error(
                        "The model backend is rate-limited or unavailable "
                        "(HTTP 502). Try again shortly."
                    )
                else:
                    st.error(f"The API returned an error (HTTP {exc.response.status_code}).")
            except httpx.HTTPError:
                st.error(f"Could not reach the API at {base_url}. Is it running?")
            else:
                st.subheader("Answer")
                st.write(result["answer"])
                if result.get("trace_id"):
                    st.caption(f"trace_id: {result['trace_id']}")
                render_contexts(result.get("contexts", []))

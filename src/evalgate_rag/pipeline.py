"""RAG pipeline: retrieve → prompt → generate, with optional Langfuse tracing.

Tracing is a no-op unless LANGFUSE__ENABLED=true, so unit tests and local
runs never require a Langfuse instance.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import LangfuseSettings, LLMSettings
from .retrieval import HybridRetriever
from .store import ScoredChunk

MAX_RETRIES = 5
INITIAL_BACKOFF_S = 1.0
# Above this, a 429's suggested wait isn't a short RPM/TPM blip our backoff is
# meant to absorb — it's almost certainly a hard daily quota (e.g. Groq's
# tokens-per-day cap), which won't clear by sleeping once more within this
# process. Fail fast instead of blocking for tens of minutes.
MAX_RETRYABLE_WAIT_S = 60.0

SYSTEM_PROMPT = """You are a precise assistant answering questions about the EU \
Artificial Intelligence Act (Regulation (EU) 2024/1689).
Answer ONLY from the provided context. If the context does not contain the \
answer, say "I cannot answer this from the provided context."
Cite the article identifiers (e.g. [Article 5]) that support each claim."""

PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""


@dataclass
class RAGResult:
    answer: str
    contexts: list[ScoredChunk]
    trace_id: str | None = None


class LLMClient:
    """Minimal OpenAI-compatible chat client (OpenAI, Azure via gateway,
    Groq, Ollama, vLLM, LiteLLM).

    Retries on HTTP 429 (rate limit) honouring the `retry-after` header when
    present, falling back to exponential backoff otherwise — needed for free
    tiers like Groq's (30 RPM / ~12K TPM). An optional minimum inter-request
    interval (`cfg.min_interval_s`) further throttles calls to stay under a
    TPM cap during eval runs. A 429 whose suggested wait exceeds
    `MAX_RETRYABLE_WAIT_S` is treated as a hard quota (e.g. a daily token
    cap) rather than a transient limit and is raised immediately instead of
    retried — waiting once more inside this process won't clear it.
    """

    def __init__(self, cfg: LLMSettings, transport: httpx.BaseTransport | None = None) -> None:
        self._cfg = cfg
        self._client = httpx.Client(
            base_url=cfg.base_url,
            headers={"Authorization": f"Bearer {cfg.api_key}"},
            timeout=cfg.timeout_s,
            transport=transport,
        )
        self._last_request_at: float | None = None

    def _throttle(self) -> None:
        if self._cfg.min_interval_s <= 0 or self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self._cfg.min_interval_s - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def chat(self, system: str, user: str) -> str:
        backoff = INITIAL_BACKOFF_S
        for attempt in range(MAX_RETRIES + 1):
            self._throttle()
            resp = self._client.post(
                "/chat/completions",
                json={
                    "model": self._cfg.model,
                    "temperature": self._cfg.temperature,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
            self._last_request_at = time.monotonic()
            if resp.status_code == 429 and attempt < MAX_RETRIES:
                wait_s = self._retry_delay(resp, backoff)
                if wait_s > MAX_RETRYABLE_WAIT_S:
                    print(
                        f"[LLMClient] 429 rate limited, suggested wait {wait_s:.0f}s exceeds "
                        f"{MAX_RETRYABLE_WAIT_S:.0f}s -- treating as a hard quota (not retrying): "
                        f"{self._describe_limit(resp)}",
                        file=sys.stderr,
                    )
                    resp.raise_for_status()
                print(
                    f"[LLMClient] 429 rate limited (attempt {attempt + 1}/{MAX_RETRIES}), "
                    f"retrying in {wait_s:.1f}s -- {self._describe_limit(resp)}",
                    file=sys.stderr,
                )
                time.sleep(wait_s)
                backoff *= 2
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        raise RuntimeError("unreachable")  # pragma: no cover

    @staticmethod
    def _retry_delay(resp: httpx.Response, backoff: float) -> float:
        retry_after = resp.headers.get("retry-after")
        if retry_after is None:
            return backoff
        try:
            return float(retry_after)
        except ValueError:
            return backoff

    @staticmethod
    def _describe_limit(resp: httpx.Response) -> str:
        """Best-effort description of which cap was hit, for the retry log line."""
        for header in ("x-ratelimit-remaining-tokens", "x-ratelimit-remaining-requests"):
            if resp.headers.get(header) == "0":
                return header.removeprefix("x-ratelimit-remaining-") + " limit"
        try:
            error = resp.json().get("error")
            message = error.get("message") if isinstance(error, dict) else error
        except ValueError:
            message = None
        return str(message) if message else "limit unspecified"


class Tracer:
    """Thin wrapper so the pipeline never imports langfuse unless enabled."""

    def __init__(self, cfg: LangfuseSettings) -> None:
        self._lf: Any = None
        if cfg.enabled:
            from langfuse import Langfuse  # lazy import

            self._lf = Langfuse(host=cfg.host, public_key=cfg.public_key, secret_key=cfg.secret_key)

    def trace_query(
        self,
        question: str,
        contexts: list[ScoredChunk],
        answer: str,
        model: str,
    ) -> str | None:
        if self._lf is None:
            return None
        trace = self._lf.trace(name="rag-query", input={"question": question})
        trace.span(
            name="retrieval",
            input={"question": question},
            output=[
                {"doc_id": c.chunk.doc_id, "seq": c.chunk.seq, "score": c.score} for c in contexts
            ],
        )
        trace.generation(
            name="generation",
            model=model,
            input={"question": question},
            output=answer,
        )
        trace.update(output={"answer": answer})
        return str(trace.id)

    def flush(self) -> None:
        if self._lf is not None:
            self._lf.flush()


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        tracer: Tracer,
        top_k: int = 4,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._tracer = tracer
        self._top_k = top_k

    def answer(self, question: str) -> RAGResult:
        contexts = self._retriever.retrieve(question, top_k=self._top_k)
        context_block = "\n\n---\n\n".join(f"[{c.chunk.doc_id}]\n{c.chunk.text}" for c in contexts)
        answer = self._llm.chat(
            SYSTEM_PROMPT, PROMPT_TEMPLATE.format(context=context_block, question=question)
        )
        trace_id = self._tracer.trace_query(question, contexts, answer, model=self._llm._cfg.model)
        return RAGResult(answer=answer, contexts=contexts, trace_id=trace_id)

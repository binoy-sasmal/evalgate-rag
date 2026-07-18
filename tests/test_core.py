"""Offline unit tests: hash embedder, in-memory store, fake LLM.

Integration tests against real pgvector are in test_pg_integration.py and
run only when RUN_PG_TESTS=1 (the CI integration job).
"""

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from evalgate_rag.api import build_app
from evalgate_rag.chunking import chunk_fixed, chunk_recursive, chunk_semantic
from evalgate_rag.config import LLMSettings
from evalgate_rag.embeddings import HashEmbedder
from evalgate_rag.pipeline import LLMClient, RAGPipeline, RAGResult, Tracer
from evalgate_rag.retrieval import HybridRetriever
from evalgate_rag.store import InMemoryStore

# ---------------------------------------------------------------- chunking


def test_fixed_chunks_cover_text_with_overlap():
    text = "abcdefghij" * 50  # 500 chars
    chunks = chunk_fixed(text, "doc", size=100, overlap=20)
    assert all(len(c.text) <= 100 for c in chunks)
    assert chunks[0].text[80:] == chunks[1].text[:20]  # overlap preserved
    assert [c.seq for c in chunks] == list(range(len(chunks)))


def test_fixed_rejects_overlap_ge_size():
    with pytest.raises(ValueError):
        chunk_fixed("hello", "doc", size=10, overlap=10)


def test_recursive_respects_max_chars_and_paragraphs():
    text = "Para one sentence. Another one.\n\n" + "Long paragraph. " * 100
    chunks = chunk_recursive(text, "doc", max_chars=200)
    assert all(len(c.text) <= 200 for c in chunks)
    assert chunks[0].text.startswith("Para one")


def test_semantic_chunks_break_on_topic_shift():
    embedder = HashEmbedder()
    text = (
        "Cats purr softly. Cats sleep all day. Cats chase mice. "
        "Quantum computers use qubits. Qubits enable superposition. Qubits are fragile."
    )
    chunks = chunk_semantic(text, "doc", embed_fn=embedder.embed, max_chars=500)
    assert len(chunks) >= 2  # topic shift creates at least one boundary
    assert "".join(c.text for c in chunks).count("qubits") >= 1


# --------------------------------------------------------------- retrieval


def _build_retriever() -> HybridRetriever:
    embedder = HashEmbedder()
    store = InMemoryStore()
    docs = {
        "Article 5": "Prohibited practices include social scoring and subliminal techniques.",
        "Article 99": "Fines of up to 35 million EUR or 7 percent of turnover apply.",
        "Article 14": "Human oversight requires a stop button and awareness of automation bias.",
    }
    for doc_id, text in docs.items():
        chunks = chunk_recursive(text, doc_id, max_chars=500)
        store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    retriever = HybridRetriever(store, embedder)
    retriever.refresh_bm25()
    return retriever


def test_hybrid_retrieval_finds_lexical_match():
    retriever = _build_retriever()
    results = retriever.retrieve("What fines apply for violations?", top_k=2)
    assert results[0].chunk.doc_id == "Article 99"


def test_hybrid_retrieval_returns_at_most_top_k():
    retriever = _build_retriever()
    assert len(retriever.retrieve("oversight", top_k=1)) == 1


def test_rrf_scores_are_descending():
    retriever = _build_retriever()
    results = retriever.retrieve("social scoring prohibition", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------- api


class FakePipeline:
    def answer(self, question: str) -> RAGResult:
        return RAGResult(answer=f"echo: {question}", contexts=[], trace_id=None)


@pytest.fixture()
def client():
    app = build_app(pipeline=FakePipeline())  # type: ignore[arg-type]
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_query_roundtrip(client):
    resp = client.post("/query", json={"question": "What is Article 5?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"].startswith("echo:")
    assert body["contexts"] == []


def test_query_validates_short_question(client):
    assert client.post("/query", json={"question": "hi"}).status_code == 422


# ------------------------------------------------------------------ tracer


def test_tracer_noop_when_disabled():
    from evalgate_rag.config import LangfuseSettings

    tracer = Tracer(LangfuseSettings(enabled=False))
    assert tracer.trace_query("q", [], "a", model="m") is None


# ---------------------------------------------------------------- pipeline


def test_pipeline_composes_context_and_calls_llm():
    class FakeLLM:
        class _Cfg:
            model = "fake"

        _cfg = _Cfg()

        def __init__(self):
            self.last_user = ""

        def chat(self, system: str, user: str) -> str:
            self.last_user = user
            return "answer [Article 99]"

    from evalgate_rag.config import LangfuseSettings

    retriever = _build_retriever()
    llm = FakeLLM()
    pipeline = RAGPipeline(retriever, llm, Tracer(LangfuseSettings()), top_k=2)  # type: ignore[arg-type]
    result = pipeline.answer("What fines apply?")
    assert "Article 99" in llm.last_user  # retrieved context injected into prompt
    assert result.answer == "answer [Article 99]"
    assert len(result.contexts) == 2


def test_embeddings_are_normalised():
    embedder = HashEmbedder()
    vecs = embedder.embed(["hello world", "another text"])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------- llm client


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def test_llm_client_retries_once_on_429_then_succeeds(monkeypatch, capsys):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "0"},
                json={"error": {"message": "Rate limit reached for tokens"}},
            )
        return httpx.Response(200, json=_chat_response("ok"))

    sleeps: list[float] = []
    monkeypatch.setattr("evalgate_rag.pipeline.time.sleep", lambda s: sleeps.append(s))

    client = LLMClient(
        LLMSettings(base_url="https://fake/v1"), transport=httpx.MockTransport(handler)
    )
    result = client.chat("system", "question")

    assert result == "ok"
    assert calls["n"] == 2
    assert sleeps == [0.0]  # honoured retry-after: 0 rather than the default backoff

    captured = capsys.readouterr()
    assert captured.out == ""  # log line must not pollute stdout (eval JSON goes there)
    assert "429 rate limited (attempt 1/5)" in captured.err
    assert "retrying in 0.0s" in captured.err
    assert "Rate limit reached for tokens" in captured.err


def test_llm_client_fails_fast_on_hard_quota_instead_of_long_sleep(monkeypatch, capsys):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            429,
            headers={"retry-after": "1980"},  # e.g. Groq's daily-token-quota style wait
            json={"error": {"message": "Rate limit reached ... tokens per day (TPD)"}},
        )

    sleeps: list[float] = []
    monkeypatch.setattr("evalgate_rag.pipeline.time.sleep", lambda s: sleeps.append(s))

    client = LLMClient(
        LLMSettings(base_url="https://fake/v1"), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.chat("system", "question")

    assert calls["n"] == 1  # gave up after the first attempt, no retry loop
    assert sleeps == []  # never slept for the 1980s — treated as a hard quota
    assert "hard quota (not retrying)" in capsys.readouterr().err


def test_llm_client_describes_limit_from_headers_when_no_body_message(monkeypatch, capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        if handler.calls == 0:
            handler.calls += 1
            return httpx.Response(429, headers={"x-ratelimit-remaining-requests": "0"})
        return httpx.Response(200, json=_chat_response("ok"))

    handler.calls = 0
    monkeypatch.setattr("evalgate_rag.pipeline.time.sleep", lambda s: None)

    client = LLMClient(
        LLMSettings(base_url="https://fake/v1"), transport=httpx.MockTransport(handler)
    )
    client.chat("system", "question")

    assert "requests limit" in capsys.readouterr().err


def test_llm_client_gives_up_after_max_retries(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    monkeypatch.setattr("evalgate_rag.pipeline.time.sleep", lambda s: None)

    client = LLMClient(
        LLMSettings(base_url="https://fake/v1"), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.chat("system", "question")


def test_llm_client_throttles_to_min_interval(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_response("ok"))

    clock = iter([100.0, 100.5, 102.0])
    monkeypatch.setattr("evalgate_rag.pipeline.time.monotonic", lambda: next(clock))
    sleeps: list[float] = []
    monkeypatch.setattr("evalgate_rag.pipeline.time.sleep", lambda s: sleeps.append(s))

    cfg = LLMSettings(base_url="https://fake/v1", min_interval_s=2.0)
    client = LLMClient(cfg, transport=httpx.MockTransport(handler))
    client.chat("system", "question")
    client.chat("system", "question")

    assert sleeps == [pytest.approx(1.5)]

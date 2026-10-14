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


def test_hybrid_retrieval_on_empty_store_returns_no_results():
    # A store with nothing ingested yet (fresh deployment, failed ingest) must
    # degrade to an empty result, not crash -- BM25Okapi's fallback empty-doc
    # index used to leave self._bm25_chunks (len 0) out of sync with its score
    # array (len 1), raising ValueError in zip(..., strict=True).
    retriever = HybridRetriever(InMemoryStore(), HashEmbedder())
    retriever.refresh_bm25()
    assert retriever.retrieve("anything", top_k=4) == []


class KeywordReranker:
    """Scores a chunk by whether it mentions a keyword, so the expected order
    is known regardless of what BM25 and the dense leg think."""

    def __init__(self, keyword: str) -> None:
        self.keyword = keyword
        self.seen: list[list[str]] = []

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        self.seen.append(list(texts))
        return [1.0 if self.keyword in t else 0.0 for t in texts]


def test_reranker_decides_the_final_order():
    retriever = _build_retriever()
    # Lexically and semantically this question is about fines; the fake
    # reranker prefers the oversight article, so it must come first.
    retriever._reranker = KeywordReranker("oversight")  # type: ignore[attr-defined]
    results = retriever.retrieve("What fines apply for violations?", top_k=2)
    assert results[0].chunk.doc_id == "Article 14"
    assert results[0].score == 1.0  # the reranker's score, not the RRF score


def test_reranker_sees_a_candidate_pool_wider_than_top_k():
    embedder = HashEmbedder()
    store = InMemoryStore()
    for i in range(30):
        chunks = chunk_recursive(f"Article text number {i} about obligations.", f"A{i}", 500)
        store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    reranker = KeywordReranker("never-present")
    retriever = HybridRetriever(store, embedder, reranker=reranker, rerank_candidates=12)
    retriever.refresh_bm25()

    results = retriever.retrieve("obligations", top_k=3)
    assert len(results) == 3
    assert len(reranker.seen[0]) == 12  # reranks the pool, then cuts to top_k


def test_reranker_pool_is_never_smaller_than_top_k():
    reranker = KeywordReranker("x")
    retriever = HybridRetriever(
        InMemoryStore(), HashEmbedder(), reranker=reranker, rerank_candidates=2
    )
    store = retriever._store  # type: ignore[attr-defined]
    for i in range(6):
        chunks = chunk_recursive(f"Provider duty {i} applies.", f"B{i}", 500)
        store.upsert(chunks, HashEmbedder().embed([c.text for c in chunks]))
    retriever.refresh_bm25()
    assert len(retriever.retrieve("provider duty", top_k=5)) == 5
    assert len(reranker.seen[0]) == 5


# --------------------------------------------------------------------- api


class FakePipeline:
    def __init__(self) -> None:
        self.seen_top_k: list[int | None] = []

    def answer(self, question: str, top_k: int | None = None) -> RAGResult:
        self.seen_top_k.append(top_k)
        return RAGResult(answer=f"echo: {question}", contexts=[], trace_id=None)


@pytest.fixture()
def fake_pipeline():
    return FakePipeline()


@pytest.fixture()
def client(fake_pipeline):
    app = build_app(pipeline=fake_pipeline)  # type: ignore[arg-type]
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


def test_ready_reports_ready_with_injected_pipeline(client):
    body = client.get("/ready").json()
    assert body["status"] == "ready"
    assert body["chunks"] is None


def test_top_k_override_does_not_leak_between_requests(client, fake_pipeline):
    # top_k used to be written onto the shared pipeline object per request
    # (pl._top_k = req.top_k). FastAPI runs the sync /query handler in a
    # threadpool over one pipeline instance, so an override from one caller
    # stayed set for every later caller that omitted it.
    client.post("/query", json={"question": "with an override", "top_k": 17})
    client.post("/query", json={"question": "without one"})
    assert fake_pipeline.seen_top_k == [17, None]


def test_pipeline_top_k_argument_overrides_configured_default():
    class CountingLLM:
        class _Cfg:
            model = "fake"

        _cfg = _Cfg()

        def chat(self, system: str, user: str) -> str:
            return "ok"

    from evalgate_rag.config import LangfuseSettings

    pipeline = RAGPipeline(_build_retriever(), CountingLLM(), Tracer(LangfuseSettings()), top_k=1)  # type: ignore[arg-type]
    assert len(pipeline.answer("fines").contexts) == 1  # configured default
    assert len(pipeline.answer("fines", top_k=3).contexts) == 3  # per-call override
    assert len(pipeline.answer("fines").contexts) == 1  # default restored, not mutated


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


def test_llm_client_sends_max_tokens():
    """Without a declared ceiling, providers reserve the model default against
    a per-minute output budget -- Groq reserves 2048 against a 1000 OTPM cap,
    which rejects every request regardless of how idle the account is."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_chat_response("ok"))

    client = LLMClient(
        LLMSettings(base_url="http://test", api_key="k", max_tokens=512),
        transport=httpx.MockTransport(handler),
    )
    assert client.chat("sys", "user") == "ok"
    assert seen["max_tokens"] == 512


def test_llm_client_does_not_retry_an_oversized_request(monkeypatch):
    """A 429 about request size is not transient. Retrying cannot help -- the
    request is the same size next time -- and each attempt reserves more of the
    per-minute budget it is waiting on, so backoff prolongs the outage."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            429,
            json={
                "error": {
                    "message": (
                        "Request too large for model `qwen/qwen3.8-27b` on output tokens "
                        "per minute (OTPM): Limit 1000, Requested 2048. reduce max_tokens"
                    )
                }
            },
        )

    sleeps: list[float] = []
    monkeypatch.setattr("evalgate_rag.pipeline.time.sleep", lambda s: sleeps.append(s))

    client = LLMClient(
        LLMSettings(base_url="http://test", api_key="k"),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.chat("sys", "user")

    assert calls["n"] == 1, "must fail on the first response, not burn retries"
    assert sleeps == [], "must not back off"


def test_llm_client_still_retries_a_genuine_rate_limit(monkeypatch):
    """The size check must not swallow ordinary transient 429s."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "0"},
                json={"error": {"message": "Rate limit reached for requests"}},
            )
        return httpx.Response(200, json=_chat_response("recovered"))

    monkeypatch.setattr("evalgate_rag.pipeline.time.sleep", lambda s: None)
    client = LLMClient(
        LLMSettings(base_url="http://test", api_key="k"),
        transport=httpx.MockTransport(handler),
    )
    assert client.chat("sys", "user") == "recovered"
    assert calls["n"] == 2


# -------------------------------------------------------------------- auth


def test_llm_client_sends_api_key_as_bearer_by_default():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return httpx.Response(200, json=_chat_response("ok"))

    LLMClient(
        LLMSettings(base_url="https://fake/v1", api_key="sk-test"),
        transport=httpx.MockTransport(handler),
    ).chat("s", "u")
    assert seen == ["Bearer sk-test"]


def test_llm_client_asks_the_token_provider_on_every_request():
    tokens = iter(["first", "second"])
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return httpx.Response(200, json=_chat_response("ok"))

    client = LLMClient(
        LLMSettings(base_url="https://fake/v1", api_key="ignored"),
        transport=httpx.MockTransport(handler),
        token_provider=lambda: next(tokens),
    )
    client.chat("s", "u")
    client.chat("s", "u")
    assert seen == ["Bearer first", "Bearer second"]  # a rotated token is picked up


def test_llm_client_rejects_unknown_auth_mode():
    with pytest.raises(ValueError, match="LLM__AUTH"):
        LLMClient(LLMSettings(auth="kerberos"))


def _identity(handler, clock, **kwargs):
    from evalgate_rag.auth import ManagedIdentityToken

    return ManagedIdentityToken(
        "https://cognitiveservices.azure.com",
        endpoint="http://identity.local/token",
        header="platform-secret",
        transport=httpx.MockTransport(handler),
        clock=clock,
        **kwargs,
    )


def test_managed_identity_token_is_requested_cached_and_refreshed_before_expiry():
    now = {"t": 1000.0}
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        n = len(requests)
        return httpx.Response(
            200, json={"access_token": f"token-{n}", "expires_on": str(int(now["t"]) + 3600)}
        )

    token = _identity(handler, lambda: now["t"], client_id="my-identity")

    assert token() == "token-1"
    first = requests[0]
    assert first.headers["x-identity-header"] == "platform-secret"
    assert first.url.params["resource"] == "https://cognitiveservices.azure.com"
    assert first.url.params["client_id"] == "my-identity"

    now["t"] += 3000  # 600 s of life left: still outside the refresh margin
    assert token() == "token-1"
    assert len(requests) == 1

    now["t"] += 400  # 200 s left: inside the 300 s margin, so refresh early
    assert token() == "token-2"
    assert len(requests) == 2


def test_managed_identity_outside_azure_fails_with_an_actionable_message(monkeypatch):
    from evalgate_rag.auth import ManagedIdentityToken

    monkeypatch.delenv("IDENTITY_ENDPOINT", raising=False)
    monkeypatch.delenv("IDENTITY_HEADER", raising=False)
    with pytest.raises(RuntimeError, match="LLM__AUTH=api_key"):
        ManagedIdentityToken("https://cognitiveservices.azure.com")

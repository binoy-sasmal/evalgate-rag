"""Semantic answer cache: offline tests with the hash embedder and a fake LLM.

The cache exists to skip the LLM call for a question that is a near-duplicate
of one already answered. The tests pin down both halves of that bargain: a
genuine repeat is served from the cache, and the near-misses that would make a
cached answer *wrong* (a different article number, a flipped negation, a
changed prompt, model or top_k) are never served.
"""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from evalgate_rag.api import build_app
from evalgate_rag.chunking import Chunk
from evalgate_rag.config import CacheSettings, LangfuseSettings
from evalgate_rag.embeddings import HashEmbedder
from evalgate_rag.pipeline import RAGPipeline, Tracer, cache_namespace
from evalgate_rag.retrieval import HybridRetriever
from evalgate_rag.semantic_cache import SemanticCache, anchors
from evalgate_rag.store import InMemoryStore, ScoredChunk

REFUSAL = "I cannot answer this from the provided context."


class FixedEmbedder:
    """Maps known questions to chosen unit vectors, so similarity is exact."""

    dimension = 3

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed(self, texts):
        out = np.array([self._vectors[t] for t in texts], dtype=np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _ctx(doc_id: str = "Article 5") -> list[ScoredChunk]:
    return [ScoredChunk(chunk=Chunk(doc_id=doc_id, seq=0, text="text"), score=1.0)]


# ------------------------------------------------------------------ anchors


def test_anchors_capture_numbers_and_roman_numerals_after_legal_units():
    assert anchors("What does Article 5 say?") == frozenset({"5"})
    assert anchors("Which areas are listed in Annex III?") == frozenset({"iii"})
    assert anchors("Compare Article 6(2) and Article 50") == frozenset({"6", "2", "50"})
    assert anchors("Which AI practices are prohibited?") == frozenset()


# -------------------------------------------------------------------- cache


def test_exact_repeat_is_served_from_cache():
    cache = SemanticCache(HashEmbedder(), threshold=0.95)
    q = "Which AI practices are prohibited under the AI Act?"
    assert cache.lookup(q, "ns") is None
    assert cache.store(q, "ns", "Manipulation, social scoring ...", _ctx())
    hit = cache.lookup(q, "ns")
    assert hit is not None
    assert hit.answer == "Manipulation, social scoring ..."
    assert hit.similarity > 0.999


def test_paraphrase_above_threshold_hits_and_below_misses():
    emb = FixedEmbedder({"a": [1, 0, 0], "close": [1, 0.1, 0], "far": [1, 1, 0]})
    cache = SemanticCache(emb, threshold=0.95)
    cache.store("a", "ns", "answer", _ctx())
    assert cache.lookup("close", "ns") is not None  # cosine ~0.995
    assert cache.lookup("far", "ns") is None  # cosine ~0.707


def test_different_article_number_never_hits_even_when_vectors_match():
    q1 = "What does Article 5 prohibit?"
    q2 = "What does Article 6 prohibit?"
    emb = FixedEmbedder({q1: [1, 0, 0], q2: [1, 0, 0]})  # identical vectors on purpose
    cache = SemanticCache(emb, threshold=0.5)
    cache.store(q1, "ns", "answer about Article 5", _ctx())
    assert cache.lookup(q2, "ns") is None


def test_flipped_negation_never_hits():
    q1 = "Which AI systems are prohibited?"
    q2 = "Which AI systems are not prohibited?"
    emb = FixedEmbedder({q1: [1, 0, 0], q2: [1, 0, 0]})
    cache = SemanticCache(emb, threshold=0.5)
    cache.store(q1, "ns", "answer", _ctx())
    assert cache.lookup(q2, "ns") is None


def test_namespace_isolates_model_prompt_and_top_k():
    assert cache_namespace("m1", 4) != cache_namespace("m2", 4)
    assert cache_namespace("m1", 4) != cache_namespace("m1", 5)
    cache = SemanticCache(HashEmbedder(), threshold=0.9)
    cache.store("What is a regulatory sandbox?", cache_namespace("m1", 4), "a", _ctx())
    assert cache.lookup("What is a regulatory sandbox?", cache_namespace("m2", 4)) is None


def test_refusals_and_empty_contexts_are_not_cached():
    cache = SemanticCache(HashEmbedder(), threshold=0.9)
    assert not cache.store("What is the role of the AI Office?", "ns", REFUSAL, _ctx())
    assert not cache.store("What is the role of the AI Office?", "ns", "an answer", [])
    assert len(cache) == 0


def test_entries_expire_after_ttl():
    clock = Clock()
    cache = SemanticCache(HashEmbedder(), threshold=0.9, ttl_s=10, clock=clock)
    cache.store("How does the AI Act define an AI system?", "ns", "a", _ctx())
    clock.now = 9.9
    assert cache.lookup("How does the AI Act define an AI system?", "ns") is not None
    clock.now = 10.1
    assert cache.lookup("How does the AI Act define an AI system?", "ns") is None
    assert len(cache) == 0  # expired entry purged, not just skipped


def test_capacity_evicts_least_recently_used():
    cache = SemanticCache(HashEmbedder(), threshold=0.99, max_entries=2)
    cache.store("first question about fines", "ns", "1", _ctx())
    cache.store("second question about sandboxes", "ns", "2", _ctx())
    assert cache.lookup("first question about fines", "ns") is not None  # touch first
    cache.store("third question about deepfakes", "ns", "3", _ctx())
    assert cache.lookup("second question about sandboxes", "ns") is None  # evicted
    assert cache.lookup("first question about fines", "ns") is not None
    assert len(cache) == 2


def test_rejects_invalid_settings():
    import pytest

    with pytest.raises(ValueError):
        SemanticCache(HashEmbedder(), threshold=1.5)
    with pytest.raises(ValueError):
        SemanticCache(HashEmbedder(), max_entries=0)


# ----------------------------------------------------------------- pipeline


class CountingLLM:
    class _Cfg:
        model = "fake-model"

    _cfg = _Cfg()

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, system: str, user: str) -> str:
        self.calls += 1
        return "Fines reach EUR 35 million [Article 99]"


def _pipeline(cache: SemanticCache | None) -> tuple[RAGPipeline, CountingLLM]:
    embedder = HashEmbedder()
    store = InMemoryStore()
    chunks = [
        Chunk(doc_id=f"Article {i}", seq=0, text=f"fines penalties article {i}") for i in range(5)
    ]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    retriever = HybridRetriever(store, embedder)
    retriever.refresh_bm25()
    llm = CountingLLM()
    return RAGPipeline(retriever, llm, Tracer(LangfuseSettings()), top_k=2, cache=cache), llm  # type: ignore[arg-type]


def test_pipeline_skips_the_llm_on_a_cache_hit():
    pipeline, llm = _pipeline(SemanticCache(HashEmbedder(), threshold=0.95))
    first = pipeline.answer("What fines apply for violations?")
    second = pipeline.answer("What fines apply for violations?")
    assert llm.calls == 1
    assert not first.cached and second.cached
    assert second.answer == first.answer
    assert [c.chunk.doc_id for c in second.contexts] == [c.chunk.doc_id for c in first.contexts]


def test_pipeline_top_k_override_misses_the_cache():
    pipeline, llm = _pipeline(SemanticCache(HashEmbedder(), threshold=0.95))
    pipeline.answer("What fines apply for violations?")
    pipeline.answer("What fines apply for violations?", top_k=3)
    assert llm.calls == 2


def test_pipeline_without_cache_always_calls_the_llm():
    pipeline, llm = _pipeline(None)
    pipeline.answer("What fines apply for violations?")
    pipeline.answer("What fines apply for violations?")
    assert llm.calls == 2


def test_cache_hit_is_recorded_without_question_text_on_spans():
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from evalgate_rag.telemetry import Telemetry

    spans = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(spans))
    reader = InMemoryMetricReader()
    telemetry = Telemetry(tp, MeterProvider(metric_readers=[reader]))

    pipeline, _ = _pipeline(SemanticCache(HashEmbedder(), threshold=0.95))
    pipeline._telemetry = telemetry
    question = "What fines apply for violations?"
    pipeline.answer(question)
    pipeline.answer(question)

    query_spans = [s for s in spans.get_finished_spans() if s.name == "rag.query"]
    assert [s.attributes["rag.cache.hit"] for s in query_spans] == [False, True]
    for span in spans.get_finished_spans():
        assert question not in str(dict(span.attributes or {}))

    points = {
        dict(p.attributes)["rag.cache.result"]: p.value
        for m in reader.get_metrics_data().resource_metrics[0].scope_metrics[0].metrics
        if m.name == "rag.cache.lookups"
        for p in m.data.data_points
    }
    assert points == {"miss": 1, "hit": 1}


# ---------------------------------------------------------------------- api


def test_api_reports_whether_an_answer_came_from_the_cache():
    pipeline, _ = _pipeline(SemanticCache(HashEmbedder(), threshold=0.95))
    with TestClient(build_app(pipeline=pipeline)) as client:
        first = client.post("/query", json={"question": "What fines apply for violations?"})
        second = client.post("/query", json={"question": "What fines apply for violations?"})
    assert first.json()["cached"] is False
    assert second.json()["cached"] is True


def test_cache_is_disabled_by_default():
    assert CacheSettings().enabled is False

"""Offline telemetry tests: in-memory span exporter and metric reader, so no
collector, network, or OTLP endpoint is involved."""

import httpx
import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from evalgate_rag.chunking import chunk_recursive
from evalgate_rag.config import LangfuseSettings, LLMSettings, TelemetrySettings
from evalgate_rag.embeddings import HashEmbedder
from evalgate_rag.pipeline import LLMClient, RAGPipeline, Tracer
from evalgate_rag.retrieval import HybridRetriever
from evalgate_rag.store import InMemoryStore
from evalgate_rag.telemetry import Telemetry, make_telemetry

SECRET_TEXT = "Fines of up to 35 million EUR apply."


class Recorder:
    """A Telemetry wired to in-memory exporters, plus helpers to read them."""

    def __init__(self) -> None:
        self._spans = InMemorySpanExporter()
        self._reader = InMemoryMetricReader()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(self._spans))
        self.telemetry = Telemetry(tracer_provider, MeterProvider(metric_readers=[self._reader]))

    def spans(self) -> dict:
        return {s.name: s for s in self._spans.get_finished_spans()}

    def points(self, metric_name: str) -> list:
        data = self._reader.get_metrics_data()
        return [
            point
            for rm in (data.resource_metrics if data else [])
            for sm in rm.scope_metrics
            for metric in sm.metrics
            if metric.name == metric_name
            for point in metric.data.data_points
        ]


@pytest.fixture()
def rec() -> Recorder:
    return Recorder()


def _chat_response(content: str = "answer [Article 99]") -> dict:
    return {
        "model": "served-model-v2",
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
    }


def _llm(rec: Recorder, handler) -> LLMClient:
    return LLMClient(
        LLMSettings(base_url="https://llm.example/v1", model="test-model"),
        transport=httpx.MockTransport(handler),
        telemetry=rec.telemetry,
    )


def _pipeline(rec: Recorder, handler) -> RAGPipeline:
    embedder = HashEmbedder()
    store = InMemoryStore()
    chunks = chunk_recursive(SECRET_TEXT, "Article 99", max_chars=500)
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    retriever = HybridRetriever(store, embedder)
    retriever.refresh_bm25()
    return RAGPipeline(
        retriever,
        _llm(rec, handler),
        Tracer(LangfuseSettings()),
        top_k=2,
        telemetry=rec.telemetry,
    )


def test_query_produces_one_trace_with_retrieval_and_llm_children(rec):
    pipeline = _pipeline(rec, lambda request: httpx.Response(200, json=_chat_response()))
    pipeline.answer("What fines apply?")

    spans = rec.spans()
    assert set(spans) == {"rag.query", "rag.retrieval", "chat test-model"}
    root = spans["rag.query"]
    assert root.parent is None
    assert root.attributes["rag.top_k"] == 2
    for child in ("rag.retrieval", "chat test-model"):
        assert spans[child].parent.span_id == root.context.span_id
        assert spans[child].context.trace_id == root.context.trace_id
    assert spans["rag.retrieval"].attributes["rag.retrieval.returned"] == 1


def test_llm_span_follows_genai_conventions_and_records_usage(rec):
    _llm(rec, lambda request: httpx.Response(200, json=_chat_response())).chat("sys", "user")

    span = rec.spans()["chat test-model"]
    assert span.kind is SpanKind.CLIENT
    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["gen_ai.request.model"] == "test-model"
    assert span.attributes["gen_ai.response.model"] == "served-model-v2"
    assert span.attributes["gen_ai.usage.input_tokens"] == 120
    assert span.attributes["gen_ai.usage.output_tokens"] == 30
    assert span.attributes["server.address"] == "llm.example"

    tokens = {p.attributes["gen_ai.token.type"]: p.sum for p in rec.points(TOKEN_USAGE)}
    assert tokens == {"input": 120, "output": 30}
    (duration,) = rec.points(LLM_DURATION)
    assert duration.count == 1
    assert "error.type" not in duration.attributes


def test_no_prompt_context_or_answer_text_reaches_a_span(rec):
    pipeline = _pipeline(rec, lambda request: httpx.Response(200, json=_chat_response("SECRET")))
    pipeline.answer("What fines apply for violations?")

    recorded = " ".join(
        str(value)
        for span in rec.spans().values()
        for source in (span.attributes, *(event.attributes for event in span.events))
        for value in source.values()
    )
    for leaked in ("What fines apply", SECRET_TEXT, "SECRET"):
        assert leaked not in recorded


def test_rate_limit_retry_is_counted_and_marked_on_the_span(rec, monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "slow down"})
        return httpx.Response(200, json=_chat_response())

    monkeypatch.setattr("evalgate_rag.pipeline.time.sleep", lambda s: None)
    _llm(rec, handler).chat("sys", "user")

    (retries,) = rec.points("rag.llm.retries")
    assert retries.value == 1
    assert retries.attributes["reason"] == "rate_limited"
    span = rec.spans()["chat test-model"]
    assert [event.name for event in span.events] == ["rate_limited"]
    assert span.status.status_code is not StatusCode.ERROR  # it recovered
    assert rec.points(LLM_DURATION)[0].count == 1  # one call, however many attempts


def test_upstream_failure_marks_spans_and_metrics_with_error_type(rec):
    pipeline = _pipeline(rec, lambda request: httpx.Response(500, json={"error": "boom"}))
    with pytest.raises(httpx.HTTPStatusError):
        pipeline.answer("What fines apply?")

    spans = rec.spans()
    assert spans["rag.query"].status.status_code is StatusCode.ERROR
    assert spans["chat test-model"].status.status_code is StatusCode.ERROR
    assert spans["chat test-model"].attributes["error.type"] == "500"
    assert rec.points(LLM_DURATION)[0].attributes["error.type"] == "500"
    assert rec.points(QUERY_DURATION)[0].attributes["error.type"] == "HTTPStatusError"
    assert rec.points(TOKEN_USAGE) == []  # no usage is invented for a failed call


def test_default_telemetry_is_a_noop():
    telemetry = make_telemetry(TelemetrySettings(enabled=False))
    with telemetry.span("anything") as span:
        assert not span.is_recording()
    telemetry.record_query(0.1)
    telemetry.record_llm_call(0.1, {}, input_tokens=1, output_tokens=1)
    telemetry.shutdown()


def test_enabled_telemetry_builds_otlp_exporters_without_connecting():
    # Construction must not touch the network: export is batched in the
    # background, so a collector that is down cannot stop the API from booting.
    telemetry = make_telemetry(
        TelemetrySettings(enabled=True, otlp_endpoint="http://127.0.0.1:9/", export_interval_s=3600)
    )
    # Checked on the provider rather than by opening a span, which would queue
    # an export to the dead endpoint and spend the test retrying it.
    assert isinstance(telemetry._tracer_provider, TracerProvider)
    assert isinstance(telemetry._meter_provider, MeterProvider)


QUERY_DURATION = "rag.query.duration"
LLM_DURATION = "gen_ai.client.operation.duration"
TOKEN_USAGE = "gen_ai.client.token.usage"


@pytest.mark.parametrize("stage", ["retrieval", "llm"])
def test_exception_content_never_reaches_telemetry(rec, stage, monkeypatch):
    secret = "PRIVATE_QUESTION_CONTEXT_ANSWER"
    original = RuntimeError(secret)

    def fail(*args, **kwargs):
        raise original

    pipeline = _pipeline(rec, fail if stage == "llm" else lambda r: httpx.Response(200))
    if stage == "retrieval":
        monkeypatch.setattr(pipeline._retriever, "retrieve", fail)
    with pytest.raises(RuntimeError) as raised:
        pipeline.answer(secret)
    assert raised.value is original
    for span in rec.spans().values():
        assert secret not in span.to_json()
        if stage == "llm" and span.name == "rag.retrieval":
            assert span.status.status_code is StatusCode.UNSET
        else:
            assert span.status.status_code is StatusCode.ERROR
            assert span.attributes["error.type"] == "RuntimeError"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": ["not text"]}}]},
    ],
)
def test_malformed_success_response_is_an_observed_failure(rec, body):
    pipeline = _pipeline(rec, lambda request: httpx.Response(200, json=body))
    with pytest.raises(ValueError, match="Invalid chat completion response"):
        pipeline.answer("What fines apply?")
    assert rec.points(LLM_DURATION)[0].attributes["error.type"] == "ValueError"
    assert rec.points(QUERY_DURATION)[0].attributes["error.type"] == "ValueError"
    assert rec.spans()["chat test-model"].status.status_code is StatusCode.ERROR

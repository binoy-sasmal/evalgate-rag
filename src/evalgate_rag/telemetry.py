"""OpenTelemetry traces and metrics for the query path.

Langfuse (see `pipeline.Tracer`) answers "what did the model see and say";
this answers "how long did it take, what did it cost, and how often did it
fail" in the vendor-neutral form an operations stack already speaks. The same
OTLP stream feeds the local Prometheus/Tempo/Grafana profile in
docker-compose and any managed backend that accepts OTLP.

Telemetry is a no-op unless TELEMETRY__ENABLED=true. Providers are passed in
rather than read from OpenTelemetry's process globals, so tests hand over
in-memory exporters and nothing leaks between them.

Prompts, retrieved text and answers are never put on a span: the trace
backend is an operations tool with a wider audience than the Langfuse
project, and token counts and timings are what it needs.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.metrics import MeterProvider
from opentelemetry.trace import Span, SpanKind, StatusCode, TracerProvider

from .config import TelemetrySettings

_SCOPE = "evalgate_rag"

# Bucket boundaries recommended by the OpenTelemetry GenAI semantic
# conventions. The SDK default tops out at 10 s, which would flatten every
# rate-limited call (a single retry can wait up to 60 s) into one bucket.
_DURATION_BUCKETS_S = [
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
    10.24,
    20.48,
    40.96,
    81.92,
]
_TOKEN_BUCKETS = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]


class Telemetry:
    """Spans and instruments for the RAG query path. No-op by default."""

    def __init__(
        self,
        tracer_provider: TracerProvider | None = None,
        meter_provider: MeterProvider | None = None,
    ) -> None:
        self._tracer_provider = tracer_provider
        self._meter_provider = meter_provider
        self._tracer = (tracer_provider or trace.NoOpTracerProvider()).get_tracer(_SCOPE)
        meter = (meter_provider or metrics.NoOpMeterProvider()).get_meter(_SCOPE)

        self._query_duration = meter.create_histogram(
            "rag.query.duration",
            unit="s",
            description="End-to-end time to answer one question",
            explicit_bucket_boundaries_advisory=_DURATION_BUCKETS_S,
        )
        self._retrieval_duration = meter.create_histogram(
            "rag.retrieval.duration",
            unit="s",
            description="Hybrid retrieval time (embed + dense + BM25 + fusion)",
            explicit_bucket_boundaries_advisory=_DURATION_BUCKETS_S,
        )
        # The next two names and their attributes follow the GenAI semantic
        # conventions, so dashboards built for any conforming client work here.
        self._llm_duration = meter.create_histogram(
            "gen_ai.client.operation.duration",
            unit="s",
            description="LLM call time, including rate-limit retries",
            explicit_bucket_boundaries_advisory=_DURATION_BUCKETS_S,
        )
        self._llm_tokens = meter.create_histogram(
            "gen_ai.client.token.usage",
            unit="{token}",
            description="Tokens per LLM call, split by gen_ai.token.type",
            explicit_bucket_boundaries_advisory=_TOKEN_BUCKETS,
        )
        self._llm_retries = meter.create_counter(
            "rag.llm.retries",
            unit="{retry}",
            description="LLM requests retried after a rate-limit response",
        )

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        """Mark failures without exporting exception messages or stack traces."""
        with self._tracer.start_as_current_span(
            name,
            kind=kind,
            attributes=attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                yield span
            except Exception as exc:
                # HTTP callers may already have supplied a status code.
                if not (getattr(span, "attributes", None) or {}).get("error.type"):
                    span.set_attribute("error.type", type(exc).__name__)
                span.set_status(StatusCode.ERROR)
                raise

    def record_query(self, duration_s: float, error_type: str | None = None) -> None:
        self._query_duration.record(duration_s, _with_error({}, error_type))

    def record_retrieval(self, duration_s: float) -> None:
        self._retrieval_duration.record(duration_s)

    def record_llm_call(
        self,
        duration_s: float,
        attributes: dict[str, Any],
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        error_type: str | None = None,
    ) -> None:
        self._llm_duration.record(duration_s, _with_error(attributes, error_type))
        for token_type, count in (("input", input_tokens), ("output", output_tokens)):
            if count is not None:
                self._llm_tokens.record(count, {**attributes, "gen_ai.token.type": token_type})

    def record_llm_retry(self, attributes: dict[str, Any], reason: str) -> None:
        self._llm_retries.add(1, {**attributes, "reason": reason})

    def shutdown(self) -> None:
        """Flush and stop the exporters. Without this the last export interval
        of metrics and any batched spans are lost when the process exits."""
        for provider in (self._tracer_provider, self._meter_provider):
            shutdown = getattr(provider, "shutdown", None)
            if shutdown is not None:
                shutdown()


def _with_error(attributes: dict[str, Any], error_type: str | None) -> dict[str, Any]:
    return {**attributes, "error.type": error_type} if error_type else attributes


def make_telemetry(cfg: TelemetrySettings) -> Telemetry:
    """Build OTLP-exporting telemetry, or the no-op when disabled."""
    if not cfg.enabled:
        return Telemetry()

    # Lazy: the SDK and exporter are an optional extra (`.[telemetry]`).
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider as SdkMeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    # With no endpoint configured the exporters fall back to the standard
    # OTEL_EXPORTER_OTLP_* variables, which is how a managed collector (for
    # example the Azure Container Apps OpenTelemetry agent) injects its address.
    base = cfg.otlp_endpoint.rstrip("/") if cfg.otlp_endpoint else None
    span_exporter = OTLPSpanExporter(endpoint=f"{base}/v1/traces" if base else None)
    metric_exporter = OTLPMetricExporter(endpoint=f"{base}/v1/metrics" if base else None)

    resource = Resource.create({"service.name": cfg.service_name})
    tracer_provider = SdkTracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    meter_provider = SdkMeterProvider(
        resource=resource,
        metric_readers=[
            PeriodicExportingMetricReader(
                metric_exporter, export_interval_millis=cfg.export_interval_s * 1000
            )
        ],
    )
    return Telemetry(tracer_provider, meter_provider)

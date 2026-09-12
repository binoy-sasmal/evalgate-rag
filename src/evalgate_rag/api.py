"""FastAPI service. Wiring lives in build_app() so tests can inject fakes."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .embeddings import make_embedder
from .pipeline import LLMClient, RAGPipeline, Tracer
from .retrieval import HybridRetriever
from .store import PgVectorStore


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class ContextOut(BaseModel):
    doc_id: str
    seq: int
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    contexts: list[ContextOut]
    trace_id: str | None


def build_app(settings: Settings | None = None, pipeline: RAGPipeline | None = None) -> FastAPI:
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = None
        if pipeline is not None:
            app.state.pipeline = pipeline
        else:
            embedder = make_embedder(cfg.embedding)
            store = PgVectorStore(cfg.db.dsn, dimension=embedder.dimension)
            retriever = HybridRetriever(store, embedder, rrf_k=cfg.rrf_k)
            retriever.refresh_bm25()
            app.state.store = store
            app.state.pipeline = RAGPipeline(
                retriever,
                LLMClient(cfg.llm),
                Tracer(cfg.langfuse),
                top_k=cfg.retrieval_top_k,
            )
        yield
        if app.state.store is not None:
            app.state.store.close()

    app = FastAPI(title="evalgate-rag", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        """Liveness only: the process is up. Deliberately does no I/O so a
        database blip never gets the container killed and restarted."""
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict:
        """Readiness: the store is reachable *and* has been ingested.

        Separate from /health because the two failures want opposite responses.
        An empty store is not a crash -- retrieval returns nothing and the model
        dutifully answers "I cannot answer this from the provided context" for
        every question, so a deployment where ingest never ran looks like a bad
        model rather than a missing step. Failing readiness makes it obvious.
        """
        store = app.state.store
        if store is None:  # injected pipeline (tests): nothing to probe
            return {"status": "ready", "chunks": None}
        try:
            chunks = store.count()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"store unreachable: {exc}") from exc
        if chunks == 0:
            raise HTTPException(
                status_code=503,
                detail="store is empty -- run scripts/ingest.py against this database",
            )
        return {"status": "ready", "chunks": chunks}

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        pl: RAGPipeline = app.state.pipeline
        try:
            result = pl.answer(req.question, top_k=req.top_k)
        except Exception as exc:  # surface upstream failures as 502
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return QueryResponse(
            answer=result.answer,
            contexts=[
                ContextOut(doc_id=c.chunk.doc_id, seq=c.chunk.seq, score=c.score, text=c.chunk.text)
                for c in result.contexts
            ],
            trace_id=result.trace_id,
        )

    return app


app = build_app()

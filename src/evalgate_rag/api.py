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
        if pipeline is not None:
            app.state.pipeline = pipeline
        else:
            embedder = make_embedder(cfg.embedding)
            store = PgVectorStore(cfg.db.dsn, dimension=embedder.dimension)
            retriever = HybridRetriever(store, embedder, rrf_k=cfg.rrf_k)
            retriever.refresh_bm25()
            app.state.pipeline = RAGPipeline(
                retriever,
                LLMClient(cfg.llm),
                Tracer(cfg.langfuse),
                top_k=cfg.retrieval_top_k,
            )
        yield

    app = FastAPI(title="evalgate-rag", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        pl: RAGPipeline = app.state.pipeline
        if req.top_k is not None:
            pl._top_k = req.top_k  # per-request override
        try:
            result = pl.answer(req.question)
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

"""Cross-encoder reranking, run locally.

A bi-encoder (the dense leg) embeds the question and each chunk separately and
compares the vectors; a cross-encoder reads the question and one chunk
together and scores how well that chunk answers it. That is far more
accurate per pair and far too slow to run over the whole corpus, so it only
reorders the pool hybrid retrieval already narrowed down (see retrieval.py).

Runs on fastembed's ONNX runtime, the same engine as the embeddings, so it
needs no torch and no API: reranking costs CPU time, never tokens.
"""

from __future__ import annotations

from .config import RerankSettings


class CrossEncoderReranker:
    def __init__(self, model: str, cache_dir: str | None = None) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # lazy: optional dependency

        self._model = TextCrossEncoder(model, cache_dir=cache_dir)

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, texts)]


def make_reranker(cfg: RerankSettings) -> CrossEncoderReranker | None:
    return CrossEncoderReranker(cfg.model, cfg.cache_dir) if cfg.enabled else None

"""Embedding providers behind one protocol.

- OpenAIEmbedder   — any OpenAI-compatible /embeddings endpoint
- FastEmbedEmbedder — local ONNX BGE model (pip install .[embed-local])
- HashEmbedder     — deterministic, offline; used in unit tests and CI smoke runs
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol

import httpx
import numpy as np

from .config import EmbeddingSettings


class Embedder(Protocol):
    dimension: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class OpenAIEmbedder:
    def __init__(self, cfg: EmbeddingSettings) -> None:
        self._cfg = cfg
        self.dimension = cfg.dimension
        self._client = httpx.Client(
            base_url=cfg.base_url,
            headers={"Authorization": f"Bearer {cfg.api_key}"},
            timeout=30.0,
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        resp = self._client.post(
            "/embeddings", json={"model": self._cfg.model, "input": list(texts)}
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return np.array([d["embedding"] for d in data], dtype=np.float32)


class FastEmbedEmbedder:
    """Local, torch-free embeddings via fastembed (BAAI/bge-small-en-v1.5)."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding  # lazy: optional dependency

        self._model = TextEmbedding(model)
        self.dimension = 384

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.array(list(self._model.embed(list(texts))), dtype=np.float32)


class HashEmbedder:
    """Deterministic bag-of-hashed-tokens embedding. Offline; for tests only.

    Not semantically meaningful, but identical tokens map to identical
    dimensions, so lexical overlap produces cosine similarity — enough to
    exercise retrieval logic end-to-end without a model or network.
    """

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                h = int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little")
                out[i, h % self.dimension] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


def make_embedder(cfg: EmbeddingSettings) -> Embedder:
    if cfg.provider == "openai":
        return OpenAIEmbedder(cfg)
    if cfg.provider == "fastembed":
        return FastEmbedEmbedder()
    if cfg.provider == "hash":
        return HashEmbedder()
    raise ValueError(f"Unknown embedding provider: {cfg.provider}")

"""Three chunking strategies benchmarked in scripts/benchmark_chunking.py.

1. fixed      — fixed-size character windows with overlap
2. recursive  — split on paragraph > sentence > word boundaries, greedily packed
3. semantic   — greedy packing that starts a new chunk when the embedding
                similarity between consecutive sentences drops below a threshold
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Chunk:
    text: str
    doc_id: str
    seq: int
    meta: dict = field(default_factory=dict)


_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(\d])")


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_RE.split(text)]
    return [s for s in parts if s]


def chunk_fixed(text: str, doc_id: str, size: int = 1000, overlap: int = 200) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks: list[Chunk] = []
    step = size - overlap
    for seq, start in enumerate(range(0, max(len(text), 1), step)):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(Chunk(text=piece, doc_id=doc_id, seq=seq))
        if start + size >= len(text):
            break
    return chunks


def chunk_recursive(text: str, doc_id: str, max_chars: int = 1000) -> list[Chunk]:
    """Greedy packing of paragraph/sentence units up to max_chars."""
    units: list[str] = []
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
        else:
            units.extend(split_sentences(para))

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    seq = 0
    for unit in units:
        if buf and buf_len + len(unit) + 1 > max_chars:
            chunks.append(Chunk(text=" ".join(buf), doc_id=doc_id, seq=seq))
            seq += 1
            buf, buf_len = [], 0
        # a single unit longer than max_chars falls back to fixed windows
        if len(unit) > max_chars:
            for piece in chunk_fixed(unit, doc_id, size=max_chars, overlap=0):
                chunks.append(Chunk(text=piece.text, doc_id=doc_id, seq=seq))
                seq += 1
            continue
        buf.append(unit)
        buf_len += len(unit) + 1
    if buf:
        chunks.append(Chunk(text=" ".join(buf), doc_id=doc_id, seq=seq))
    return chunks


def chunk_semantic(
    text: str,
    doc_id: str,
    embed_fn: Callable[[Sequence[str]], np.ndarray],
    max_chars: int = 1200,
    breakpoint_percentile: float = 25.0,
) -> list[Chunk]:
    """Start a new chunk where consecutive-sentence similarity falls into the
    bottom `breakpoint_percentile` of the document's similarity distribution."""
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [Chunk(text=text.strip(), doc_id=doc_id, seq=0)] if text.strip() else []

    vecs = embed_fn(sentences)
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
    threshold = float(np.percentile(sims, breakpoint_percentile))

    chunks: list[Chunk] = []
    buf: list[str] = [sentences[0]]
    seq = 0
    for i, sent in enumerate(sentences[1:]):
        boundary = sims[i] <= threshold  # <=: with few sentences the percentile can equal the min
        too_long = sum(len(s) for s in buf) + len(sent) > max_chars
        if boundary or too_long:
            chunks.append(Chunk(text=" ".join(buf), doc_id=doc_id, seq=seq))
            seq += 1
            buf = []
        buf.append(sent)
    if buf:
        chunks.append(Chunk(text=" ".join(buf), doc_id=doc_id, seq=seq))
    return chunks


STRATEGIES = ("fixed", "recursive", "semantic")

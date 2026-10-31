"""Semantic answer cache: serve a stored answer for a near-duplicate question.

A repeated or reworded question ("What practices does the AI Act prohibit?"
after "Which AI practices are prohibited under the AI Act?") costs a full LLM
call and a share of the provider's daily token quota for an answer the
service has already produced. This cache embeds each answered question and,
on a later question whose embedding is close enough, returns the stored
answer and contexts without calling the model.

The risk is the opposite error: a *false hit* serves a confident answer to a
different question. Embeddings rate "What does Article 5 prohibit?" and
"What does Article 6 prohibit?" as near-identical, and the same holds for a
flipped negation. Three guards keep a hit honest:

1. **Anchors must match exactly.** Numbers and roman numerals that follow a
   legal unit (Article 5, Annex III, Article 6(2)) are extracted from both
   questions; any difference is a miss, whatever the similarity.
2. **Negation must match.** A question with "not", "never", "without" ...
   never matches one without.
3. **Namespaces.** Entries are keyed by the model, a hash of the prompt and
   the retrieval top_k (see `pipeline.cache_namespace`), so changing any of
   them can never serve an answer produced under the old configuration.

Refusals and answers with no retrieved context are never stored, entries
expire after `ttl_s`, and the cache is LRU-bounded at `max_entries`. It is
in-process and per-replica by design: nothing persists across restarts.

The similarity threshold is chosen by measurement, not by feel:
`scripts/benchmark_semantic_cache.py` sweeps it over labelled paraphrase and
near-miss pairs and reports hit rate against false-hit rate.

The evaluation gate never uses this cache: `eval/run_eval.py` builds its
pipeline without one, so every golden-set question reaches the model.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .embeddings import Embedder
from .store import ScoredChunk

REFUSAL_MARKER = "cannot answer this from the provided context"

_UNIT = r"(?:article|articles|annex|annexes|recital|recitals|chapter|title|section|paragraph|point)"
# The closing lookahead, not \b: after "Article 6(2)" the next character is a
# space, and there is no word boundary between ")" and " ".
_ANCHOR_RE = re.compile(
    rf"\b{_UNIT}\s+([0-9]+(?:\s*\([0-9a-z]+\))*|[ivxlc]+)(?![0-9a-z])", re.IGNORECASE
)
_NUMBER_RE = re.compile(r"[0-9]+|[ivxlc]+", re.IGNORECASE)
_NEGATIONS = frozenset({"not", "no", "never", "without", "except", "cannot", "isn't", "aren't"})
_WORD_RE = re.compile(r"[a-z']+")


def anchors(question: str) -> frozenset[str]:
    """Numbers and roman numerals attached to legal units, lower-cased.

    "Compare Article 6(2) and Article 50" -> {"6", "2", "50"}.
    """
    found: set[str] = set()
    for match in _ANCHOR_RE.finditer(question):
        found.update(token.lower() for token in _NUMBER_RE.findall(match.group(1)))
    return frozenset(found)


def is_negated(question: str) -> bool:
    words = set(_WORD_RE.findall(question.lower()))
    return bool(words & _NEGATIONS) or "n't" in question.lower()


@dataclass(frozen=True)
class CacheHit:
    answer: str
    contexts: list[ScoredChunk]
    similarity: float


@dataclass
class _Entry:
    namespace: str
    vector: np.ndarray
    anchors: frozenset[str]
    negated: bool
    answer: str
    contexts: list[ScoredChunk]
    expires_at: float


class SemanticCache:
    """Thread-safe, in-process, LRU- and TTL-bounded semantic answer cache."""

    def __init__(
        self,
        embedder: Embedder,
        threshold: float = 0.96,
        max_entries: int = 1000,
        ttl_s: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")
        if ttl_s <= 0:
            raise ValueError(f"ttl_s must be > 0, got {ttl_s}")
        self._embedder = embedder
        self.threshold = threshold
        self._max_entries = max_entries
        self._ttl_s = ttl_s
        self._clock = clock
        self._entries: OrderedDict[int, _Entry] = OrderedDict()
        self._next_id = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _embed(self, question: str) -> np.ndarray:
        vec = np.asarray(self._embedder.embed([question])[0], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def _purge_expired(self, now: float) -> None:
        expired = [key for key, entry in self._entries.items() if entry.expires_at <= now]
        for key in expired:
            del self._entries[key]

    def lookup(self, question: str, namespace: str) -> CacheHit | None:
        """Best stored answer for `question` in `namespace`, or None."""
        vector = self._embed(question)  # outside the lock: embedding is the slow part
        q_anchors = anchors(question)
        q_negated = is_negated(question)
        with self._lock:
            self._purge_expired(self._clock())
            best_key: int | None = None
            best_sim = -1.0
            for key, entry in self._entries.items():
                if (
                    entry.namespace != namespace
                    or entry.anchors != q_anchors
                    or entry.negated != q_negated
                ):
                    continue
                sim = float(np.dot(vector, entry.vector))
                if sim > best_sim:
                    best_key, best_sim = key, sim
            if best_key is None or best_sim < self.threshold:
                return None
            self._entries.move_to_end(best_key)
            hit = self._entries[best_key]
            return CacheHit(answer=hit.answer, contexts=list(hit.contexts), similarity=best_sim)

    def store(
        self, question: str, namespace: str, answer: str, contexts: list[ScoredChunk]
    ) -> bool:
        """Remember an answer. Returns False when it is deliberately not cached."""
        if not contexts or REFUSAL_MARKER in answer.lower():
            return False
        entry = _Entry(
            namespace=namespace,
            vector=self._embed(question),
            anchors=anchors(question),
            negated=is_negated(question),
            answer=answer,
            contexts=list(contexts),
            expires_at=self._clock() + self._ttl_s,
        )
        with self._lock:
            self._entries[self._next_id] = entry
            self._next_id += 1
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return True

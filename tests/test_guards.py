"""Offline guard tests for the ingest / fetch / eval scripts.

These lock in the defence-in-depth added after a broken corpus fetch silently
produced an empty store, which slipped past ingest, past retrieval (which
degrades to no-context by design), and only surfaced as a total eval-score
collapse at the final gate. Each layer must now fail loudly at its true cause.

scripts/ and eval/ aren't importable packages (only src/evalgate_rag is), so
we load them by file path. Everything here stays offline -- HashEmbedder,
InMemoryStore, and fakes only; no network, no database.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import types

import pytest

from evalgate_rag.embeddings import HashEmbedder
from evalgate_rag.store import InMemoryStore

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(relpath: str, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, _ROOT / relpath)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_doc(dir_: pathlib.Path, doc_id: str, text: str) -> None:
    safe = doc_id.replace(" ", "_").lower()
    (dir_ / f"{safe}.json").write_text(
        json.dumps({"doc_id": doc_id, "title": doc_id, "text": text}, ensure_ascii=False),
        encoding="utf-8",
    )


# --------------------------------------------------------------- ingest guard


def test_ingest_corpus_chunks_and_upserts_docs(tmp_path):
    ingest = _load("scripts/ingest.py", "ingest_script")
    _write_doc(tmp_path, "Article 5", "Prohibited AI practices. " * 30)
    _write_doc(tmp_path, "Article 99", "Fines up to 35 million EUR. " * 30)

    store = InMemoryStore()
    total = ingest.ingest_corpus(tmp_path, "recursive", HashEmbedder(), store)

    assert total > 0
    assert len(store.all_chunks()) == total


def test_ingest_corpus_empty_dir_returns_zero(tmp_path):
    ingest = _load("scripts/ingest.py", "ingest_script")
    store = InMemoryStore()
    assert ingest.ingest_corpus(tmp_path, "recursive", HashEmbedder(), store) == 0
    assert store.all_chunks() == []


def test_ingest_main_exits_on_empty_corpus(tmp_path, monkeypatch, capsys):
    ingest = _load("scripts/ingest.py", "ingest_script")
    monkeypatch.setattr(ingest, "CORPUS_DIR", tmp_path)  # empty -> 0 chunks
    monkeypatch.setattr(ingest, "make_embedder", lambda cfg: HashEmbedder())
    monkeypatch.setattr("evalgate_rag.store.PgVectorStore", lambda *a, **k: InMemoryStore())
    monkeypatch.setattr(sys, "argv", ["ingest.py", "--strategy", "recursive"])

    with pytest.raises(SystemExit) as exc:
        ingest.main()
    assert exc.value.code == 1
    assert "ingested 0 chunks" in capsys.readouterr().err


def test_ingest_main_succeeds_on_populated_corpus(tmp_path, monkeypatch):
    ingest = _load("scripts/ingest.py", "ingest_script")
    _write_doc(tmp_path, "Article 5", "Prohibited AI practices. " * 30)
    monkeypatch.setattr(ingest, "CORPUS_DIR", tmp_path)
    monkeypatch.setattr(ingest, "make_embedder", lambda cfg: HashEmbedder())
    monkeypatch.setattr("evalgate_rag.store.PgVectorStore", lambda *a, **k: InMemoryStore())
    monkeypatch.setattr(sys, "argv", ["ingest.py", "--strategy", "recursive"])

    ingest.main()  # must not raise SystemExit


# --------------------------------------------------- run_eval empty-store guard


def test_require_populated_store_rejects_empty(capsys):
    run_eval = _load("eval/run_eval.py", "run_eval_script")
    with pytest.raises(SystemExit) as exc:
        run_eval.require_populated_store(0, min_chunks=50)
    assert exc.value.code == 1
    assert "empty store" in capsys.readouterr().err


def test_require_populated_store_rejects_below_threshold():
    run_eval = _load("eval/run_eval.py", "run_eval_script")
    with pytest.raises(SystemExit):
        run_eval.require_populated_store(49, min_chunks=50)


def test_require_populated_store_accepts_populated():
    run_eval = _load("eval/run_eval.py", "run_eval_script")
    run_eval.require_populated_store(50, min_chunks=50)  # must not raise
    run_eval.require_populated_store(500, min_chunks=50)


# --------------------------------------------------------------- fetch guard


_ARTICLE_HTML = "".join(
    f"<p>Article {n}</p><p>{'Body of the article text. ' * 20}</p>" for n in range(1, 60)
)
_WAF_HTML = "<html><body><h1>Just a moment...</h1><p>Checking your browser</p></body></html>"


def test_split_documents_parses_articles():
    fetch = _load("scripts/fetch_corpus.py", "fetch_script")
    docs = fetch.split_documents(_ARTICLE_HTML)
    assert len(docs) >= fetch.MIN_DOCS
    assert docs[0]["doc_id"] == "Article 1"


def test_split_documents_on_waf_page_yields_nothing():
    fetch = _load("scripts/fetch_corpus.py", "fetch_script")
    assert fetch.split_documents(_WAF_HTML) == []


def test_fetch_main_exits_on_waf_page(monkeypatch, capsys):
    fetch = _load("scripts/fetch_corpus.py", "fetch_script")
    monkeypatch.setattr(fetch, "fetch_html", lambda: _WAF_HTML)

    with pytest.raises(SystemExit) as exc:
        fetch.main()
    assert exc.value.code == 1
    assert "WAF" in capsys.readouterr().err


def test_fetch_main_writes_corpus_on_good_page(tmp_path, monkeypatch):
    fetch = _load("scripts/fetch_corpus.py", "fetch_script")
    monkeypatch.setattr(fetch, "fetch_html", lambda: _ARTICLE_HTML)
    monkeypatch.setattr(fetch, "OUT_DIR", tmp_path)

    fetch.main()  # must not raise
    assert len(list(tmp_path.glob("*.json"))) >= fetch.MIN_DOCS

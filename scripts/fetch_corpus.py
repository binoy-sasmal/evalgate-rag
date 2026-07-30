"""Download the EU AI Act (Regulation (EU) 2024/1689) from EUR-Lex and split
it into one JSON document per article/annex.

Usage:
    python scripts/fetch_corpus.py            # writes data/corpus/*.json

The consolidated HTML is public; ~113 articles + 13 annexes gives a corpus
of roughly 200 documents once long annexes are split by section.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import httpx
from bs4 import BeautifulSoup

EURLEX_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689"
OUT_DIR = pathlib.Path("data/corpus")

# The consolidated Act yields ~125 article/annex documents. A fetch that
# parses far fewer means we got something other than the Act -- most often a
# WAF/consent challenge page that still returns HTTP 200, so raise_for_status()
# doesn't catch it. Refuse to overwrite a good committed corpus with garbage.
MIN_DOCS = 50

ARTICLE_RE = re.compile(r"^Article\s+(\d+[a-z]?)\b", re.IGNORECASE)
ANNEX_RE = re.compile(r"^ANNEX\s+([IVXL]+)\b", re.IGNORECASE)


def fetch_html() -> str:
    resp = httpx.get(
        EURLEX_URL,
        timeout=120.0,
        follow_redirects=True,
        headers={"User-Agent": "evalgate-rag corpus fetcher"},
    )
    resp.raise_for_status()
    return resp.text


def split_documents(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")
    lines = [ln.strip() for ln in text.splitlines()]

    docs: list[dict] = []
    current_id: str | None = None
    current_title: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        if current_id and buf:
            body = "\n".join(ln for ln in buf if ln)
            if len(body) > 200:  # skip heading-only fragments
                docs.append({"doc_id": current_id, "title": current_title or "", "text": body})
        buf = []

    for ln in lines:
        m_art, m_annex = ARTICLE_RE.match(ln), ANNEX_RE.match(ln)
        if m_art:
            flush()
            current_id, current_title = f"Article {m_art.group(1)}", ln
        elif m_annex:
            flush()
            current_id, current_title = f"Annex {m_annex.group(1)}", ln
        elif current_id:
            buf.append(ln)
    flush()
    return docs


def main() -> None:
    docs = split_documents(fetch_html())
    if len(docs) < MIN_DOCS:
        print(
            f"ERROR: parsed only {len(docs)} documents (expected >= {MIN_DOCS}). "
            "EUR-Lex likely returned a WAF/consent page instead of the Act. "
            "Leaving the existing data/corpus/ untouched.",
            file=sys.stderr,
        )
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        safe = doc["doc_id"].replace(" ", "_").lower()
        (OUT_DIR / f"{safe}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"Wrote {len(docs)} documents to {OUT_DIR}/")


if __name__ == "__main__":
    main()

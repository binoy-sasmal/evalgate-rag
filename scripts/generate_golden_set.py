"""Draft additional golden-set candidates from the corpus using the LLM.

The 20 hand-curated pairs in data/golden_set.jsonl are the trusted core.
This script drafts candidates for the remaining ~30 by sampling corpus
documents and asking the LLM for one question whose answer is fully
contained in that document. ALL generated pairs are written to
data/golden_set_candidates.jsonl and MUST be human-reviewed before being
promoted into golden_set.jsonl — never gate CI on unreviewed ground truth.

Usage:
    python scripts/generate_golden_set.py --n 30
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

from evalgate_rag.config import get_settings
from evalgate_rag.pipeline import LLMClient

CORPUS_DIR = pathlib.Path("data/corpus")
OUT = pathlib.Path("data/golden_set_candidates.jsonl")

PROMPT = """Read the following excerpt from the EU AI Act ({doc_id}).

Write ONE factual question that a compliance officer might ask, whose answer
is fully contained in this excerpt, and the answer itself.

Respond as JSON only: {{"question": "...", "ground_truth": "..."}}

Excerpt:
{text}"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    llm = LLMClient(get_settings().llm)
    docs = sorted(CORPUS_DIR.glob("*.json"))
    random.Random(args.seed).shuffle(docs)

    existing = {
        json.loads(ln)["gold_doc_id"]
        for ln in pathlib.Path("data/golden_set.jsonl").read_text().splitlines()
        if ln.strip()
    }

    written = 0
    with OUT.open("w") as fh:
        for path in docs:
            if written >= args.n:
                break
            doc = json.loads(path.read_text())
            if doc["doc_id"] in existing:
                continue  # prefer coverage of articles not yet in the golden set
            raw = llm.chat(
                "You draft evaluation questions. Respond with JSON only.",
                PROMPT.format(doc_id=doc["doc_id"], text=doc["text"][:4000]),
            )
            try:
                pair = json.loads(raw.strip().removeprefix("```json").removesuffix("```"))
            except json.JSONDecodeError:
                continue
            fh.write(
                json.dumps(
                    {
                        "id": 100 + written,
                        "question": pair["question"],
                        "gold_doc_id": doc["doc_id"],
                        "ground_truth": pair["ground_truth"],
                        "review_status": "UNREVIEWED",
                    }
                )
                + "\n"
            )
            written += 1
    print(f"Wrote {written} candidates to {OUT} — review before promoting.")


if __name__ == "__main__":
    main()

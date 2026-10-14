"""Calibrate the answer-correctness judge before trusting it.

Runs the judge (evalgate_rag.judge) over eval/judge_calibration.jsonl, where
each candidate answer's correct verdict is known by construction:

  paraphrase -- the reference's facts in different, often shorter, words: PASS
  perturbed  -- plausible and on topic, with one central fact changed:   FAIL
  swapped    -- a correct reference answer to a different question:      FAIL

and reports how often the judge agrees, overall and per kind, with the
disagreements listed so each can be read. The perturbed cases are the ones
that matter: they are what a fluent, confident, wrong RAG answer looks like.

Paraphrases and perturbations are hand-written and marked UNREVIEWED until a
human has checked each label; the report states how many were reviewed.

Constructed cases are not enough on their own. The first run against real
pipeline answers showed the judge failing answers that were correct but
carried extra correct detail, a kind of answer no constructed case covered.
So eval/judge_real_answers.jsonl holds real answers for a human to label
(`"label": "PASS"` or `"FAIL"`); every labelled one joins the calibration as
kind `real`. Existing real labels were not blind; agreement is exploratory,
not independent human validation. See docs/judge-rubric.md.

Verdicts are cached in eval/.cache/ (keyed by case, invalidated when the
judge prompt, model or calibration set changes), so a run stopped by a rate
limit resumes without re-spending tokens.

Usage:
    python eval/calibrate_judge.py [--run 1] [--out eval/judge_calibration_report_v2.json]

`--run N` keeps a separate cache per run number, so two runs over the same
cases measure the judge's agreement with itself.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys

from evalgate_rag.config import get_settings, judge_llm_settings
from evalgate_rag.judge import (
    JUDGE_RUBRIC_VERSION,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_TEMPLATE,
    AnswerJudge,
    JudgeParseError,
    agreement_report,
)
from evalgate_rag.pipeline import LLMClient

CALIBRATION_SET = pathlib.Path("eval/judge_calibration.jsonl")
REAL_ANSWERS = pathlib.Path("eval/judge_real_answers.jsonl")
GOLDEN_SET = pathlib.Path("data/golden_set.jsonl")
CACHE_DIR = pathlib.Path("eval/.cache")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--out", default="eval/judge_calibration_report_v2.json")
    args = parser.parse_args()

    cfg = get_settings()
    judge_cfg = judge_llm_settings(cfg)
    golden = {
        g["id"]: g
        for g in (json.loads(ln) for ln in GOLDEN_SET.read_text(encoding="utf-8").splitlines())
    }
    cases = [json.loads(ln) for ln in CALIBRATION_SET.read_text("utf-8").splitlines() if ln.strip()]
    if REAL_ANSWERS.exists():
        real = [json.loads(ln) for ln in REAL_ANSWERS.read_text("utf-8").splitlines() if ln.strip()]
        cases += [r for r in real if r.get("label") in ("PASS", "FAIL")]

    # Invalidates everything when the judge itself changes. Data changes are
    # caught per case by a content hash below, so labelling more real answers
    # does not re-spend tokens on cases already judged.
    fingerprint = hashlib.sha256(
        json.dumps(
            [
                JUDGE_RUBRIC_VERSION,
                JUDGE_SYSTEM_PROMPT,
                JUDGE_TEMPLATE,
                judge_cfg.model_dump(exclude={"api_key"}),
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    cache_path = CACHE_DIR / f"judge_calibration_run{args.run}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if cache.get("_fingerprint") != fingerprint:
        cache = {"_fingerprint": fingerprint}

    judge = AnswerJudge(LLMClient(judge_cfg))
    for case in cases:
        ref = golden[case["golden_id"]]["ground_truth"]
        content = hashlib.sha256(
            json.dumps([case["question"], ref, case["candidate"]], ensure_ascii=False).encode()
        ).hexdigest()
        cached = cache.get(case["case_id"])
        if cached is not None and cached.get("content") == content:
            continue
        try:
            v = judge.judge(case["question"], ref, case["candidate"])
            cache[case["case_id"]] = {"passed": v.passed, "reason": v.reason, "content": content}
        except JudgeParseError as exc:
            cache[case["case_id"]] = {"unparsed": str(exc), "content": content}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2))
        result = cache[case["case_id"]]
        shown = "UNPARSED" if "unparsed" in result else ("PASS" if result["passed"] else "FAIL")
        print(f"{case['case_id']:<14} label {case['label']:<4} judge {shown}", file=sys.stderr)

    scored = [c for c in cases if "passed" in cache.get(c["case_id"], {})]
    unparsed = [c["case_id"] for c in cases if "unparsed" in cache.get(c["case_id"], {})]

    def report_for(subset: list[dict]) -> dict:
        return agreement_report(
            [c["label"] == "PASS" for c in subset],
            [cache[c["case_id"]]["passed"] for c in subset],
        )

    by_kind = {}
    for kind in ("paraphrase", "perturbed", "swapped", "real"):
        subset = [c for c in scored if c["kind"] == kind]
        if subset:
            agree = sum((c["label"] == "PASS") == cache[c["case_id"]]["passed"] for c in subset)
            by_kind[kind] = {"n": len(subset), "agreement": round(agree / len(subset), 4)}

    disagreements = [
        {
            "case_id": c["case_id"],
            "label": c["label"],
            "judge": "PASS" if cache[c["case_id"]]["passed"] else "FAIL",
            "judge_reason": cache[c["case_id"]]["reason"],
            **({"planted_error": c["error"]} if "error" in c else {}),
        }
        for c in scored
        if (c["label"] == "PASS") != cache[c["case_id"]]["passed"]
    ]

    overall = {k: round(v, 4) if isinstance(v, float) else v for k, v in report_for(scored).items()}
    report = {
        "date": dt.date.today().isoformat(),
        "judge_model": judge_cfg.model,
        "rubric_version": JUDGE_RUBRIC_VERSION,
        "judge_fingerprint": fingerprint,
        "validation_status": "exploratory_not_independent_human_validation",
        "reasoning_effort": judge_cfg.reasoning_effort,
        "run": args.run,
        "cases": len(cases),
        "scored": len(scored),
        "unparsed": unparsed,
        "labels_human_reviewed": sum(c["review_status"] == "REVIEWED" for c in cases),
        "real_answers_labelled": sum(c["kind"] == "real" for c in cases),
        "overall": overall,
        "by_kind": by_kind,
        "disagreements": disagreements,
    }
    pathlib.Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("scored", "overall", "by_kind")}, indent=2))


if __name__ == "__main__":
    main()

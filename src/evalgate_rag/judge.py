"""LLM-as-judge for answer correctness against the golden set's reference answers.

Ragas' faithfulness asks "is the answer supported by what was retrieved?",
which a confidently wrong answer built on a wrongly retrieved chunk passes.
This judge asks the question the golden set exists for: does the answer state
what the reference says, without contradicting it?

A judge is a measuring instrument, so it is calibrated before it is trusted:
`eval/calibrate_judge.py` runs it over `eval/judge_calibration.jsonl`, whose
labels are known by construction, and reports agreement with `agreement_report`.

The judge is deliberately a different model family from the answering model
(see JudgeSettings), so it is not grading answers written by itself.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .pipeline import LLMClient

JUDGE_SYSTEM_PROMPT = """You grade answers to questions about the EU Artificial \
Intelligence Act. You are given a question, a reference answer that is correct \
by definition, and a candidate answer. Judge the candidate ONLY against the \
reference; do not use outside knowledge to add or remove requirements.

PASS when the candidate states the reference's central facts that answer the \
question (articles, numbers, thresholds, actors, conditions, yes/no) and \
contains nothing that contradicts the reference. Different wording, less \
detail, or extra correct context are fine.

FAIL when the candidate gets any central fact wrong (a different number, \
article, actor, condition or scope), contradicts the reference, answers a \
different question, refuses, or omits the central fact the question asks for.

Respond with a JSON object only:
{"verdict": "PASS" or "FAIL", "reason": "<one sentence naming the deciding fact>"}"""

JUDGE_TEMPLATE = """Question:
{question}

Reference answer:
{reference}

Candidate answer:
{candidate}"""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class JudgeParseError(ValueError):
    """The judge's reply held no readable verdict."""


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str


def parse_verdict(text: str) -> Verdict:
    """Read a verdict from the judge's reply. Anything unreadable raises,
    rather than defaulting to PASS or FAIL: a silent default would bias the
    agreement figures this module exists to measure."""
    match = _JSON_OBJECT.search(text)
    if match is None:
        raise JudgeParseError(f"no JSON object in judge reply: {text[:200]!r}")
    try:
        body = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"malformed JSON in judge reply: {text[:200]!r}") from exc
    verdict = str(body.get("verdict", "")).strip().upper()
    if verdict not in ("PASS", "FAIL"):
        raise JudgeParseError(f"verdict must be PASS or FAIL, got {body.get('verdict')!r}")
    return Verdict(passed=verdict == "PASS", reason=str(body.get("reason", "")))


class AnswerJudge:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def judge(self, question: str, reference: str, candidate: str) -> Verdict:
        reply = self._llm.chat(
            JUDGE_SYSTEM_PROMPT,
            JUDGE_TEMPLATE.format(question=question, reference=reference, candidate=candidate),
        )
        return parse_verdict(reply)


def agreement_report(labels: list[bool], judged: list[bool]) -> dict[str, float | int]:
    """Agreement between known labels (True = PASS) and the judge's verdicts.

    `false_pass_rate` is reported on its own because it is the error that
    matters for a gate: a wrong answer the judge lets through. Cohen's kappa
    corrects accuracy for the agreement two raters would reach by chance,
    which a skewed label mix (here, more FAIL than PASS) otherwise inflates.
    """
    if len(labels) != len(judged) or not labels:
        raise ValueError("labels and judged must be non-empty and the same length")
    n = len(labels)
    tp = sum(lab and jud for lab, jud in zip(labels, judged, strict=True))
    tn = sum(not lab and not jud for lab, jud in zip(labels, judged, strict=True))
    fp = sum(not lab and jud for lab, jud in zip(labels, judged, strict=True))
    fn = sum(lab and not jud for lab, jud in zip(labels, judged, strict=True))
    positives, negatives = tp + fn, tn + fp

    accuracy = (tp + tn) / n
    tpr = tp / positives if positives else 0.0
    tnr = tn / negatives if negatives else 0.0
    p_label, p_judge = positives / n, (tp + fp) / n
    expected = p_label * p_judge + (1 - p_label) * (1 - p_judge)
    kappa = 1.0 if expected == 1 else (accuracy - expected) / (1 - expected)
    return {
        "n": n,
        "accuracy": accuracy,
        "balanced_accuracy": (tpr + tnr) / 2,
        "cohen_kappa": kappa,
        "false_pass_rate": fp / negatives if negatives else 0.0,
        "false_fail_rate": fn / positives if positives else 0.0,
        "true_pass": tp,
        "true_fail": tn,
        "false_pass": fp,
        "false_fail": fn,
    }

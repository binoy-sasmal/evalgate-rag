"""Offline tests for the answer-correctness judge and its calibration maths."""

import json

import httpx
import pytest

from evalgate_rag.config import LLMSettings
from evalgate_rag.judge import (
    AnswerJudge,
    JudgeParseError,
    agreement_report,
    parse_verdict,
)
from evalgate_rag.pipeline import LLMClient


def test_parse_verdict_accepts_plain_json():
    v = parse_verdict('{"verdict": "PASS", "reason": "states 35 million and 7 %"}')
    assert v.passed is True
    assert v.reason == "states 35 million and 7 %"


def test_parse_verdict_tolerates_fences_and_surrounding_prose():
    text = 'Here is my grading:\n```json\n{"verdict": "fail", "reason": "wrong FLOPs"}\n```'
    assert parse_verdict(text).passed is False


@pytest.mark.parametrize(
    "text",
    ["PASS", '{"reason": "no verdict"}', '{"verdict": "MAYBE", "reason": "x"}', "{not json"],
)
def test_parse_verdict_refuses_anything_it_cannot_read(text):
    # An unreadable verdict must be an error, never silently a PASS or FAIL:
    # either default would bias the very agreement numbers being measured.
    with pytest.raises(JudgeParseError):
        parse_verdict(text)


def test_judge_sends_question_reference_and_candidate_but_no_retrieved_context():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"verdict": "PASS", "reason": "ok"}'}}]},
        )

    llm = LLMClient(LLMSettings(base_url="https://fake/v1"), transport=httpx.MockTransport(handler))
    verdict = AnswerJudge(llm).judge("Q?", "REFERENCE TEXT", "CANDIDATE TEXT")

    assert verdict.passed
    user = seen["messages"][1]["content"]
    assert "Q?" in user and "REFERENCE TEXT" in user and "CANDIDATE TEXT" in user


def test_llm_client_sends_reasoning_effort_and_json_mode_only_when_set():
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    plain = LLMClient(LLMSettings(base_url="https://f/v1"), transport=httpx.MockTransport(handler))
    tuned = LLMClient(
        LLMSettings(base_url="https://f/v1", reasoning_effort="low", json_mode=True),
        transport=httpx.MockTransport(handler),
    )
    plain.chat("s", "u")
    tuned.chat("s", "u")

    assert "reasoning_effort" not in bodies[0] and "response_format" not in bodies[0]
    assert bodies[1]["reasoning_effort"] == "low"
    assert bodies[1]["response_format"] == {"type": "json_object"}


def test_agreement_report_counts_the_dangerous_error_separately():
    labels = [True, True, False, False, False, False]
    judged = [True, False, False, False, True, False]  # one false fail, one false pass
    r = agreement_report(labels, judged)
    assert r["n"] == 6
    assert r["accuracy"] == pytest.approx(4 / 6)
    assert r["false_pass_rate"] == pytest.approx(1 / 4)  # wrong answers it let through
    assert r["false_fail_rate"] == pytest.approx(1 / 2)  # right answers it rejected
    assert r["balanced_accuracy"] == pytest.approx((0.5 + 0.75) / 2)
    # kappa: po = 4/6; pe = (2/6)(2/6) + (4/6)(4/6) = 20/36
    assert r["cohen_kappa"] == pytest.approx((4 / 6 - 20 / 36) / (1 - 20 / 36))


def test_agreement_report_on_perfect_agreement():
    r = agreement_report([True, False, False], [True, False, False])
    assert r["accuracy"] == 1.0 and r["cohen_kappa"] == 1.0

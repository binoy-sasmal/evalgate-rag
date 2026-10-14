"""Offline calibration runs exercise cache reuse without model calls."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from evalgate_rag.config import Settings


def test_reference_change_and_legacy_entry_force_rejudging(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "calibrate_review", Path(__file__).parents[1] / "eval/calibrate_judge.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    golden = tmp_path / "golden.jsonl"
    cases = tmp_path / "cases.jsonl"
    out = tmp_path / "report.json"
    case = {
        "case_id": "1-real",
        "golden_id": 1,
        "question": "Q?",
        "candidate": "Answer",
        "label": "PASS",
        "kind": "real",
        "review_status": "REVIEWED",
    }
    cases.write_text(json.dumps(case) + "\n")
    golden.write_text(json.dumps({"id": 1, "ground_truth": "Original"}) + "\n")
    monkeypatch.setattr(module, "GOLDEN_SET", golden)
    monkeypatch.setattr(module, "CALIBRATION_SET", cases)
    monkeypatch.setattr(module, "REAL_ANSWERS", tmp_path / "absent")
    monkeypatch.setattr(module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(module, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(module, "LLMClient", lambda cfg: None)
    calls = []

    class FakeJudge:
        def __init__(self, client):
            pass

        def judge(self, question, reference, candidate):
            calls.append(reference)
            return SimpleNamespace(passed=True, reason="fixture")

    monkeypatch.setattr(module, "AnswerJudge", FakeJudge)
    monkeypatch.setattr("sys.argv", ["calibrate", "--out", str(out)])
    module.main()
    module.main()
    assert calls == ["Original"]
    golden.write_text(json.dumps({"id": 1, "ground_truth": "Corrected"}) + "\n")
    module.main()
    assert calls == ["Original", "Corrected"]
    cache_path = module.CACHE_DIR / "judge_calibration_run1.json"
    cache = json.loads(cache_path.read_text())
    del cache["1-real"]["content"]
    cache_path.write_text(json.dumps(cache))
    module.main()
    assert calls == ["Original", "Corrected", "Corrected"]
    # Labels affect comparison, not the judge input: changing one must not
    # spend tokens or silently rewrite a human's decision.
    case["label"] = "FAIL"
    cases.write_text(json.dumps(case) + "\n")
    module.main()
    assert len(calls) == 3
    assert json.loads(out.read_text())["overall"]["false_pass"] == 1

"""CI eval gate: compare eval/results.json against eval/baseline.json.

Exit code 1 (build fails) if any metric drops more than TOLERANCE below its
baseline. A small tolerance absorbs LLM-judge noise between runs; tighten it
once you have variance data from a few runs.

Usage:
    python eval/check_regression.py                # gate (CI)
    python eval/check_regression.py --promote      # accept current results as new baseline
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

BASELINE = pathlib.Path("eval/baseline.json")
RESULTS = pathlib.Path("eval/results.json")
TOLERANCE = 0.03
METRICS = ("faithfulness", "answer_relevancy", "context_precision")
# Below this fraction of n_questions actually scored, an average is judge-call
# noise, not a real measurement (a run that dropped 85% of its samples to
# TPM-pressure NaNs once produced a 0.66 that looked like a regression but was
# really 3 lucky/unlucky samples) -- fail the gate outright rather than trust it.
MIN_SAMPLE_FRACTION = 0.8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    results = json.loads(RESULTS.read_text())

    if args.promote:
        missing = [m for m in METRICS if results.get(m) is None]
        if missing:
            print(
                f"Refusing to promote -- eval/results.json has null/missing metrics "
                f"(no successful judge samples) for: {', '.join(missing)}",
                file=sys.stderr,
            )
            sys.exit(1)
        n_questions = results.get("n_questions")
        unreliable = [
            m
            for m in METRICS
            if n_questions
            and (n_samples := results.get(f"{m}_n")) is not None
            and n_samples < MIN_SAMPLE_FRACTION * n_questions
        ]
        if unreliable:
            print(
                f"Refusing to promote -- too few successful judge samples "
                f"(<{MIN_SAMPLE_FRACTION:.0%} of {n_questions}) for: {', '.join(unreliable)}",
                file=sys.stderr,
            )
            sys.exit(1)
        baseline = {m: results[m] for m in METRICS}
        baseline["_comment"] = "Promoted from eval/results.json"
        BASELINE.write_text(json.dumps(baseline, indent=2))
        print("Baseline updated:", json.dumps(baseline, indent=2))
        return

    baseline = json.loads(BASELINE.read_text())
    n_questions = results.get("n_questions")
    failures: list[str] = []
    print(f"{'metric':<20}{'baseline':>10}{'current':>10}{'delta':>10}{'samples':>10}")
    for m in METRICS:
        n_samples = results.get(f"{m}_n")
        samples_str = f"{n_samples}/{n_questions}" if n_samples is not None else ""
        if results.get(m) is None:
            print(f"{m:<20}{'--':>10}{'null':>10}{'--':>10}{samples_str:>10}  FAIL (no samples)")
            failures.append(m)
            continue
        if n_samples is not None and n_questions and n_samples < MIN_SAMPLE_FRACTION * n_questions:
            print(
                f"{m:<20}{'--':>10}{results[m]:>10.4f}{'--':>10}{samples_str:>10}"
                f"  FAIL (only {n_samples}/{n_questions} judge calls succeeded -- unreliable)"
            )
            failures.append(m)
            continue
        base, cur = float(baseline[m]), float(results[m])
        delta = cur - base
        flag = "  FAIL" if delta < -TOLERANCE else ""
        print(f"{m:<20}{base:>10.4f}{cur:>10.4f}{delta:>+10.4f}{samples_str:>10}{flag}")
        if delta < -TOLERANCE:
            failures.append(m)

    if failures:
        print(f"\nEval gate FAILED -- regression beyond {TOLERANCE} in: {', '.join(failures)}")
        sys.exit(1)
    print("\nEval gate passed.")


if __name__ == "__main__":
    main()

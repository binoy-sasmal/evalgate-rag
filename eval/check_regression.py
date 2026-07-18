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
        baseline = {m: results[m] for m in METRICS}
        baseline["_comment"] = "Promoted from eval/results.json"
        BASELINE.write_text(json.dumps(baseline, indent=2))
        print("Baseline updated:", json.dumps(baseline, indent=2))
        return

    baseline = json.loads(BASELINE.read_text())
    failures: list[str] = []
    print(f"{'metric':<20}{'baseline':>10}{'current':>10}{'delta':>10}")
    for m in METRICS:
        if results.get(m) is None:
            print(f"{m:<20}{'--':>10}{'null':>10}{'--':>10}  FAIL (no successful judge samples)")
            failures.append(m)
            continue
        base, cur = float(baseline[m]), float(results[m])
        delta = cur - base
        flag = "  FAIL" if delta < -TOLERANCE else ""
        print(f"{m:<20}{base:>10.4f}{cur:>10.4f}{delta:>+10.4f}{flag}")
        if delta < -TOLERANCE:
            failures.append(m)

    if failures:
        print(f"\nEval gate FAILED -- regression beyond {TOLERANCE} in: {', '.join(failures)}")
        sys.exit(1)
    print("\nEval gate passed.")


if __name__ == "__main__":
    main()

# Judge rubric and validation status

The current rubric is `reference-supported-v2`. It measures whether an answer
contains the reference's central facts and makes no unsupported factual claims.
Paraphrases and omission of nonessential detail are allowed. Additional factual
claims absent from the reference fail as unsupported, not necessarily false.
This follows the previously documented human review policy. It intentionally
does not measure general factual correctness beyond the supplied reference.

The original prompt allowed extra correct context while the human review used
the stricter policy. That mismatch is fixed in the prompt, not by changing labels.
All original labels, review statuses and measured results are preserved.

## Historical results

`eval/judge_calibration_report.json` and `eval/results/judged_baseline.json`
are historical results from the old prompt. Their agreement/pass rates do not
validate v2. Of the 80 cases, 40 constructed labels are unreviewed, 20 are
mechanically constructed and 20 real-answer labels endorsed visible first-pass
judge verdicts. The 80 cases derive from only 20 golden questions; they are not
80 independent questions. No independent human-validation claim is supported.

## Next validation run

1. Have a human review constructed labels using this rubric. Never revise a
   label merely to agree with a model.
2. Collect fresh real answers and label them without seeing judge verdicts.
   Keep this evaluation set separate from prompt development. Record rubric,
   reviewer provenance, disagreements and adjudication.
3. Run `python eval/calibrate_judge.py`. It evaluates the full existing set and
   writes `eval/judge_calibration_report_v2.json` without replacing the original.
   This costs model tokens. Until blind labels are added, the report remains
   exploratory regardless of its agreement score.
4. Report constructed and real-answer agreement separately. Only consider a
   gate after human review and a deliberate baseline promotion.

The v2 live run has not been performed as part of this offline code fix. Cache
keys now include the reference, question, candidate, prompt, template, rubric
version and non-secret judge settings. Old entries without content hashes are
recomputed; changing comparison labels alone does not spend model tokens.

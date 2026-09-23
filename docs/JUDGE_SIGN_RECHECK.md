# Milestone 4 correction: saved scientific-judge sign evidence

The live critic pilot completed six revisions/fits and retained four children.
Its reports also exposed a deterministic evidence bug: `-k*x` was described as a
positive outer term whose supposedly unsigned expression still contained `-k`,
while the equivalent `-(k*x)` was correctly described as negative. `-z/tau`
had the same issue. This can contaminate both atomic direction checks and the
comparative stage that receives their findings. It does not change the equations
that were fitted, and it does not establish that all judge findings are wrong.

## Bounded correction

The shared negative-factor helper now supports a second caller: judge evidence.
The judge extracts explicit minus signs only along the outer product/quotient
shell and retains their parity. With `n` removed minus signs, an additive term
with sign `s` is represented as `s * (-1)**n * unsigned_expression`.
Construction retains its existing topology-owned sign convention. Candidate
models are never rewritten by the judge. Calls, powers, and internal sums and
differences remain opaque. Parameter positivity, numerical values and physical
meaning are not inferred. In particular, `1*(Tj-Tjf)` remains context dependent;
this fix does not decide which consumer receives a source or sink.

`outer-factor-sign-2` is an explicit evidence version threaded through atomic
plans, structural facts, paired-judge cache identities and named references.
Both atomic and comparative stages use that version. Historical defaults remain
`top-level-sign-1`, so archived campaigns and calibration requests keep their
original meaning. The correction pilot explicitly selects the new version;
this commit does not silently change every historical evaluation entry point.

## Saved-pair recheck

The CLI `scripts/judge_sign_recheck.py` imports the completed M4 campaign. It
verifies sealed plan/stage/request/receipt linkage and copies only symbolic
judge requests plus historical review records into a separate output. It reads
plan metadata but no trajectory files, and never starts a proposer or optimizer.

The audit aligns additive occurrences from the old and corrected extraction.
A review is affected if an occurrence's unsigned expression, identity or actual
polarity changes. The same extraction supplies exact-repeat and structural facts,
so the affected set includes changes to these downstream questions. Reused
parent reviews are deduplicated by their complete request hash. Selection does
not depend on NMSE, historical finding severity, or any new judge answer.

For each affected unique pair, rerun both atomic and comparative stages in both
orientations. Reuse the source's exact public prompt, symbolic models, context,
seed, GPT-OSS-120B revision, serving image and judge settings. Use the established
low reasoning, temperature 0.2, 6144 output tokens, two-stage consensus and
fallback seed policy. Successful paired reviews ordinarily need four physical
requests; retries can increase this count. Old atomic responses are not patched
or reused because the questions can have changed.

Unaffected reviews are labelled `unchanged_evidence`; they get no new LLM call.
All historical fits, models, reviews and selections remain untouched. This
experiment does not retroactively undo revisions influenced by old feedback.
It neither tests critic benefit nor transfers the old calibration automatically
to the corrected evidence version. A matched critic ablation remains a later
milestone.

## Execution and outputs

Use the pinned commit and group-scratch archive commands supplied with the change.
The existing Python virtual environment and vLLM SIF can be reused. Then submit:

```bash
module load GCCcore/13.2.0 Python/3.11.5
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT"
export PYTHONDONTWRITEBYTECODE=1
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_judge_sign_recheck.py" \
  --source "$AF_SOURCE_ROOT" --root "$AF_OUTPUT_ROOT"
```

The source is `general-critic-v1`; use a new `judge-sign-recheck-v1` output.
`AF_HF_HOME` names the existing group cache and `AF_VLLM_IMAGE` the original SIF.
The model revision and image hash are imported, not resolved from a moving branch.
The submitter validates saved prerequisites and records a snapshot hash before
scheduling. CPU preparation reruns tests/smoke, verifies that snapshot, freezes
the audit, checks the SIF, and ensures the exact judge weights are cached. One
job with two H100s reviews affected pairs, followed by a CPU report. If there are
no affected pairs, no GPU job is submitted. No fitting jobs are submitted.

Completed reviews are reused on resume. An interrupted in-flight review retains
partial logs and is marked interrupted without resetting its budget. Repeating
submission returns its existing manifest; ambiguous partial scheduler submissions
require inspecting retained receipts rather than deleting intents. Do not mix
new code into an already frozen output directory.

Inspect:

```bash
cat "$AF_OUTPUT_ROOT/SUMMARY.md"
jq '{unique_reviews, affected_unique_reviews, status_counts, cost,
     optimizer_calls, proposer_calls, model_changes, selection_changes}' \
  "$AF_OUTPUT_ROOT/summary.json"
jq '[.rows[] | {task, stage, affected, status,
     old_findings: [.historical_findings[] | {code, observation}],
     new_findings: (if .new_findings == null then null else
       [.new_findings[] | {code, observation}] end)}]' \
  "$AF_OUTPUT_ROOT/summary.json"
```

`plan.json` records frozen requests and occurrence-level before/after evidence.
`reviews/<historical-request-hash>/judge/` contains new cached calls and events;
its sibling `result.json` links the new and old request identities. Cost totals
count each unique new review once, including recorded partial calls. Unavailable
usage and missing results remain explicit. No findings is not scientific approval.

Local verification: regression tests cover explicit sign parity, opaque nonlinear
laws, shared processes, real wrong-direction failures, both judge stages, unchanged
legacy identities, provenance tampering, interruption and submission idempotency.
`scripts/smoke_judge_sign_recheck.py` runs the real import/review/report path with
mocked judge responses and confirms historical files and equations are unchanged.

Verification on 2026-09-22: the full test wrapper passed 3,073 tests with eight
optional Torch skips; the separately excluded shared-process pilot file passed
all 11 tests. The offline paired-review smoke, changed-file Ruff, Python 3.11
syntax parsing of the new entry points, and shell syntax checks passed.
Repository-wide Ruff reports 37 pre-existing findings in unrelated
`analysis/claude` scripts; those files were not changed. Live model serving and
the corrected scientific reviews remain to be tested on ACES.

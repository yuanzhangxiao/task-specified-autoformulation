# Fitting-free pre-fitting experiments

This follow-up isolates construction-interface behavior while the fitter is being
finalized. It does not rerun numerical fitting or reinterpret optimizer failures
as structural defects. Benchmark prompts and tables are unchanged.

## Milestone 1: historical response replay

`scripts/prefit_response_replay.py` reads a terminal
`prefit-matched-construction-1` root and writes a sealed replay corpus outside it.
The source plan, construction results, initializer checkpoints and referenced
cached requests are checked. File hashes bind the replay to the source; missing
requests and changed artifacts fail explicitly. No CSV or fit result is opened.
Only final visible provider JSON becomes a proposed expression.

The replay rechecks interaction-function slots and causal initializers using the
production validators. Functions retain historically accepted preceding slots;
initializers retain the actual historical equations and accepted boundaries.
Every saved attempt is independent. A newly admissible response never replaces
the historical parent of a later response. The assignment-normalization view is
compared with a strict-RHS view under the same current validators. This is not an
exact re-execution of every old runtime version.

A schema-valid equation batch was historically accepted before its individual
slots were checked. Its replay rows explicitly identify that different scope;
batch acceptance must not be compared with full slot validity as a regression.
Topology requests, malformed batch shapes and missing final JSON remain in an
explicit unreplayed-request inventory. They are not counted as successful replay
or silently selected as live repair cases. This first pilot does not evaluate
topology repair or scientific correctness.

Historical construction completion, normalization recoveries and remaining
deterministic violations have separate counts. Scientific status is
`not_assessed`. Replay does not establish saved live calls, avoided future
failures, scientific improvement, or a revised historical construction rate.

The smoke creates temporary synthetic public observations and prescribed cached
responses, then exercises the actual constructor and replay. It performs no fit
or live LLM call and does not modify benchmark assets.

## Milestone 2: matched local feedback

`scripts/prefit_feedback_campaign.py` freezes remaining invalid slots and valid
controls from the replay, with identical source contexts, normalization, response
schemas, model settings, seeds and budgets in both arms. The treatment is the
feedback payload: ordinary error text versus named diagnostics with explicit scope
and syntax facts. The common instruction asks both arms to preserve a valid
response. Neither arm receives a runtime-invented replacement formula.

`configs/prefit_matched_feedback_v1.json` selects at most eight invalid slots and
four valid controls, three seeds, two arms, and three attempts per episode:
at most 72 episodes and 216 physical requests. Selection uses one failed response
per historical task/component, ordered by original attempt then case hash, with
round-robin coverage of component kind and first diagnostic code. A slot with any
saved invalid response cannot also be a valid control. Controls are selected by
case hash. Arm order alternates by case and seed. Fewer available cases produce a
smaller experiment; cases are never fabricated to fill quotas.

The model revision, reasoning effort, temperature and output-token cap must match
the historical source. Both arms use the same current runtime and a fresh matched
repair budget. The copied corpus, selected cases, compiler dependencies, code and
launchers are frozen. Historical-context inconsistencies stop the experiment;
they cannot be reclassified as proposer errors. Runtime, schema and launcher
changes require a new replay and campaign root.

Every physical request is cached, with a pre-call budget reservation. Interrupted
in-flight calls remain consumed and uncertain on resume; they are not resent.
Per-episode checkpoints bind evaluated cache records, and a changed or missing
record fails explicitly. A worker drains before its deadline and locks each
episode. Terminal failures remain terminal instead of receiving a new budget.

Reports separate invalid-case repair from valid controls. They include first-call
and eventual validity, repeated baseline codes, newly observed violation codes,
physical calls, observed tokens, unknown usage, budget charges and provider time.
Control changes compare the final accepted canonical structure with its original;
intermediate violations and failed controls have separate counts. This is not a
general symbolic-equivalence test. New codes are an operational indicator, not
proof that every new defect was identified. Paired outcomes are also reported for
each case/seed. Repeated seeds are not independent scientific tasks.

Routing, scientific judging and new full-model construction are later milestones.
These two experiments can establish normalization coverage and local deterministic
repair performance. They cannot establish scientific improvement, a new whole-model
construction success rate, or a benefit from the routing policy. Runtime-valid
identity initializers remain scientifically unassessed.

## Implementation and offline verification

The two implementation modules are `rebuttal/prefit_replay.py` and
`rebuttal/prefit_feedback.py`. Each has a standalone CLI and a synthetic smoke.
The ACES config and two new shell wrappers reuse the shared model-server launcher;
the only shared-launcher changes select and verify the new protocol. No production
constructor, fitter, judge, benchmark table or finalized benchmark prompt changes
are needed by this pilot.

The focused replay, feedback and old/new ACES submission checks passed 36 tests.
The full suite passed 1,590 tests with three optional Torch skips in 457.42 seconds;
Ruff and shell syntax/dispatch checks passed.
Both new fitting-free smokes passed, including exact terminal resume. The existing
causal-initializer numerical smoke also passed as a separate integration regression.
These synthetic outcomes verify experiment mechanics; they are not evidence that
either feedback arm performs better on the saved ACES responses.

## ACES preparation correction (2026-09-14)

Job 2129189 failed in preflight with 13 tests passed and 14 fixture errors:
`PackageNotFoundError: No package metadata was found for sympy`.
The experiment's runtime-version inventory incorrectly required SymPy even though
none of its validators or runners uses it. ACES then cancelled dependent GPU job
2129190 without starting it. Replay and live feedback had not begun.

The corrected inventory pins Python, Pydantic, NumPy and SciPy. Missing metadata
for required packages still fails explicitly. A regression exercises freeze,
execution and exact resume with SymPy metadata unavailable. The CPU wrapper now
prints the failing pytest summary into the main job error log while retaining the
complete preflight log and exit status.

Correction verification: 32 focused tests passed; the full suite passed 1,595
tests with three optional Torch skips in 524.78 seconds. Ruff, shell checks, and
both fitting-free smokes passed. The subsequently completed real ACES replay and
matched-feedback results are in [PREFIT_FEEDBACK_RESULTS_2026-09-14.md](PREFIT_FEEDBACK_RESULTS_2026-09-14.md).

Use a new detached checkout named `autoformalism-prefit-feedback-v1-fix1` and a new
output root named `prefit-feedback-v1-fix1` for this correction. Preserve the old
failure logs. `AF_RESUME=1` does not apply because preparation never completed;
submit preparation and its dependent feedback job afresh. SymPy installation is
unnecessary. The feedback arms, inference settings, case-selection rule and
budgets are unchanged.

## ACES commands

Run each command as one physical line. The following creates a detached experiment
checkout from the pushed branch. The submission manifest records its exact commit.
Use a new checkout/output name for a later implementation version; do not update a
checkout supporting an unfinished frozen campaign.

```bash
git -C /scratch/user/u.yx126462/repos/autoformalism-e432fe3 fetch origin codex/prefit-aces-v1 && git -C /scratch/user/u.yx126462/repos/autoformalism-e432fe3 worktree add --detach /scratch/user/u.yx126462/repos/autoformalism-prefit-feedback-v1 FETCH_HEAD
```

Submit the CPU preparation followed by one H100 job. Preparation runs the new
regression tests and fitting-free smoke, replays the historical source, freezes
the matched cases, verifies the pinned container and prepares its model cache.
No numerical fit job is submitted. All logs, pytest temporary files and results
use scratch; pytest's home-directory cache is disabled.

```bash
AF_REPO_ROOT=/scratch/user/u.yx126462/repos/autoformalism-prefit-feedback-v1 AF_OUTPUT_ROOT=/scratch/user/u.yx126462/phase_b/prefit-feedback-v1 bash /scratch/user/u.yx126462/repos/autoformalism-prefit-feedback-v1/scripts/hpc/submit_prefit_feedback_aces.sh
```

Repeating this command prints the existing submission manifest without new jobs.
CPU preparation requests one hour; the dependent GPU job requests four hours,
including startup and shutdown time. No automatic resubmission occurs. If replay
has no residual local failures, preparation stops with that explicit message and
the dependent GPU job cannot start; inspect the replay report and cancel the
blocked dependency rather than manufacturing a live experiment.

Read the replay counts and selected cases after CPU preparation:

```bash
jq '{counts, limitation}' /scratch/user/u.yx126462/phase_b/prefit-feedback-v1/replay.json
```

```bash
cat /scratch/user/u.yx126462/phase_b/prefit-feedback-v1/runtime/selected-cases.json
```

Read live/terminal results with the exact Python environment and source path:

```bash
module load GCCcore/13.2.0 Python/3.11.5 && PYTHONPATH=/scratch/user/u.yx126462/repos/autoformalism-prefit-feedback-v1/src /scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python /scratch/user/u.yx126462/repos/autoformalism-prefit-feedback-v1/scripts/prefit_feedback_campaign.py summary --root /scratch/user/u.yx126462/phase_b/prefit-feedback-v1
```

The CLI prints cohort counts and paired outcomes; `summary.json` includes all
episode rows. Each `results/TASK_ID/state.json` contains the attempt diagnoses and
responses; `results/TASK_ID/calls/REQUEST_HASH.json` holds the corresponding cached
request/response and cost record. Read `runtime/preflight-JOB_ID.log` for pytest
results and `logs/prepare-JOB_ID.err` or `logs/repair-JOB_ID.err` for job failures.

If the GPU job ended with pending episodes, this command verifies unchanged
provenance and terminal scheduler state before resuming only the GPU stage:

```bash
AF_RESUME=1 AF_REPO_ROOT=/scratch/user/u.yx126462/repos/autoformalism-prefit-feedback-v1 AF_OUTPUT_ROOT=/scratch/user/u.yx126462/phase_b/prefit-feedback-v1 bash /scratch/user/u.yx126462/repos/autoformalism-prefit-feedback-v1/scripts/hpc/submit_prefit_feedback_aces.sh
```

An ambiguous scheduler submission leaves an intent record under `submissions/`
and refuses blind resubmission. Inspect its saved job IDs and the scheduler before
resolving such an interruption.

To run just the fitting-free tests and smokes independently on ACES:

```bash
module load GCCcore/13.2.0 Python/3.11.5 && cd /scratch/user/u.yx126462/repos/autoformalism-prefit-feedback-v1 && TMPDIR=/scratch/user/u.yx126462 PYTHONPATH=/scratch/user/u.yx126462/repos/autoformalism-prefit-feedback-v1/src /scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_prefit_replay.py tests/test_prefit_feedback.py tests/test_prefit_feedback_submission.py
```

```bash
module load GCCcore/13.2.0 Python/3.11.5 && TMPDIR=/scratch/user/u.yx126462 PYTHONPATH=/scratch/user/u.yx126462/repos/autoformalism-prefit-feedback-v1/src /scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python /scratch/user/u.yx126462/repos/autoformalism-prefit-feedback-v1/scripts/smoke_prefit_replay.py
```

```bash
module load GCCcore/13.2.0 Python/3.11.5 && TMPDIR=/scratch/user/u.yx126462 PYTHONPATH=/scratch/user/u.yx126462/repos/autoformalism-prefit-feedback-v1/src /scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python /scratch/user/u.yx126462/repos/autoformalism-prefit-feedback-v1/scripts/smoke_prefit_feedback.py
```

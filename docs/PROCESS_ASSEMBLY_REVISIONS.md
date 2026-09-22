# Process assembly revisions — saved-v6 gate

V6 constructed seven of eight models. Three construction pairs improved in
prediction error, four worsened, and one failed. Both v5 and v6 had four of sixteen
gain arms below 0.1 on both training and validation. Five intrinsic-factor flags
included three with no overlap and two that retained an extra area division.
The failed construction exhausted local dependency/pathway repairs. These findings
motivate an interface correction, not a fitter change or a claimed shared-process win.

## This milestone

`process-assembly-revision-1` provides an opt-in, independently replayable repair
transaction and a cached bounded repair runner. The saved-v6 CPU audit and offline
construction smoke exercise this interface before another fresh construction run.
Historical policies, pilot launchers, public prompts, benchmark data and fitter
defaults are unchanged. The new runner is an explicit API, not yet enabled by the
basin pilot. A fresh pilot adapter/manifest is a subsequent milestone after this
audit; no v7 fitting jobs are submitted by these commands.

### Dependency ownership

An intrinsic law can omit a selected public fixed covariate if that covariate
belongs to its declared consumer conversion. The consumer conversion remains in
place. Runtime never deletes a factor from the proposed expression or inserts a
dummy `area/area`. All other omissions, and non-fixed dependency additions, still
require the proposer's explicit `revise_dependencies` decision. The final equation
graph, not the prose explanation, must retain public pathways and dynamic memory.

### Atomic law/conversion revision

The response contains the function, optional `consumer_conversions` edits naming
existing consumer targets, and `retain_intrinsic_for` naming actual remaining
overlaps when the proposer intentionally retains them. Signs and consumer
identities cannot change. Conversion expressions use only permitted fixed
covariates, positive constants, multiplication and division; unresolved remains
explicitly null. Actual values are checked again by the existing gain compiler.

The transaction validates source changes, conversions, all known retained functions
and new functions before committing any change. Invalid replies leave the parent
untouched. Repeated covariates are not automatically physical errors. Intentionally
retained overlaps carry `retained_intrinsic_unverified`, never a scientific or
dimensional certificate. A legacy Boolean is not converted to a new decision.

### Bounded topology repair

`process_revision_runner.run` uses the caller's existing `StagedTopologyClient` and
its remaining per-step attempts (`attempt_offset` records attempts already spent).
A public-pathway or process-target-path failure changes the next remaining request
to topology scope; it does not replenish requests, tokens or wall-clock budget.
The proposer can revise companion **ordinary existing slots**, each with a full
function and explicit dependency decision. The whole transaction is checked at
once, allowing a valid path to move between terms without requiring an invalid
intermediate graph to pass. There are no invented edges or dummy factors.

This bounded scope does not add variables, change signs or shared consumers, or
change physical boundaries. Edits requiring those changes still need the existing
larger topology-construction flow. No-call exhaustion, deferred calls, uncertain
cached outcomes, exact resume, and retained-function binding are tested. The runner
does not generate a new numerical budget or automatically fit its returned draft.

## Saved-response audit

The audit reads public campaign metadata and saved function-stage call records.
It reproduces original dependency histories, including the exact historical alias
compatibility case. Each reply is checked in its original context; newly valid
replies never advance the historical trajectory. No new expression, conversion,
topology companion, model or initial condition is invented.

The report distinguishes:

- mechanically valid replies and conversion-only dependency omissions;
- pending explicit dependency decisions, conversion decisions and topology repairs;
- confirmed actual overlaps versus intrinsic flags with no overlap;
- public-pathway rejections versus the narrower process-target-path guard;
- equation-batch delivery/schema failures, with their saved errors and responses;
- missing saved calls, expected conversion-policy changes and unexpected regressions.

`whole_models_recovered`, `llm_calls` and `optimizer_calls` remain zero. Historical
source files are hash-frozen, symlink-contained, read-only, and rechecked on resume.
Output is checkpointed per construction in a separate directory. No fit scores,
trajectory tables, hidden references or test data are used. Legacy call records can
contain a previously supplied training summary, but it is not analyzed by this audit.

Before a fresh pilot, require zero unexpected acceptance regressions, missing
constructions and unavailable saved attempts. Review remaining conversion and
topology decisions separately. These gates establish interface integrity, not
scientific correctness or predictive improvement.

## Local verification

```bash
.venv/bin/python -m pytest -q tests/test_process_assembly_revision.py \
  tests/test_process_revision_runner.py tests/test_process_revision_audit.py
.venv/bin/python -m scripts.smoke_process_revisions \
  --output /tmp/process-revisions-smoke
bash scripts/run_tests.sh full
.venv/bin/python -m pytest -q tests/test_shared_process_pilot.py
.venv/bin/ruff check .
```

The smoke uses prescribed offline replies and a toy, not benchmark data. It
checks local-to-topology routing, cached resume, independent transaction replay,
complete function binding, and the actual compiled derivatives after applying
the area conversion once. There are no optimizations or live provider calls.

## ACES: one CPU audit

Set `AF_COMMIT` to the supplied full commit first. Use the existing Python
environment and a git archive in group scratch to avoid another full clone.

```bash
(
  set -euo pipefail
  : "${AF_COMMIT:?Set AF_COMMIT to the supplied commit first}"
  export AF_COMMIT
  AF_BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  export AF_REPO_ROOT="$AF_GROUP/repos/process-revision-${AF_COMMIT:0:7}"
  export AF_SOURCE_ROOT="$AF_GROUP/detention-process-pilot-v6"
  export AF_OUTPUT_ROOT="$AF_GROUP/process-revision-audit-v1"
  export AF_PYTHON="$AF_BASE/.venv/bin/python"

  git -C "$AF_BASE" fetch origin codex/prefit-aces-v1
  git -C "$AF_BASE" cat-file -e "$AF_COMMIT^{commit}"
  if [[ ! -e "$AF_REPO_ROOT" ]]; then
    mkdir -p "$AF_REPO_ROOT"
    git -C "$AF_BASE" archive "$AF_COMMIT" | tar -x -C "$AF_REPO_ROOT"
    printf '%s\n' "$AF_COMMIT" > "$AF_REPO_ROOT/SOURCE_COMMIT"
  fi
  [[ "$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")" == "$AF_COMMIT" ]]
  [[ -f "$AF_SOURCE_ROOT/plan.json" && -x "$AF_PYTHON" ]]
  mkdir -p "$AF_OUTPUT_ROOT/logs"
  sbatch --export=ALL \
    --output="$AF_OUTPUT_ROOT/logs/audit-%j.out" \
    --error="$AF_OUTPUT_ROOT/logs/audit-%j.err" \
    "$AF_REPO_ROOT/scripts/hpc/run_process_revision_audit_aces.sh"
)
```

After completion:

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/process-revision-audit-v1
cat "$ROOT/SUMMARY.md"
jq '{classification_counts, historical_counters,
     newly_mechanically_valid_attempts, unexpected_acceptance_regressions,
     unavailable_saved_attempts, missing_constructions,
     llm_calls, optimizer_calls}' "$ROOT/summary.json"
jq '[.constructions[] as $c | $c.routes[] as $r |
     $r.rejected_events[] |
     select(.step | startswith("equation_functions_")) |
     {task: $c.task, route: $r.route, step, attempt,
      error, category, saved_response}]' "$ROOT/summary.json"
```

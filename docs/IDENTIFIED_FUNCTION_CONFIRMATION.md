# Identified function delivery and process repair — basin v7

This opt-in milestone reuses the existing equation-batch construction, restricted
expression grammar, topology-owned signs, parameter-role certification, causal
initialization, canonical compilation, paired gain policies and frozen fitter.
The original policies remain available. Benchmark prompts and data are unchanged.

The saved-v6 audit found 39 mechanically valid replies, five conversion-only
source omissions now admissible, 11 dependency decisions, four conversion decisions,
four topology repairs and three syntax failures. All three syntax failures were
assignments in an RHS field. The 13 earlier batch failures all returned too many
functions. No whole model was recovered by that read-only audit.

## Delivery changes

`identified-function-delivery-1` retains one batch request per selected equation.
Each returned function adds an `interaction_id` identifying its requested slot.
Order is immaterial. The request distinguishes requested blanks from shared-process
consumer identities supplied by the runtime. Those uses have no LLM function call.

Only unique, requested, schema-valid entries are mapped. Unknown identifiers,
duplicates and malformed entries receive delivery diagnostics; their expressions
are never assigned to a different slot. Unambiguously valid entries are retained.
Remaining delivery retries request only missing entries. Semantically invalid
functions receive the existing bounded per-slot repair opportunity. No function
is selected by array position, silently summed with another, or used to create a
new term. The existing construction-wide token/request/wall-clock gates still apply.

A single assignment to the exact selected algebraic definition may have its label
removed, with original and normalized strings recorded. It is then validated as
an ordinary scalar expression. Different labels, chained/multiple assignments,
unsafe expressions, undeclared symbols and algebraic cycles remain errors.

## Shared-process repairs

The audited `process-assembly-revision-1` transaction API is now used for focused
repairs. Intrinsic laws may omit conversion-owned fixed covariates without dummy
factors. The proposer can explicitly revise dependencies or consumer conversions.
A failed public pathway can use remaining repair attempts to revise ordinary
companion slots atomically. There are no new variables, sign changes or shared
consumer changes in this bounded scope. Larger changes still require topology
construction. Intentional conversion overlaps remain scientifically unverified.

Every transaction rebinds all retained functions. Independent reconstruction
replays original keyed batches, label normalization, automatic identities and
explicit repair transactions, then uses the existing boundary compiler and
canonical handoff. Retained functions may change only through an explicit checked
companion revision. Deferred execution resumes from cached calls; uncertainty does
not replenish budgets. A failed optional-process route retains the existing
fallback policy and shared construction-wide budget.

## Experiment and interpretation

The fresh pilot imports exactly the eight saved variable inventories and public
train/validation inputs from v6. It keeps the same seeds, full/brief-only variants,
proposer settings, total budgets and two gain policies: eight fresh constructions,
sixteen fits. Historical calls and models are report-only, never proposer inputs.
No variable selection is rerun. No automatic continuation or test access follows.

Changing prompts and response schemas can change sampled scientific models even
with matched seeds. This is a combined interface comparison, not an isolated
estimate of the physical value of sharing. Report construction completion,
fallbacks, delivery/repair counts, physical tokens, NMSE and numerical replay
separately. Legacy counters that were not recorded remain null.

## Local checks

```bash
.venv/bin/python -m pytest -q tests/test_identified_function_stage.py \
  tests/test_process_revision_runner.py tests/test_process_revision_confirmation.py
.venv/bin/python -m scripts.smoke_process_revision_confirmation \
  --output /tmp/identified-function-confirmation-smoke
bash scripts/run_tests.sh full
.venv/bin/python -m pytest -q tests/test_shared_process_pilot.py
.venv/bin/ruff check .
```

The smoke uses prescribed toy replies and synthetic toy observations, with no live
LLM calls. It exercises the actual canonical handoff, both frozen fit arms,
independent numerical replay and idempotent resume.

## ACES submission

Set `AF_COMMIT` to the full supplied commit first. Run in an ACES shell; archives
and outputs use group scratch to reduce pressure on the personal inode quota.

```bash
(
  set -euo pipefail
  : "${AF_COMMIT:?Set AF_COMMIT to the supplied commit first}"
  export AF_COMMIT
  AF_BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  export AF_REPO_ROOT="$AF_GROUP/repos/process-revision-${AF_COMMIT:0:7}"
  export AF_PYTHON="$AF_BASE/.venv/bin/python"
  export AF_SOURCE_ROOT="$AF_GROUP/detention-process-pilot-v6"
  export AF_AUDIT_ROOT="$AF_GROUP/process-revision-audit-v1"
  export AF_OUTPUT_ROOT="$AF_GROUP/detention-process-pilot-v7"

  git -C "$AF_BASE" fetch origin codex/prefit-aces-v1
  git -C "$AF_BASE" cat-file -e "$AF_COMMIT^{commit}"
  if [[ ! -e "$AF_REPO_ROOT" ]]; then
    mkdir -p "$AF_REPO_ROOT"
    git -C "$AF_BASE" archive "$AF_COMMIT" | tar -x -C "$AF_REPO_ROOT"
    printf '%s\n' "$AF_COMMIT" > "$AF_REPO_ROOT/SOURCE_COMMIT"
  fi
  [[ "$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")" == "$AF_COMMIT" ]]
  bash "$AF_REPO_ROOT/scripts/hpc/submit_process_revision_confirmation_aces.sh"
)
```

The submission prepares the frozen plan and submits the existing four-job chain:
CPU preflight, one-H100 proposer, sixteen one-CPU fit tasks (four concurrent),
and report. Submission is checkpointed; rerunning the same command reuses job IDs.

After completion:

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/detention-process-pilot-v7
cat "$ROOT/COMPARISON.md"
jq '{historical_status_counts, current_status_counts,
     historical_accounting, current_accounting}' "$ROOT/comparison.json"
```

`EQUATIONS.md` contains the fitted equations. Each construction's
`review/function_stage.json` or `fallback/function_stage.json` stores the full
`result.function_delivery.ledger`, including raw delivered batches and explicit
repair decisions. Do not overwrite or delete historical v6/audit results.

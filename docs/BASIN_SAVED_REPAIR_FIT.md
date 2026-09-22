# Saved-reply basin fitting handoff

This opt-in milestone consumes the completed `basin-equation-repair-v1`
campaign and its `basin-repair-handoff-audit-v1` audit. It makes **no new LLM
calls**. Public prompts, benchmark data, production defaults and the
`collocation-single-target-v2` fitter are unchanged.

## Binding normalization

`basin-assembled-model-repair-3` reuses the v2 equation compiler. It adds two
mechanical normalizations, recorded beside the original reply:

- An empty binding entry changes nothing. Its parameter must already exist,
  be a recorded alias, or be separately declared in that reply. This does not
  invent a parameter, assign a value, or free a previously fixed parameter.
- Compatible `same_as` cycles state parameter equality, not a dynamic algebraic
  loop. Remove one redundant edge and propagate the resulting identity through
  all uses. Prefer an existing parameter as representative, then lexical order.
  Acyclic bindings retain their requested direction. Existing role, scope,
  domain and bounds checks remain in force. Self-identities are no-ops.

Conflicting value/sharing operations, unknown equation names, incompatible
parameter declarations and covariates used as parameter identities remain
errors. The runtime does not infer reciprocal area factors from prose. Legacy
repair-v1 and repair-v2 semantics are preserved for exact historical replay.

## Frozen child choice

The prepare job independently replays cached requests and transactions against
their actual historical predecessors. It reproduces the completed v2 audit,
then evaluates those same replies under v3. A newly accepted counterfactual
patch never becomes the predecessor of a later historical reply.

One child is chosen per arm, before fitting or inspecting scores:

1. Preserve every historically confirmed attempt and its recorded result;
   do not refit it, even if the new static checks flag a problem.
2. Otherwise take the **latest eligible reply**, in its actual displayed
   historical context. Eligibility requires compilation, a changed candidate
   relative to the original parent, and no failed scoped static check.
3. If no reply qualifies, use the actual saved final draft if eligible.
4. Otherwise record `no_eligible_child` or `unavailable`; do not fit.

This rule may select an earlier eligible reply when a later reply is blocked.
That is an explicitly recorded diagnostic child, not a reconstruction of what
the old campaign actually accepted. No NMSE is used in this choice. The complete
input ledger and selected request/context hashes are sealed before fitting.
The two historical fitted outcomes remain separately labelled evidence.

## Fitting, recovery and reporting

Each selected new child receives one attempt from the established sibling-fit
runner. Compatible parent parameter values and initializer parameters are
retained; new declarations use the existing deterministic initialization policy.
Training controls fitting. Validation is reported descriptively and never fits
its own initials. No new fitting algorithm, budget extension or proposer run
is introduced. Historical parents are not equal-total-budget controls.

The runner keeps consumed-attempt semantics: a started incomplete optimization
does not get a fresh budget on resume. Completed results are verified and reused.
BDF/Radau replay follows completed fits; an interrupted replay is recorded as
such rather than silently repeated. Numerical agreement does not establish
predictive accuracy. Fitted equation checks, pre-fit static checks, NMSE and
numerical replay are reported separately. Unverified static checks do not erase
known failures at historical fitted parameters.

The scheduler uses three CPU jobs: prepare, a 16-element array with at most four
concurrent workers, and report. Ineligible and historical arms are cheap no-ops.
Each fitting worker requests one CPU, 16 GB and a 40-minute scheduler ceiling;
the numerical budget is still the frozen profile. There is no GPU allocation.
Submission receipts prevent duplicate or uncertain resubmission. No automatic
model promotion, test access, new campaign, or budget extension follows.

## ACES commands

Use the existing Python environment and group scratch. Replace the commit
placeholder with the full pushed commit supplied with this milestone.

```bash
(
  set -euo pipefail
  export AF_COMMIT=REPLACE_WITH_PUSHED_COMMIT
  AF_BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  export AF_REPO_ROOT="$AF_GROUP/repos/basin-saved-fit-${AF_COMMIT:0:7}"
  export AF_SOURCE_ROOT="$AF_GROUP/basin-equation-repair-v1"
  export AF_GATE_ROOT="$AF_GROUP/basin-repair-handoff-audit-v1"
  export AF_OUTPUT_ROOT="$AF_GROUP/basin-saved-repair-fit-v1"
  export AF_PYTHON="$AF_BASE/.venv/bin/python"

  git -C "$AF_BASE" fetch origin codex/prefit-aces-v1
  git -C "$AF_BASE" cat-file -e "$AF_COMMIT^{commit}"
  if [[ ! -e "$AF_REPO_ROOT" ]]; then
    mkdir -p "$AF_REPO_ROOT"
    git -C "$AF_BASE" archive "$AF_COMMIT" | tar -x -C "$AF_REPO_ROOT"
    printf '%s\n' "$AF_COMMIT" > "$AF_REPO_ROOT/SOURCE_COMMIT"
  fi
  [[ "$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")" == "$AF_COMMIT" ]]
  module load GCCcore/13.2.0 Python/3.11.5
  export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT"
  export PYTHONDONTWRITEBYTECODE=1
  "$AF_PYTHON" -m scripts.submit_basin_saved_fit \
    --source "$AF_SOURCE_ROOT" --gate "$AF_GATE_ROOT" --root "$AF_OUTPUT_ROOT"
)
```

After the jobs finish:

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/basin-saved-repair-fit-v1
cat "$ROOT/SUMMARY.md"
jq '{status_counts, selection_counts, classification_counts,
     newly_mechanically_valid_attempts, maximum_new_fits,
     new_fit_results, new_fitted_checks, llm_calls}' "$ROOT/summary.json"
```

`EQUATIONS.md` contains frozen selected equations. `plan.json` records each
choice and its original context; `inputs.json` preserves every audited reply,
normalization, compiled child, original parent, and historical repair result.
`results/<task>/fit/` holds the normal frozen-fitter artifacts and warm-start
provenance. Missing and interrupted results remain visible.

Check scheduler status without a remote session from this agent:

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/basin-saved-repair-fit-v1
AF_JOBS=$(jq -r '.jobs | [.prepare,.fit,.report] | join(",")' "$ROOT/submission_manifest.json")
sacct -j "$AF_JOBS" --format=JobID,JobName%26,State,ExitCode,Elapsed,MaxRSS
```

## Local verification

```bash
.venv/bin/python -m pytest -q tests/test_basin_repair_v3.py tests/test_basin_saved_fit.py
.venv/bin/python -m scripts.smoke_basin_saved_fit --output /tmp/basin-saved-fit-smoke-new
bash scripts/run_tests.sh full
.venv/bin/ruff check .
```

The smoke uses prescribed synthetic cached replies, performs one real frozen
child fit and independent solver replay, and checks exact resume and preservation
of historical source bytes. It does not assert that the fitted model is accurate.

Verification on 2026-09-22: 73 focused tests and the 11 separately excluded
shared-process tests passed. The full runner passed 2,796 bulk tests (seven
wall-clock-sensitive failures passed on serial retry), then all 66 dedicated
timing-sensitive tests. Eight optional PyTorch tests were skipped. The final
real-fit smoke passed with BDF/Radau agreement and unchanged resume/source bytes.
Changed-file Ruff, shell syntax and whitespace checks passed. Repository-wide
Ruff still reports 37 pre-existing findings under `analysis/claude/`.

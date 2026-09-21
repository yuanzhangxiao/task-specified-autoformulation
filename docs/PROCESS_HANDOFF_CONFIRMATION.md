# Fresh matched process-handoff confirmation

This is the live follow-up to the saved-response audit described in
`docs/PROCESS_HANDOFF_RELIABILITY.md`. That audit found five newly valid saved
attempts, with no unavailable attempts and no previously accepted attempts newly
blocked. Local acceptance of a saved reply does not establish a complete model,
successful fit, or scientific correctness. This pilot measures those outcomes
separately.

## Frozen comparison

The new `process-handoff-confirmation-1` experiment uses the existing
`detention-process-pilot-3` execution engine. Its output directory is v4; the
engine protocol remains v3 because the gain arms, fitter, and scheduler are
unchanged.

- Eight fresh constructions: coupled/independent basin cases, seeds 0/1, and
  full/brief-only prompts.
- Sixteen fits: each common constructed model is compiled under
  `explicit_conversion` and `independent_gains`. Tokens are counted once per
  common construction, including optional-review fallback calls.
- Identical saved variable inventories, public data, evidence settings, model
  settings, seeds, image hash, construction limits, and fitting budgets are copied
  from the historical v3 plan. Variable selection is not rerun, so its historical
  cost is excluded from both construction token totals.
- Corrected topology can return no additional terms when the runtime has already
  supplied its signed process contributions. The original process proposal and
  unresolved conversion suggestions remain advisory context for proposing its
  one shared function. The frozen fitter is unchanged.
- No historical calls, constructed candidates, or fitted parameters initialize
  the new construction. Historical outcomes are copied into `baseline.json` for
  reporting only. The new plan identity gives fresh requests their own namespace.
- Optional-process failures still use the existing fallback and total budget.
  Existing hard scientific/structural checks remain in place. There are no test
  data, new scientific judge calls, or automatic follow-up rounds.

This compares the corrected construction package with the historical run. It is
not an isolated estimate of the empty-addition fix: the function handoff context
also changed, and sampled models may differ despite matching seeds. The two gain
policies within the new run use exactly the same constructed shape model.

## Preparation and resume

Preparation verifies the historical source against every file hash recorded by
the saved-response audit. It rejects incomplete historical results, missing saved
attempts, previously accepted attempts newly blocked, runtime/package differences,
or an overlapping output directory. Historical data and results remain unchanged.

The new plan binds the package source, launch scripts, imported scientific inputs,
baseline snapshot, and successful audit identities. Workers use the frozen new
root; historical source and audit directories are not needed on compute nodes.
Repeating preparation with the same inputs is idempotent. Repeating a completed
submission prints its existing job IDs. Partial or uncertain scheduler submissions
remain blocked pending inspection rather than silently duplicating jobs.

Existing fit markers and consumed budgets are preserved. A repeated command does
not authorize a fresh numerical attempt after an interrupted fit.

## ACES commands

Set `AF_COMMIT` to the full commit supplied with this milestone, then run this
block on ACES. Both the archive and output use group scratch. Reuse the same
Python environment as v3; no new environment or model download is requested.

```bash
(
  set -euo pipefail
  : "${AF_COMMIT:?Set AF_COMMIT to the supplied commit first}"
  export AF_COMMIT
  AF_BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  export AF_REPO_ROOT="$AF_GROUP/repos/basin-confirm-${AF_COMMIT:0:7}"
  export AF_SOURCE_ROOT="$AF_GROUP/detention-process-pilot-v3"
  export AF_AUDIT_ROOT="$AF_GROUP/process-handoff-audit-v1"
  export AF_OUTPUT_ROOT="$AF_GROUP/detention-process-pilot-v4"
  export AF_PYTHON="$AF_BASE/.venv/bin/python"

  git -C "$AF_BASE" fetch origin codex/prefit-aces-v1
  git -C "$AF_BASE" cat-file -e "$AF_COMMIT^{commit}"
  if [[ ! -e "$AF_REPO_ROOT" ]]; then
    mkdir -p "$AF_REPO_ROOT"
    git -C "$AF_BASE" archive "$AF_COMMIT" | tar -x -C "$AF_REPO_ROOT"
    printf '%s\n' "$AF_COMMIT" > "$AF_REPO_ROOT/SOURCE_COMMIT"
  fi
  [[ "$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")" == "$AF_COMMIT" ]]
  bash "$AF_REPO_ROOT/scripts/hpc/submit_process_handoff_confirmation_aces.sh"
)
```

The manifest records four stages: CPU preflight; one H100 proposer allocation;
sixteen one-CPU fitting tasks (at most four concurrent); and a CPU report after
the fitting array terminates. No second construction round is submitted.

Inspect submission and scheduler status:

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/detention-process-pilot-v4
cat "$ROOT/submission_manifest.json"
JOBS=$(jq -r '.jobs | [.[]] | join(",")' "$ROOT/submission_manifest.json")
sacct -j "$JOBS" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

After completion:

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/detention-process-pilot-v4
cat "$ROOT/COMPARISON.md"
jq '{historical_status_counts, current_status_counts,
     historical_accounting, current_accounting}' "$ROOT/comparison.json"
jq '.rows[] | {task, status: .current.status,
     fallback: .current.fallback_used, fit: .current.fit,
     findings: .current.equation_diagnostics.findings,
     duplicate_laws: .current.equation_diagnostics.duplicate_process_laws,
     cancellation: .current.equation_diagnostics.transfer_cancellation}' \
  "$ROOT/comparison.json"
```

`SUMMARY.md` / `summary.json` retain the detailed existing gain-policy report.
`EQUATIONS.md` lists constructed equations. `COMPARISON.md` / `comparison.json`
add historical versus corrected construction status, token accounting, complexity,
train/validation NMSE, replay and advisory equation facts. All planned arms remain
in the reports; missing metrics are unavailable, never zeros.

Numerical replay measures solver agreement. It does not establish accurate
prediction. Shared-process counts, syntactic threshold operators and cancellation
of declared transfers do not certify full scientific mechanism compliance. Inspect
the equations and independent-control behavior along with output NMSE before
deciding on the next milestone. This experiment does not automatically select a
gain policy or promote shared processes to the full pipeline.

## Local verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/python -m scripts.smoke_process_handoff_confirmation \
  --output /tmp/process-handoff-confirmation-smoke
```

The smoke uses prescribed offline scientific replies and temporary synthetic data,
then runs real construction, both gain assemblies, fitting, numerical replay and
resume. It is not a result on the basin benchmark.

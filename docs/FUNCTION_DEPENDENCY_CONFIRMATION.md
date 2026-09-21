# Fresh function dependency confirmation — basin v5

The saved-response audit found 35 already-valid replies, 22 newly admissible
public-covariate additions, 14 replies requiring an explicit proposer dependency
revision, and nine blocked replies. No previously accepted reply became blocked;
no saved reply was unavailable. These are reply/slot counts, not recovered models.

This milestone measures complete construction and fitting after enabling
`local-function-dependencies-1`. It does not change the policy implemented in
`FUNCTION_DEPENDENCY_REPAIR.md`, the fitter, the scientific prompts, benchmark
data, model capacity, or gain definitions.

## Matched experiment

- Eight fresh constructions: coupled/independent cases, seeds 0/1, full/brief-only.
- Sixteen fits: the same fresh construction under `explicit_conversion` and
  `independent_gains`. Tokens are counted once per construction, including fallback.
- Import the exact saved variable inventories, public train/validation data,
  evidence settings, limits, model settings, serving image hash, and fit budgets
  from v4. Variable selection is not rerun or charged again.
- The proposer remains GPT-OSS-20B with its saved low-reasoning settings. The new
  request namespace prevents reuse of historical calls. Saved candidate equations,
  fitted parameters, and historical outcomes are not proposer or fitter inputs.
- Function replies may add permitted public covariates. An explicit atomic repair
  may revise dependencies while preserving required paths, causal boundaries,
  shared identities, signs and conversions. The shared function is compiled once.
- Optional-process failures retain the existing fallback within the original
  total construction budget. No extra fit is granted for a rejected construction.

Historical v4 results are the control. This measures the combined new dependency
policy and its prompt changes, not an isolated mechanical effect on identical
sampled models. Matching seeds does not make different prompts produce identical
replies. Neither gain policy is automatically selected by the experiment.

The scientific inputs and execution engine still have protocol
`detention-process-pilot-3`; the new experiment identity is
`function-dependency-confirmation-1`. Its output directory is v5.

## Preparation, provenance and resume

Preparation requires the successful sealed dependency audit and completed v4
artifacts. It checks the audit's source hashes and matrix, missing replies and
regressions, and Python/package compatibility. The baseline is snapshotted with
file hashes; the new plan binds that snapshot, all matched inputs, the new policy,
package source and launcher files. Separate directories are mandatory.

Workers read only the new frozen root. Historical directories can subsequently
be offline. Repeating preparation with identical inputs and submission after a
completed submission is idempotent. Partial or uncertain scheduler submissions
require inspection of receipts; the script does not silently duplicate them.
Consumed fit attempts and replay markers retain the existing resume semantics.

No private reference or test data are opened. No scientific judge, extra round,
automatic model promotion, or automatic follow-up is added.

## ACES submission

Set `AF_COMMIT` to the supplied commit and run on ACES. Use group scratch and the
existing environment to limit personal-scratch file consumption.

```bash
(
  set -euo pipefail
  : "${AF_COMMIT:?Set AF_COMMIT to the supplied commit first}"
  export AF_COMMIT
  AF_BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462

  export AF_REPO_ROOT="$AF_GROUP/repos/function-confirm-${AF_COMMIT:0:7}"
  export AF_SOURCE_ROOT="$AF_GROUP/detention-process-pilot-v4"
  export AF_AUDIT_ROOT="$AF_GROUP/function-dependency-audit-v1"
  export AF_OUTPUT_ROOT="$AF_GROUP/detention-process-pilot-v5"
  export AF_PYTHON="$AF_BASE/.venv/bin/python"

  git -C "$AF_BASE" fetch origin codex/prefit-aces-v1
  git -C "$AF_BASE" cat-file -e "$AF_COMMIT^{commit}"
  if [[ ! -e "$AF_REPO_ROOT" ]]; then
    mkdir -p "$AF_REPO_ROOT"
    git -C "$AF_BASE" archive "$AF_COMMIT" | tar -x -C "$AF_REPO_ROOT"
    printf '%s\n' "$AF_COMMIT" > "$AF_REPO_ROOT/SOURCE_COMMIT"
  fi
  [[ "$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")" == "$AF_COMMIT" ]]
  bash "$AF_REPO_ROOT/scripts/hpc/submit_function_dependency_confirmation_aces.sh"
)
```

The four scheduled stages are CPU preflight, one H100 proposer allocation,
sixteen one-CPU fits with at most four concurrent, and a report after the fitting
array terminates. The serving image and model cache are reused. No second round
is submitted.

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/detention-process-pilot-v5
cat "$ROOT/submission_manifest.json"
JOBS=$(jq -r '.jobs | [.[]] | join(",")' "$ROOT/submission_manifest.json")
sacct -j "$JOBS" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

## Results to inspect

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/detention-process-pilot-v5
cat "$ROOT/COMPARISON.md"
jq '{historical_status_counts, current_status_counts,
     historical_accounting, current_accounting}' "$ROOT/comparison.json"
jq '[.construction_pairs[] | {task,
     fallback: .current.fallback_used,
     repairs: .current.function_repairs}]' "$ROOT/comparison.json"
jq '[.rows[] | {task, status: .current.status,
     connectivity: .current.connectivity,
     gains: .current.fitted_process_gains,
     diagnostics: .current.equation_diagnostics}]' "$ROOT/comparison.json"
```

`SUMMARY.md`, `summary.json` and `EQUATIONS.md` retain the existing detailed gain
report and equations. `COMPARISON.md` / `comparison.json` add historical versus
fresh outcomes, whole-construction tokens, atomic-function repair tokens/calls,
fallback frequency, dependency revisions, complexity, train/validation NMSE,
solver replay, process connectivity and fitted process gains. Missing or
unmeasured calls are explicit beside the observed token sums. Missing models
remain in denominators and unavailable NMSE values are not replaced by zero.

Connectivity is syntactic reachability, not proof of fitted activity: a fitted
gain can be negligible. Threshold syntax, shared-process counts and declared
transfer cancellation do not certify full scientific compliance. Solver replay
is numerical consistency, not prediction accuracy. The independent control and
equations still need inspection. In particular, check whether the three formerly
covariate-blocked review routes complete without fallback and whether the other
coupled routes obtain explicit dependency repairs and useful fitted models.

## Local verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/python -m scripts.smoke_function_dependency_confirmation \
  --output /tmp/function-dependency-confirmation-smoke
```

The smoke uses a separate temporary toy with prescribed offline replies. It
exercises a real explicit dependency edit, automatic area inclusion, independent
reconstruction, both gain assemblies, two actual frozen-fitter runs, numerical
replay and deterministic resume. It is not a basin experiment result.

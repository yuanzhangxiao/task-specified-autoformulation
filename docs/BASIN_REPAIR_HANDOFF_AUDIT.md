# Basin repair handoff: saved-response milestone

The completed `basin-equation-repair-1` campaign produced two fitted children and
fourteen unconfirmed drafts. This milestone implements
`basin-assembled-model-repair-2` and checks the saved replies before another live
repair or fitting campaign. The original protocol, source files and results remain
unchanged. The new command does not call an LLM, optimize, simulate, select or
promote a candidate. Production defaults and the frozen fitter are unchanged.

## Changes

- An explicit valid patch authorizes its changes. No separate acceptance call is
  required. The historical `accept_displayed` field remains readable and does not
  invalidate accompanying edits. Current static failures keep a valid compilation
  as a repair draft rather than making it eligible for fitting.
- New declarations and fixed/shared bindings resolve in one transaction against
  the complete effective equations. A new shared parameter may be declared and
  referenced in `same_as` in the same reply. Existing domains and bounds remain
  binding. Both `value` and `same_as`, neither of them, cyclic identities,
  incompatible roles, and covariates used as parameter identities remain errors.
- Eliminated parameter spellings have an explicit alias ledger. Repeating the
  same binding or using the old spelling in an RHS does not resurrect a parameter.
  A contradictory binding is rejected; the proposer must refer to the current
  identity explicitly. Reusing an algebraic law propagates through its consumers.
- Existing component kinds are runtime-owned in this fixed-state pilot. An
  inconsistent `kind` is normalized and logged. Unknown component names receive
  the actual names; the runtime does not guess that `state_equations` means
  `h_down`. New algebraic processes still require explicit declarations.
- The prospective payload presents complete current equations, aliases, component
  kinds and current unresolved findings. Historical fitted-parent witnesses are
  explicitly separate. Scientific hypotheses never override executable checks.
- Static outlet probes evaluate only numbers provably independent of unknown
  parameters. For example, `k*max(0,h-warning_depth)` is zero at a public probe
  above the outlet crest but below the warning level, regardless of finite `k`.
  A correct law with an unknown gain remains unverified above its crest.
  Unsupported partial expressions remain unknown: neither `0*exp(k)` nor
  `0*unmapped_state` is silently declared safe. Existing fitted-point checks
  remain unchanged.

The implementation reuses the restricted expression parser, parameter-role
resolution, whole-model compiler, causal initializers, public pathways and
negative-control rules. It does not add hydraulic terms, choose signs, infer
reciprocal area factors, assign unknown numerical values or interpret prose.

## What the audit measures

`scripts/audit_basin_repairs.py` accepts the completed repair-v1 directory. It
verifies seals and independently replays the original transaction ledger and
cached requests under the unchanged historical policy. It then evaluates each
saved reply under v2 **against the model that reply actually saw**. A newly valid
counterfactual edit is never spliced into the context of a later saved reply.

The report separates:

1. Mechanically blocked replies, including genuine ambiguity and scientific
   contract violations.
2. Mechanically valid replies that still require static scientific repair.
3. Replies whose resulting draft is eligible for a later fitting attempt.
4. Actual saved final drafts that become eligible solely by removing the
   confirmation requirement. This is a model count, distinct from reply counts.
5. Unexpected mechanical acceptance regressions and missing artifacts.

An eligible draft has no demonstrated failure in these scoped static checks.
Unverified checks remain unverified; eligibility does not certify conservation,
general scientific validity or accurate trajectory prediction. There are no new
NMSEs. Historical parent/child scores are not used to select replies.

Each row stores the raw-reply identity, actual context hash, proposed rebuilt
bundle, mechanical normalizations, parameter aliases, current assessment and
historical outcome. Resume verifies the source-byte manifest, code, entrypoints
and runtime, and reproduces sealed rows. Changed inputs fail closed. A missing
proposal is reported as unavailable, not a successful zero-error model.

Review the audit before implementing the next fitting handoff. The v2 transition
and prompt are available for that adapter; this milestone deliberately does not
launch a new live proposer campaign or rewrite the v1 runner.

## Run on ACES

Use group scratch and the existing environment. No data upload, GPU or new clone
is needed. Replace the commit placeholder with the full pushed commit.

```bash
(
  set -euo pipefail
  export AF_COMMIT=REPLACE_WITH_PUSHED_COMMIT
  AF_BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  export AF_REPO_ROOT="$AF_GROUP/repos/repair-handoff-${AF_COMMIT:0:7}"
  export AF_SOURCE_ROOT="$AF_GROUP/basin-equation-repair-v1"
  export AF_OUTPUT_ROOT="$AF_GROUP/basin-repair-handoff-audit-v1"
  export AF_PYTHON="$AF_BASE/.venv/bin/python"

  git -C "$AF_BASE" fetch origin codex/prefit-aces-v1
  git -C "$AF_BASE" cat-file -e "$AF_COMMIT^{commit}"
  if [[ ! -e "$AF_REPO_ROOT" ]]; then
    mkdir -p "$AF_REPO_ROOT"
    git -C "$AF_BASE" archive "$AF_COMMIT" | tar -x -C "$AF_REPO_ROOT"
    printf '%s\n' "$AF_COMMIT" > "$AF_REPO_ROOT/SOURCE_COMMIT"
  fi
  [[ "$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")" == "$AF_COMMIT" ]]
  mkdir -p "$AF_OUTPUT_ROOT/logs"
  sbatch --export=ALL \
    --output="$AF_OUTPUT_ROOT/logs/audit-%j.out" \
    --error="$AF_OUTPUT_ROOT/logs/audit-%j.err" \
    "$AF_REPO_ROOT/scripts/hpc/run_basin_repair_audit_aces.sh"
)
```

The job first runs the focused regression tests, then audits the saved replies
on one CPU with 8 GB and a 20-minute scheduler ceiling. There is no follow-up job.

After completion:

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/basin-repair-handoff-audit-v1
cat "$ROOT/SUMMARY.md"
jq '{status_counts, classification_counts, newly_mechanically_valid_attempts,
     newly_eligible_saved_drafts, unexpected_acceptance_regressions,
     unavailable_saved_attempts, llm_calls, optimizer_calls, solver_rollouts}' \
  "$ROOT/summary.json"
```

If any replies remain blocked:

```bash
jq '[.rows[] | .task as $task | .attempts[] |
     select(.classification == "mechanically_blocked") |
     {task: $task, attempt, historical_error, error}]' "$ROOT/summary.json"
```

## Local checks

```bash
.venv/bin/python -m pytest -q tests/test_basin_repair_v2.py tests/test_basin_repair_audit.py
.venv/bin/python -m scripts.smoke_basin_repair_audit --output /tmp/basin-repair-audit-smoke-new
bash scripts/run_tests.sh full
.venv/bin/ruff check .
```

The smoke builds synthetic historical fixtures with prescribed cached replies,
checks release of an otherwise stranded draft, and verifies resume and original
file preservation. It performs no fitting or live provider calls.

Local verification on 2026-09-22: the full test runner passed 2,778 parallel tests
and 66 timing-sensitive tests, with eight optional PyTorch skips. The eleven
shared-process tests excluded by that runner passed separately. Final focused
verification passed 61 tests, covering both repair policies, the saved-response
audit and the equation checks, including the last alias/payload edge cases.
The final audit smoke passed with no live calls, fits or rollouts and unchanged
source files on resume. Changed-file Ruff and shell syntax pass. Repository-wide
Ruff retains 37 pre-existing findings under `analysis/claude/`.

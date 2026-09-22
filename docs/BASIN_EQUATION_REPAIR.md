# Saved basin equation repair

`basin-equation-repair-1` is one opt-in milestone after the public equation audit
of `detention-process-pilot-v7`. It reuses the whole-model revision compiler,
cached proposer transport, compatible-parameter warm start, and frozen
`collocation-single-target-v2` fitter. Production construction is unchanged.

## Decisions and runtime responsibilities

The proposer sees the public specification, complete assembled equations,
parameter declarations, unchanged initialization plan, historical equation
witnesses, and current static findings. It receives no validation scores, test
data, generating equations, or private state trajectories. Historical witnesses
are explicitly labeled as belonging to the old fitted model. A failure in a
complete derivative does not uniquely identify a faulty outlet term.

The proposer supplies scientific content through the existing equation patch
interface. It can replace several complete RHSs in one transaction, update one
named process law used by several consumers, and add/remove algebraic processes.
This pilot keeps dynamic states, target mappings, and initialization rules fixed.
It imposes no six-equation/two-variable edit limit.

Two small explicit declarations remove redundant bookkeeping:

```json
{"parameter": "existing_gain", "value": 1, "same_as": null}
{"parameter": "gain_down", "value": null, "same_as": "gain_up"}
```

The first fixes a known coefficient; the second reuses one compatible parameter
identity. These are scientific decisions, not inferred from prose or fitted
values. The runtime substitutes identifiers throughout inherited and edited
equations, removes eliminated parameters, re-lowers causal initialization, and
validates the complete transaction. Existing coefficient domains and bounds
apply. Sharing requires compatible declarations and cannot form a cycle.
Unknown hydraulic coefficients remain fitted. Different state units still need
explicit conversions: sharing coefficients alone does not prove conservation.

The runtime never reapplies the old gain compiler to the assembled child. Thus
an explicitly fixed unit coefficient stays fixed. It distinguishes directly
rewritten definitions from all transitively affected consumers, then presents
the **entire rebuilt model** on the next call. The proposer either accepts that
displayed model or supplies another coordinated patch. No duplicate process law,
consumer-by-consumer parameter bookkeeping, or remembered earlier topology is
required.

Grammar, symbols, domains, algebraic acyclicity, typed public input and memory
paths, and the independent-basin negative control are checked deterministically.
Known static public-equation violations block acceptance of a revised model;
unverified findings remain unverified. Scope and coordinate assumptions are
shown. The runtime does not interpret the hypothesis text as a scientific proof.

## Bounded pilot and interpretation

- Sixteen saved arms, arising from eight constructions. They are not sixteen
  independent construction seeds.
- At most three proposer calls per arm, inherited from the source attempt
  allowance: up to **48 calls**, including review of the rebuilt model. Invalid
  replies consume calls. An unconfirmed draft is saved and never fitted.
- Accepting the unchanged parent retains it without refitting. Confirmed revised
  children get at most **one** fit each, hence at most **16 new fits**.
- Compatible fitted parameters and latent initialization coefficients survive
  through `sibling_fit`; new parameters use existing deterministic starts.
  The initializer/refinement algorithms, tolerances, budgets and initialization
  contract are unchanged.
- Every completed fit receives the same public equation assessment and BDF/Radau
  replay. Replay agreement is numerical consistency, not prediction accuracy.
- Reports retain parent and child NMSE, every scoped check, provider calls and
  observed token usage. Missing measurements and unavailable fitted assessments
  are explicit. No single scientific-compliance score is manufactured.
- Historical gain-policy labels identify parents. Because repairs can change
  several equations, this is not an isolated comparison of gain policies or an
  equal-budget repair-versus-refit experiment.
- The scientific critic remains off for this milestone. There is no automatic
  model promotion, following round, or expansion of the experiment.

## Integrity and recovery

Preparation verifies the sealed v7 ledger, its public equation audit, the public
prompt profile, and independent reconstruction of all sixteen models. It freezes
copied public development data and parent models in a separate directory.
Historical files remain untouched. Worker checks bind package source, launcher
source, numerical runtime, model settings, and audit identity.

Replies, failures, and previews are cached and sealed individually. Before a
child is fitted, its full transaction sequence is independently replayed from
raw replies. A shutdown resumes from the cache; uncertain provider deliveries
remain charged and are not silently reissued. Consumed fits do not receive fresh
budgets on resume. Interrupted numerical replay is reported explicitly. The
scheduler retains submission intents and receipts; do not delete these to bypass
an uncertain submission.

## ACES execution

Use a new archive under group scratch, a full pushed commit, and the existing
environment and container. No new public-data upload is required.

```bash
(
  set -euo pipefail
  export AF_COMMIT=REPLACE_WITH_PUSHED_COMMIT
  AF_BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  export AF_REPO_ROOT="$AF_GROUP/repos/basin-repair-${AF_COMMIT:0:7}"
  export AF_SOURCE_ROOT="$AF_GROUP/detention-process-pilot-v7"
  export AF_AUDIT_ROOT="$AF_GROUP/basin-equation-audit-v1"
  export AF_OUTPUT_ROOT="$AF_GROUP/basin-equation-repair-v1"
  export AF_PYTHON="$AF_BASE/.venv/bin/python"

  git -C "$AF_BASE" fetch origin codex/prefit-aces-v1
  git -C "$AF_BASE" cat-file -e "$AF_COMMIT^{commit}"
  if [[ ! -e "$AF_REPO_ROOT" ]]; then
    mkdir -p "$AF_REPO_ROOT"
    git -C "$AF_BASE" archive "$AF_COMMIT" | tar -x -C "$AF_REPO_ROOT"
    printf '%s\n' "$AF_COMMIT" > "$AF_REPO_ROOT/SOURCE_COMMIT"
  fi
  [[ "$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")" == "$AF_COMMIT" ]]
  bash "$AF_REPO_ROOT/scripts/hpc/submit_basin_repair_pilot_aces.sh"
)
```

The submission is CPU preflight → one H100 proposer → up to sixteen one-CPU
fits (four concurrent) → summary. The preflight verifies the serving image and
runs the targeted tests. The proposer uses the same model/settings as v7.
If the audit was stored elsewhere, change only `AF_AUDIT_ROOT` to its directory
containing `freeze.json` and `summary.json`.

After completion:

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/basin-equation-repair-v1
cat "$ROOT/submission_manifest.json"
cat "$ROOT/SUMMARY.md"
jq '{status_counts, repair_status_counts, physical_calls, observed_total_tokens,
     unmeasured_calls, fitted_assessments_available,
     checks_before, checks_after_fitting}' "$ROOT/summary.json"
cat "$ROOT/EQUATIONS.md"
```

Read each result's `proposal.json` for the exact decisions, affected components,
and rebuilt draft. The adjacent `fit/freeze.json` records retained and new
parameter starts. Unconfirmed children and missing results are not improvements.

## Local verification

```bash
.venv/bin/python -m pytest -q tests/test_basin_model_repair.py tests/test_basin_repair_pilot.py
.venv/bin/python -m scripts.smoke_basin_repair_pilot --output /tmp/basin-repair-smoke-new
```

The smoke uses prescribed replies and a synthetic fixture, then performs an
actual frozen child fit and independent numerical replay. It checks cache
resume and byte-for-byte preservation of historical fixture files. It does not
call a live LLM or assert that the repair improves prediction accuracy.

Verified locally on 2026-09-22: the full test runner passed 2,757 parallel tests
and 66 timing-sensitive tests, with eight optional PyTorch tests skipped because
PyTorch is unavailable locally. The eleven shared-process tests excluded by that
runner passed separately. All 28 new focused tests are included in the parallel
count. The real child-fit/replay smoke passed, including deterministic resume and
historical-file preservation. Changed Python files and shell syntax checks pass;
repository-wide Ruff still reports 37 existing findings in `analysis/claude/`.
The live proposer repair outcomes remain to be measured on ACES.

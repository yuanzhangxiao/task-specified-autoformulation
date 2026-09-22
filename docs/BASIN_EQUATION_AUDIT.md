# Saved basin equations: public physics witnesses

This milestone audits the saved v7 gain arms before another construction experiment.
It changes neither construction nor fitting. There are no LLM calls, optimizer
calls, ODE rollouts, model edits, rejection decisions, or promotions. The calibrated
scientific critic is not called yet. Per-model public feedback packets establish a
reviewable input for a later integration milestone.

The motivating v7 comparison completed eight constructions and sixteen fits, using
415,807 measured tokens versus 595,002 in v6 (earlier variable selection excluded).
Only five of sixteen gain arms had both train and validation NMSE at most 0.1.
Saved equations included absent outlets, same-sign internal transfers, incorrect
depth conversions and repeated inflow contributions. Completion and numerical
replay did not establish mechanism compliance.

## Rules and interpretation

The versioned `basin-public-equation-witnesses-1` profile is tied to the exact two
existing public basin specifications (ignoring boundary whitespace removed by
their schema). It does not read the benchmark generator,
reference equations, hidden states, or test data. It does not infer equations from
scientific-role prose. No specific discharge formula or coefficient is required.

The audit checks the **final gain-assembled equations** and expands algebraic
definitions through the restricted parser. Named and equivalent inlined terms get
the same analysis. It reuses the production scalar interpreter for point probes;
there is no string evaluation, generated code or new symbolic dependency.
Formal algebra is bounded in expanded nodes, terms, powers and rational size.
Unsupported identities or resource limits remain unverified.

Checks are reported separately:

- **External inflow conversion:** in physical depth coordinates, the coefficient
  of volumetric inflow should be `1/area`. An exact algebraic identity passes.
  A concrete mismatch at saved parameters and public training geometry fails that
  check. A free coefficient without a counterexample is unverified, even if sampled
  values happen to agree; parameter units are not guessed.
- **Storage-dependent outlet presence:** absence of downstream-depth dependence
  after algebraic expansion fails the necessary presence check. Dependence alone
  passes only this presence check, not outlet direction or scientific validity.
  If other unmapped states could mediate the dependence, it remains unverified.
- **Free-outlet threshold probes:** at zero input and upstream depth zero, evaluate
  downstream depths 0, half the surveyed crest, the crest, and
  `1.25*crest + 0.01 m`. Below/at crest the derivative should be zero; above crest
  it should be negative. The absolute tolerance is `1e-8 m/min`. A reported failure
  includes the actual geometry, probe states and derivatives. A pass covers only
  these points, not all levels, state invariance or a proof of a threshold law.
  Nonzero above-crest rates within the tolerance remain unverified; a tiny
  positive discharge is not falsely classified as a violation. Exactly zero
  discharge at that probe is recorded as inactive at the retained parameters.
- **Internal transfer cancellation:** inspect
  `area_up * h_up' + area_down * h_down'`. Upstream-depth contributions should
  cancel under the public assumptions. If formal cancellation is not proved,
  compare total storage derivatives at upstream depths `crest_up/2` and
  `1.25*crest_up + 0.01 m`, holding downstream depth and inputs fixed. A change
  greater than `1e-8 * max(1, abs(derivatives))` is a counterexample in m³/min.
  This examines ordinary/inlined transfers too, not only declared shared terms.
- **Multiple inflow contributions:** report expanded additive uses for review.
  Multiple summands are not automatically double counting: their total coefficient
  may be correct. This finding is advisory/unverified.

Downstream depth must be exposed by an identity observation mapping. Upstream
checks use a state named `h_up` with unspecified or metre units, **conditionally
interpreted as physical depth**. This does not certify latent-state meaning.
Different latent coordinates need an explicit storage mapping and are unverified;
the audit does not invent a mapping. Formal identities hold only on their defined
nonzero-denominator domain. No overall mechanism-compliance score is emitted.
Nonnegative storage, all dimensional identities, all state/parameter values and
total physical adequacy are not certified by this finite audit.
Failure at retained parameters does not prove that different fitted parameters
cannot satisfy the requirement. The packet makes this qualification explicit.

## Source identity and feedback boundary

The audit independently reconstructs each selected common construction from its
saved topology/function decisions, recompiles both existing gain policies, and
checks the proposal/result hash links. Missing source records remain unavailable.
Digest/ledger disagreement is an execution error, never a scientific failure.
Source paths are contained in the historical root, including symlink resolution.

The source plan contains train/validation arrays. The audit reads its sealed
container but uses **only training fixed survey covariates**, the existing public
specification, equations, and saved training-fitted parameter vectors. Observed
input/output samples, time arrays, validation scores, validation geometry and
validation trajectories never enter the checks or feedback. Missing fitted vectors
do not trigger optimization or guessed parameters; applicable numeric checks stay
unverified. Static checks can still run without a fit.

`freeze.json` binds present and absent source files plus audit implementation.
Each arm has an immutable checkpoint, and identical reruns reuse it. A source
change, including a previously missing result appearing, requires a new output
directory. Historical files are never edited. Fit arms, constructions, and scoped
check outcomes retain their distinct denominators.

Outputs:

- `SUMMARY.md`: status and per-check counts, without a compliance percentage.
- `FINDINGS.md`: per-arm diagnostic messages.
- `summary.json`: complete identities, equations, witness values, assumptions,
  missing/invalid cases, and original construction-route errors.
- `feedback/*.json`: public specification, assembled equations and scoped facts
  for later proposer/critic integration. No fit scores or measured trajectories.

The next milestone should inspect these findings before routing them into the
existing repair/critic flow. Scientific changes remain proposer-owned. Neither
passing probes nor critic approval can override a failed deterministic fact, and
an unverified check is not a demonstrated failure.

## ACES commands

Use the full pushed commit supplied with this milestone. This submits one CPU job,
8 GB, at most 20 minutes; no GPU or new fitting experiment is needed.

```bash
(
  set -euo pipefail
  : "${AF_COMMIT:?Set AF_COMMIT to the supplied full commit}"
  AF_BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  export AF_REPO_ROOT="$AF_GROUP/repos/basin-equations-${AF_COMMIT:0:7}"
  export AF_PYTHON="$AF_BASE/.venv/bin/python"
  export AF_SOURCE_ROOT="$AF_GROUP/detention-process-pilot-v7"
  export AF_OUTPUT_ROOT="$AF_GROUP/basin-equation-audit-v1"

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
    "$AF_REPO_ROOT/scripts/hpc/run_basin_equation_audit_aces.sh"
)
```

After completion:

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/basin-equation-audit-v1
cat "$ROOT/SUMMARY.md"
cat "$ROOT/FINDINGS.md"
jq '{status_counts, checks_by_code, llm_calls, optimizer_calls,
     solver_rollouts, scientific_compliance_score}' "$ROOT/summary.json"
```

Share `SUMMARY.md` and `summary.json`. Keep v7 and this audit separate.

## Local verification

```bash
.venv/bin/python -m pytest -q tests/test_basin_equation_checks.py \
  tests/test_basin_equation_audit.py
.venv/bin/python -m scripts.smoke_basin_equation_audit \
  --output /tmp/basin-equation-audit-smoke
bash scripts/run_tests.sh full
.venv/bin/python -m pytest -q tests/test_shared_process_pilot.py
.venv/bin/ruff check .
```

The smoke uses existing prescribed toy construction replies and performs no live
LLM calls or fitting. It exercises independent reconstruction, both gain arms,
missing-arm accounting, feedback output, read-only source handling and resume.

Verification for this milestone: the full regression run passed (2,726 bulk tests,
66 timing-sensitive tests; eight optional Torch skips). After final numerical
tolerance and snapshot hardening, all 45 focused audit/gain/confirmation tests
passed. The 11 separately excluded shared-process tests also passed. The audit
smoke passed with source bytes unchanged and identical resume. Changed-file Ruff
and shell syntax passed; repository-wide Ruff retains 37 pre-existing findings
in unrelated `analysis/claude` files.

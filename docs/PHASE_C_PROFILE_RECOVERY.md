# M20: finish saved-profile verification before further scaling

Protocol `phase-c-profile-recovery-1`. This is a new, separately budgeted CPU
diagnostic of all 24 completed M19 endpoints. It does not change historical runs,
production fitting defaults, construction, benchmark data or prompts. There are
no LLM calls, GPUs or test-data access. The original M19 run remains frozen.

## What this milestone must resolve

M19's scoring recovery completed all 24 results. Coefficient recovery within 1%
improved from 15/24 to 16/24 and initial-value recovery within 0.001 from 13/24 to
15/24. Six fast/slow six-state endpoints have verified alternative-vector
witnesses, but every one of the 862 in-domain profile checks exhausted its time
allowance. Two ordinary six-state solutions still have 323% coefficient error.
Those retrospective errors motivate this diagnostic; they do not choose starts,
profiles or stopping decisions inside it.

M20 first verifies every saved profile candidate. It then spends bounded extra
optimization only on training-selected unresolved profiles and endpoints whose
training loss remains above the numerical target. It distinguishes verification
of a feasible candidate from convergence of the nuisance minimization and from
any claim about parameter determination.

## Frozen export and information boundary

Source: `review-20261008-084818.tar.gz`, original M19 plan
`318cdc03bc281a226a76d4f66f08a44166ab62f54b0d7a642eb9fc5a4fce3738`.
The portable export contains all 24 endpoints and all 862 saved profile candidates,
plus the two out-of-domain points. The exporter verifies source results, completed
backends, fitting identities, sealed profile grids and checkpoint operation
locations. Each imported progress record receives a content digest. No endpoint
is removed based on prediction or parameter error.

The numerical worker receives an explicit `RecoveryInput` schema: candidate
equations, public training arrays, training-derived coordinates, bounds, generic
start, retained incumbent, original profile grid and saved parameter vectors.
Validation and reference values are separate evaluator inputs. They are read for
scoring only after the new backend and selected vector are sealed. Schema tests
reject extra evaluation fields; campaign tests check the selection-before-scoring
boundary. All original parameter and initializer domains remain unchanged.

## Equation-derived affine propagation

The new solver symbolically certifies that the supplied dynamics have the form

\[
\dot x=A(\theta)x+B(\theta)u+c(\theta),
\]

with no explicit time dependence and no state/input dependence in A, B or c.
Coefficients may be nonlinear functions of parameters. No benchmark-specific
coefficient values, state dimensions or coupling matrix enter the solver.
Nonlinear or time-varying dynamics are rejected, not locally approximated.

Over each public piecewise-linear input interval, augment the state with the
input, its constant interval slope and a constant one. The resulting autonomous
linear system advances by `expm(G * dt)`. Every sampled forcing interval is
retained, without rounding its duration. Initial states use the existing causal
initialization rules; hidden initial parameters remain fitted globals.

Parameter sensitivities use the Fréchet derivative of the matrix exponential,
including derivatives of the initial state. This is a derivative of the actual
affine rollout, not a finite-difference approximation or a regression against
estimated latent trajectories. It uses SciPy's
[expm_frechet](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm_frechet.html).
Matrix exponentials still have floating-point error; “exact” refers to the
constant-coefficient interval solution, not arbitrary precision.

Independent verification evaluates the same saved parameters through both the
matrix-exponential solver and the existing DOP853 solver of the original
equations. **Neither verification call computes a full sensitivity matrix.**
Verification requires both normalized prediction and loss agreement under M19's
declared tolerance rules. A final joint coefficient/initial sensitivity analysis
is computed once at the selected vector. Production integrators are unchanged.

## Ordered computation and budgets per endpoint

1. Recheck the incumbent and up to three saved M19 reliability alternatives.
2. Recheck all 24 or 48 frozen profile points with surviving candidates. Retain
   explicit domain-limited/missing/unresolved records. Each independent check
   receives 30 seconds, separately from optimization.
3. Freeze a shortlist of at most **four** profiles, at most one per parameter.
   Points already demonstrating a materially different near-equivalent solution
   at tolerance 1e-12 need no refit. Among remaining verified points, try those
   closest to the retained training loss first; unavailable points sort last.
   Stable profile index breaks ties. No coefficient truth or validation ranking
   participates.
4. Each shortlisted profile receives **60 seconds / 200 complete training
   residual evaluations**, shared by two nuisance starts: its saved candidate
   (or the current verified point) and the original generic start. Its coordinate
   remains fixed. All other coefficients and hidden initials remain free. A
   separate 30-second verification checks the best result. A failed/worse refit
   does not discard a better verified saved candidate.
5. If the best verified training loss is above M19's numerical target 1e-18,
   run two free joint searches: the best verified vector and the original generic
   start. Each receives **120 seconds / 300 evaluations**, then its own independent
   check. This targets observable fitting difficulty rather than selecting only
   endpoints known retrospectively to have wrong coefficients.
6. Select using the worse of the two independent training losses. Compute one
   final joint sensitivity matrix. Seal the backend before separate prediction,
   coefficient and initial-value evaluation at `after_saved` and `selected`.

The scalar target is a numerical accuracy target for these noiseless controls,
not a universal tolerance for experimental data. The original M19 profile grid
is preserved. If the selected vector moves, the report says so; it does not
pretend the old grid certifies the new center.

Both original method labels receive this same follow-up optimizer. The `arm`
field describes their M18 provenance, not a new comparison of joint fitting
against variable projection. Nuisance/free searches here are joint bounded
least squares with affine-rollout derivatives.

## Interpretation and accounting

The three numerical loss-increase tolerances remain 1e-12, 1e-10 and 1e-8.
A materially different independently verified near-equivalent solution is a
constructive ambiguity witness even if its optimizer stopped early. A high-loss
candidate is only an upper bound on the profile minimum. Verification alone
cannot exclude a better nuisance solution.

`profiles.verified` counts checked candidates, **not converged profile minima**.
This milestone does not certify full-grid minimization or global exclusion and
therefore cannot promote an endpoint to `locally_supported_on_tested_grid`.
It can report weak constraint, numerical disagreement or incomplete search.
There are no probability/95% labels and no coverage claim. Incorrect-skeleton
and noisy-data calibration remain future work.

Each check and optimization operation records its wall time, process CPU time,
allowance, actual dataset rollout calls and checkpoint. Matrix/sensitivity calls
and independent ODE checks can be distinguished by their operation stage; one
call covers the whole training split, not one trajectory or RHS evaluation.
Setup, saved-candidate verification, profile refits, free searches and final
sensitivity are reported separately. Original M19 costs are retained under
`source_cost`; they are never charged again. Retrospective scoring has separate
timing. This is additional-budget recovery, not a matched algorithm comparison.

Completed operations are reused exactly. Interrupted operations remain unresolved
with explicit incomplete accounting and their original allowance charge; resume
does not reset their budgets. New operations may continue. Task locks exclude
concurrent writers. Submission journals refuse to repeat an uncertain scheduler
submission. Source/runtime/plan identities are checked on resume.

## Verification and scope

Equation-derived propagation and joint derivatives were checked against DOP853
and finite differences on all four controls, including latent initial parameters.
Tests reject nonlinear/time-varying dynamics, changed fixed profile coordinates,
changed source training data and evaluator fields. They cover bounded shortlist
selection, failed verification, exact resume, post-seal scoring and idempotent CPU
submission.

A deliberately shortened six-state fast/slow smoke verified all 48 saved profile
candidates in 23.4 local seconds; its complete numerical work took 28.6 seconds.
It allowed only one six-call nuisance fit and two six-call free searches. These
are verification/resume smoke measurements, not prospective Delta performance
or coefficient-recovery results. The full campaign has not been run locally.

Final verification: **4,297 passed, eight skipped** in the full primary-checkout
suite (the skips require unavailable Torch); 45 focused tests and 17 tests in the
portable bundle passed. A complete controller smoke, including both independent
postfit scoring stages, finished and resumed with an identical sealed result.
Changed-file Ruff and `git diff --check` passed. Repository-wide Ruff still has
37 pre-existing unrelated findings.

## Delta commands

Upload `transfers/phase-c-fitting-m20.tar.gz` to
`/work/hdd/bibo/yxiao2/phase_c/`. The bundle includes code, tests, the frozen
configuration and the self-contained M19 export. No access to the original M19
directory is needed by the new worker.

```bash
bash <<'BASH'
set -euo pipefail
AF_CODE=/work/hdd/bibo/yxiao2/phase_c/code/fitting-m20
mkdir -p "$AF_CODE"
tar -xzf /work/hdd/bibo/yxiao2/phase_c/phase-c-fitting-m20.tar.gz -C "$AF_CODE"
bash "$AF_CODE/scripts/hpc/submit_phase_c_profile_recovery_delta.sh"
BASH
```

Default root: `/work/hdd/bibo/yxiao2/phase_c/profile-recovery-v1`.
There are 24 one-CPU tasks, six concurrent, each with a one-hour scheduler ceiling
covering the worst permitted default check/search/scoring allowances. Jobs stop
when their work finishes. Existing `AF_PYTHON` and `AF_CASADI_ROOT` environments
are reused. Set `AF_CONCURRENCY` only to change scheduling, not numerical budgets.

After completion:

```bash
bash /work/hdd/bibo/yxiao2/phase_c/code/fitting-m20/scripts/hpc/inspect_phase_c_profile_recovery_delta.sh
```

The inspector regenerates `summary.json` and a dated review archive, showing
before/after recovery, verified profile counts, confidence categories and costs.
Missing results remain explicit. No remote sessions or jobs are launched by the
local coding agent.

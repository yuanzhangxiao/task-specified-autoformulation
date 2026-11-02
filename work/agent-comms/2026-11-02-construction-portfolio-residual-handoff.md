# Four-model residual continuation: implementation and local qualification

2 November 2026. Implements the next milestone recommended in
`2026-11-02-construction-portfolio-profile-results.md`. The runbook is
`docs/PHASE_C_CONSTRUCTION_PORTFOLIO_RESIDUAL.md`.

## Scope

- Models 12/15 warm-start from the independently verified successful endpoints
  at training NMSE .449981937/.439421177. Conditional bounded least squares
  continues to estimate the 32 training initial roots. Shared coefficients and
  both timescales remain free. Maximum 2,000 objective calls and 1,800 numerical
  seconds are new, explicit reservations; internal Powell directions are not
  restored from the old run.
- Models 10/14 use the existing certified 24-coordinate reductions, with a
  separate residual-only qualification and derivative-free Powell control.
  Qualification checks the exact lift/domain proof, two fresh BDF residual
  evaluations and original-equation Radau replay. No parameter Jacobian is
  requested. Gradient, stationarity and parameter-confidence gates remain
  unresolved; this is a control of a different optimizer, not a weakened
  derivative check.
- Root-profile trials now optionally retain bounded failure histograms and
  representative shared points, trajectory identifiers and KKT/pairing evidence.
  Existing default behavior is preserved.
- Models 12/15 also test the declared shared coefficient
  `k_Uii_term_12_0_244d72574a` at exact zero, holding every other original shared
  coefficient fixed and reprofiling the roots. Zero must be feasible in both
  charts. A separate original-equation replay and meaningful training gain are
  required before selection; the control never removes a state or equation.
- Reporting separates numerical ties from improvements and evaluation limits
  from convergence. A source incumbent remains selected for a tie. The guard is
  `max(1e-10, 1e-8*loss_scale, 4*(old+new solver_loss_discrepancy))` by default.
  This is a numerical distinction, not a claim of practically important gain.

The source root is `construction-portfolio-profile-v1`, pinned to plan
`91aca141ae159930ef5d521759806346168cdd7b7f66999e3244c98eff68fcf3`.
The new root is `construction-portfolio-residual-v1`. Preparation authenticates
all seven source terminals before projecting the four selected models. The
compact input file is 2,918,150 bytes, compared with 138,865,403 bytes in the
preceding nested archive. Four source-record hashes, the training payload and
numerical settings are independently pinned. Historical records remain intact.

Numerical settings come from each model's authenticated setup receipt. This
preserves rtol/atol 1e-11/1e-13 for models 10/14 and 1e-9/1e-11 for 12/15;
using the common original policy would inadvertently loosen the nonlinear
controls by 100 times, and is explicitly covered by a regression test.

The launcher uses four one-CPU, 8-GB workers, at most two concurrently, each with
a four-hour scheduler ceiling. Largest inclusive numerical-stage reservation:
4,620 seconds. The dependent report has 30 minutes. No GPU, LLM, validation,
test evaluation, benchmark change or production-default change is included.
Model 03 remains unresolved and is outside this run.

## Bounded saved-endpoint checks

All four production endpoints passed local qualification; no long search was
run locally. Evidence is in ignored
`output/construction-portfolio-residual-smoke-20261102.json`.

| Model | Local check | Result |
| --- | --- | --- |
| 10 | Fresh residual qualification plus setup | Passed in 24.8 s; repeat gap zero; original Radau residual gap 1.93e-10; NMSE 1.749666619 |
| 14 | Fresh residual qualification plus setup | Passed in 19.8 s; repeat gap zero; original Radau residual gap 2.25e-10; NMSE 1.689072972 |
| 12 | Root profile and exact-zero control/replay | Passed in 4.0 s; zero-control NMSE .449981807 |
| 15 | Root profile and exact-zero control/replay | Passed in 4.6 s; zero-control NMSE .439421075 |

The zero-control improvements are only about 1e-7 NMSE and should not be
portrayed as a substantial performance gain. They establish feasibility and
numerical resolution of this boundary comparison. The outer search remains for
Delta. The root-control original replays agree to about 1e-9 in normalized
residuals. These are training-only reconstruction checks, not new forecasts.

The production compact projection recreates the four-task roster and passes
prepare/verify/report smoke checks without running jobs. Final status/inspection
instructions recover the actual checkout from
`launcher_execution.json.value.repo`, avoiding the prior empty/current-directory
failure.

## Accounting and verification

208 targeted tests pass, including actual toy ODEs, no-Jacobian assertions,
precision preservation, bounds, numerical ties, rejected trials, source-data
integrity, checkpoint/interrupt behavior, deterministic resume and Slurm launch
recovery. Changed Python files pass Ruff; repository-wide Ruff retains 37
pre-existing findings in unrelated `analysis/claude` files. Bash syntax and CLI
help pass.

Fresh residual qualification intentionally bypasses the cached `evaluate`
wrapper twice. Its authoritative verification ledger records all three solves;
`stage_backend_calls` records the two reduced solves. The original Radau call
is the third. Reported backend, view, profile and operation scopes may overlap
and must not be summed as disjoint physical calls.

No fitting-adequacy, global-optimality, parameter-confidence or scientific
correctness conclusion is promoted. New retained endpoints still need their
own causal-prefix forecasts before assessing generalization or issuing
equation-family revision feedback.

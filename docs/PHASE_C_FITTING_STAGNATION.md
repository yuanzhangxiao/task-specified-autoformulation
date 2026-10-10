# M25: derivative and scaling diagnostics at saved nonlinear endpoints

The user approved the M24 follow-up on 2026-10-10. This opt-in milestone asks why
profiled rollout can retain a poor loss with non-small reported gradients after
many evaluations. It audits the derivative first, then compares two optimizer
scalings. It changes no production defaults, equations, prompts or datasets and
does not extend construction or add a new collocation method.

## Fixed sources and scientific boundary

Source is M24 plan
`562879ac15f0614cc69a7110536fe5dbe2d369f75687165cf581e7b2bd525391`.
Import the `incumbent_first` arm's **incumbent-continuation/check** point for
each original start, not the final retained point or whichever method has the
best retrospective score. Thus start 2 uses the stalled `0.73493` point rather
than its better subsequent restart. Start 1 uses `0.35383`; the successful
start-0 endpoint (`1.6935e-14`) is a positive control. All three are mandatory.

The exporter verifies the original M23/M24 input identities, source problem,
shared starting vector, policy, operation status, best vector, independent check
and backend digest. The portable handoff digest is
`68a64d3d56cf31aa2a65008c2a3e851b88772f57f1c09ae485e5d13f698ad759`.
It contains no source evaluation scores. The original evaluation data remain
separate from the strictly allowlisted `TrainingProblem` passed to the audit
and optimizer. Both continuation backends are sealed before retrospective
validation/coefficient scoring starts. No test observations or LLM calls are used.

The anchored alien-device block still has six states, thirteen dynamic
coefficients and five shared hidden initials. Four output gains are solved by
bounded linear least squares inside every rollout evaluation; the outer problem
has fourteen unknowns. The public common-preparation and continuous-input
contracts are unchanged. This experiment studies three saved points on one
known, noiseless skeleton, not a new generic-start success rate.

## Shared derivative audit

Each endpoint receives one training-only audit, charged once. Let the exact
profiled residual be `r(theta)`, with output gains reoptimized for every `theta`.
Let `S` contain the fixed public/training-derived parameter coordinate units.
For a unit direction `d` in scaled coordinates compare

```text
a(d)   = J(theta) S d
f(h,d) = [r(theta + h S d) - r(theta - h S d)] / (2h)
```

Four directions are tested: the coordinate with largest absolute scaled loss
gradient, the normalized scaled gradient, and two fixed-seed random directions.
At an exactly zero gradient the coordinate direction substitutes for it.
The fixed step sizes are `1e-3`, `1e-4`, `1e-5`. Each full audit needs at most
25 profiled residual/Jacobian evaluations: one center and 24 perturbations.
The wall guard is 1,800 seconds including formulation setup.

For each direction/step, compare RMS derivative error against
`2e-7 + 5e-4 * max(RMS(a), RMS(f))`. Save the base and perturbed inner active
sets. A perturbation outside an outer parameter bound is unavailable, without
clipping or silently changing the direction. Inner active-set transitions are
explicitly inconclusive for this fixed-active-set derivative check.

A direction passes only if its two finest available same-active-set steps both
agree. Two discrepant finest steps fail; insufficient or mixed evidence is
inconclusive. All four directions must pass before either scaling fit runs.
Failed, incomplete or interrupted audits retain the source vector and report
`diagnostic_blocked`; they do not imply that the equation skeleton is wrong.
This is a bounded local check, not a proof of all Jacobian columns or derivatives
across the whole search domain. A finite-difference disagreement can also reflect
integration error or conditioning and requires inspection before changing code.

## Matched continuation after a passing audit

Each scaling policy starts a fresh TRF optimizer from exactly the same source
parameter vector. The policies share exact output profiling, fitted hidden
initials, integration settings, bounds, residual multiplier, stopping criteria,
independent DOP853/Radau checks, tighter checks when needed, bounded joint
fallback on numerical profiling failure, and conservative incumbent retention.
Neither policy gets the other's endpoint or uses validation to choose a start.

1. **`jacobian`:** current optimizer, with `x_scale="jac"` inside the existing
   centered/scaled coordinate system.
2. **`fixed_coordinates`:** use `x_scale=1` in that same coordinate system, so
   the optimizer does not further rescale step geometry from Jacobian columns.

Both policies have 120 started evaluation attempts and a 2,400-second safety
guard, including any bounded fallback. There are no new random starts. The
mean training target is `1e-12` and worst-trajectory target is `1e-11`; reaching
both must be supported by independent numerical verification. The already
accurate positive control should skip search after verification. Each check and
final joint sensitivity has 180 seconds; post-fit scoring has 300 per arm.
Equal ceilings do not imply equal actual runtime or completed evaluations.

These are alternative step geometries, not alternative loss functions or
different physical parameter units. Continuation restores parameters, not an
optimizer's internal trust-region state. No adaptive allocation rule is inferred
or deployed in this milestone.

## Progress, accounting and resume

The existing per-evaluation trace keeps started/completed counts, training loss,
best-so-far loss, parameter digest, projected gradient and active outer bounds.
SciPy's public callback additionally records distinct accepted iterates without
performing extra integrations. For displacement `dq` it reports:

```text
actual reduction    = mean(r_old^2) - mean(r_new^2)
linearized reduction = mean(r_old^2) - mean((r_old + J_q_old dq)^2)
ratio                = actual / linearized, if linearized > 0
```

The step norm is in the fixed scaled coordinates. This reconstruction is the
Gauss-Newton residual prediction, not SciPy's internal bound-corrected model or
trust-region acceptance ratio. The trust-region radius is not exposed by the
public callback and is not claimed to be recorded. A terminal evaluation that
triggers the accuracy exception can precede an accepted-iterate callback; it
remains in the evaluation trace and requires independent verification.

Completed tasks and operations resume by exact identity. Interrupted audits
are charged their full allowance and not repeated; they block both fits.
Interrupted fitting operations retain the independently verified source point
and preserve unknown call accounting. Changing policy/source/runtime requires
a separate campaign root. Retrospective evaluation is separately checkpointed.
Shared audit, new fitting, verification, sensitivity and retrospective costs are
reported separately; M24's historical costs are not hidden in the new totals.

## Delta commands

Upload `transfers/phase-c-fitting-m25.tar.gz` to
`/work/hdd/bibo/yxiao2/phase_c`, then run:

```bash
bash <<'BASH'
set -euo pipefail
cd /work/hdd/bibo/yxiao2/phase_c
tar -xzf phase-c-fitting-m25.tar.gz
bash phase-c-fitting-m25/scripts/hpc/submit_phase_c_fitting_stagnation_delta.sh
BASH
```

This creates `fitting-stagnation-v1` and submits CPU preflight, three array
tasks (one endpoint/audit/two-scaling comparison each), and a dependent report.
Default concurrency is three and each task reserves 2.5 hours to cover worst-case
search/check/evaluation allowances. Tasks stop when their bounded work ends;
the reservation does not force them to run that long. No GPUs are requested.
The existing Delta Python/CasADi environments are reused. SciPy 1.16 or newer
with the public `least_squares` callback is required and exercised in preflight.
No prior campaign folder is required after unpacking the portable input bundle.

```bash
bash /work/hdd/bibo/yxiao2/phase_c/phase-c-fitting-m25/scripts/hpc/inspect_phase_c_fitting_stagnation_delta.sh
```

Inspection reports derivative gate outcomes separately from fitting accuracy
and creates a downloadable review archive with stage results and traces.
Repeated submission reuses recorded job IDs; uncertain scheduler replies require
reconciliation instead of blindly resubmitting.

## Local qualification

Focused tests cover correct/incorrect derivatives, bound-limited and active-set
transition checks, matching initial points/budgets, accepted-step prediction,
blocked/interrupted gates, immutable source identity, post-seal evaluation,
deterministic finished-task resume and idempotent CPU submission.

The small two-state nonlinear smoke passes its 25-evaluation derivative audit.
Jacobian scaling reaches training NMSE `5.083e-13` in 27 evaluations; fixed
coordinates reach `1.492e-12` in seven. Both resume identically. These are wiring
checks on a toy fixture under short budgets, not benchmark comparison results.
Full alien-device numerical work remains for Delta.

The smoke also exposes a plausible numerical mechanism worth testing: the first
accepted step has scaled-coordinate norm `0.000322` with Jacobian scaling versus
`0.353342` with fixed coordinates. The current search multiplies residuals and
Jacobians by `1e6` for tight polishing. SciPy 1.16.3 computes its Jacobian column
scales from that amplified Jacobian and initializes the trust radius to one
when centered optimizer coordinates are zero. Thus residual amplification can
affect initial step geometry under Jacobian scaling even though it leaves the
mathematical minimizer unchanged. This is a candidate explanation for slow
startup, not an established explanation for the difficult benchmark endpoints.
M25 keeps the amplification unchanged in both arms to isolate the scaling policy.

## Completed M25 review — review-20261010-191931

Source archive: `review-20261010-191931.tar.gz`; code `54791d2`; plan
`b40ff87cb1199a5264a19b4bab928b56a8a85f88b1486eba0a89de4afda2cf41`.
All 33 sealed JSON records verify, and rebuilding the report reproduces it
exactly. All three tasks and all 75 audit evaluations completed, with complete
cost accounting. The campaign's `complete` status means all planned tasks have
records; two are deliberately `diagnostic_blocked`.

| Saved start | Training NMSE at audit point | Derivative gate | Continuation outcome |
|---|---:|---|---|
| 0 | 1.6935e-14 | Passed, four directions | Both arms verify and retain the identical accurate point; zero optimization calls |
| 1 | 0.353834 | Inconclusive | Neither scaling arm launched |
| 2 | 0.734933 | Failed under the frozen three-step rule | Neither scaling arm launched |

The start-0 independent evaluation remains train/validation
`1.6935e-14 / 2.2565e-14`; maximum coefficient relative error is `5.50e-7`.
This is preservation of the M24 result, not an M25 fitting improvement. No actual
optimization was performed anywhere in M25, so this run provides no comparison
of the two scaling methods on the difficult endpoints.

### Why the blocked audits do not yet establish a derivative defect

All sampled perturbations stayed within the outer parameter bounds and retained
the same inner active set. Start 1 has no active gain bounds; start 2 keeps two
output gains at their lower bounds throughout. Neither active-set transitions
nor a provider/worker failure caused the blocks.

The residual derivative discrepancies show the convergence pattern of central
finite-difference truncation. For a smooth residual along a direction,

```text
[r(theta + h S d) - r(theta - h S d)] / (2h)
    = J(theta) S d + O(h^2).
```

At the difficult endpoints, shrinking `h` from `1e-4` to `1e-5` reduces the
absolute discrepancy by factors of 97.2 to 100.0 across all four directions.
That supports an under-refined finite-difference check, rather than establishing
an incorrect analytic derivative. An error floor at smaller steps has not yet
been measured, so this remains an inference rather than a passed audit.

For the normalized-gradient direction, the relative RMS discrepancies are:

| Start | h = 1e-4 | h = 1e-5 | Nominal relative tolerance |
|---|---:|---:|---:|
| 1 | 0.1677% | 0.001683% | 0.05% plus absolute allowance |
| 2 | 8.4595% | 0.09415% | 0.05% plus absolute allowance |

At `h=1e-5`, **every start-1 direction passes individually**, but three lack a
second agreeing scale and are therefore inconclusive. Start 2's coordinate and
gradient directions still fail the two finest available steps; their finest
relative errors are 0.06505% and 0.09415%, respectively. They are approaching the
tolerance rapidly. The current `failed` label records failure of this bounded
check, not proof of a software bug, invalid equations or nonidentifiability.

### Costs and next recommendation

Shared audit costs are 403.37, 379.06 and 342.08 seconds (25 profile evaluations
each), totaling 18.74 worker-minutes. Start 0 additionally spends 251.32 seconds
across the two arms on setup, independent verification and final sensitivity,
plus 277.39 seconds on retrospective evaluation. Total recorded audit/arm/
evaluation time is about 27.55 worker-minutes, excluding scheduler/preflight/
report overhead. There are zero optimizer evaluations, GPUs or LLM calls.

Refine the derivative check before changing the fitting algorithm or raising
its budget. The smallest follow-up can use `h = 1e-5, 1e-6, 1e-7` with the
existing configurable policy, on the same three frozen points and in a fresh
campaign root. Keep the agreement tolerance, two-scale requirement, active-set
checks and matched continuation budgets unchanged. Only a passing gate should
release the already implemented scaling comparison. If discrepancies plateau
or worsen, investigate integration accuracy and projection conditioning before
attributing them to the analytic Jacobian. An adaptive perturbation ladder is
a sensible later improvement; it has not been implemented by this review.

No benchmark fitting was rerun locally, and no production default, model
structure, dataset or numerical tolerance was changed during this review.

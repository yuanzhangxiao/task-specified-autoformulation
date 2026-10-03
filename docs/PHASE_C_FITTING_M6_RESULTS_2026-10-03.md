# Phase C M6: harder latent fitting results

Reviewed 2026-10-03 (Pacific/Honolulu). This closes the results review for
[the harder-case campaign](PHASE_C_FITTING_CHALLENGING.md). All 36 endpoints are
accounted for. Coupled basin passes all 18 fits; alien-device passes four of 18,
comprising the same two starts under rollout-only and the hybrid. The hybrid's
successes come from rollout fitting at the original parameter guesses, not from
improved collocation initializations.

This review changes documentation only. Campaign fits and independent rollouts
were not rerun; there is no benchmark change or production fitter promotion.

## Provenance and verification

- Archive: `review-20261003-230225.tar.gz`, extracted to the ignored local folder
  `artifacts/fitting-m6-review-20261003`.
- Archive SHA256:
  `529c236d34058d43c75b1cc59180d9a8c1977f952f3a0acb1768a55484e6c04f`.
- Experiment commit: `6ba4edb3ee1c81244df21fdd3effa365a2bf8618`.
- Plan SHA256:
  `d2e5bb179b82a59107bd74186341c7b0a8a9297ef844cf85665f3cfa5da1f3c5`.
- Input SHA256:
  `97aacd4d137ad3631b11798795ad857d4febcae5780565d351277806119adfc6`.
- Delta jobs: prepare `22634862`, fit array `22634863`, report `22634864`.
- All 149 sealed JSON records verify. The report reproduces exactly. All 36
  results reconcile with common-start identities, expected worker payloads,
  backend hashes, coefficient/initial-error calculations and saved replay scores.
  The local source identity matches the frozen campaign identity.
- Both reference-replay and training-sensitivity gates pass. The alien fitted
  block has 18 nonzero singular values and normalized singular ratio 0.006420;
  basin has two, with ratio 0.876688. These are local checks at the reference,
  not global uniqueness or noise-robustness certificates.
- All 36 retained endpoints have complete independent Radau/DOP853 replays.
  Maximum normalized solver disagreement is 8.475e-6, below the 1e-4 gate.
  `complete` therefore means available evaluation, not successful optimization.
- No LLM calls or test access. Validation/reference values are used for
  qualification and post-selection evaluation, not fitting or checkpoint choice.

## What was fitted

Basin has two dynamic coefficients, known geometry/laws, and public initial
conditions. Alien has six states, one observed channel, 13 free dynamic factors
and five shared hidden initial values, fitted from 16 training trajectories.
Its four validation trajectories use the same retained coefficients and initials.

Correct equations, internal couplings and internal/output nonlinear shapes are
supplied. Those fixed constants anchor latent coordinates. This is explicitly
assisted, conditional parameter recovery, not autonomous identification of every
coefficient in the generator. No hidden trajectories are supplied to fitting.
M6 does not score latent trajectory recovery separately.

All methods receive the same generic start within each case/seed. There are
three starts, no best-of-three selection, a 900-second total fit ceiling and a
separate independent-replay allowance. Start 2 is drawn from a broader generic
range; it is not selected using reference error.

## Results

For alien, prediction requires train and validation NMSE <=0.01 and independent
solver agreement. Coefficient recovery requires every free dynamic coefficient
within 1%; initial recovery requires every shared initial within 0.01 absolute
units in the anchored coordinates. The three pass counts coincide in this run.

| Alien strategy | Prediction / coefficient / initial passes | Validation NMSE, start 0 | Start 1 | Start 2 | Mean fitting seconds |
|---|---:|---:|---:|---:|---:|
| Rollout only | 2/3 each | 7.78e-16 | 9.69e-15 | 0.12354 | 584.3 |
| Collocation then rollout | 2/3 each | 7.78e-16 | 9.69e-15 | 0.12950 | 820.8 |
| Adaptive collocation | 0/3 each | 0.82928 | 2.12495 | 0.91908 | 891.3 |
| Adaptive shooting | 0/3 each | 0.66486 | 0.83164 | 0.60487 | 813.6 |
| Cached, latest primal | 0/3 each | 1.05967 | 1.28895 | 0.79515 | 667.3 |
| Cached, screened primal | 0/3 each | 1.05967 | 0.60063 | 0.86261 | 684.5 |

Times include formulation, screening and process overhead, excluding independent
endpoint replay. Raw summary group times are sums over three starts. Native solve
schedules differ between strategy bundles; wall-limited jobs also advance through
different numbers of iterations. These are not controlled hardware speed ratios.

The successful rollout fits recover much more than outputs:

| Start | Training NMSE | Maximum dynamic coefficient relative error | Maximum shared-initial absolute error | Rollout-only seconds |
|---|---:|---:|---:|---:|
| 0 | 1.26e-15 | 5.38e-7 (0.0000538%) | 4.00e-7 | 347.7 |
| 1 | 9.74e-15 | 1.15e-6 (0.000115%) | 6.40e-7 | 505.2 |
| 2 | 0.12405 | 10.33 (1033%) | 2.52 | 900.0 |

The hybrid has exactly the same retained vectors as rollout-only for starts 0
and 1. For the other four alien arms, maximum coefficient relative error in
each fit ranges from 0.927 to 283.35, and maximum initial absolute error ranges
from 0.645 to 10.209. Their finite predictions do not imply parameter recovery.

Basin passes prediction and coefficient recovery for all six methods and all
three starts. Worst validation NMSE is 4.92e-9; worst maximum dynamic-coefficient
relative error is 0.0002611 (0.02611%). It remains a useful regression control.

## Why the alien outcomes differ

### Rollout fits: one difficult region, without recorded integration failure

Successful starts use 25 and 37 joint optimization evaluations after the initial
screen. Every recorded joint evaluation completes all training trajectories.
For start 2, all 69 recorded joint evaluations also complete; the outer wall
guard then stops the process. A further in-flight evaluation, if any, is not
counted as complete. We cannot infer that every attempted operation completed.

Start-2 training NMSE falls from 0.873 to 0.163 at evaluation 10, to 0.128 at
20, and to 0.12447 at 30. It reaches only 0.12405 by evaluation 69. Over the last
six completed evaluations the objective improves about 0.0087%. One output gain,
`output_gain_1`, approaches its zero lower bound, while `decay_x0` reaches 0.7772
against reference 0.06861. Other coefficients and hidden initials compensate.

This is evidence of slow progress in an incorrect parameter region, not evidence
of a missing mechanism, failed integration, proven nonidentifiability or a
certified local minimum. Full local rank at the true vector does not guarantee
that a distant start reaches it. A longer continuation might help, but success
is not established merely because some descent remains.

### The hybrid does not obtain a useful new start

The first two hybrid collocation initializers each hit their 300-second limit.
For all three starts, the best screened parameter vector is exactly the original
generic vector. At start 1 that vector is labeled
`collocation_checkpoint:least_violation`, but comparison of its values confirms
it is the unmodified original start. A provenance label alone is insufficient to
establish numerical progress.

The joint rollout evaluation paths match rollout-only exactly: 25 evaluations
for start 0, 37 for start 1 and the first 41 for start 2. Thus the hybrid adds
about 345 and 365 seconds to the two successes without changing their vectors.
At start 2 it leaves time for 41 evaluations instead of 69, and ends slightly
worse. This diagnoses this initializer/budget policy, not all possible hybrids.

### Collocation and shooting are limited by the current numerical implementation

Initial alien collocation has 115,218 decision variables, including 115,200
nodal variables and 18 global quantities. It has 115,200 dynamical equalities
plus 13 parameter-bound constraints. Mesh refinement raises its size to 180,114
variables. Only 11--28 iterations are recorded on each mesh before interruption;
no successful adaptive-collocation native solve is recorded.

Shooting begins with 2,304 variables and 2,299 constraints. Despite its much
smaller dimension, its first mesh records only one or two iterations; every later
mesh records only iteration 0 before the stage timeout. Some short-window errors
decrease while continuity defects remain large and full rollouts remain poor.
Those scores do not demonstrate that the assembled ODE has been fitted.

Explicit graph construction takes about 13--20 seconds per collocation mesh and
5--6 seconds per shooting mesh. That alone does not explain the hundreds of
seconds consumed. Lazy solver setup, integrated derivative evaluation, linear
algebra and checkpoint work are not fully separated in these interrupted logs.
The current smooth shooting problem requests exact IPOPT Hessians through CVODES;
their cost is a candidate to profile, not a measured explanation yet.

The controller refines the mesh after an unfinished solve. This can increase an
already expensive problem before the coarse problem is adequately optimized.
Separate optimization error from discretization error before increasing the mesh.
The present results do not establish that shooting or collocation intrinsically
cannot fit this case.

### Screening exercises the veto, but does not solve fitting

For alien, the screened second start is an earlier admissible primal in start 0
and the original cold start in starts 1 and 2. The unconditional arm imports its
latest finite, parameter-bounded primal in all three. Saved decisions show actual
vetoes for unavailable training rollouts, dynamical inconsistency, and insufficient
training improvement. For example, an almost feasible checkpoint at start 2 has
training NMSE 1.0295, worse than the original 0.8727, and is correctly rejected by
the frozen rule.

Compared with unconditional reuse, screened reuse gives the same final vector
for start 0; validation improves from 1.289 to 0.601 at start 1; but worsens from
0.795 to 0.863 at start 2 despite a small training improvement. None passes.
First-solve wall budgets yield different checkpoint pools across paired jobs
(for example, 93 versus 126 iterations at start 1). Therefore these results show
policy behavior and mixed outcomes, not a clean causal estimate of veto benefit.

Some fixed-mesh solves report `Solve_Succeeded` while retained model accuracy is
poor. Conversely, four whole-fit timeouts retain evaluable endpoints. Native
convergence, native budget limits, outer budget limits, and prediction/recovery
are separate statuses. `budget_exhausted:false` does not imply every native
stage converged: it can mean a fixed schedule ended before the outer deadline.

## Recommended next milestone

Retain all strategies and keep this qualified problem as the diagnostic target.
Do not add harder families or change the equations to explain these failures yet.

1. Diagnose shooting cost on a fixed mesh, recording objective, Jacobian, Hessian,
   setup and solve timings before termination. Compare exact Hessians with a
   limited-memory alternative under matched starts/budgets. Do not automatically
   refine a mesh merely because its native optimization was interrupted.
2. Diagnose rollout start 2 with a predeclared continuation-versus-restart
   comparison. Use training stagnation and boundary activity to decide when to
   spend on a fresh generic start; keep every start and all consumed work in the
   accounting. Reference error cannot trigger restarts or select their outcome.
3. If primal screening is tested again, replay both policies from the exact same
   saved checkpoint pool. Keep formulation reuse separate from choosing a start.
   Feasible-but-poor points and promising-but-inconsistent points require distinct
   treatment; restarting the identical cold problem is not automatically useful.

This is a proposed bounded numerical diagnostic, not an implemented M7 or an
instruction to relaunch M6. Rollout fitting is the strongest current option on
this case, but its two-of-three success is not sufficient for reliable default
promotion. Successful parameter recovery establishes an attainable result under
the stated assistance; failed starts establish an optimization reliability gap.

## Review checks and changed files

The 55 focused strategy/reuse/challenging-case tests pass. The four-arm native
reuse smoke passes independent replay and exact completed-result resume on its
small control. No alien/basin campaign refits were performed locally. Repository
`ruff check .` reports the same 37 pre-existing findings in unrelated
`analysis/claude/` scripts; no Python implementation is changed in this review.

Changed files: this results note, `PHASE_C_FITTING_PLAN.md`, and
`PHASE_C_START_HERE.md`. No implementation or benchmark artifacts are committed.

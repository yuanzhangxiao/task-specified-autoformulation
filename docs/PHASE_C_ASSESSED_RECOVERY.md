# M21: assessed fitting and matched joint/profiled recovery

Protocol `phase-c-assessed-recovery-1`; numerical result schema
`fitting-assessment-1`. This is an isolated Phase C fitting experiment. It uses
all 24 frozen M19 endpoints, each with two recovery methods, for **48 CPU tasks**.
It does not change construction, production fitting defaults, benchmark data,
or prompts. No LLM calls, GPUs, or test observations are needed.

## Questions and scope

1. Preserve usable incumbents when a stricter uncertainty check is unresolved.
2. Diagnose and correct the five M20 parameter-domain failures without relaxing
   scientific bounds.
3. Compare direct warm joint rollout polishing with exact profiling of the
   input gain and hidden initials under matched stage budgets.
4. Try genuinely different feasible starts when warm recovery remains poor.
5. Return a structured fitting assessment, separating numerical reliability,
   remaining optimization work, and evidence about parameter determination.

The controls are the same three- and six-state linear systems, with ordinary
and separated timescales, three starts, and two historical fitting methods. They
have known correct equation skeletons and noiseless observations. This milestone
does not yet establish performance on nonlinear benchmark blocks, noisy data,
incorrect skeletons, or arbitrary latent-coordinate ambiguities.

## Frozen starting points and information boundary

Reuse M20's portable `profile-recovery-inputs.json` export of M19. Content digest:
`e3eb8390a348024b22b2007774226128f3680c0f35ce8b158982fe77496ad64a`.
Source M19 plan:
`318cdc03bc281a226a76d4f66f08a44166ab62f54b0d7a642eb9fc5a4fce3738`.

Both methods start from the **same M19 retained vector**, before M20's profile
refits and warm searches. Thus this is a prospective matched comparison, not
another comparison of the historical source labels. Every endpoint is included;
no selection is made using known coefficient errors. `source_arm` records
historical provenance and `method` records this experiment's treatment.

Only equations, public training arrays, parameter domains, training-derived
coordinate units, original generic start, and retained incumbent enter the
numerical worker. Saved profile vectors, source scores, validation arrays and
reference coefficients are excluded from its input schema. Source validation
checks hashes, roster and unchanged public training inputs. The fitting backend
is sealed before the separate evaluator reads validation and reference values.
These values can measure coefficient recovery afterward, but cannot route
search, choose starts, select endpoints, or stop fitting.

## The five domain failures: established cause

A local replay of the five failed M20 nuisance-start operations reproduced all
five errors. In every case, reconstructing physical parameters from centered
optimizer coordinates gave

```
parameter: a4
lower bound: 0.01
reconstructed value: 0.009999999999999787
distance below bound: 2.1337098754514727e-16
```

The domain check correctly detected an out-of-domain floating-point value. The
optimizer coordinate was at its permitted boundary; subtraction and addition
introduced the discrepancy. This is now a confirmed reconstruction-roundoff
failure, not a hypothesis about scientific bounds or a bad initial vector.

For a reconstructed component `p = anchor + units*q`, M21 permits a projection
onto its bound only when the violation is no larger than

\[
8\epsilon_{\rm machine}
\max\{1, |\mathrm{anchor}|+|\mathrm{units}\,q|+|\mathrm{bound}|\}.
\]

The rejected value, bounds, distance, allowance and action are recorded. Larger
violations remain failures. Imported initial vectors are checked strictly;
there is no general-purpose clipping of invalid input or widening of a domain.
The new search uses this restoration for both methods. Historical M20 artifacts
and its implementation remain unchanged.

## Retention and uncertainty require different numerical tests

Each candidate receives two output-only rollouts: the equation-derived matrix
exponential and independent DOP853 integration of the original ODE. Let their
normalized residuals be `r_A`, `r_B`, and losses `L_A`, `L_B`.

A candidate is **usable for fitting** when all are finite,

\[
\|r_A-r_B\|_\infty\le 10^{-7},\qquad
|L_A-L_B|\le \max(10^{-12},10^{-8}\max(L_A,L_B)).
\]

The unchanged stringent requirement for interpreting loss differences at the
1e-12 uncertainty scale is `|L_A-L_B| <= 1e-13`, as well as the same prediction
agreement condition. If this stricter test fails, a bounded tighter DOP853
check uses rtol 1e-12 and atol 1e-14. If still unresolved, the usable incumbent
remains eligible with an explicit uncertainty warning.

A usable new candidate replaces a usable incumbent only if

\[
\max(L_{A,\rm new},L_{B,\rm new})
 < \min(L_{A,\rm incumbent},L_{B,\rm incumbent}).
\]

Ties or overlapping two-solver loss ranges retain the incumbent. These ranges
are a conservative comparison rule, not rigorous integration error bounds.
Unavailable checks cannot promote a candidate. If no candidate can be verified
as usable, the original parameter vector is still saved with status
`retained_unverified`; the run does not claim a verified fit.

## Two recovery methods

**Joint:** optimize every matrix coefficient, forcing gain, and hidden initial
using bounded trust-region reflective least squares on actual affine rollouts.
Coordinates are centered and scaled, and the residual Jacobian differentiates
the interval matrix exponential and initial conditions.

**Profiled:** separate the unknowns into matrix coefficients `beta` and linear
unknowns `gamma` (input gain and affine hidden initials). At each outer vector,
propagate the observation offset and basis columns so that

\[
r(\beta,\gamma)=b(\beta)+\Phi(\beta)\gamma,
\quad
\widehat\gamma(\beta)=\underset{\ell\le\gamma\le u}{\arg\min}
\|b(\beta)+\Phi(\beta)\gamma\|_2^2.
\]

The bounded inner least-squares problem is solved at each outer residual
evaluation. The outer optimizer updates only `beta`, differentiating the
projected residual using the existing fixed-active-set derivative. At bound
changes the objective is piecewise smooth; rank-deficient inner problems remain
explicit failures. These controls reduce from six to three, or twelve to six,
outer unknowns. Every final diagnosis uses the full joint Jacobian, including
profiled coefficients and initials, rather than diagnosing only outer variables.

The symbolic certifier checks autonomous affine state/input dynamics and exact
linear separability of the selected inner unknowns. A compact `3*n` generator
advances state, affine source, and source slope for all offset/basis columns.
Fréchet derivatives include mixed forcing and initializer dependence on outer
parameters. Every interval of the public piecewise-linear forcing is retained.
No reference coefficient values or benchmark-specific matrices enter this
kernel. It is exact for the certified interval solution up to floating-point
matrix-exponential error; it is not an approximation of a nonlinear block.

## Adaptive allocation and matched budgets

| Stage | Allowance for each method |
|---|---:|
| Warm search from frozen incumbent | 180 seconds, 400 residual evaluations |
| Diverse trials, if warm fit misses target | Three starts, each 60 seconds / 150 evaluations |
| Continue best verified trial, if still above target | 180 seconds, 400 evaluations |
| Independent check of each returned candidate | 30 seconds; optional 30-second tighter check |
| Full joint sensitivity at selected endpoint | 30 seconds |
| Separate retrospective scoring | 180 seconds per stage, after-warm and selected |

Stop a search at the first wall-time or call limit, solver termination, or the
training NMSE numerical target **1e-20**. This is deliberately tighter than M20's
1e-18 target to investigate remaining hidden-initial errors. It is shared by both
methods, declared before outcomes, and applies to these noiseless controls only.
It is not a claim that noisy experimental fits should reach such a target.

After a warm search misses the target, freeze and evaluate all three short
trials. Parameter starts use stratified log sampling within strictly positive
declared bounds; signed coordinates use stratified sampling in the original
training start plus/minus two optimizer units, intersected with their bounds.
Unbounded signed initials keep their unbounded optimization domains; the finite
window controls starting-point sampling only. A seed derived from the common
training-problem digest ensures both methods get
exactly the same portfolio. These are new feasible points, not repeated copies
of a previously unsuccessful generic start or perturbations chosen from truth.

If needed, continue the trial with the lowest independently verified training
loss, **even when it has not yet beaten the incumbent**. A short trial may enter
a useful basin before reaching a good loss. This continuation never discards the
incumbent unless it produces a verified improvement. Weak sensitivity alone
does not trigger random restarts after the numerical fitting target is reached.

Ceilings, stopping rules, proposed portfolios and source endpoints are matched;
actual work may differ under adaptive early stopping. Each method's residual
call contains different work, so report wall and CPU times as well as call
counts. Maximum optimizer allowance is 540 seconds per task, excluding checks,
setup, final sensitivity and scoring. The one-hour scheduler ceiling includes
headroom and is not a fixed fitting duration.

## The fitting assessment

The parameter vector remains part of the result. Its accompanying assessment
has independent fields for fit quality, numerical status, search status,
parameter information, reasons, and a recommended next action. The campaign also
writes `assessment.md`, a compact table for all endpoints alongside the detailed
`summary.json`. This table contains training-only diagnoses, not ground-truth
parameter error.

| Evidence | Output/action |
|---|---|
| No usable independent rollout | Retain original as unverified; verify numerics |
| Usable fit but stringent numerical agreement unresolved | Retain fit; refine numerical verification before uncertainty claims |
| Strictly verified fit above numerical target | Diversify or extend optimization |
| Target reached, but weak sensitivity or alternative vectors | Keep predictive fit; investigate parameter uncertainty |
| Target reached and local sensitivity resolved | Retain with local evidence; profiles/global uniqueness remain untested |

For normalized training residuals and explicit optimizer units `s`, the joint
sensitivity diagnostic uses

\[
J_s=\frac{1}{\sqrt N}\frac{\partial r}{\partial p}
\operatorname{diag}(s),\qquad J_s=U\Sigma V^\top.
\]

It reports all singular values, column norms and the weakest joint direction.
Numerical rank uses `max(1e-8*sigma_max, 1e-15)`; the assessment flags weak
sensitivity if rank is deficient or condition number exceeds 1e6. The thresholds
are declared numerical diagnostics, not structural identifiability proofs.

Independently verified search alternatives can be ambiguity witnesses if their
loss is within 1e-12 of the selected loss and a coordinate differs by at least
0.01 optimizer units. No full profile grid is minimized in M21. If the fit is
still poor, the report states that witnesses concern that local solution, not a
certified global optimum. `probability` is null, global identifiability is not
certified, and retrospective coefficient error is absent from this assessment.

Routing currently automates numerical rechecks, target-based warm/trial/
continuation allocation, and conservative retention. It does not automatically
change equations, acquire data, or spend another profile budget in response to
weak parameter information. Those are future construction/fitting decisions.

## Checkpoints, accounting and verification

Each operation journals its identity, allowance, observed calls, best point,
domain events, wall time and process CPU time. Completed operations are reused.
An interrupted operation is marked unresolved and charged its original
allowance, without restarting optimization. Accounting incompleteness remains
visible; summed observed timings are not represented as exact totals in that
case. Setup, checks, searches, final sensitivity, historical source cost and
post-seal scoring are reported separately. Locks prevent concurrent writers and
submission receipts prevent a duplicate uncertain scheduler submission.

Tests cover actual boundary-roundoff arithmetic and material violations,
retention after stringent-check failures, two-solver improvement dominance,
bounded projected solutions, derivative agreement on all four controls,
identical diverse starts, continuation without losing the incumbent, exact
resume, interrupted allowances, source identities, training-only input fields,
selection before scoring, and CPU submission dependencies.

Local verification: 44 focused regression tests passed, including the existing
affine propagation, profile recovery and coupled-profiling tests. The portable
bundle's 19 preparation tests passed in isolation. Four deliberately shortened
real-endpoint runs (both methods on a three-state and a six-state fast/slow
endpoint) completed independent scoring and exact result reuse. Both methods
preserved the historically excluded three-state incumbent; no worse point
replaced it. These short runs check execution and retention, not full-budget
algorithm performance. Changed-file Ruff passed; repository-wide Ruff retained
37 unrelated findings.

## Delta commands

Upload `transfers/phase-c-fitting-m21.tar.gz` to
`/work/hdd/bibo/yxiao2/phase_c/`. It includes the frozen source export, code,
configuration and tests. No old campaign directory is required.

```bash
bash <<'BASH'
set -euo pipefail
AF_CODE=/work/hdd/bibo/yxiao2/phase_c/code/fitting-m21
mkdir -p "$AF_CODE"
tar -xzf /work/hdd/bibo/yxiao2/phase_c/phase-c-fitting-m21.tar.gz -C "$AF_CODE"
bash "$AF_CODE/scripts/hpc/submit_phase_c_assessed_recovery_delta.sh"
BASH
```

Default output: `/work/hdd/bibo/yxiao2/phase_c/assessed-recovery-v1`.
There are 48 one-CPU tasks, six concurrent, with 16 GB memory each, plus
preparation and reporting jobs. `AF_CONCURRENCY` can adjust scheduling without
changing numerical budgets. The existing Delta Python/CasADi environments are
reused. No remote session is started by the local coding agent.

After completion:

```bash
bash /work/hdd/bibo/yxiao2/phase_c/code/fitting-m21/scripts/hpc/inspect_phase_c_assessed_recovery_delta.sh
```

The inspector prints before/after prediction and coefficient/initial recovery,
assessments and costs, then creates a dated `review-*.tar.gz` for download.
Missing tasks remain explicit. The completed campaign is reviewed below;
implementation and local smoke checks alone are not recovery evidence.

## Completed Delta results — 2026-10-08

Reviewed `review-20261008-235055.tar.gz` (archive SHA256
`1ce793756510d062514029472f71c71d141e68d3cb978f2a502ec48837ebe6ad`).
The plan identity is
`111a6998b8cc5c74ae7d23706ede5fe913de815e3615806877421fe809d730c2`.
All **48 tasks and 177 recorded numerical operations completed**; cost accounting
is complete. Verification checked 789 sealed JSON artifacts, the frozen input
digest, the complete paired roster, and result/backend identities. Regenerating
the summary and assessment table reproduced both exactly. All 24 paired
portfolios contain identical proposed starts. No new fitting was performed in
this review.

Review verification: 36 focused regression tests passed, the real-archive report
replay matched, and `git diff --check` passed. Repository-wide Ruff still reports
37 unrelated findings. Only this runbook and the fitting-plan record are edited;
the uploaded artifacts remain outside Git.

The baseline is the same **24 M19 endpoints**, not M20's improved outputs. These
are four control systems, three seeds and two historical source methods, not 24
independent benchmark problems. The two new methods each receive every endpoint.

| Retained result | Prediction recovered | Coefficients recovered | Hidden initials recovered | All three |
|---|---:|---:|---:|---:|
| Frozen M19 incumbent | 23/24 | 16/24 | 15/24 | 15/24 |
| Joint, after warm search | 23/24 | 21/24 | 21/24 | 21/24 |
| Profiled, after warm search | 23/24 | 21/24 | 21/24 | 21/24 |
| Joint, after adaptive portfolio | 24/24 | 22/24 | 22/24 | 22/24 |
| Profiled, after adaptive portfolio | **24/24** | **24/24** | **24/24** | **24/24** |

Prediction recovery requires training and validation NMSE at most 1e-6 with the
independent numerical check. Coefficient recovery requires every dynamic
coefficient within 1% relative error; initial recovery requires every hidden
initial within 0.001 absolute error. Reference errors and validation are computed
after selection; neither supplies a search objective nor a stopping rule.

### Where recovery came from

Fifteen endpoints already met the tighter numerical target, so neither method
searched them. Both methods warm-polished the remaining nine endpoints. The six
fast/slow six-state endpoints all recovered their coefficients **and initials**
in this stage. Their worst coefficient errors were 0.08034% (joint) and 0.09289%
(profiled); worst initial errors were 0.0009363 and 0.0007588 respectively. Thus
the tighter training target resolves the remaining M20 initial-error threshold
failures on these controls. It does not make parameter error monotonic: one
already excellent coefficient vector becomes slightly less accurate while its
initials improve, and both methods still satisfy the recovery thresholds.

Warm searches stalled on the same three difficult endpoints. Both methods then
evaluated all three frozen diverse starts:

- The fast/slow three-state `s2_rollout_only` endpoint recovered with both
  methods. Its previous validation NMSE of 0.001615 fell to approximately
  5.05e-22. Warm polishing alone did not leave the poor local solution.
- The ordinary six-state `s0_coupled_profiled_rollout` and
  `s1_coupled_profiled_rollout` endpoints recovered only with the new profiled
  searches. Different starts found the accurate solution, with validation NMSE
  approximately 3.32e-22 and largest coefficient relative error below 1.32e-10.
- Joint searches retained those two original incumbents: validation NMSE
  3.675e-7, maximum coefficient relative error 322.69%, and maximum hidden-initial
  error 1.543. Every short trial exhausted its 150-call allowance, and continuing
  the best trial for another 400 calls still failed to improve the incumbent.
  Longer repeated joint polishing is therefore not the supported first remedy.

The comparison supports **profiling plus diverse starts**, not a claim that
profiling eliminates local minima: its warm searches also stalled, and four of
its six diverse trials on the two ordinary six-state endpoints returned to the
poor solution. Profiling reduces the nonlinear search from twelve unknowns to
six decay coefficients, solving gain and five initials conditionally at each
step. This removes coupled nuisance-variable search from the outer optimizer;
it does not make the remaining decay-coefficient problem convex.

### Retention, numerical checks and assessment

Every changed endpoint improves the two-solver training-loss comparison over
its starting incumbent. The previously excluded three-state incumbent survives
the warm stage until a verified better trial is available. Both unresolved joint
six-state endpoints retain their original fitted vectors. All final endpoints
pass the strict numerical check. No domain failures or boundary-projection
events occurred in this run; the separate regression test covers the specific
M20 roundoff failure and rejection of material violations.

The assessment remains more informative than prediction NMSE alone:

| Endpoint class | Training-only assessment | Interpretation |
|---|---|---|
| Two unresolved joint six-state fits | `diversify_or_extend_search` | Prediction passes the coarse 1e-6 gate, but the 1e-20 numerical target is unmet. Locally full-rank sensitivity does not certify a correct optimum. |
| Six fast/slow six-state fits, each method | `assess_parameter_uncertainty` | Accurate retained solution, but verified alternatives and weak joint sensitivity remain. |
| Other recovered endpoints | `retain_with_local_evidence` | Numerical target and local sensitivity checks pass; global uniqueness and statistical coverage remain untested. |

For the fast/slow six-state endpoints, the scaled joint Jacobian still has
numerical rank 11/12 under the declared threshold and condition numbers
1.99e8–2.57e8. Search alternatives remain within the 1e-12 loss-difference
tolerance despite materially different coordinates. This is compatible with
accurate parameters at the selected point: precise noiseless fitting can find
them, while the data remain weakly informative under finite perturbations.
Neither warning nor success establishes structural identifiability or a
calibrated confidence probability. No profile grid was minimized in M21.

### Additional computation

These are summed measured task times, not elapsed campaign or queue time. They
exclude historical M19 computation, which is common to both methods.

| Additional work | Joint | Profiled |
|---|---:|---:|
| Fitting, numerical checks and sensitivity: wall minutes | 12.53 | 6.06 |
| Same work: CPU minutes | 8.41 | 4.24 |
| Observed rollout calls, including diagnostics | 3,128 | 1,275 |
| Separate retrospective scoring: wall minutes | 9.16 | 9.13 |
| Wall minutes including scoring | 21.69 | 15.19 |

Profiling uses 51.6% less added fitting/diagnostic wall time in this run, with
more recoveries. It uses 30.0% less when common retrospective scoring is included.
Both methods have matched stage ceilings and proposed starts, but adaptively
consume different effort; these are not fixed-cost or equal-convergence-accuracy
comparisons. Joint fitting sometimes returns more accurate coefficients within
the accepted tolerance. The 1e-20 target is specific to noiseless controls and
must not become a universal stopping rule for experimental data.

### Recommended next milestone

Retain direct joint rollout as the general baseline and exact profiling as an
option for symbolically certified affine blocks. For those blocks, these results
favor a bounded profiled restart portfolio after warm stagnation over repeatedly
extending the same joint search. Preserve the assessment and independent-rollout
retention rules regardless of fitting strategy.

Return next to a known-structure **nonlinear benchmark block** with generic
training-only starts. Compare direct rollout fitting against coefficient
profiling conditional on collocation trajectories followed by rollout polishing,
recording setup, search and diagnostic overhead separately. Such conditional
profiling is not exact rollout profiling: nonlinear state coupling usually
removes affine dependence of the full trajectories on coefficients. Keep
coefficient/initial recovery retrospective and report solver failures and all
starts, including unsuccessful ones. Noise robustness, arbitrary skeletons and
construction integration remain later qualifications. No production defaults or
new experiment implementation are changed by this completed-results review.

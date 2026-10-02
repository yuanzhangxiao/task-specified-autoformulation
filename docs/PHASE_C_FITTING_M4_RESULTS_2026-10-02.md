# Phase C M4: coefficient recovery with more fitting time

Reviewed 2026-10-02 (Pacific/Honolulu). This closes the numerical comparison in
[the budget/reuse protocol](PHASE_C_FITTING_BUDGET_REUSE.md). This review changes
documentation only: no fitting implementation, benchmark, production default,
or historical result is modified.

## Provenance and verification

- Source archive: `review-20261002-093801.tar.gz`, extracted under
  `artifacts/phase-c-fitting-m4-review-20261002-093801` (untracked).
- Archive SHA256:
  `8aeb768e8a985d321de91921c4ac6673f74fca8f5b05a00768b0b05df395ff9b`.
- Experiment commit: `5089c4e466d661d7f3ddd549e8c69abfbb2af300`.
- Plan SHA256:
  `dfa5fb2f910dcb3282191f87bc39ea1b075d9c705ae94a3eb8ef2f8ed72550b8`.
- Delta jobs: prepare `22611074`, fit array `22611075`, report `22611077`.
- All 186 sealed JSON records verify. All 45 result rows agree with their saved
  backends, worker payload hashes, common-start hashes, coefficient calculations,
  independent replay scores, and summary groups.
- All nine common records and three case payloads exactly match the sealed M3
  archive. No regeneration or fitted-endpoint warm start was introduced.
- All 45 endpoints have complete independent replay; maximum recorded
  Radau/DOP853 disagreement is `5.24152e-6`, below the `1e-4` agreement gate.
  No new integration was run to produce this review's metrics.

The archive reports 45 complete endpoints, **43 prediction passes and 37
coefficient passes**. Complete means an endpoint and its evaluation are
available; it does not imply optimizer convergence or scientific recovery.
Reference values enter the post-selection evaluator only. No test data or LLM
calls were used.

## Prediction, coefficient and time comparison

For each fit, take the maximum absolute relative error across its dynamic
coefficients. Tables report the median and worst of those maxima over three
starts, expressed as percentages. Hidden initial parameters are excluded and
reported separately. A coefficient pass requires every dynamic coefficient
within 1% and independent solver agreement. CSTR prediction passes use the
frozen `0.01` NMSE threshold; small-control prediction uses `1e-6`.

Time is mean saved fitting wall time per start, including worker/screening
overhead and excluding the separate endpoint replay. It is descriptive, not a
hardware-controlled speed benchmark. The group `seconds` field is a sum across
three fits and must not be mistaken for one fit's duration. Every arm receives
the same 600-second ceiling; successful training stopping may finish sooner.

### CSTR hard: seven coefficients and two shared hidden initials

| Method | Median train NMSE | Median validation NMSE | Coefficient error median / worst (%) | Coefficient passes | Mean fit seconds |
|---|---:|---:|---:|---:|---:|
| Rollout only | 2.56e-13 | 9.79e-14 | 0.001529 / 0.002852 | 3/3 | 211.9 |
| Collocation then rollout | 8.72e-13 | 3.23e-13 | 0.002852 / 0.005256 | 3/3 | 342.5 |
| Original collocation | 7.31e-9 | 3.50e-9 | 0.1376 / 0.1893 | 3/3 | 447.4 |
| Adaptive multiple shooting | 2.58e-4 | 1.46e-4 | 27.73 / 37.20 | 0/3 | 569.2 |
| Collocation reuse variant | 0.522 | 0.310 | 73.04 / 615.81 | 1/3 | 469.9 |

All hard-CSTR methods except the reuse variant pass prediction on 3/3 starts;
reuse passes 1/3. Rollout-only finishes its three fits in 191.3, 193.0 and
251.3 seconds. None exhausts its 600-second allowance.

### CSTR easy: seven coefficients, public initial readings

| Method | Median validation NMSE | Coefficient error median / worst (%) | Coefficient passes | Mean fit seconds |
|---|---:|---:|---:|---:|
| Rollout only | 8.90e-12 | 0.0007821 / 0.004813 | 3/3 | 162.5 |
| Collocation then rollout | 1.56e-16 | 0.000002394 / 0.000009284 | 3/3 | 212.0 |
| Original collocation | 2.62e-12 | 0.001812 / 0.05041 | 3/3 | 330.4 |
| Adaptive multiple shooting | 6.63e-5 | 11.67 / 32.19 | 0/3 | 568.2 |
| Collocation reuse variant | 2.73e-12 | 0.001908 / 0.003396 | 3/3 | 277.3 |

All 15 easy-CSTR fits pass the prediction gate. Multiple shooting illustrates
why an output threshold alone is insufficient evidence of coefficient recovery.

### Saturating-shape control

All five methods recover prediction, coefficients and latent trajectory on
all three starts. Worst maximum coefficient errors are below 0.00155% for
every method. Mean fitting times are 13.4 seconds (rollout), 19.3 (hybrid),
45.5 (original collocation), 41.8 (reuse) and 68.2 (shooting).

## Hidden initial conditions

Worst absolute errors over three hard-CSTR starts:

| Method | Initial concentration error | Initial jacket-temperature error (K) |
|---|---:|---:|
| Rollout only | 1.286e-6 | 1.139e-4 |
| Collocation then rollout | 1.325e-6 | 2.878e-4 |
| Original collocation | 7.918e-4 | 0.05762 |
| Adaptive multiple shooting | 0.1171 | 14.55 |
| Collocation reuse variant | 0.3825 | 16.27 |

Concentration errors use the supplied benchmark's concentration units. Absolute
temperature errors avoid dependence on a chosen temperature zero. Good
coefficient recovery here accompanies good initial-value recovery; it is not
merely compensating for incorrect hidden initials at the measured output.
This does not establish global uniqueness or latent-trajectory recovery for
arbitrary candidate equations.

## What the longer budget establishes

In M3 (180 seconds), hard-CSTR median/worst coefficient errors were 6.77/35.8%
for rollout, 99.9/169.7% for the hybrid and 97.8/127.9% for direct collocation.
All three now pass coefficient recovery in all starts. Data and starts are
identical; the original arms retain their algorithms, with larger time and call
ceilings. Node timing and both ceilings differ, so this is evidence of a budget
limitation, not an isolated estimate of the causal effect of extra seconds.

The earlier result did not establish that collocation's polynomial representation
was intrinsically too inaccurate. Original collocation now recovers the correct
coefficients closely. Rollout remains the fastest successful option on these
cases, but integration accuracy alone cannot explain the ranking: shooting
also integrates and still underperforms at the allotted budget.

The hybrid's hard-CSTR initializer reports no native success in any start:
starts 0 and 1 hit iteration limits; start 2 hits its 200-second time allowance.
Nevertheless, starts 0 and 1 retain excellent collocation checkpoints, and their
subsequent joint refinement stages require only two residual calls each. These
are optimization-stage calls, not total calls including checkpoint screening.
Start 2 instead selects the ordinary starting vector and makes 39 refinement
calls, yielding the same final parameters as rollout-only start 2. Thus two
runs support the value of intermediate checkpoints; the third demonstrates
the value of preserving the ordinary-start fallback. Hybrid success is not
always a successful collocation initialization.

## Why the reuse arm should not be promoted

It successfully reuses one constructed graph across native solver calls. The
failure is in the combined optimization policy, not evidence that caching a
graph is mathematically unsound. Relative to the original arm, it also changes
native initialization (`warm_start_init_point=yes` even on the first solve),
50-iteration chunking with explicit primal/dual transfer, mesh time allocation,
and the requirement of native success before refinement. This experiment does
not identify which change caused the regressions.

- **Hard start 0:** six completed 50-iteration calls all hit their iteration
  limit, followed by another interrupted call. The latest saved iteration is
  320. Native work reaches its roughly 447-second stage cap. It achieves a tiny
  defect (`1.41e-8`) but a poor nodal loss (`1.0213`), consistent with its poor
  independent rollout (`1.0213` train / `0.4153` validation). This is a nearly
  dynamically consistent poor fit, not merely a nodal-versus-rollout mismatch.
  The overall worker stops after about 477 seconds because the coarse solve is
  unfinished, leaving some of the 600-second outer allowance unused.
- **Hard start 1:** the coarse solve converges after 50+14 iterations, then two
  refined meshes converge. Coefficient error is 0.408%, passing the 1% gate.
- **Hard start 2:** all eight 50-iteration calls hit their iteration limit.
  Native work ends after about 235.7 seconds including process overhead. Yet
  its latest accepted snapshot is still iteration 32, recorded at 58.7 seconds.
  The snapshot routine can reject nonfinite or out-of-bound points, but logs do
  not record rejection reasons; the cause of the stale checkpoint is unproven.
  Screening then runs until the outer worker is killed at 600 seconds. Only two
  completed screening records remain, and the retained parameters are exactly
  the ordinary start (train `0.5219`, validation `0.3100`). This is a safely
  preserved endpoint, not a successful fit.

The screening code assigns an individual point the entire remaining screening
allowance. A difficult point can therefore monopolize it. The archive establishes
the unfinished screening phase, but lacks a flushed start record identifying
the interrupted point. Future diagnostics should record started evaluations and
cap each point while reserving time for alternatives and final metadata.

Reported graph-building time in original hard collocation totals about 25--27
seconds over three meshes, roughly 5--8% of its overall fitting wall time.
Eliminating this overhead cannot explain or remedy errors of the magnitude seen
in the reuse variant. Repeated primal/dual warm starts also do not retain IPOPT's
complete state (including limited-memory curvature history). A clean reuse
comparison must separate caching from restart and mesh-selection policies.

## Multiple shooting and status interpretation

All nine hard-CSTR shooting mesh stages reach their allocated time caps. Latest
recorded native iterations are only 16--17, 9--10 and 7 across the three passes.
Small variable count does not imply cheap iterations: each involves integrated
window maps and their derivatives. The records support unfinished optimization,
not an impossibility of coefficient identification by multiple shooting.

`budget_exhausted=false` refers to the outer budget. It does not negate recorded
mesh-stage timeouts. Likewise, `complete` means independently evaluated output
exists, not native solver success. Read stop reason, stage timeouts, coefficient
accuracy and prediction accuracy together. Preserve both levels in later reports.

## Recommended next milestone

1. Keep rollout-only and the existing hybrid as development reference methods;
   retain original collocation and shooting as alternatives. Do not promote the
   current reuse variant or replace production defaults based on three starts.
2. Before rerunning reuse, isolate unchanged-mesh graph retention from native
   restarts: ordinary cold-start settings first, an uninterrupted solve control,
   and warm starts only when an actual previous solution exists. Compare with
   identical meshes and allowances. Instrument skipped checkpoint reasons and
   bounded per-point screening; preserve original M4 outcomes.
3. Then test conditional affine-coefficient least squares within an explicit
   soft-defect formulation, followed by independent rollout. Keep fixed correct
   equations and include harder starts and qualified noisy/sparse controls.
   Do not introduce coefficient truth into screening, stopping or selection.

These results establish attainability on the present noiseless, fixed-structure
controls. They do not settle noisy inference, badly scaled candidate equations,
global identifiability, long-horizon instability, or construction integration.
No further remote jobs were launched during this review.

## Verification performed for this documentation review

- The archive/hash/numerical-accounting checks above passed. Coefficient errors
  were recomputed from the retained vectors, without another optimization.
- Focused pytest: 31 passed in 11.47 seconds.
- A fresh five-arm local linear smoke gave prediction passes for four methods.
  Multiple shooting hit its 60-second wall limit: training NMSE `9.52803e-7`,
  validation NMSE `1.14362e-6`, just above the `1e-6` gate. The smoke therefore
  failed overall; this is not presented as a passing check. The previous
  milestone recorded a passing smoke; this run does not reproduce it. This is
  consistent with wall-budget sensitivity; its timing cause was not isolated.
  This local check neither changes nor invalidates the sealed Delta scores.
- Repository-wide Ruff still reports the same 37 unrelated `analysis/claude`
  findings. Patch whitespace checks pass. No implementation files changed.

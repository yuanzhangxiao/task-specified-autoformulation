# M28 results: the common fitting adapter preserves recovery; numerical retry works

Reviewed `review-20261012-172005.tar.gz` on 2026-10-12. All 31 tasks completed:
30 new fits and one fixed-parameter replay. Both fitting strategies recover all
15 case/start combinations, pass independent numerical verification and satisfy
the retrospective coefficient thresholds. The tighter evaluator retry resolves
the previously flagged basin discrepancy without changing parameters.

**The fixed-scaling alternative never activates.** These results qualify the
common guarded fitting interface on these inputs. They do not establish a
benefit for switching scaling or justify attributing timing differences to that
switch. Production integration remains a separate milestone.

## Provenance and audit

- Protocol: `phase-c-rollout-adapter-1`; [M28 runbook](PHASE_C_FITTING_ADAPTER.md).
- Source commit: `f0445f6444fd2d3bb6c4905376f466d50d1f7446`.
- Source-tree SHA256: `4427f61e3041a9ac05ef759b04b19c67a5dec04cf774728921beba8499e70b1c`.
- Plan SHA256: `abe824c37daa10f407b1a7875f0d78c02725dddba59a47d2a879ccb17cd7ae89`.
- Input SHA256: `eb1d681faa7d98af0d0130e010f7099513b71e041f41c0ae574632770ae142a2`.
- Archive SHA256: `0174e9a3e94145f5cfbaa8eaceaebd39462dee470c754aaad12765bc40eb7bc4`.
- Delta jobs: prepare `22828976`, array `22828977`, report `22828978`.
- Output root: `/work/hdd/bibo/yxiao2/phase_c/fitting-rollout-adapter-v1`.
- Runtime: Python 3.12.13, NumPy 2.3.5, SciPy 1.16.3, CasADi 3.7.2,
  Pydantic 2.12.5. One CPU per task, three concurrent tasks, no GPU or LLM calls.

All 711 sealed JSON artifacts validate; rebuilding the report reproduces the
supplied summary exactly. Source identity matches the committed implementation.
The archived input equals the frozen local export. Every retained parameter
vector matches its selected verified checkpoint and evaluator parameter hash.
Coefficient and initial errors recompute exactly from the evaluator references.

All fifteen paired experiments have identical original/restart lists, fitting
policies apart from strategy, and first evaluated parameter-vector hashes.
Typed training-problem hashes match the explicit training-only inputs. The
saved endpoint is used only by its separate evaluator task. No validation or
reference values enter fitting/retention; test data remain unopened.

All 164 fitting operations have terminal records: 163 complete, one explicitly
unavailable training check. All 34 independent evaluator attempts complete,
covering 526 model/trajectory records with both Radau and DOP853. There are no
recorded domain failures. Of 827 started search calls, 825 complete; two calls
reach their wall-clock allowance and remain charged. Cost accounting is complete.

## Recovery and cost

Each task starts from one original generic vector and has three predetermined
domain-only alternatives. Both arms share the same ceiling of six blocks,
forty evaluations or 800 seconds per block, and at most two consecutive blocks
per start. Independent training verification requires aggregate NMSE <= 1e-12,
worst-trajectory NMSE <= 1e-11 and solver agreement before early stopping.

| Case | Guarded-only recovery | Portfolio recovery | Median fit minutes, guarded-only | Median fit minutes, portfolio |
|---|---:|---:|---:|---:|
| Alien-device hard | 5/5 | 5/5 | 9.45 | 9.42 |
| Nonlinear CSTR hard | 5/5 | 5/5 | 5.37 | 5.47 |
| Coupled basin | 5/5 | 5/5 | 1.88 | 1.95 |

Recovery here includes the strict training target, independent evaluator solver
agreement, validation NMSE <= 1e-8, dynamic coefficient relative error <= 1%,
and shared initial-parameter absolute error <= 1e-3 where applicable. The
initials are checked separately in physical units, not combined into a
relative-error percentage. Basin has no fitted hidden initials in this control.

Worst errors across both arms and all five starts in each case:

| Case | Maximum train NMSE | Maximum validation NMSE | Maximum dynamic coefficient relative error | Maximum hidden-initial absolute error |
|---|---:|---:|---:|---:|
| Alien-device hard | 4.28e-13 | 9.07e-13 | 0.000682% | 5.97e-6 |
| Nonlinear CSTR hard | 3.39e-13 | 1.21e-13 | 0.001807% | 7.56e-5 |
| Coupled basin | 5.14e-14 | 8.86e-14 | 0.000129% | Not applicable |

The NMSE columns are the independent evaluator's primary (Radau) scores, with
the separate two-solver agreement gate passed. This is known-structure,
noiseless recovery across repeated initializations, not thirty independent
scientific problems. Fourteen of fifteen paired endpoints are numerically
identical; the exception is explained below.

Fitting takes 250.78 task-minutes in total: 131.15 for guarded-only and 119.63
for the portfolio. Independent evaluation adds 32.00 task-minutes including the
fixed-endpoint diagnostic. These sums are recorded process wall times, not
campaign elapsed time or billed service units. Evaluation is reported outside
the optimizer budget, including both attempts when retried.

## The apparent timing improvement is not a fallback effect

All 38 executed search blocks use guarded relative scaling; zero use fixed
scaling. Accordingly, the portfolio's smaller time sum cannot demonstrate that
its alternative scaling strategy is better.

Alien start 3 accounts for most of the difference. Both arms leave the original
start after two blocks and recover from the first predetermined restart.

| Alien start 3 | Guarded-only | Portfolio |
|---|---:|---:|
| Fitting minutes | 57.14 | 44.98 |
| Search blocks | 4 | 4 |
| Search calls started/completed | 132/130 | 137/137 |
| Final training NMSE | 4.45e-19 | 4.28e-13 |
| Final validation NMSE | 6.69e-19 | 9.07e-13 |

In guarded-only, the second original-start block and first restart block each
hit the 800-second wall-clock cap, finishing 31 and 27 calls respectively. The
second original-start verification also reaches its 180-second limit. The
verified incumbent survives; the controller advances to the reserved restart.
The final restart continuation reaches the target.

The paired portfolio run finishes forty calls in each of those blocks within
its time allowance, then reaches the target with seventeen further calls.
It never changes scaling. Wall-clock cutoffs alter where optimization stops and
resumes even with the same deterministic policy and starts. The records explain
the differing paths; they do not identify a hardware or scheduler cause for
the runtime variation. The smaller final error in guarded-only is also not a
reason to select that arm retrospectively: both meet the frozen accuracy target.

This is positive evidence for preserving incumbents and reserving genuinely
different starts. It provides no empirical evidence yet for the fixed-scaling
fallback. Its budget/preservation behavior remains covered by unit tests.

## The basin retry resolves the numerical disagreement

The exact flagged M27 endpoint reproduces the original discrepancy on Delta.
At rtol=1e-9 and atol=1e-11, the maximum normalized Radau/DOP853 difference is
3.033e-4, exceeding the unchanged 1e-4 limit. The largest discrepancy is on
`val_001`; Radau's NMSE is 3.09e-16 and DOP853's is 1.81e-9 there. A small
primary-solver NMSE alone therefore would not certify numerical agreement.

The reserved retry at rtol=1e-12, atol=1e-14 reduces the maximum discrepancy to
5.08e-10, retaining exactly the same parameter vector. Final aggregate
train/validation NMSE is 1.56e-16/2.65e-16. For this separate diagnostic, the
initial attempt takes 24.20 seconds and the retry 49.28 seconds, totaling 73.48.

Both new basin start-0 fits independently return that same vector and trigger
the same successful retry. There are thus three retry executions of one
endpoint, not three different numerical failure cases. Every other evaluation
passes on its first attempt. Original failures and their costs remain recorded;
M27 artifacts are unchanged. This supports bounded tighter verification, not a
relaxed agreement criterion or additional parameter fitting on validation.

## What the fitter can report without the reference

All thirty returned assessments report `numerical_target_reached`,
`strictly_verified` and `local_sensitivity_resolved_profiles_not_tested`.
The joint parameter/initial sensitivity ranks are 18/18 for alien, 9/9 for
CSTR and 2/2 for basin. Coordinate-scaled condition-number ranges are about
890–1078, 540–707 and 2.5–31.9 respectively.

These are local checks at the retained points. The optimizer does not know its
retrospective coefficient error, and full local rank does not certify global
uniqueness, noisy-data precision or correctness of a proposed mechanism.
Probability remains null; profile-likelihood intervals are not computed here.

## Recommended next milestone

Proceed to a **small integration qualification on frozen constructed models**,
with no new proposer calls and no test access. Keep the existing accurate
known-structure cases as controls. This would test the remaining boundary between
ordinary public fitting requests and the typed adapter: training-only coordinate
and restart generation, parameter domains, initializer bindings, multiple
targets, profiling applicability and retained-incumbent handoff.

Freeze the candidate roster and generic starts before fitting. Include supported
constructed models with imperfect equations; inability to reach a noiseless
target must remain a fit outcome, not an automatic proof of structural
infeasibility. Unsupported initializers or expressions should be reported
explicitly rather than silently reinterpreted. Keep validation evaluation after
sealing the fit and use reference recovery only for the known-structure controls.

Use guarded-only as the empirically qualified default for that pilot. Keep the
fixed-scaling alternative optional until a predeclared plateau stress test
actually invokes it under matched budgets. Noise-aware stopping and broader
structural coverage are still needed before a general production promotion.
No next-milestone implementation or remote submission was performed in this review.

## Review verification

The local preparation regression suite passes 47 tests; the archived Delta
preparation log also records 47 passes. The archived-report reconstruction and
all identity/coefficient checks above pass without refitting benchmark models.
Repository-wide `ruff check .` still reports 37 pre-existing issues in four
unrelated `analysis/claude` files. No implementation, benchmark data or prompts
were changed by this review.

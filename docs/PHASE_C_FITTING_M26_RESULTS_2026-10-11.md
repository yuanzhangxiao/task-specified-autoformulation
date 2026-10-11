# M26 results: fixed and guarded scaling recover all three generic starts

Reviewed `review-20261011-154745.tar.gz` on 2026-10-11. All nine tasks and
retrospective evaluations completed, with complete recorded cost accounting.
Eight fits reach the independently verified numerical target. Fixed coordinates
and guarded relative scaling each recover all three original generic starts;
iterative Jacobian scaling recovers two and remains inaccurate on start 2 after
using all six search blocks. Task completion is not fitting success.

This establishes recovery from the tested original starts, extending earlier
evidence from saved warm endpoints. It does not establish universal scaling
superiority or robustness across arbitrary starts, noise or equation structures.
Production fitting defaults are unchanged.

## Frozen experiment and archive checks

- Protocol: `phase-c-generic-scaling-1`; [M26 runbook](PHASE_C_FITTING_SCALING.md).
- Source commit: `c4c9bd599cfcbe2f6b85a6ac33c9591284bceb40`.
- Plan SHA256: `5c2c5bba54c1fee583c6880b2a1b91bff1c4252064880163e4f59d326ed76cc6`.
- Input SHA256: `ed71b7ee77608f94b66e7c16a91c0a0c29f9c05ec7b25bc594694ba079d42d09`.
- Archive SHA256: `08a838dee51c32735e6c402ed23f129be754d4bda6b31703880c3b440b902295`.
- Delta receipts: prepare `22804541`, array `22804542`, report `22804543`.
- Recorded runtime: Python 3.12.13, NumPy 2.3.5, SciPy 1.16.3,
  CasADi 3.7.2, Pydantic 2.12.5.
- Known nonlinear alien-device hard skeleton: six states, thirteen dynamic
  coefficients and five shared latent initial values. Exact terminal-output
  profiling removes four linear gains, leaving fourteen outer unknowns.
- Six search blocks maximum per task, each capped at forty evaluations or
  eight hundred seconds. All arms have the same frozen original/restart vectors
  and budget ceilings; actual use is adaptive.

The local review verifies all 159 sealed JSON records, plan/input agreement,
the nine-task roster, result/backend identities and exact summary/result
agreement. Frozen start lists match across arms within each original seed.
The first evaluated parameter-vector hash also matches across each paired arm.
Retained vectors agree with their selected verified checkpoints.

Every final model has complete independent DOP853/Radau replay across sixteen
training and four validation trajectories, with 601 samples per trajectory:
180 complete model/trajectory records. This includes the inaccurate model.
Backend records retain the training-only selection boundary; reference and
validation values are used only in retrospective evaluation. No test data or
LLM calls are involved. This review performs no new optimization or remote run.

The archived preparation log reports 27 tests passed. Worker stderr contains
the warning that `jac` has no effect for the explicit RK solver, rather than a
worker exception. This warning does not mean the optimizer's residual Jacobian
was unused; it concerns an ODE integrator option.

## All nine fitted endpoints

Scores are final independent replay scores. Coefficient error is the maximum
relative error across the thirteen dynamic coefficients, expressed as a
percentage. Initial error is the maximum absolute error across the five hidden
initial values. Fit time includes setup, search, training verification and final
sensitivity, but excludes retrospective validation/reference evaluation.

| Start | Scaling | Train NMSE | Validation NMSE | Max. coefficient error | Max. initial error | Search evaluations | Fit minutes |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | Iterative Jacobian | 3.714e-13 | 1.385e-12 | 0.000884% | 1.372e-6 | 19 | 8.84 |
| 0 | Fixed coordinates | 7.772e-19 | 3.976e-19 | 0.000000906% | 8.075e-9 | 12 | 7.08 |
| 0 | Guarded relative | 8.357e-15 | 1.913e-14 | 0.000122% | 8.271e-7 | 12 | 7.08 |
| 1 | Iterative Jacobian | 2.340e-13 | 9.869e-13 | 0.000898% | 7.157e-6 | 61 | 21.92 |
| 1 | Fixed coordinates | 1.570e-17 | 3.783e-17 | 0.00000990% | 5.667e-8 | 33 | 12.36 |
| 1 | Guarded relative | 3.468e-16 | 1.640e-15 | 0.0000346% | 1.986e-7 | 23 | 9.71 |
| 2 | Iterative Jacobian | 0.109776 | 0.203706 | 714.87% | 39.574 | 224 | 75.46 |
| 2 | Fixed coordinates | 1.132e-16 | 2.407e-16 | 0.0000238% | 1.430e-7 | 29 | 10.55 |
| 2 | Guarded relative | 2.185e-15 | 7.717e-15 | 0.0000876% | 4.516e-7 | 36 | 12.72 |

Prediction, coefficient and initial-recovery gates pass for the same eight
tasks. Coefficient recovery is measured using the known reference after fitting;
the optimizer did not select candidates by coefficient error. Values far below
the stopping threshold are incidental overshoot of the final accepted step,
not evidence that one model is scientifically better than another.

## Recovery and compute cost

| Scaling | Recovered starts | Evaluations for starts 0 / 1 / 2 | Median search minutes | Median fit minutes | Summed fit minutes |
|---|---:|---|---:|---:|---:|
| Iterative Jacobian | 2/3 | 19 / 61 / 224 | 16.51 | 21.92 | 106.22 |
| Fixed coordinates | 3/3 | 12 / 33 / 29 | 7.18 | 10.55 | 29.99 |
| Guarded relative | 3/3 | 12 / 23 / 36 | 6.16 | 9.71 | 29.50 |

All six fixed/guarded fits succeed in their first forty-evaluation block, without
continuation or a restart. The iterative arm needs a second block on start 1.
On start 2 it uses the original plus all three predetermined restart vectors and
all six blocks. One block reaches its time ceiling after twenty-four evaluations;
the other five use forty each. The 224 calls therefore do not indicate missing
accounting against the 240-call maximum.

Start 2 initially improves from profiled training NMSE 0.742334 to only 0.712312
under iterative Jacobian scaling. The controller detects the plateau and moves
to a different start. Its best final score, 0.109776, comes from the last restart.
Worse intermediate branches preserve the earlier verified incumbent. Thus the
restart/retention machinery executes, but the tested search portfolio does not
recover an accurate fit under that scaling.

Guarded scaling is faster than fixed scaling on start 1 and slower on start 2;
start 0 costs essentially the same. Their summed fit time differs by only about
1.6%. The guarded median is lower, but three paired starts do not establish a
reliable speed winner. Both have a clear advantage over the iterative arm here.

For the successful fixed/guarded arms, median training-verification time is
about 194 seconds and sensitivity about 19 seconds per task. Retrospective
evaluation adds about 128–140 seconds per task across the whole study. Total
recorded fitting time across all nine tasks is 165.71 minutes, with another
20.00 minutes for retrospective evaluation. These are summed task wall times,
not elapsed campaign time, billed service units or exclusively search time.

## What scaling and sensitivity establish

All three arms use the same existing dimensionless base coordinates, exact
profiling, objective normalization and stopping rules. The additional optimizer
scale is the controlled difference within M26:

- Fixed coordinates uses an identity adjustment.
- Guarded relative scaling uses partially balanced initial Jacobian column
  norms, with the adjustment frozen within each search block.
- Iterative Jacobian scaling updates its Jacobian-based scale during the solve.

Actual guarded adjustments range from about 0.28 to 4.90 across the three
starts. None reaches the configured 0.1/10 clipping limits, and no weak-column
guard fires. This run therefore tests partial relative scaling and freezing,
but does not empirically test whether clipping or weak-column safeguards improve
benchmark recovery. Their unit tests remain separate evidence.

This is evidence that optimization coordinates affect recovery on the same
nonlinear objective. It does not prove that all iterative rescaling is harmful,
identify a specific local minimum, or isolate freezing from partial balancing.
M26 also changes objective normalization/stopping relative to M25, so a
historical comparison cannot attribute all improvement to scaling alone.

All eight accurate solutions have full local joint sensitivity rank 18/18,
including hidden initials, with reported scaled-Jacobian condition numbers
about 890–1078. The inaccurate solution also has rank 18/18, with condition
number about 23,519. Full local rank is neither a fit-quality certificate nor
proof of global identifiability. Its assessment correctly requests further
search and does not claim coefficient accuracy or a calibrated confidence level.

## Recommended next milestone

Keep fixed and guarded scaling as the two primary alternatives; retain the
iterative arm as a comparison. Freeze a broader generic-start qualification on
additional correct benchmark skeletons and additional predetermined starts,
with matched ceilings and the same training-only retention and independent
rollout checks. Where terminal-output profiling is inapplicable, use the existing
direct-rollout route and identify that route explicitly in reports.

Continue reporting coefficients and initials retrospectively, rather than using
them to route fitting. Include restart and verification overhead, and record
whether scale clipping/weak directions are actually exercised. More starts on
one noiseless skeleton do not replace testing other structures. No new campaign
or production integration is implemented or submitted by this review.

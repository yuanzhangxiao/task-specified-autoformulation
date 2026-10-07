# M14 results: profiling output gains recovers the difficult generic start

Reviewed `review-20261006-233435.tar.gz` on 2026-10-06. Profiling the four
conditionally linear output gains passes independent prediction checks from all
three original starts. Joint rollout fitting passes from two. The profiled arm
uses less fitting time on each matched start. This is a positive result for
initialization robustness on this one problem, not a general recovery guarantee.

## Scope and integrity

- Protocol: `phase-c-profiled-output-1`; [M14 runbook](PHASE_C_FITTING_PROFILED_OUTPUT.md).
- Execution commit: `a78ea692fdf7e3273ce208c96fa98e5013093e2d`.
- Plan: `1d4d103f8805a39e6604ab643051ae68332d63d7f2b3c5bbfae8788f98cac91e`.
- Archive: `77f06d562431b6951ce77d38ff23aed56ff2b1c6f851e7d4479be4e50910447a`.
- Input content: `ed71b7ee77608f94b66e7c16a91c0a0c29f9c05ec7b25bc594694ba079d42d09`,
  unchanged from M12/M13.
- Delta preparation `22711460`, array `22711461`, report `22711462`.
- Correct alien-device hard equations and anchored latent coordinates: six
  states, 13 dynamic coefficients, five shared latent initials, 16 training
  trajectories and four validation trajectories. Three starts on one problem
  are not three independent benchmark cases. Correct equations are supplied
  assistance; construction is not evaluated.
- Both arms have a 1,200-second fitting ceiling, including training certification,
  and separate final evaluation. No test data or LLM calls are used.

Verified 57 sealed JSON records, all six result/backend identities and summary
rows, 12 subprocess receipts and 27 recorded output hashes. The archived plan's
source identity matches the original portable M14 bundle. Every worker has
confirmed termination. Final replay covers all 120 model/trajectory combinations,
601 samples each, using both Radau and DOP853 on the original complete equations.
The maximum recorded normalized solver disagreement is `6.525e-9`. Independently
recomputed the reported coefficient/initial absolute errors from the selected
vectors and the post-selection reference. No hard-case refitting was done locally.

## Results

NMSE is from independent final replay. Coefficient error is the maximum relative
error among the 13 dynamic coefficients. Initial error is the maximum absolute
error among the five shared latent initials. Time includes the training check but
excludes final validation/reference evaluation.

| Start | Arm | Train NMSE | Validation NMSE | Max. coefficient error | Max. initial error | Fit minutes | Residual calls |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | Joint rollout | 3.337e-9 | 3.635e-9 | 0.0428% | 0.000383 | 7.30 | 25 |
| 0 | Profiled rollout | 3.282e-7 | 2.274e-6 | 0.5737% | 0.001970 | 3.82 | 10 |
| 1 | Joint rollout | 1.548e-9 | 4.747e-9 | 0.1156% | 0.000598 | 10.24 | 37 |
| 1 | Profiled rollout | 2.428e-7 | 8.651e-7 | 1.3390% | 0.006712 | 8.29 | 31 |
| 2 | Joint rollout | 0.124009 | 0.121275 | 1,033.15% | 2.719644 | 19.07 | 82 |
| 2 | Profiled rollout | 3.080e-7 | 6.062e-7 | 1.2542% | 0.007683 | 7.23 | 27 |

| Gate | Joint rollout | Profiled rollout |
|---|---:|---:|
| Train and validation NMSE at most 0.01 | 2/3 | 3/3 |
| Strict independent training prediction certificate | 2/3 | 3/3 |
| Every dynamic coefficient within 1% | 2/3 | 1/3 |
| Every latent initial within 0.01 absolute error | 2/3 | 3/3 |

Fitting time decreases by 47.6%, 19.0% and 62.1% for starts 0, 1 and 2. The
third comparison includes a budget-exhausted unsuccessful control, so it is not
an equal-accuracy speed comparison. The first two controls end more accurately
than the profiled arm despite sharing the same stopping threshold. Report both
accuracy and time; do not describe profiling as dominating coefficient recovery.
Calls are complete-algorithm accounting, not equal-cost primitive operations.
The optimization workers take 326/506/1,020 seconds for joint fitting and
129/379/315 seconds for profiling. Both retain independent training checks
costing approximately 99--123 seconds; their cost is included above.

## Why the third start improves

The terminal observed-state equation has the form

\[
\dot y=-d_5y+\sum_{j=0}^3g_j\phi_j(z),\qquad g_j\geq0.
\]

The latent dynamics do not depend on `y` or these gains. Integrating filtered
features therefore gives a rollout linear in the four gains for any fixed outer
vector. Profiling solves their bounded least-squares problem across all training
trajectories at every outer evaluation. The nonlinear optimizer searches over
14 unknowns rather than 18; all 18 quantities are still estimated. This changes
the optimization path without changing the model family or allowed signs.

The joint start-2 fit again suppresses `output_gain_1` to `1.39e-17`. The profiled
fit returns `0.4610424`, close to the reference `0.4611828`. Its initial inner
solution also sets this gain to zero: the first five evaluations keep it at the
lower bound, evaluation six releases it, and later evaluations eventually recover
its contribution. Profiling does not artificially prevent zero gains or encode
their reference values. It repeatedly reoptimizes the whole output block as
latent dynamics change. This trace supports reduced coefficient coupling as an
explanation, without proving which geometric feature caused the joint failure.

All four final profiled gains are interior. The final column-normalized design
condition numbers are 6.317--6.319 and inner optimality residuals are at most
`1.25e-13`. These characterize the conditional four-gain solve, not identifiability
or conditioning of the full nonlinear inverse problem.

## Why two coefficient gates narrowly fail

All three profiled runs stop because they satisfy the training prediction target,
not because their budget expires or the outer optimizer declares convergence.
The mean training threshold is `1e-6`, with per-trajectory and solver-agreement
checks. Nothing in that criterion guarantees coefficients within 1%.

Only `decay_x0` exceeds 1% in profiled starts 1 and 2. The maximum errors are
1.3390% and 1.2542%; every other dynamic coefficient is within 1%. All five
latent initials pass the separate absolute-error gate.

The last two training evaluations are:

| Start | Penultimate NMSE | Final NMSE |
|---:|---:|---:|
| 0 | 2.7592e-5 | 3.2821e-7 |
| 1 | 8.9185e-5 | 2.4280e-7 |
| 2 | 1.3860e-5 | 3.0796e-7 |

The run ends during rapid improvement, making premature stopping for the stricter
coefficient objective a plausible explanation. Further improvement is untested.
The successful joint controls happen to cross the same prediction threshold at
much smaller residuals; this does not establish that profiling has a worse
attainable coefficient optimum. Good predictions alone still cannot certify
parameter recovery, especially with noisy data or weakly observed directions.

## Recommended next milestone

Retain joint rollout as the reference and profiling as the promising specialized
option. Next test a stricter training-only polishing target, for example mean
NMSE `1e-8` with a corresponding per-trajectory target, on both arms and all three
original starts under the same total ceiling. Preserve the first certified
prediction checkpoint, and measure the additional cost and coefficient change
after polishing. Do not stop using reference coefficient errors or validation
scores. A saved-checkpoint continuation can be a cheaper preliminary diagnostic,
but must retain its original fitting cost and be labeled as continuation.

Then qualify on additional structures, noise levels and generic starts. The
current exact profiling rule applies only to a certified terminal linear output
with no feedback into the latent subsystem. A coupled glucose-insulin model, for
example, cannot automatically use this same decomposition. Broader applicability,
global uniqueness, noisy-data stopping and production integration remain open.

## Review verification

Only this results note and the fitting plan are changed. Full `pytest -q -n 4`
passes 4,135 tests, with eight optional Torch skips. The small two-arm numerical
smoke passes independent replay, parameter recovery (maximum absolute errors
`1.41e-6` joint and `1.83e-6` profiled), and resume without renewed fitting cost.
Repository-wide `ruff check .` reports the same 37 existing findings in unrelated
untracked `analysis/claude` files. `git diff --check` passes. The supplied hard-case
records are inspected and hashed, not refitted; no remote session is opened.

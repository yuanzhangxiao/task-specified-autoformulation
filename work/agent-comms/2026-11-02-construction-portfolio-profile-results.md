# Construction portfolio profiling: verified gains and remaining fitting work

2 November 2026. Review of the seven-task `construction-portfolio-profile-1`
campaign implemented at `93c59d04b313b2ca839e40d83a2ed0ca2c63e1ee`.

## Conclusion

Models 12/15 improve substantially without changing their equations. Their
successful searches reach the function-evaluation limit before convergence.
The original starting points remain numerically unchanged. Models 10/14 still
cannot finish the sensitivity prerequisite, despite the certified reductions;
their continuations did not start. Model 03 completes the missing probes but
retains unresolved derivative precision.

The immediate priority remains numerical fitting. These results do not justify
telling construction that the current equation families have been optimized to
a demonstrated best fit. They also do not establish physiological correctness
or parameter identification.

All scores below are **training reconstruction NMSE**. Initial roots are fitted
using training trajectories. No new causal-prefix forecast, validation, or test
evaluation occurred; old forecast scores cannot be assigned to new endpoints.

## Evidence and verification

- Archive: `/Users/yuanzhangxiao/Downloads/construction-portfolio-profile-inspection.tar.gz`.
- Archive SHA256: `bc8a71042664c806f20e858b2ec48d32f28e9a9873e22671488a0e308ecf07cf`.
- Plan SHA256: `91aca141ae159930ef5d521759806346168cdd7b7f66999e3244c98eff68fcf3`.
- All seven terminal records reconcile exactly against operation receipts.
- Local receipt-only report reconstruction exactly reproduces the supplied
  summary. All 144 archived files remain byte-identical after review.
- Local evidence: `output/construction-portfolio-profile-review-20261102/`,
  including `local-review-integrity.json`.
- No fitting, numerical rollout, remote access, or LLM calls were executed in
  this review. Independent-rollout conclusions below come from authenticated
  saved verification receipts, not new local simulations.

## Models 12/15: initial-root profiling and a different start help

| Model / start | Starting NMSE | After conditional initial-root solve | Final NMSE | Termination |
| --- | ---: | ---: | ---: | --- |
| 12 / incumbent | .564509059 | .564509059 | .564509059 | Powell stops after one iteration |
| 12 / slower insulin action | .556565972 | .534137382 | **.449981937** | 1,000 calls; five iterations |
| 15 / incumbent | .564608325 | .564608325 | .564608325 | Powell stops after one iteration |
| 15 / slower insulin action | .558997699 | .537017157 | **.439421177** | 1,000 calls; five iterations |

The final successful fits improve 20.3% and 22.2% over the original incumbents.
Every measured output improves, unlike the earlier short timescale profiles:

| Model | Gp: old → new | I: old → new | U: old → new |
| --- | --- | --- | --- |
| 12 | .451761 → .293832 | .647464 → .534800 | .594302 → .521314 |
| 15 | .449880 → .247471 | .647511 → .549925 | .596433 → .520868 |

At each shared-parameter trial, the profiler solves the 32 initial roots as 16
bounded two-variable least-squares problems. This leaves nine/eight shared
coordinates in the outer search. Both timescales remain free. At the successful
endpoints, all 16 inner problems pass bounds and optimality checks; maximum
normalized KKT residuals are 2.67e-14/2.99e-14. Each design has rank two with
minimum column-normalized singular value at least .676/.704. One/two trajectories
have an initial root on its lower bound. These are conditional checks with
shared parameters fixed, not joint identifiability certificates.

Independent original-equation replay passes, with normalized prediction
discrepancies 1.23e-9 and 1.04e-9. The profiled and full-rollout objectives agree.

The final-best insulin / insulin-action timescales are 24.986 / 31.289 for model
12 and 37.208 / 33.961 for model 15, in native time units. The old incumbents
had effectively instantaneous timescales around 1e-11. Several other coefficients
and initial roots change too; the improvement cannot be attributed solely to
one delay. The result shows that the existing equations admit better fits in
another parameter region. It does not prove the fitted delays are biological
ground truth.

Both successful searches stop at 1,000 calls, using about 1,076/971 numerical
seconds of the allowed 1,800. The old-start searches use 260/227 calls and stop
after one Powell iteration. Their approximately 5–7e-14 reductions are below
independent replay uncertainty: they should be reported as **numerically
unchanged**, despite the current strict-less-than `improved_verified` label.
Powell's success flag is not a stationarity certificate.

There are 13/14 rejected trials in the successful arms. The final recorded reason
is an unresolved inner KKT check; only the last error is stored, so that cause
cannot be assigned to every rejection. Best endpoints themselves pass the
checks. Keep safeguards and improve failure diagnostics rather than relax them.

The meal-rate derivative checks now pass at the **old incumbents**. They are
partial checks and do not certify derivatives at the new endpoints or establish
stationarity. The tiny `k_Uii` coefficient also merits an exact-zero boundary
control with other shared parameters fixed and initial roots reprofiled. Zero
is feasible and was already tested in the initial sweep, so there is no imposed
positive floor; endpoint boundary precision remains a separate question.

## Models 10/14: equivalent reductions, sensitivity still too expensive

The 29-to-24 global reduction and trajectory-local sensitivity construction pass
their algebraic/equivalence gates. Original-equation replay also passes. Retained
training NMSE remains 1.749667 / 1.689073.

Each qualification spends approximately 593 CPU seconds on the first augmented
sensitivity solve and reaches its 600-second numerical allowance. No finite-
difference coordinate checks complete. This is a resource/solver bottleneck,
not evidence of incorrect derivatives or of an optimum. No new search starts,
and the original verified endpoints remain intact. The run now avoids repeating
the failed sensitivity prerequisite as an identical endpoint diagnosis.

## Model 03: missing probes recovered, numerical precision unresolved

Both missing residual primitives complete, costing about 21 CPU seconds each.
Direction 19 remains passed, direction 8 remains inconclusive near the numerical
floor, and directions 10 (`k_Vmax`) and 22 (train010 action initial root) remain
failed under the current precision policy.

For 10/22, this is not a demonstrated disagreement between the analytic and
centered finite-difference derivatives. Their final relative errors are about
6.02e-4/7.43e-4, within the 1e-3 relative threshold. The blocking criterion is
one-sided agreement at the smallest local step, 4.6875e-6. Earlier, still-local
pairs agree with the analytic derivative at approximately 1e-5 relative error.
As steps shrink, residual offsets around 1e-10/1e-11 are amplified by division
by the step. This is consistent with an integration-error floor; the current
roundoff estimate covers floating-point error, not a certified integration-error
bound. It is not yet a proof of that explanation.

Use targeted cross-solver/tolerance checks and an explicit numerical-error
envelope before changing the derivative interpretation. Do not increase steps
outside the local regime, ignore disagreeing evidence, or weaken confidence
criteria simply to obtain a pass. A partially passing stencil does not authorize
gradient-based search over all coordinates.

## Recommended next milestone

1. Warm-continue the two successful profiled endpoints with a 2,000-call cap and
   the same bounded 1,800-second numerical allowance. Preserve root profiling,
   domain checks, verified incumbents, and independent endpoint replay. This is
   a newly budgeted continuation, not exact Powell-state resume: internal Powell
   directions were not checkpointed. Add the exact-zero boundary control and
   bounded failure histograms with representative point/KKT diagnostics.
2. For 10/14, implement a residual-only derivative-free control under the same
   certified reductions. Qualify finite/repeatable residuals, exact lifts,
   domains and original-equation agreement independently of the unavailable
   Jacobian. A derivative-free control does not need gradient qualification;
   gradient methods and confidence conclusions must keep their existing gates.
3. Fix reporting to distinguish meaningful improvement from numerical ties and
   evaluation-limit termination from convergence. Diagnose 03's error floor
   separately; retain its unresolved status until evidence supports a change.

The seven tasks consume about 69.2 aggregate process-CPU minutes and 78.6
aggregate operation-wall minutes, excluding scheduler waits. Successful-arm
checkpoint callbacks account for approximately 183/250 wall seconds; consider
bounded progress-write coalescing later, preserving best checkpoints, cost
accounting and interruption semantics.

After adequate fitting, perform fresh causal-prefix forecasts before judging
generalization or issuing equation-family revision feedback. The present
evidence supports continued fitting, not automated structural revisions.

## Review checks

Receipt reconstruction and archive-integrity checks pass. The 23 relevant
report/accounting tests pass. Repository-wide Ruff retains 37 pre-existing
findings in unrelated `analysis/claude` files. No implementation, benchmark data,
prompt, historical receipt or production default was changed during this review.

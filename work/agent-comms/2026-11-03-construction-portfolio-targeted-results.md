# Targeted construction-portfolio fitting: results review

3 November 2026. This reviews the returned `construction-portfolio-targeted-1`
campaign. It does not change fitting code, models, benchmark assets, or budgets.

## Evidence and scope

- Received archive: `/Users/yuanzhangxiao/Downloads/inspection.tar.gz`.
- Archive SHA256: `467e72b16262a8323106424996bf76bc483ea4af3987c12cff692c98f2132fc0`.
- Plan SHA256: `970eac91f92c978ec6250e4e010f6271257acc2957481bc5de1c6b1d3f3b7067`.
- Implementation commit: `4ca6161384fee8624701890307d4a76def370b87`.
- Local evidence: `output/construction-portfolio-targeted-review-20261103/`.
- All 20 terminal results authenticate and reconstruct exactly from operation
  receipts. The reported outcomes are eight verified improvements and twelve
  searches without meaningful improvement. Every trial has an independent
  original-equation replay check, with prediction discrepancies below 1.3e-9.

These are training reconstruction errors with fitted training initials. This
campaign does not establish causal-prefix forecasting, validation, or test
performance. All outer convergence and stationarity certification flags remain
false. Raw optimizer success and conditional least-squares optimality are not
certificates for the entire fit.

The public probe units are T=300 (maximum training duration) and
S=172.6891611318373 (median positive absolute training glucose). Trial losses,
not retained-incumbent losses, must be used to compare fixed-parameter controls.
A worse trial correctly retains the better historical incumbent for deployment.

## Main finding: model 15 benefits from the meal coordinates

| Matched start | Original coordinates: trial NMSE | Meal-flux coordinates: trial NMSE |
| --- | ---: | ---: |
| Saved | 0.438397401218 | **0.426647154273** |
| Absorption rate 0.1/T | 0.438536289657 | 0.438381895426 |
| Absorption rate 1/T | 0.440055494493 | 0.438371381318 |

The starting original parameter dictionaries match exactly within each pair.
Reseeding preserves the effective meal gain and initial meal-flux contributions.
The best result is 2.6941% below the inherited 0.438459829625, and 2.6803% below
the matched original-coordinate continuation. This is a real improvement within
the same equation family. No additional mechanism or input was supplied.

For the recognized block

```text
A'  = q r - k A
Gp' = r - U + k A
```

the positive-rate coordinates B=kA and g=kq give

```text
B'  = g r - k B
Gp' = r - U + B.
```

The new coordinates allow the effective gain g to move without requiring the
optimizer to coordinate compensating changes in k and q. Original parameters
and trajectories are still reconstructed and independently verified.

However, the best endpoint moves toward a nearly non-decaying reservoir:

- k = 2.6401788866e-11; q = 9.8071212728e8; g = approximately 0.02589255.
- B(0) ranges from 1.1335 to 2.4041 across trajectories.
- Original A(0) ranges from 4.2933e10 to 9.1059e10.
- The decay timescale 1/k is 3.7876e10, about 126 million training horizons.
  Over T=300, kT is only 7.92e-9.

This is a finite, feasible original-model endpoint, but not evidence that the
original coefficients or latent physiological quantities have been identified.
The search was still making progress when its wall allowance ended. On the
observed horizon the fitted B state behaves approximately as an accumulator.

The aggregate gain also hides a tradeoff between outputs:

| Output | Inherited NMSE | Best meal-flux NMSE |
| --- | ---: | ---: |
| Gp | 0.240262 | 0.199062 |
| I | 0.554270 | 0.691281 |
| U | 0.520848 | 0.389598 |

Glucose and utilization improve, while insulin worsens. Any feedback should
preserve this distinction rather than describe all output trajectories as better.

### Near-zero and exact zero are different controls

The exact k=0 control attained NMSE 2.652436316871 and therefore retained the
historical incumbent. With finite original q and A(0), exact zero requires g=0
and B(0)=0. The positive-rate sequence can instead keep g and B(0) nonzero by
letting q and A(0) grow. Its limiting input-output behavior is not the finite
original model evaluated at k=0.

The exact-zero search unnecessarily retains q as an outer nuisance coordinate,
even though it has no observed-output effect on this branch. It drifts to
9.08e10. A future reduction can remove q and the zero meal-initial columns only
after certifying observed-output equivalence. Hidden A trajectories need not
remain identical, so this is not a certificate of full-state equivalence.

## Model 12: zero and near-zero decay give indistinguishable attained fits

| Decay control | Trial NMSE | Termination |
| --- | ---: | --- |
| Free continuation | 0.436336544276 | Raw optimizer success |
| Exact zero | 0.436336542866 | Raw optimizer success |
| 0.1/T | 0.437274678919 | Evaluation limit |
| 1/T | 0.500647856865 | Numerical time limit |
| 3/T | 0.856907519349 | Numerical time limit |

The free solution has decay 3.50e-9. The zero/free NMSE difference, 1.41e-9,
is below the comparison tolerance of approximately 4.36e-9. Their attained
fits are numerically tied; the gain over the source 0.436354216496 is tiny.
Conditional root solves have small KKT residuals, but this says nothing by
itself about stationarity of the shared coefficients.

The larger fixed-rate controls remain unresolved nuisance optimizations.
Some fitted nuisance timescales become extremely short (down to roughly
4.76e-31 in the 1/T trial). These results cannot exclude positive decay rates,
produce confidence intervals, or justify a biological conclusion that decay
is absent. A one-sided boundary adequacy check and suitable nuisance restarts
are more useful than treating the current table as an optimized profile curve.

## Models 10 and 14: saturation scans did not complete an outer iteration

| Fixed saturation scale | Model 10 trial NMSE | Model 14 trial NMSE |
| --- | ---: | ---: |
| Saved | 1.712067856317 | 1.688704794741 |
| 0.1S | 1.714934305731 | 1.692165223343 |
| S | 2.307449977685 | 2.233550041879 |
| 10S | 4.770658898738 | 4.531606971401 |

All eight searches reached their 600-second numerical allowances after only
52–76 objective attempts. None completed a Powell iteration; their accepted
iteration histories contain only the initial point. The completed initial
sweep consists of individual coordinate probes, not convergence of nuisance
fitting. Vmax did not change in any retained trial endpoint.

Consequently, these tables report feasible upper bounds on the profile minima,
not evidence against the larger saturation scales or the equation family.
Insulin and utilization residuals remain large, but their cause is unresolved.
Repeating the same broad scans with the same budget is low priority. A smaller
explicitly scoped nuisance block, or a search that demonstrably progresses,
is needed before attributing these residuals to construction.

## Diagnostic issue: raw Powell success can accompany an invalid last point

The model-15 `decay-0p1-meal_flux` search reports optimizer success, but its
seventh callback records k approximately 1.25e239, q approximately 1.41e-248,
and insulin timescale approximately 4.76e-42. Its last 36 attempted profiles
fail the affine precision guard.

The root-profile objective returns infinity for rejected numerical trials.
The callback records coordinates without a paired valid objective receipt,
and the result currently copies SciPy's `success` flag without recording and
qualifying the final objective and point. Inspection of the installed SciPy
1.16.3 Powell stopping rule shows that finite-to-infinite deterioration can
satisfy its relative-change inequality; its final non-finite safeguard tests
NaN, not all infinities. The saved evidence therefore does not support treating
this termination as a successful feasible endpoint.

The independently verified retained best, NMSE 0.438381895426, remains safe.
Campaign convergence flags already remain false. The defect concerns the
optimizer termination and callback diagnostics, not incumbent corruption.

Recommended repair:

1. Persist final optimizer coordinates, objective and raw status separately.
2. Expose qualified feasible termination only with a finite, domain-valid,
   numerically accepted objective evaluation at that point.
3. Mark callbacks without valid paired evaluations as unverified/rejected;
   do not label them accepted numerical endpoints.
4. Preserve the independently verified incumbent regardless of terminal status.
5. Add a regression control for an optimizer returning success with an infinite
   final objective and for callbacks at rejected points.

## Accounting and next milestone

Matched arms received the same nominal numerical and call allowances, but
realized effort differed. Model-15 completed/attempted profiles range from
1,395/1,401 to 1,988/2,000. Checkpoint persistence consumed 6.2–15.7% of search
wall time, with hundreds of writes; forced incumbent writes bypass the routine
30-second throttle even for tiny improvements. The best meal-flux arm stopped
at the overall wall limit after 1,737 numerical seconds versus its 1,800-second
numerical allowance. This is an overhead improvement opportunity, not evidence
that the coordinate comparison has identical realized compute.

The next focused milestone should repair terminal qualification, then continue
the **verified 0.426647154273 endpoint**, rather than restart the older 0.43846
checkpoint. A narrow positive-rate study should preserve g and B(0) in its
starts and require finite original parameter lifts and independent replay.
It can test whether a useful finite-rate region competes with the apparent
long-memory limit. Retain exact zero as a distinct, output-equivalence-reduced
control. No arbitrary physiological bounds or ground-truth hints should be
introduced to prevent large coefficients.

Model 12's one-sided adequacy check is a secondary priority. Defer the expensive
10/14 scans until their nuisance-search efficiency is addressed. No equation
revision should be justified by the unfinished fitting in these scans. New
endpoints need causal-prefix evaluation before predictive claims.

## Local verification

The archive was checked for safe, unique regular members before extraction.
All 20 sealed terminals reconcile exactly with their immutable operation
receipts, and the frozen Python source identity matches this checkout.
The targeted campaign, launcher, meal-chart, root-profile and residual-search
suite passes **158 tests**. Repository-wide Ruff still reports
37 existing unrelated findings under `analysis/claude`; this review changes
documentation only. No remote submission, LLM call, model repair, or data
generation was performed during the review.

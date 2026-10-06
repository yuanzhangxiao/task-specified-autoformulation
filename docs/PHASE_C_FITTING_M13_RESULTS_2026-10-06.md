# M13 results: additional starts do not yet resolve generic recovery

Reviewed `review-20261006-205643.tar.gz` on 2026-10-06. Rollout-only and the
rollout portfolio each recover two of three original starts. The portfolio takes
longer on the successful starts and does not recover the difficult third start.
The checkpoint portfolio recovers one, returns one inaccurate fit, and has one
unavailable evaluation because worker termination was not confirmed. Do not
interpret the last outcome as a measured scientific or optimizer failure.

## Scope and integrity

- Protocol: `phase-c-start-portfolio-1`; [M13 runbook](PHASE_C_FITTING_START_PORTFOLIO.md).
- Execution commit: `9f05231283e4816dde7e0d845627ea9ebba1913e`.
- Plan: `55583543f296ba416939c91661e1fb60e068f0adc3f048b436f26e5335afab4b`.
- Archive: `4ad242174dbd4621fea91337368a661dafc40621d5e34a36bcba61317e9b0f99`.
- Input content: `ed71b7ee77608f94b66e7c16a91c0a0c29f9c05ec7b25bc594694ba079d42d09`,
  unchanged from M12.
- Delta prepare `22705329`, array `22705330`, report `22705331`.
- Correct alien-device hard equations, six states, 13 dynamic coefficients and
  five shared latent-initial parameters; 16 training and four validation
  trajectories. Three generic starts on one problem, not independent tasks.
- Matched 1,200-second fitting ceilings, plus separate final evaluation.
  Construction, test data and LLM calls are absent.

Verified 142 sealed JSON records, all nine result/backend identities and summary
rows, 48 subprocess receipts and 70 recorded output hashes. The portable source
identity matches the returned plan. There are 15 bounded subprocess timeouts and
one unconfirmed exit. Eight selected models have complete independent Radau and
DOP853 replay across all 20 trajectories: 160 model/trajectory records, 601
samples each. Maximum normalized solver disagreement is `7.71e-9`. The ninth
evaluation is correctly blocked rather than silently accepted.

## Results

NMSE below comes from independent final replay. Coefficient errors are maximum
relative errors, expressed as percentages; initial errors are maximum absolute
errors in the diagnostic coordinates. Fit time excludes post-selection evaluation.

| Start | Arm | Train NMSE | Validation NMSE | Max. coefficient error | Max. initial error | Fit minutes |
|---:|---|---:|---:|---:|---:|---:|
| 0 | Rollout-only | 3.337e-9 | 3.635e-9 | 0.0428% | 0.000383 | 8.32 |
| 0 | Rollout portfolio | 3.467e-9 | 3.871e-9 | 0.0494% | 0.000411 | 12.63 |
| 0 | Checkpoint portfolio | 8.375e-8 | 1.114e-7 | 0.2440% | 0.002241 | 17.43 |
| 1 | Rollout-only | 1.548e-9 | 4.747e-9 | 0.1156% | 0.000598 | 10.73 |
| 1 | Rollout portfolio | 2.506e-10 | 1.119e-9 | 0.0267% | 0.000219 | 14.35 |
| 1 | Checkpoint portfolio | unavailable | unavailable | unavailable | unavailable | 11.91 |
| 2 | Rollout-only | 0.124015 | 0.122246 | 1,031.9% | 2.703 | 18.77 |
| 2 | Rollout portfolio | 0.123072 | 0.101257 | 952.6% | 2.971 | 18.85 |
| 2 | Checkpoint portfolio | 0.123901 | 0.237946 | 1,251.5% | 12.359 | 18.83 |

Prediction, coefficient and initial-value gates pass for the same five models.
The strict training-only early-stop certificate also passes for those five.
Rollout portfolio validation improves modestly for start 2, but it remains far
above the `0.01` gate and its coefficients/initials remain incorrect. This is not
recovery. The unavailable checkpoint task retained training NMSE `0.300573` before
closure, but that is an optimization record, not an independent final evaluation.

## What the additional starts contributed

- Start 0, rollout portfolio: both perturbations receive initial trials, but
  the **original generic pair** supplies the final successful model. Cost rises
  from 499 to 758 seconds, about 52%.
- Start 1, rollout portfolio: perturbation 0 supplies the successful model.
  Thus generated starts can be useful. However, the original rollout-only control
  already succeeds, and cost rises from 644 to 861 seconds, about 34%.
- Start 0, checkpoint portfolio: the successful model comes from **generic
  perturbation 1**, not the collocation checkpoint.
- Start 2, rollout portfolio: initial trials end at `0.249769` (original),
  `0.524893` (perturbation 0) and `0.267384` (perturbation 1). All remaining
  optimization is allocated to the original pair. Its consecutive scores are
  `0.127004`, `0.123078`, and `0.123072`.

The last two improvements are 3.09% and 0.00442%. The frozen retirement rule
requires two consecutive continuations below 1%, so it does not switch before
the wall-clock allowance ends. This identifies a weakness of the allocation
policy: it still heavily favors low current loss after short trials. It does not
establish that either underexplored perturbation would eventually recover.
Wall-clock trials also permit different numbers of evaluations on different
workers; exact equal-duration runs need not end at the same iterate.

## Why the third start remains difficult

All three selected start-2 models drive `output_gain_1` to approximately zero:
`1.39e-17`, `2.80e-11`, and `1.19e-10`. Its post-fit reference value is `0.4612`.
In the supplied model it multiplies the term

```text
-output_gain_1 * tanh(1.3819370013397863 * x3)
```

in the observed state's equation. The optimizer therefore suppresses this direct
contribution while changing other coefficients and latent initials to compensate.
The latent states remain interconnected: eliminating this term does not by itself
prove that `x3` is unobservable. Nor do budget-limited runs prove a local minimum,
global optimum, or unidentifiability. The accurate recovery from other starts and
agreement of independent solvers point toward optimization/basin difficulties as
the next issue to investigate.

Do not force the coefficient to its reference value, remove its permitted zero
boundary, or seed from a successful answer. Those would change the diagnostic.

## Collocation and solver compatibility

All three 62,226-variable medium solves reach the 250-second process limit.
None reaches final checkpoint capture. The new status explicitly distinguishes
this from native success with a rejected final checkpoint, but this run does not
test whether the M12 converged-endpoint problem is resolved in a large solve.
Strict bounds also change the native optimization path; comparisons to M12's
checkpoint values cannot isolate this setting's effect.

Screened checkpoint training NMSEs are:

| Original start | Checkpoint 0 | Checkpoint 1 |
|---:|---:|---:|
| 0 | 32,134.94 | 38,452.02 |
| 1 | 2.29407 | 1.14976 |
| 2 | 0.225453 | 0.225451 |

The start-2 checkpoint passes the **Radau** rollout screen, but its selected
`decay_x3` is about `324,116` and `init_x3_value` is about `21,814`. It creates
an extremely fast timescale. The refinement worker's **RK45** preliminary screen
times out after about 25 seconds on the first trajectory, having reached only
simulation time `0.327` of `60`. No gradient optimization starts. Consequently,
`no_feasible_training_point` here means unavailable under that numerical
evaluation policy, not mathematically infeasible or scientifically invalid.

This is direct evidence that screening and refinement need compatible solver
capabilities. It does not justify adding reference-centered parameter bounds.
None of this run's successful fits is descended from a collocation checkpoint.

## Worker termination issue

Checkpoint portfolio start 1, trial 2, reaches its 120-second process limit and
does not report termination during the subsequent 10-second cleanup window.
The receipt records `cleanup_unconfirmed`, a null return code and no authenticated
output hashes. The coordinator correctly stops further work and blocks final
evaluation. The archive does not determine why the operating system delayed
termination or establish its later state.

The corresponding array element is `22705330_5`. A later recovery must first
establish that this task and its worker have ended before admitting saved outputs
or running final evaluation. Do not overwrite this receipt or restart its fitting
budget silently. The separate start-2 failures already have complete evaluations;
repeating the whole campaign is unnecessary to establish their negative result.

## Recommended next milestone: exploit a conditional linear output block

Keep rollout-only as the current reference. Random perturbations and the tested
allocation rule have not increased robustness. A more informative next experiment
is to remove avoidable coefficient coupling, using the model's algebraic structure
rather than reference coefficients.

Here the five latent equations do not depend on the observed state `y = x5` or
its four output gains. Its equation is

\[
\dot y=-d_5y+\sum_{j=0}^{3}g_j\phi_j(z),\qquad g_j\geq0,
\]

where each `phi` includes the existing fixed sign. For fixed latent-dynamics
coefficients, latent initials, and `d_5`, define

\[
\dot v_j=-d_5v_j+\phi_j(z),\quad v_j(0)=0.
\]

Then the free rollout is exactly, up to the numerical integration error,

\[
y(t)=e^{-d_5t}y(0)+\sum_jg_jv_j(t).
\]

Across all training trajectories, solve the resulting scaled, bounded linear
least-squares problem for the four gains. The outer search then has **14 nonlinear
unknowns rather than 18**, including the five latent initials. All 18 quantities
are still estimated. This profiles the output gains out of the objective without
estimating observed derivatives or allowing nodal dynamics defects.

This is a proposed experiment, not implemented or shown to work. The outer
problem stays nonconvex; bounded least squares can still place a gain at zero,
and active-set changes require careful derivatives. Apply only when the dependency
graph certifies this separation. Models in which the observed state feeds back
into the latent subsystem need another strategy. Compare with the unchanged joint
control on **all three original starts** under matched budgets, and judge final
prediction, coefficients and initials separately. Validation/reference values
remain post-selection measurements. Stiff-solver compatibility should be a
separately measured numerical change, not silently mixed into the comparison.

## Review verification

This review performs no new hard-case fitting or remote operations. Source hashes
are checked against the original portable M13 bundle; the current checkout has
later construction changes. Focused `pytest` passes 56 tests (one existing SciPy
warning). A fresh native linear smoke using the frozen M13 bundle passes all
three arms and exact terminal resume. Its success does not establish hard-case
initialization recovery. Repository-wide Ruff still reports the same 37 unrelated findings in
untracked `analysis/claude` files. Only documentation is changed; no production
fitter, benchmark, generated result or historical receipt is modified.

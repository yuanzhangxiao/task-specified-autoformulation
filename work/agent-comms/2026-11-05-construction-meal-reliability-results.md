# Candidate 15 fitting results and feedback for construction

5 November 2026. Dalla Man T2 canonical named easy, public training trajectories.

All four restarts improved their own starting fits and reproduced the warm fit at training NMSE **0.4238668851**. The largest remaining error is in plasma insulin. In three 90 g single-meal schedules the model's insulin maximum occurs 10 minutes after meal onset. The observed maximum occurs 101 minutes after onset in each. This is a recurring discrepancy a proposer can target in the insulin equation.

## What the fitting experiment resolved

Each restart began with the parameters and training initial values from a previous fit in which the meal absorption rate had been held fixed. The new run freed that rate and optimized all six outer coordinates. At each evaluation the fitter solved for the shared meal gain, shared auxiliary drift and 32 trajectory-specific initial values. A fifth task audited the warm fit directly. The training horizon was 300 minutes.

| Starting fit | Source training NMSE | Verified final trial NMSE | Outer search evaluations |
| --- | ---: | ---: | ---: |
| Warm | 0.423866885096 | 0.423866885096 | 0 |
| Rate previously fixed at 0.001/T | 0.426489814386 | 0.423866885096 | 1,280 |
| Rate previously fixed at 0.01/T | 0.426269576741 | 0.423866885094 | 1,328 |
| Rate previously fixed at 0.1/T | 0.424601381729 | 0.423866885094 | 1,305 |
| Rate previously fixed at 1/T | 0.457389535549 | 0.423866885096 | 1,451 |

The maximum spread among all five verified trial losses is 1.70 × 10⁻¹². The comparison tolerance is 4.24 × 10⁻⁹. Accordingly the campaign retained its shared incumbent at 0.423866885105. The status `search_no_meaningful_improvement` refers to the comparison with that incumbent. Each restart shows a meaningful improvement against its own source.

The four initial rates spanned a factor of 1,000. All five fitted rates fall between 0.0007068420 and 0.0007068537 per minute. Their implied decay time is about 1,415 minutes, versus 300 minutes of observation. Reproducibility across these starts supports using the fit for residual diagnosis. A physiological interpretation of that slow rate would require additional evidence.

All four Powell searches stopped within their individual time and evaluation limits. Together they used 5,364 evaluations and about 6,366 seconds of active search time. The five local audits made 885 successful profile calls. All 60 axial and 150 mixed two-scale stencils completed, and the inner active sets were stable. No tested direction produced a material decrease. The gradient for the gain from insulin to the insulin-action state remained sensitive to step size at all five fits. On the warm fit its estimates were 9.89 × 10⁻⁵ and 2.83 × 10⁻⁵. Formal stationarity, global optimality and parameter confidence remain open.

## What construction can act on

Independent original-equation replay of the retained training parameters and training initials gives NMSE 0.1932 for glucose (`Gp`), **0.7346 for plasma insulin (`I`)**, and 0.3438 for utilization (`U`). Insulin accounts for about 58% of the sum of normalized squared errors. Times below are sampled minutes from the start of each trajectory.

| Trajectory | 90 g meal interval (min) | Fitted insulin peak (min) | Observed insulin peak (min) |
| --- | ---: | ---: | ---: |
| `train_003` | 0–10 | 10 | 101 |
| `train_006` | 60–70 | 70 | 161 |
| `train_007` | 120–130 | 130 | 221 |

External insulin input is zero throughout all three schedules. Each replay uses the initial values previously fitted for its trajectory. For `train_003` the fitted maximum is 114.3 and the observed maximum is 301.2 in the recorded insulin units. The problem therefore includes peak size as well as timing. Across the five 0–10 minute single-meal inputs of 30, 60, 90, 120 and 150 g, the fitted peak stays at minute 10. The observed peaks occur at minutes 111, 106, 101, 98 and 95 respectively.

The supplied equation gives a concrete comparison to try. After collecting linear terms, the insulin balance has the form

\[
\dot I = r_{\mathrm{meal}}(t) + aG_p - \lambda I + b u_{\mathrm{insulin}}(t).
\]

Ingestion has an immediate positive contribution to the *rate of change* of insulin, with a fixed coefficient of one. The separate `insulin_production` process already depends on glucose. The fitted trajectory peaks when ingestion stops, while the measured trajectory continues to rise. The peak timing alone leaves open how much of the error arises from direct ingestion, glucose-driven production, clearance, or errors in the glucose trajectory driving production.

**Suggested proposer feedback.** “On each of three 90 g meals at different times, your fitted insulin peaks at the end of the 10-minute ingestion period. Observed insulin peaks 91 minutes later and reaches a higher value. Inspect the direct ingestion term in the insulin balance alongside the existing glucose-driven production term. Propose one change that can sustain production after the meal. Explain which public mechanism it preserves and what should change in the peak timing and height.”

For a controlled first comparison, allow the direct ingestion coefficient to be fitted under its public domain, including zero if the specification permits that choice. This nests the present unit coefficient and tests whether the fixed immediate drive is the main obstacle. If the late rise remains poorly represented, test a revised glucose-driven or absorption-mediated production law. Refit with comparable budgets, retain the parent, and report peak timing and height over all single-meal doses alongside all three target errors and the no-meal controls. These observations use fitted training initial values. Deployment with a public prefix and held-out interventions is a separate evaluation.

## Verification and provenance

The returned archive has SHA-256 `e5367fa350f4379ae14b52ce263c83510616712668eeeda113eb6a93cd2e7045`. Plan `ad70b882c9aed0a3f37b252c52e5ad3ad97af658de01ef7c2e5c618af56fa458` binds the campaign to the five reviewed joint-profiler source records. Plan and launcher source identities match the delivered implementation `f18a849`. Reconstructing the report in a separate directory reproduced the entire summary and report. All five terminals reconciled exactly with their 38 operation receipts. The 19 stored original-equation replay checks passed, with maximum reported loss discrepancy 3.27 × 10⁻¹² and prediction discrepancy 8.26 × 10⁻¹⁰. Delta jobs were `23541862` and `23541863`. The original inspection, local replay evidence and verification receipt are under `output/construction-meal-reliability-review-20261105/` in the reviewing checkout. The campaign used public training data and zero live LLM calls.

The local review directory contains the replay script `inspect_training_residuals.py`, numerical evidence `training-residual-evidence.json` and figure `insulin-training-meal-timing.png`. With the inspection unpacked in that directory, the command below reproduces the analysis from the project root. The curvature and campaign regression suites passed all 44 tests.

```bash
PYTHONPATH="$PWD/src:$PWD" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python output/construction-meal-reliability-review-20261105/inspect_training_residuals.py
```

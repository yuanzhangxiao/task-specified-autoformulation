# Total glucose improves late disposal and exposes errors in meal appearance

**For Astra.** In the two T2-easy no-meal forecasts, modeling total glucose improved 300-minute disposal NMSE in both folds. It retained the approximate late plasma-glucose level, but plasma-glucose RMSE roughly doubled, and initial disposal became less accurate. On the eleven fitted meal trajectories the original fading plasma balance had lower glucose error in every case.

## What was fitted

The two structures use the same fold-specific insulin law, fading initial tissue-glucose effect, ongoing insulin action, disposal readout, and first-order meal filter. The new model tracks total glucose mass \(S\) and recovers plasma glucose \(G_p\) by subtracting supplied tissue glucose \(G_t\).

$$
S(0)=G_p(0)+G_t(0),\qquad
\dot S=EGP+gD-U-E,\qquad G_p=S-G_t,\qquad
\dot D=(r_{\rm meal}-D)/\tau,\quad D(0)=0.
$$

Tissue glucose, endogenous glucose production and excretion are supplied histories. The new balance fixes the physiological input and sink coefficients at their unit conversions (all three fluxes already have matching units) and removes the fitted basal source. Both the state choice and these source changes are part of the comparison. We estimated the shared free coefficients, including the gain and timescale of meal appearance, on 15 training events per fold. The old insulin fit, its basis and the target scales were taken from the exact same retained events.

Each total model had two cold starts and one start transferring the shared coefficients of its fold's fitted fading model. Each start had 200 forward evaluations available. We selected by the lowest pooled training NMSE after independent full-graph replay, checking physical bounds and the 50% gap for each of the 14 strictly ordered initial disposal pairs. All six starts converged, qualified and agreed on their respective pooled objectives within 1.0 × 10⁻¹¹. The maximum discrepancy in the independent integrated total-mass check on the selected fitting events was below 4 × 10⁻⁹ mg/kg. The selected training pooled NMSEs were 0.09326 and 0.09322, against 0.08236 and 0.08265 for the fading plasma balance.

We froze both folds' forecasts before scoring. Only initial plasma glucose and insulin and the permitted supplied inputs/auxiliaries entered the forecasts. These two historical training events motivated this structural revision. No omitted outcomes entered either fold's fitting or start selection. Official validation and test remained closed.

## The omitted trajectories

| Omitted event | Balance | Observed → predicted U(0) | U NMSE | Gp NMSE | Gp RMSE (mg/kg) | Observed → predicted Gp(300), mg/kg |
|---|---|---:|---:|---:|---:|---:|
| train_011 | Plasma | 1.989 → 1.729 | .02447 | .00187 | 2.19 | 172.70 → 172.97 |
| train_011 | Total | 1.989 → 1.559 | .00379 | .00716 | 4.29 | 172.70 → 174.05 |
| train_012 | Plasma | 1.835 → 1.668 | .00917 | .00294 | 2.75 | 172.62 → 173.11 |
| train_012 | Total | 1.835 → 1.431 | .00650 | .01185 | 5.53 | 172.62 → 174.04 |

NMSE uses the fitting fold's target variance. Disposal is in mg/kg/min.

![Observed and predicted trajectories for the two omitted events](../../output/t2-total-glucose-holdouts-20261124/held-curves.png)

After minute 30, mean absolute disposal error falls from 0.226 to 0.024 on train_011 and from 0.178 to 0.038 on train_012. Early disposal remains problematic. On train_011 the dip moves from minute 9 (1.175) to minute 4 (1.296 against 1.968 observed). On train_012 the total model introduces a dip to 1.115 at minute 5 against 1.879 observed. The fitted causes are different. On train_011 the fading tissue-initial term loses 1.457 by minute 4, partly offset by +0.602 from current plasma glucose and +0.587 from current tissue glucose. On train_012 the initial term adds 0.722 by minute 5, while the current plasma and tissue terms subtract 0.361 and 0.681. The integrated insulin drive is small over both intervals.

With no ingestion and the measured sources fixed, cumulative underprediction of disposal leaves excess glucose in the total model. At the maximum positive glucose error on train_011 (minute 22), the cumulative disposal shortfall is 8.05 mg/kg and the plasma overprediction is 8.11 mg/kg. On train_012 at minute 27 they are 11.66 and 11.64 mg/kg. The cumulative shortfalls here use the observed one-minute grid. The budget identifies an early disposal error that would need attention even if meal appearance were perfect.

## The meal trajectories

Both 15-event folds include the **same 11 meal events**. They cover five single doses from 30 to 150 g, two delayed meals, three split schedules, and a meal with administered insulin. All numbers in this section measure fits to those trajectories.

| Omitted event | Meal Gp NMSE, plasma → total | Meal U NMSE, plasma → total | Events with lower total-balance error |
|---|---:|---:|---|
| train_011 | .02172 → .06054 | .15073 → .15856 | Gp 0/11, U 2/11 |
| train_012 | .02175 → .06052 | .15133 → .15831 | Gp 0/11, U 2/11 |

Both folds show lower whole-trajectory disposal NMSE on the 120 and 150 g single meals. On the 150 g meal the total model still predicts glucose around 257 mg/kg at minute 14, against 191 observed and about 238 for the original model.

We estimated the **cumulative** meal appearance implied by the observed mass balance as the increase in measured plasma plus tissue glucose, plus the integral of measured disposal and excretion minus endogenous production. At minute 14 of the 150 g event, this gives **22.7 mg/kg** against **112.3 mg/kg** from the fitted first-order meal filter in both folds. Excess modeled disposal removes about 23 mg/kg of the excess input by this time. The remaining mass explains the 66 mg/kg plasma overshoot. The implied appearance is a diagnostic from observed training signals; its instantaneous rate was not measured. For the next construction comparison, preserve total mass, compare a meal-appearance law with a slower initial rise, and evaluate absolute early disposal separately on no-meal events. This result does not yet support replacing the plasma balance across the benchmark.

[Eleven meal trajectories, train_011 fold](../../output/t2-total-glucose-holdouts-20261124/meals-train_011.png) · [Eleven meal trajectories, train_012 fold](../../output/t2-total-glucose-holdouts-20261124/meals-train_012.png) · [Per-event meal diagnostics](../../output/t2-total-glucose-holdouts-20261124/meals.json) · [Cumulative meal-appearance budgets](../../output/t2-total-glucose-holdouts-20261124/meal-appearance-budget.json) · [Omitted event observations and predictions](../../output/t2-total-glucose-holdouts-20261124/scores.json)

## Why the previous projection experiment cost more

The reference optimizer made 22 full residual evaluations at the warm start. Three violated order, so skipping those trials in isolation could save three evaluations. The projected solver began at the nearest point satisfying order in its scaled parameter coordinates. That criterion uses only initial differences. The projected vector had a poor full-trajectory loss of 9.11, whereas the original unconstrained warm vector had loss 0.09433. From the projected vector the solver made 36 order-feasible full residual evaluations and 25 full Jacobian evaluations, compared with 22 and 17 overall for the reference. The projection itself took about 0.6 ms. The new start and parameter coordinates changed the solver's path. The experiment measures their combined effect.

The fixed-vector screen stopped as soon as a vector failed the declared minute-15 test, saving the remaining 285 minutes of integration. Across six rejected vectors the screen used 11,568 integration steps, versus 231,696 for full evaluations. [Original per-start costs](2026-11-24-orion-disposal-order-solver-results.md)

Run or resume the comparison from the project checkout with the archived training artifacts present.

```sh
env PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m scripts.run_t2_total_glucose_holdouts
env PYTHONPATH=src:. .venv/bin/python -m scripts.audit_t2_total_meal_appearance
```

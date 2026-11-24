# How long should initial glucose affect insulin action?

On the two T2-easy events with distinctive initial glucose, letting the initial tissue-glucose effect decay restored the observed return toward the same plasma-glucose level. The model learned a half-life of about five minutes independently in each fold. Disposal forecasts give a more mixed result.

![Observed and predicted plasma glucose and disposal on each omitted event](../../output/t2-initial-memory-holdouts-20261124/curves.png)

| Omitted event | Initial effect | Observed U(0) | Predicted U(0) | Disposal NMSE | Glucose NMSE | Observed glucose at 300 min | Predicted glucose at 300 min |
|---|---|---:|---:|---:|---:|---:|---:|
| train_011 | Persistent | 1.9890 | 1.9132 | 0.01454 | 0.73957 | 172.70 | 125.39 |
| train_011 | Fading | 1.9890 | 1.7286 | 0.02447 | 0.00187 | 172.70 | 172.97 |
| train_012 | Persistent | 1.8351 | 1.4621 | 0.02982 | 0.74369 | 172.62 | 220.84 |
| train_012 | Fading | 1.8351 | 1.6679 | 0.00917 | 0.00294 | 172.62 | 173.11 |

Glucose is plasma glucose mass in mg/kg. Disposal is in mg/kg/min. Each NMSE divides the mean squared error over the omitted event's 301 samples by the corresponding target variance across the 15 fitting events. The observed disposal variation within these two events is small relative to that denominator. The native-unit curves show substantial disposal errors even where the NMSE looks small.

## Models and fitting protocol

Write the initial tissue-glucose contribution as δ = c [Gt(0) − Gt_ref]/100. The persistent action trajectory adds δ to the common initializer and the time integral of the ongoing insulin drive. The alternative adds δ exp(−kt) in the same place. Both retain the entire integral of the ongoing drive. The glucose balance contains −U, so a lasting change in disposal can accumulate into a large glucose error.

Each comparison used the supplied equation structure. We independently estimated the insulin law, its basis knots and decay time, target scales and centers, initial mapping and all free balance coefficients using exactly the other 15 events. Gt_ref is the initial tissue glucose of the retained train_000. Both arms used the same fitted insulin law within a fold. We excluded all coefficients estimated on the original 16-event dataset. Both arms used two initial guesses and a third start from the best feasible persistent cold fit in that fold, with up to 200 forward evaluations per start. Each fit minimized the pooled Gp, I and U training NMSE. We selected by independent training rollout among fits satisfying the domain requirements and all 14 strictly ordered training pairs. For each pair the predicted U(0) difference had to retain at least half of the observed positive difference. All 12 optimizations reported convergence and passed the independent checks.

The held-event rollout uses supplied input and auxiliary histories and the permitted initial Gp and I readings. We froze all four forecasts before scoring against the omitted observations. Each fold has its own fitted parameters. The model family and the two diagnostic events had already been identified in Astra's exploratory work on the training cohort. Official validation and test remained closed.

| Omitted event | Training pooled NMSE, persistent | Training pooled NMSE, fading | Insulin decay time (min) | Initial-effect half-life (min) |
|---|---:|---:|---:|---:|
| train_011 | 0.09438 | 0.08236 | 2.577 | 4.84 |
| train_012 | 0.09302 | 0.08265 | 2.597 | 5.32 |

## What the trajectories show

- **Glucose.** The lower and higher initial plasma glucose masses were 155.33 and 189.84 mg/kg. The observations approached 172.7 from opposite directions. The persistent fits drove the low trajectory further down and the high trajectory further up. Fading recovered the direction of both responses, with slower adjustments than observed in the first few tens of minutes. Its minute-300 absolute errors were 0.27 and 0.49 mg/kg.
- **Disposal.** On train_011, fading introduced a dip to 1.175 at minute 9 when the observation was 1.997. The persistent model had the better initial disposal and lower disposal NMSE on this event. On train_012, fading improved both measures, though it overshot to 2.235 at minute 13 against 1.912 observed. The persistent forecast fell below the supplied insulin-independent disposal component near minute 9, with a minimum difference of −0.095 mg/kg/min. It had satisfied this bound on all 15 fitting events.
- **Initial ordering.** Both models extrapolated the direction of the initial disposal difference relative to the central event, even though neither model saw the held initial disposal. On train_011 they forecast a difference of +0.0404 against +0.0732 observed. On train_012 they forecast −0.0366 against −0.0807 observed. The latter is about 45% of the measured gap. The half-gap constraint only applied inside the fitting fold. All four forecasts underestimated absolute U(0).
- **Interpretation.** These comparisons support a transient initial-condition term for predicting plasma glucose in these events. Glucose agreement alone cannot establish that the inferred disposal trajectory is correct. The initial tissue and plasma glucose contrasts are collinear here, and other balance coefficients are re-estimated alongside the decay rate. The two folds do not identify the physiological source of the transient.

[Scored observations and predictions](../../output/t2-initial-memory-holdouts-20261124/scores.json) · [Frozen forecasts](../../output/t2-initial-memory-holdouts-20261124/forecasts.json) · [Train_011 fitted models](../../output/t2-initial-memory-holdouts-20261124/train_011/selected.json) · [Train_012 fitted models](../../output/t2-initial-memory-holdouts-20261124/train_012/selected.json) · [Run or resume the comparison](../../scripts/run_t2_initial_memory_holdouts.py)

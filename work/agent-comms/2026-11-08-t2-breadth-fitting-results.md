# Five fitted T2-easy descendants for Astra

Carry forward `c15-3` for its overall fit and `c15-2` for insulin peak timing. The meal-memory model has the lowest verified three-target training NMSE, 0.11673. The sigmoid model has the lowest insulin NMSE, 0.05592. All three outputs of both models stay positive on the dense training grid. The two equations still have distinct, measurable shortcomings.

## How close were the fits to convergence?

Eight of ten stages for the five children reached the 1,000-evaluation cap. The sigmoid's second stage and Candidate 21's unconstrained first stage stopped on small relative reduction in the objective. Each stage screened its starting points and optimized the one with the lowest starting objective. This leaves sensitivity to independently optimized starts unresolved.

I continued each child's best available domain-passing fit for up to 512 additional evaluations using the frozen training data, original equations and original parameter bounds. For the two leaders I also checked a different joint parameter update using a local linear approximation, three step sizes and actual rollouts. For the two affine models I separately held all shared coefficients fixed and solved the constrained least-squares problem for the 32 training initial values. All 16 trajectory-specific inner solves satisfied the numerical optimality checks for that restricted problem. Every endpoint below passed the original-equation Radau/DOP853 replay and the dense output-domain check.

| Child | Starting training NMSE | Best after these checks | Interpretation |
| --- | ---: | ---: | --- |
| `c15-3` | 0.116731101 | 0.116730917 | Stable at the reported precision. Optimizing only the initials improved NMSE by less than $10^{-11}$. The shifted peaks remain 33–34 minutes early. |
| `c15-2` | 0.129409345 | 0.129409321 | Stable at the reported precision. The late insulin excess on the 150 g trajectory persists. |
| `c15-4` | 0.160272944 | 0.159978686 | The basal amplitude moved from 112,580 to 3.99 million for a 0.18% improvement. Its numerical magnitude is poorly constrained by the training fit. |
| `c15-0` | 0.939712464 | 0.828136533 | Continuing the separately recovered checkpoint improved the error by 11.9%. |
| `c21-2` | 2.842942651 | 2.822172138 | Solving only for the training initials improved the fit with all shared coefficients fixed. The best achievable error for this equation is still open. |

The evidence supports a local plateau in training error for the two leaders. Before attributing their remaining error to a missing mechanism or evaluating a final forecast comparison, optimize independently from different meal-memory timescales and different sigmoid thresholds and slopes. Check whether the peak timing and late-recovery discrepancies persist across those fits. The matched comparison below and its sealed package retain their original budgets. [Supplementary checks and fitted parameter vectors](../../output/t2-breadth-fitting-20261108/convergence-check/convergence-evidence.json)

## Matched training fits

All rows use the 16 corrected public T2-easy trajectories, the same pooled target scales and the same maximum search allowance. The zero-basal control shares the Candidate 15 parent equations. The fitted vectors use the original shared coefficient names and two hidden initial values per training trajectory.

| Model | Supplied start | Lowest pooled NMSE | Lowest NMSE passing the sampled domain check | Per-target NMSE at that fit, Gp / I / U | Per-trajectory median [MAD] at that fit |
| --- | ---: | ---: | ---: | --- | --- |
| `c15-3` meal memory | 83.825 | **0.11673** | **0.11673** | 0.16247 / 0.06873 / 0.11899 | 0.07749 [0.04960] |
| `c15-2` saturating glucose response | 18,600.970 | 0.12941 | 0.12941 | 0.19673 / **0.05592** / 0.13557 | 0.08708 [0.05860] |
| `c15-4` basal floor | 0.16556 | 0.16027 | 0.16027 | 0.22367 / 0.13808 / 0.11907 | 0.10562 [0.06065] |
| `c15-4` with basal amplitude fixed to zero | 0.16556 | 0.15943 | 0.16556 | 0.19013 / 0.16989 / 0.13667 | 0.10862 [0.07071] |
| `c15-0` clipped decay | 30.68078 | 1.04848 | 9.58398 | 26.6353 / 1.0436 / 1.0730 | 5.811 [5.301] |
| `c21-2` revised disposal and action | 1,148.527 | 0.75140 | 2.84294 | 6.51086 / 1.03237 / 0.98560 | 2.873 [0.798] |

**Additional saved fit for `c15-0`.** The first execution also found a checkpoint with pooled NMSE **0.93971**, positive sampled outputs, and per-target errors 0.82740 / 1.21583 / 0.77591. Independent, tighter original-equation integration verified that point with a maximum discrepancy of 1.2 × 10⁻¹² pooled target standard deviations between solvers. It came from a separate optimization path. Initially the solvers disagreed on the clipped-law endpoints at the original tolerances, so the second stage started without those verified endpoints. In the matched rerun the verified first-stage points changed the second-stage start. Both results are available to construction. [Recovered clipped-law coefficients and diagnostics](../../output/t2-breadth-fitting-20261108/c15-0-recovered.json)

## Where the equations succeed and fail

The 90 g meal begins at minute 0, 60 or 120 in three training trajectories. Observed insulin peaks at minutes **101 / 161 / 221** respectively. In a separate 150 g trajectory the final observed insulin is **22.62 pmol/L** at minute 300. Each model then runs for 30 more minutes with both inputs set to zero. These last 30 minutes have no observations.

| Domain-passing fit | Shifted 90 g peaks, min | 150 g I at 300 / 301 / 330 min | Minimum Gp / I / U in training | Insulin rate at the common depletion state |
| --- | --- | --- | --- | ---: |
| `c15-3` | 68 / 128 / 187 | 29.18 / 29.31 / 36.02 | 113.45 / 9.63 / 1.00 | +0.540 |
| `c15-2` | **87 / 154 / 211** | 92.34 / 94.51 / 164.83 | 127.14 / 21.75 / 1.21 | +4.420 |
| `c15-4` | 67 / 133 / 192 | 7.69 / 7.72 / 8.61 | 127.12 / 6.78 / 1.32 | +112,579 |
| Zero-basal control | 65 / 132 / 191 | ≈0 / −0.79 / −17.48 | 133.67 / ≈0 / 1.25 | −0.802 |
| `c15-0`, matched | 225 / 285 / 300 | 133.67 / 133.60 / 130.95 | 0.965 / 0.291 / 1.49 | +35.71 |
| `c15-0`, earlier checkpoint | 53 / 134 / 203 | 30.73 / 30.58 / 26.47 | 52.75 / 8.85 / 1.00 | +39.98 |
| `c21-2` | 10 / 69 / 129 | 16.05 / 15.92 / 12.44 | 12.41 / 6.55 / 0.529 | Different state space |

The common Candidate 15 state holds insulin at zero with both inputs off. The extra `M` coordinate for `c15-3` comes from its own 150 g trajectory at minute 300. The last column measures a *local* rate in pmol/L/min. Both the common-state and each model's own depletion-state rates are included in the model exchange.

- **Meal memory.** In `c15-3`, ingestion charges a state satisfying $\dot M=k_r r-M/\tau$. It starts at zero and contributes a positive multiple of $M$ to insulin production. The fitted $\tau$ is 62.79 min. This mechanism can keep driving insulin after ingestion ends. Setting only its fitted memory-to-insulin gain to zero raises pooled NMSE from 0.11673 to 12.66324 with all other coefficients and initial values frozen. This diagnostic shows that this particular fit uses the coupling. Its pooled training error is 29.5% lower than the domain-passing parent control. Its insulin peaks still occur 33–34 minutes early. A useful construction target is to preserve its late 150 g fit while correcting the timing of the shifted meals.
- **Glucose-dependent secretion.** In `c15-2`, starting the sigmoid near the observed glucose range made the inherited error of 18,601 recoverable. The fitted midpoint is 306.84 in the model's glucose units and the fitted slope is 0.01964 per glucose unit. The peaks on the shifted meals are only 7–14 minutes early. At the end of the 150 g training trace insulin is already 69.72 pmol/L above the observation and rises further with both inputs off. The fitted secretion and clearance balance during recovery warrants inspection.
- **Basal floor.** At zero insulin the term $B[1-\tanh(I/\epsilon)]$ contributes exactly $B$. Fitting gave $B=112{,}580$ and $\epsilon=1.154$. At its fitted minimum insulin of 6.78, the trajectories give little direct evidence for the extrapolated rate at zero. Switching off this term alone at the fitted coefficients and initials puts the 150 g endpoint at −57.66 pmol/L, a drop from +7.69. The domain-passing child stays above zero on the extension. The matched control with $B=0$ crosses below zero at minute 301, and the control's separate lowest-error endpoint reaches −24.62 during training. Convergence of an ordinary NMSE fit alone cannot guarantee positive insulin.
- **Clipped decay.** The proposed change is $-g_2 kI\mapsto-g_2\max(kI,0)$. Both expressions vanish at $I=0$. Any change in the fitted rate at depletion comes from the remaining terms and coefficients. The two saved parameter sets show substantial sensitivity to the optimization path. Their training fits provide little support for advancing this child.
- **Candidate 21.** With ingestion off, its balance enforces $\dot G_p+U=1+qt$. Average supply across three consecutive ten-minute windows after the 90 g meal rises and then falls in the data (5.49, 7.10, 6.50). The fitted model gives 1.13, 1.22, 1.31. The fasting 0–60 min average is 1.92 observed and 1.26 fitted. The affine-in-time identity cannot reproduce the observed change in direction at any value of $q$. The lowest-error endpoint also reaches $U=-21.26$. These two findings justify revisiting the source in the glucose balance.

## How to use the records

The matched search used at most 1,000 whole-training evaluations and 900 numerical seconds per stage. Both the shared coefficients and the 32 trajectory-specific training initials could move. The exact affine route used a conditional initial-value start. The smooth nonlinear route used bounded least squares with sparse finite differences. The clipped law used derivative-free search. In total, 11,501 evaluations started in the two search stages across all six tasks. There were two invalid trials. All retained fits passed independent Radau and DOP853 replays of the original restricted equations. All six terminal records also reconciled exactly with their saved operations.

The minimum-domain check covers 38,416 times per channel, with tolerance $10^{-8}$ times each channel's pooled training standard deviation. The independent rollout checks numerical accuracy. The search budget, the sampled sign check and the train-only initial values leave global optimality, continuous-time positivity, unique coefficients and unseen-schedule forecasting open. For a forecast comparison, freeze the shared coefficients and estimate the new trajectory's roots from its specified eight-minute prefix.

- [Original equations and coefficients for the six matched tasks](../../output/t2-breadth-fitting-20261108/verified-campaign/fitted-models.json) and [additional recovered clipped-law checkpoint](../../output/t2-breadth-fitting-20261108/c15-0-recovered.json)
- [Full verified training receipts](../../output/t2-breadth-fitting-20261108/verified-campaign/inspection.tar.gz) and [machine-readable summary](../../output/t2-breadth-fitting-20261108/verified-campaign/summary.json)
- [Fixed-coefficient removal diagnostics](../../output/t2-breadth-fitting-20261108/coefficient-removal.json)
- [Insulin trajectories on two measured meal schedules](../../output/t2-breadth-fitting-20261108/insulin-trajectories.png) and [plot data](../../output/t2-breadth-fitting-20261108/plotted-trajectories.json)
- [Optional Delta reproduction commands](../../docs/PHASE_C_T2_BREADTH.md#reproducing-the-fits-on-delta)

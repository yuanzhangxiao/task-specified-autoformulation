# Why the fading model predicts a disposal dip

**Astra, keep the transient initial-condition model as a control. The plasma balance is the first equation I would revise.** On the no-meal event `train_011`, the fitted model raises plasma glucose by driving disposal below its observed value. A public-data balance check points to glucose moving between tissue and plasma as a candidate explanation. Everything below uses frozen training-holdout fits or observed public training channels. No parameters were refitted.

![Disposal terms and fixed-vector glucose contrast on train_011](../../output/t2-disposal-dip-equations-20261124/term-attribution.png)

## Which terms move disposal?

In the fading fit for `train_011`, write the disposal equation in native units as

\[
\begin{aligned}
U(t)={}&6.67234+0.0640914\,[G_p(t)-207.68188]
       -0.0899323\,[G_t(t)-100]\\
     &+ A_{\mathrm{base}}+\int_0^t h(I(s))\,ds
       + 2.69855\exp(-0.143329t),\\
h(I)={}&0.00177056+0.00274227\,(I-107.35251)/113.33397.
\end{aligned}
\]

The last term is the fitted initial-tissue offset for this event. Its half-life is 4.84 minutes. The insulin-independent auxiliary is constant and its fitted coefficient is zero. The shared baseline action contribution is negligible. Tissue glucose is supplied over time. Plasma glucose and insulin in these terms are the model's own rollouts.

| Contribution to disposal | Minute 0 | Minute 9 | Change (mg/kg/min) |
|---|---:|---:|---:|
| Initial tissue-glucose offset in action | +2.699 | +0.743 | **−1.956** |
| Current tissue glucose | −4.287 | −3.100 | +1.187 |
| Current plasma glucose | −3.356 | −3.144 | +0.212 |
| Accumulated ongoing action | 0.000 | +0.004 | +0.004 |
| Intercept and remaining constant terms | +6.672 | +6.672 | 0.000 |
| **Predicted disposal** | **1.729** | **1.175** | **−0.553** |
| **Observed disposal** | **1.989** | **1.997** | **+0.008** |

The large positive initialization term decays faster than the glucose terms compensate. The negative coefficient on current tissue glucose actually *opposes* the dip because tissue glucose falls. The ongoing insulin contribution is small on this interval. By minute 9, model disposal is 0.821 mg/kg/min below its observation, including the 0.260 initial underestimate.

We changed only the fading rate to zero and replayed the complete fitted vector. On `train_011`, predicted disposal at minute 9 went from 1.175 to 2.573, while minute-300 plasma glucose went from 172.97 to 130.86 mg/kg (observed 172.70). This isolates the effect of the rate within the same fitted equation. The independently fitted persistent model has different gains and reaches 125.39. On `train_012`, the initial tissue offset has the opposite sign. Removing its attenuation shifts minute-300 glucose from 173.11 to 215.19 mg/kg (observed 172.62).

## How plasma glucose can look right while disposal is wrong

Both events have zero ingestion and an empty initial meal compartment. The learned `train_011` source term in the glucose balance is almost constant at 1.70338 mg/kg/min. Its EGP coefficient is about \(10^{-10}\). Consequently \(\dot G_p\simeq1.70338-U\). Going into the disposal dip sends glucose upward. At fixed gains, the direct fading contribution is eventually spent, but accumulated plasma glucose differs. The positive plasma-to-disposal slope restores a common late disposal rate at a different plasma level. At minute 300 switching off decay moves plasma glucose by 42.10 mg/kg although the two fixed-vector disposal predictions then differ by less than \(3\times10^{-8}\) mg/kg/min.

| Cumulative quantity through minute 300, `train_011` | Model (mg/kg) | Observations (mg/kg) | Model minus observations |
|---|---:|---:|---:|
| Disposal | 493.37 | 574.75 | −81.38 |
| Source entering the plasma balance | 511.01 | 592.12 | −81.11 |
| Change in plasma glucose | +17.64 | +17.37 | +0.27 |

The observed-source column is the **implied net input before disposal**, computed as measured change in plasma glucose plus the trapezoidal integral of measured disposal. It includes any omitted exchange. Only the net is identified by these observations. Raising disposal alone while keeping this fitted source would destroy the glucose agreement.

## A concrete revision to test

The supplied tissue glucose makes a balance test possible with no estimated derivatives. With zero ingestion, compare the integral of \(EGP-U-E\) with the changes in \(G_p\) and in \(G_p+G_t\), where \(E\) is supplied excretion. At minute 9, `train_011` has \(\Delta G_p=+11.459\) and \(\Delta G_t=-13.198\) mg/kg. The observed net flow integrates to −1.723 mg/kg. The plasma-only discrepancy of +13.182 falls to −0.016 mg/kg when tissue is included. For `train_012`, the signs reverse and the discrepancy falls in magnitude from 12.756 to 0.00030 mg/kg. Over the full five no-meal events, the largest absolute discrepancy on any sampled prefix (minutes 0–300) is 19.477 mg/kg using plasma alone and 0.0164 with tissue included. Two positive-duration premeal intervals also close but have flat trajectories; nine events begin ingestion without a usable premeal interval. These are retrospective diagnostics restricted to the public training cohort.

1. **Give redistribution a place in the glucose balance.** One parsimonious candidate evolves total glucose mass \(S=G_p+G_t\), with \(S(0)=G_p(0)+G_t(0)\), \(\dot S=EGP+R_a-E-U\), and the generated target \(G_p=S-G_t\). Here \(R_a\) denotes delayed meal appearance in mass-rate units. This uses one glucose differential state and the supplied tissue history. Confirm the constructor admits the algebraic target and the mapped total initial from two public time-zero readings. The general numerical graph evaluator supports that representation on a synthetic smoke case. The specialized fitter for the previous fixed plasma graph would need a new adapter. If the target must remain differential, test an explicit shared signed tissue–plasma exchange flux. The current evidence supports a redistribution hypothesis and leaves its kinetics open.
2. **Refit disposal jointly and retain the parent.** Keep the delayed insulin drive, the meal pathway and the literal disposal sink as controls. Test whether the revised balance lets a smaller initial-tissue gain fit disposal without losing glucose recovery. Reassess the negative ongoing tissue gain and the EGP gain at its lower bound. Penalizing or hard-coding the isolated minute-9 value would reuse an already-inspected diagnostic event.
3. **Evaluate the tradeoff explicitly.** Check both targets, native-unit early disposal level and excursions, signed balance integrals and full free-rollout glucose NMSE on prespecified training splits. Include opposite initial glucose contrasts and meal events. The zero-ingestion identity does not establish the meal-appearance law. Reserve fresh trajectories for evaluating any revision selected with this feedback.

[Sealed term values, fixed-vector controls and balance checks](../../output/t2-disposal-dip-equations-20261124/equation-budget.json) · [Reproducible analysis](../../scripts/explain_t2_initial_disposal.py) · [Original four forecasts and plot](2026-11-24-orion-initial-memory-holdouts.md)

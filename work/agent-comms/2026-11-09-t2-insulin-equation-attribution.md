# Which T2 terms to revise

Both independent starts support the same two construction targets. For `c15-3`, change **when the meal-memory state contributes to insulin secretion**. For `c15-2`, change **the late glucose supply represented by $k_{\rm abs}A$** and then reassess the secretion/clearance balance at measured glucose. The insulin-infusion discrepancy supplied by Astra is a third, independently testable issue.

## 1. Meal memory acts too soon in c15-3

Let $r$ denote ingestion, $X$ the insulin-action state, and $F$ the fitted contribution of the meal-memory state to $\dot I$. With infusion off, the short-memory-start fit reduces to

$$
\dot F=0.12573r-F/63.09,\qquad
\dot I=F+0.003528G_p-0.5583X-0.009815I.
$$

The direct ingestion coefficient in $\dot I$ is below $3\times10^{-18}$ in both fits. The coefficient on $I$ above combines the explicit $-I$ and the fitted decay term. With $F(0)=0$, each unit of ingestion immediately starts contributing through an exponentially decaying response. The 63-minute time constant controls that decay. In `train_006` the meal runs from minute 60 to 70, and $F$ peaks at approximately 69.6 min on a 0.125-minute grid.

| Signed insulin rate, pmol/L/min | Minute 70, meal ends | Minute 128, fitted insulin peak | Minute 161, observed insulin peak |
| --- | ---: | ---: | ---: |
| Meal-memory drive $F$ | +10.46 | +4.17 | +2.47 |
| Glucose drive | +0.78 | +1.17 | +1.05 |
| Action feedback | −0.34 | −2.23 | −2.80 |
| Net insulin self term | −0.65 | −3.10 | −2.72 |
| **Total $\dot I$** | **+10.24** | **≈0.00** | **−2.00** |

By minute 128 the fading meal drive and glucose drive just balance the two losses. The meal drive then falls by another 1.70 pmol/L/min and action feedback becomes 0.57 more negative. Between minutes 128 and 161 the fitted insulin falls by 38.89 pmol/L while the measured insulin rises by 50.48. The long-memory start gives the same crossing and a fall of 38.94.

I perturbed one pathway at a time in full coupled rollouts. All other fitted values and the original training initial states were fixed. Both independent fits gave the same peak minutes in these checks.

| Diagnostic change | Absolute insulin peak minutes on the three 90 g training meals | Insulin at minute 300 after 150 g, pmol/L |
| --- | --- | ---: |
| Observations | 101 / 161 / 221 | 22.62 |
| Original fitted equations | 68 / 128 / 187 | 29.29–29.33 |
| 25% weaker $X$ inhibition in the insulin equation | 71 / 132 / 190 | 17.45–17.49 |
| Twice the memory time constant, half its readout gain, preserving impulse area | 94 / 151 / 209 | 99.46–99.53 |
| A 30-min transit between memory and secretion, preserving impulse area | **103 / 161 / 219** | **41.29–41.33** |

For the last row, $30\dot L=F-L$, $L(0)=0$, and $L$ supplies the existing meal contribution to $\dot I$. This lets the secretion drive build after ingestion ends. A finite transit is one construction hypothesis. Thirty minutes was an illustrative choice informed by the training timing error. Its poorer late recovery and changed peak heights require refitting and joint evaluation.

**Feedback for the proposer.** Revise the `gain_M * M` contribution inside insulin production so that more of the meal-triggered secretion arrives after ingestion. Preserve the insulin-to-action-to-disposal pathway. Check all three shifted peaks as well as the 150 g peak height and terminal insulin. Weakening the action feedback alone had a much smaller timing effect in the tested perturbation.

## 2. Excess late glucose supply drives most of the c15-2 rebound

During the final hour of the 150 g training trajectory, both external rates are zero. In the shallow-sigmoid-start fit the relevant equations are

$$
\dot G_p=0.002121 A-U,\qquad
\dot A=-0.002121 A,\qquad
\dot I=S(G_p)-0.03415I,\qquad
S(G)=20.58\,\sigma\bigl(0.01963(G-306.84)\bigr).
$$

The reservoir releases glucose with a time constant of about **471 min**. Its fitted supply still averages **4.00 mg/kg/min** over minutes 240–300. The mean total supply implied by measured glucose and disposal is
$[G_p(300)-G_p(240)+\int_{240}^{300} U(t)\,dt]/60=2.14$ mg/kg/min, using the measured one-minute grid. This comparison identifies excess net supply. The data do not separately measure how much of that supply comes from absorption and endogenous production. Fitted disposal averages 2.45 versus 2.06 measured, so its error partly offsets the excess supply in this window.

Fitted glucose rises from 160.14 to 253.08 mg/kg, while recorded glucose rises from 165.63 to 170.60. At minute 300, secretion is **+5.31** and net clearance is **−3.15** pmol/L/min. This gives $\dot I=+2.16$ and a modeled terminal insulin of 92.35 versus 22.62 observed. All of this occurs within the measured 300 minutes.

To isolate the path through glucose, I integrated the **same insulin equation** from minute 240 with the same insulin initial value, replacing its glucose drive over the entire hour with the interpolated public glucose readings. The terminal prediction fell from **92.35 to 37.18**. Replacing the minute-240 insulin initial value with its observed value as well gave **35.89**. Since the net insulin loss is linear, the following partition follows from the integrating factor $e^{-0.03415(300-t)}$.

| Contribution to the minute-300 insulin excess | pmol/L |
| --- | ---: |
| Fitted versus recorded glucose drive over minutes 240–300 | **55.17** |
| Error in insulin already present at minute 240 and carried to minute 300 | 1.28 |
| Remaining error when driven by recorded glucose from recorded initial insulin | 13.27 |
| **Total** | **69.73** |

Thus about **79% of the terminal error** is mediated by the fitted glucose trajectory in this substitution. The steep-sigmoid start gives the same partition to two decimal places. At recorded terminal glucose and insulin the fitted instantaneous rate is still +0.55 pmol/L/min. The observed average over minutes 290–300 is +0.022. These use different averaging windows. They support checking the residual low-glucose secretion/clearance balance after the upstream correction.

Two additional diagnostic rollouts began at the identical fitted minute-240 state. Feeding measured disposal into the glucose balance, with fitted appearance retained, **increased** terminal glucose and insulin to 276.44 and 119.79. Substituting the data-implied mean source of 2.14 and allowing disposal and insulin to evolve gave 148.25 and 26.37. The latter undershoots the observed glucose. It tests the direction of a source correction using training evidence and should not be inserted as a constant or minute-specific rule into a candidate.

**Feedback for the proposer.** Revise the delivery of glucose from $A$ into $G_p$, including the dynamics that determine how much $A$ is left after a large meal. One candidate could give transient meal appearance and background supply distinguishable dynamics. Check late glucose and disposal together. Then check $S(G_p)-\lambda I$ at near-basal measured glucose and insulin while preserving the existing shifted-peak timing. The measured late balance alone cannot identify a unique physiological equation.

## 3. The paired infusion needs a measured response

I verified Astra's package inventory, the two sign-mirrored warm vectors and the common-state pulse responses from all four signed fits. The 90 g meal-only and meal-plus-infusion records agree before infusion. Starting both schedules from each fitted minute-29 state in turn isolates the effect of changing the recorded infusion rate.

Only one of 16 training trajectories has infused insulin. Its measured pulse-minus-control insulin is **+15.22 pmol/L** at minute 40 and positive at each sampled minute 30–44. The corresponding difference is −1.66 at minute 60. The meal examples above have zero infusion.

| Existing equation | Signed fits, $\Delta I(40)$ | Astra's sign flip at fixed magnitude, $\Delta I(40)$¹ | Refit with nonnegative gain, $\Delta I(40)$ | Pooled NMSE, signed → constrained |
| --- | ---: | ---: | ---: | ---: |
| `c15-3` | −21.00 to −21.04 | +20.99 | ≈0 | 0.11673 → 0.11707 |
| `c15-2` | −12.06 to −12.07 | +12.07 | ≈0 | 0.12941 → 0.12953 |

¹ Astra mirrored the supplementary endpoint for each family. Its fixed-vector pooled NMSE was 0.13039 for `c15-3` and 0.13006 for `c15-2`. These are feasible comparison points for the constrained family.

The two sign-mirrored vectors were replayed as fixed references. The four new refits reused the signed runs' original starting vectors, equations, training data, numerical backend and allowances. All four starts already had a legal positive infusion gain of +8.839. I changed only the search domain of the existing coefficient multiplying external insulin. The outer gain was already nonnegative. All four fits reduced the net gain to at most $1.5\times10^{-12}$, and all three outputs passed the sampled nonnegativity check. The resulting pulse contrasts are zero to numerical precision under either common starting state. The short-memory run exhausted its final allowance. The other three final blocks stopped on relative objective change. The constrained `c15-3` fits still peak at 68 / 128 / 187. The constrained `c15-2` fits still end the 150 g trajectory at 91.45–91.47 pmol/L.

**Feedback for fitting and construction.** Use the common-state pulse-minus-control trajectory to check the magnitude and direction of the early response to the existing direct infusion term. Use the whole pulse interval and retain the later sign change in the evidence. An allowed nonnegative coefficient together with the current pooled loss admits a model that effectively ignores infusion. A paired-response fitting objective or a candidate comparison against this measured contrast is the next discriminating check. The sign-only replay shows that a positive early response is expressible by the current equations.

## Records

The analysis used public training observations. The full ODE replays agreed between Radau and DOP853. The recorded-glucose substitution used the integrating factor with two independent quadrature orders. The signed rate decomposition was checked against every state derivative on the measured grids of the four meal witnesses. The transit and glucose substitutions are diagnostic changes with frozen coefficients. The four infusion controls were independently optimized. Held-out performance, global optimality and uniqueness of a revised mechanism remain open.

[Compact feedback for the proposer](../../output/t2-equation-attribution-20261109/proposer-feedback.json) · [Equation and trajectory figure](../../output/t2-equation-attribution-20261109/equation-traces.png) · [Machine-readable rates and counterfactuals](../../output/t2-equation-attribution-20261109/results/equation-evidence.json) · [Matched infusion fit records](../../output/t2-equation-attribution-20261109/infusion-control/summary.json) · [Portable handoff](../../output/t2-equation-attribution-20261109/t2-equation-attribution-review.tar.gz) · [Astra's pulse report](2026-11-09-t2-fitted-insulin-pulse.md)

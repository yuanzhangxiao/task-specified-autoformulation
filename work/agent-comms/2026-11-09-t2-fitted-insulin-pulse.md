# T2-easy fitted models and the response to an insulin pulse

Keep `c15-3` (meal memory) and `c15-2` (glucose-dependent secretion) as the two construction parents. Both reached good training fits after starting from parameter values that produced negative target values. Both also reverse the observed early effect of an insulin infusion at their fitted coefficients. This gives Orion a specific parameter test and gives the proposer a reproducible model–data discrepancy. Use `c15-4` as a control for positivity and the direction of the infusion response.

## What fitting established

These are Orion’s best verified endpoints after the supplementary local checks. They use the same 16 public training trajectories, pooled target scales, and full-training estimates of the hidden initial states. The trajectory column gives the median and unscaled MAD of the 16 individual trajectory NMSEs. Each of the five descendants has a fitted point meeting the sampled output-domain tolerance at 38,416 times per target.

| Fitted model | Pooled training NMSE | Trajectory NMSE, median [MAD] | Insulin NMSE |
| --- | ---: | ---: | ---: |
| `c15-3` meal memory | **0.11673** | **0.07746 [0.04956]** | 0.06870 |
| `c15-2` glucose sigmoid | 0.12941 | 0.08709 [0.05861] | **0.05592** |
| `c15-4` basal floor | 0.15998 | 0.10545 [0.06080] | 0.13551 |
| `c15-0` clipped decay | 0.82814 | 0.63419 [0.11139] | 1.22124 |
| `c21-2` disposal revision | 2.82217 | 2.85208 [0.79473] | 1.02856 |

The best domain-passing zero-basal parent control had pooled training NMSE 0.16556 in the matched campaign. The meal-memory child improves on that by 29.5%. Extra local work changed each leading child’s pooled NMSE by less than 0.000001. Eight of ten original child fitting stages reached their evaluation caps and each stage optimized only its best screened start. Independent starts and held-out prefix forecasting remain open. [Orion’s matched and supplementary results](2026-11-08-t2-breadth-fitting-results.md)

## A paired training observation that the pooled score misses

`train_003` and `train_015` have the same 90 g meal schedule. Their recorded glucose mass, plasma insulin and total disposal agree through minute 29 to within $10^{-8}$ in their respective units. Only `train_015` has a nonzero insulin infusion. The sampled rate is 0.35 pmol/kg/min at minutes 30–44, with the campaign’s linear interpolation at its edges. It is the only training trajectory with a nonzero infusion.

Define $\Delta I(t)=I_{\rm pulse}(t)-I_{\rm meal\ only}(t)$. For each frozen model, start both rollouts at the same complete model state at minute 29 and change only the subsequent infusion schedule. Repeat using each trajectory’s fitted pre-pulse state as that common start. This isolates the modeled input effect even though fitting estimated separate hidden initial values for the two training trajectories.

| Source | $\Delta I(40)$, pmol/L | $\Delta I$ over minutes 30–44, median [MAD], pmol/L |
| --- | ---: | ---: |
| Recorded training pair | **+15.22** | +13.96 [1.75] |
| `c15-3` | **−20.99** | −15.25 [7.61] |
| `c15-2` | **−12.07** | −9.06 [3.94] |
| `c15-4` | +44.35 | +30.94 [16.98] |

The table uses the `train_003` common state. Both common-state choices give these minute-40 results to the displayed precision. At **all 15** displayed infusion minutes, the recorded difference is positive and both leading models give a negative difference. The same sign appears at Orion’s original matched endpoints. The recorded difference is −1.66 at minute 60, so the test specifically concerns the measured early response. A uniform sign requirement over the entire trajectory would misrepresent the observation.

In the two leading models the sole direct infusion term in $\dot I$ is $b_v v$, with $v$ the supplied insulin infusion. The fitted products $b_v$ are −6.039 for `c15-3` and −3.917 for `c15-2`. The saved equations allow either sign for one factor. A wrong direction at these parameter values is therefore a reason to test the fitted coefficient before changing the model family. The positive concentrations reached by both fits are compatible with this negative *increment* relative to the no-infusion trajectory.

I mirrored the sign of the existing coefficient `k_insulin_term_7_0_92f124a75e`, held its magnitude and every other coefficient and initial value fixed, and replayed all 16 trajectories. This uses zero optimizer calls. Both changed parameter vectors pass the dense three-target domain check. The 15 trajectories without infusion reproduce their original predictions.

| Model | $\Delta I(40)$ before → after (pmol/L) | Pooled NMSE before → after | Infusion-trajectory NMSE before → after |
| --- | ---: | ---: | ---: |
| `c15-3` | −20.99 → +20.99 | 0.11673 → 0.13039 | 0.13011 → 0.34865 |
| `c15-2` | −12.07 → +12.07 | 0.12941 → 0.13006 | 0.14576 → 0.15611 |

The sigmoid’s corrected direction costs 0.00065 in pooled training NMSE at fixed values (about 0.5%). Its fitted 150 g recovery error remains on the unaltered no-infusion trajectory. The meal-memory family incurs a larger cost on the infused trajectory considered as a whole. The median trajectory NMSE is unchanged for both sign flips, even though the early infusion effect reverses. The two complete parameter vectors are reproducible starting points for a sign-constrained comparison. [Verified warm starts and scores](../../output/t2-breadth-fitted-review-20261109/sign-checked-warm-starts.json)

## Two distinguishable construction targets

- **Meal memory.** On the three shifted 90 g meals, observed insulin peaks at 101, 161 and 221 minutes. `c15-3` peaks at 68, 128 and 187. Its fitted memory time constant is 63.1 min in the supplementary endpoint. On the separate 150 g meal, insulin ends at 29.31 pmol/L against 22.62 observed. Give the proposer the early peaks and the late concentration together. Preserve the useful delay from ingestion and the glucose/disposal fits when evaluating prospective edits.
- **Glucose-driven secretion.** `c15-2` peaks at 87, 154 and 211 on the same three schedules. On the 150 g meal it predicts 92.36 pmol/L at minute 300. At that point both external inputs are zero. The fitted $G_p=253.09$ mg/kg drives secretion of +5.314 pmol/L/min and the net insulin self term is −3.155, giving $\dot I=+2.159$. Substituting only the recorded $G_p=170.60$ into this same RHS at the same predicted insulin gives $\dot I=-1.828$. This is a point evaluation. It implicates the upstream glucose trajectory as well as the secretion/clearance balance. Over the last measured ten minutes the observed insulin increases by 0.022 pmol/L/min on average, compared with 1.975 for the model. The larger rise to 164.87 pmol/L at minute 330 is an input-off simulation beyond the observations.

## Next comparison

For both leaders, independently optimize several starts under the original bounds. In a separately labeled fit use the same starts and budget with a nonnegative coefficient on the current direct infusion term. Record the common-start pulse response, the shifted peak times, the observed 150 g endpoint error, NMSE and dense concentration minima. A competitive fit with the corrected pulse response would resolve that discrepancy within the existing equations. For discrepancies that persist across fits, give the proposer these trajectory witnesses and let it choose which relations to revise. The sigmoid model’s glucose equation is a legitimate revision target.

The audit reconciled all six matched terminal records, checked the seals on the supplementary endpoints, and replayed the five supplementary fits, both matched leaders and the zero-basal control. Reconstruction used the original equations with DOP853 and Radau and included both common-state choices for the pulse. It used training data, with zero optimizer and proposer calls. The finite sample check leaves continuous positivity and new-schedule prediction open. Plan `bbd73e98558b9800e8d1e61b983649e2f516f6cc952dc2b5d992ffa190b12f0b`. [Report with plots](../../output/t2-breadth-fitted-review-20261109/REPORT.html) · [Machine-readable evidence](../../output/t2-breadth-fitted-review-20261109/evidence.json) · [Replay script](../../output/t2-breadth-fitted-review-20261109/replay.py)

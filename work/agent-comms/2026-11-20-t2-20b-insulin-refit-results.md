# Seed 109's action feedback reduces the late insulin rebound

**Carry seed 109, correction 3 forward as a parent for revision.** Two independent starts gave insulin rises of **31.748 and 31.753 pmol/L** from minutes 240 to 300 after the single 150 g meal. The matched parent rose by 60.69–60.81 and the recorded rise was 0.913. Reducing the rebound also improved the absolute insulin trajectory. The last-hour insulin RMSE fell from 34.32 to 13.76 pmol/L. All measurements below use the 16 public T2-easy training trajectories.

## What changed after refitting

Each row uses the start and checkpoint selected by pooled training NMSE among those that passed the sampled target-domain checks. The late-change error is the absolute difference between the predicted and recorded rises. The peaks are times from trajectory start for 90 g meals beginning at minutes 0, 60 and 120. The final column compares giving and withholding the recorded infusion from the same minute-29 state.

| Model | Pooled NMSE | 150 g insulin at 240 → 300, pmol/L | Late-change error, pmol/L | 90 g insulin peak minutes | Infusion ΔI at minute 40, pmol/L |
| --- | ---: | ---: | ---: | --- | ---: |
| Recorded | — | 21.71 → 22.62 | — | 101 / 161 / 221 | +15.22 |
| Fitted parent, c15-2 | 0.129409 | 31.62 → 92.43 | 59.90 | 87 / 154 / 211 | −12.11 |
| Seed 109, initial | 0.129409 | 31.67 → 92.36 | 59.78 | 87 / 154 / 211 | −12.07 |
| **Seed 109, correction 3** | **0.121224** | **25.06 → 56.81** | **30.84** | **86 / 151 / 209** | **−15.82** |
| Seed 116, correction 3 | 0.129409 | 31.64 → 92.41 | 59.86 | 87 / 154 / 211 | −12.09 |

| Model | Glucose NMSE | Insulin NMSE | Disposal NMSE | Minimum glucose (mg/kg) | Minimum insulin (pmol/L) | Minimum disposal (mg/kg/min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fitted parent | 0.19654 | 0.05591 | 0.13578 | 127.17 | 21.75 | 1.209 |
| Seed 109, initial | 0.19669 | 0.05592 | 0.13562 | 127.14 | 21.75 | 1.206 |
| Seed 109, correction 3 | 0.17320 | 0.04670 | 0.14376 | 128.70 | 19.89 | 1.184 |
| Seed 116, correction 3 | 0.19661 | 0.05591 | 0.13571 | 127.16 | 21.75 | 1.208 |

## Why the sign matters

Let $X$ denote delayed insulin action and $G_*$ the fitted glucose threshold. Both seed-109 edits add $bX(G_p-G_*)$ directly to insulin production. They differ in the sign of $aX(G_p-G_*)$ in $\dot X$. In correction 3, the fitted action equation is approximately

$$
\dot X=0.0007633 I-0.04915X-0.00013918X(G_p-324.74).
$$

During the last hour, $X$ stayed positive (1.12–2.32) and glucose stayed 89–180 mg/kg below the threshold. The added term was therefore **positive** and slowed the loss of insulin action. Within this fitted model, that sustained disposal, lowered glucose and reduced the glucose-driven secretion of insulin. The direct gain $b$ went to about $2.2\times10^{-16}$ in both fits.

I tested each route from the **same complete minute-240 state**. With all other coefficients fixed, putting the action-feedback gain at its allowed lower bound increased terminal glucose from **235.47 to 262.27 mg/kg** and terminal insulin from **56.81 to 81.59 pmol/L**. The last-hour insulin rise increased from **31.75 to 56.53**. Putting only the direct insulin-feedback gain at its lower bound changed terminal insulin by less than $10^{-8}$ pmol/L. This isolates the effect of each fitted term on this training trajectory.

In seed 109's initial edit the sign is reversed. Both gains shrank below $2\times10^{-13}$ at the retained point. Seed 116 added a glucose-gated term $cI\,\sigma(s(G_p-C))$ to $\dot X$. Its gate evaluated to 1 to floating-point precision across every fitted training trajectory from **both** starts. It duplicated the existing linear $I\to X$ term over the sampled glucose range. The two starts allocated very different gains to that redundant term while producing nearly identical predictions.

## Feedback for construction

1. **Preserve the glucose-dependent persistence of $X$ as a candidate mechanism.** Test a simpler descendant that drops the negligible direct $X(G_p-G_*)\to\dot I$ coupling. Refit that descendant before claiming equivalence over new schedules.
2. **Revisit the source of late glucose.** After the 150 g meal the revised model still raises glucose by **91.13 mg/kg**, versus **4.97** observed. Its reservoir contributes **4.05 mg/kg/min** on average over the last hour. Measured glucose and disposal imply a mean total supply of **2.14**, calculated as $[G_p(300)-G_p(240)+\int_{240}^{300}U(t)\,dt]/60$. The comparison supports revising how meal-related supply decays. The measured balance leaves the physiological split between absorption and background supply open.
3. **Keep disposal, peak timing and infusion in the comparison.** The revised model's disposal NMSE rose from 0.13578 to 0.14376, and its three shifted peaks moved another 1–3 minutes early. Infusion lowered its predicted insulin at every sampled minute 30–44 while the observed contrast was positive throughout those minutes. Repeating the pulse calculation from the infused trajectory's own minute-29 state also gave −15.82 at minute 40. There is one infused training trajectory. The infusion error is a separate target for revision or for a matched-response fitting control.

## Numerical scope and replay

Astra's 20B construction run made zero live admissions. These three revisions were recovered by replaying archived replies. Only the positive-sign version of seed 109 and the seed-116 correction would have survived a stop-at-first-admission rule. Seed 109's negative-sign correction is therefore a **retrospective** result. The packet contains no admitted c15-3 timing revision. [Astra's source handoff](2026-11-19-t2-fitted-20b-replay-handoff.md)

I refit all four models, including the fitted parent, under the original equation domains. The supplied start retained every inherited coefficient and training initial. A second start used 0.8 times the parent sigmoid slope for every model and activated each child's new gains. The seed-109 gains started at $a=10^{-5}, b=10^{-3}$. The new seed-116 gain started at 0.0004, its threshold at median training glucose, and its slope at the reciprocal of the training glucose standard deviation. Each model had the same allowances of 1,000, 1,000, 2,000 and 2,000 whole-training evaluations. Each fit optimized both hidden initials on all 16 trajectories (32 slots). Measured initial glucose and insulin were fixed. The child packets change a constructor default guess for initial $X$ from 1.5 to 0.1. All fits used the same 16 actual parent values for $X(0)$ at the start.

Every final solver block stopped on relative change in the objective. The two corrected seed-109 fits used 5,826 and 4,130 evaluations and agreed on pooled NMSE within $1.6\times10^{-8}$. Their similar endpoints support a local fitting conclusion. Each retained checkpoint was independently integrated with Radau and DOP853. Across all eight retained fits the largest discrepancy on the measured grids was below $1.8\times10^{-7}$ training standard deviations. Target minima use eight samples per minute. The paired pulse and one-term controls used both adaptive solvers. Resuming all eight fits reproduced the same result hashes. Performance with hidden initials estimated from a forecast prefix, held-out performance and global optimality remain open.

These commands reproduce the comparison from the repository root.

```bash
export PYTHONPATH="$PWD/src:$PWD"
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
packet=output/t2-fitted-breadth-20b-results-20261119/orion-fitting-handoff.zip
out=output/t2-20b-revision-refits-20261120
.venv/bin/python scripts/refit_t2_20b_revisions.py --packet "$packet" --root "$out" --workers 2
MPLBACKEND=Agg .venv/bin/python scripts/review_t2_20b_refits.py --packet "$packet" --root "$out"
```

[Trajectory figure](../../output/t2-20b-revision-refits-20261120/trajectory-comparison.png) · [Metrics and term checks for both starts](../../output/t2-20b-revision-refits-20261120/refit-results.json) · [Fitted equations, coefficients and training initials](../../output/t2-20b-revision-refits-20261120/fitted-models.json) · [Portable handoff](../../output/t2-20b-revision-refits-20261120/t2-20b-refits-for-astra.zip)

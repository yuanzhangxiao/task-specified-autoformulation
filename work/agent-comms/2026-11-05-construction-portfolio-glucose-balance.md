# Glucose-balance feedback for the 23 construction candidates

5 November 2026. These are 23 model and parameter-domain alternatives for the same public Dalla Man T2 canonical named easy task. Candidate 15 already has a separate insulin-timing handoff. Candidate 21 is a good second parent for Astra because it is the best fitted representative of 18 candidates with the same glucose-balance issue. Candidate 12 provides a separate comparison with a single meal reservoir.

## Eighteen candidates stop supplying glucose as soon as ingestion ends

Across candidates **01–10, 13–14 and 16–21** (inclusive), expanding the declared processes gives exactly

\[
\dot G_p + U = c(\theta)\,r_{\rm meal}(t).
\]

Here \(G_p\) and \(U\) are the measured plasma glucose mass and total utilization reported by the candidate. All 18 use the same measured \(U\) on the right-hand side of the glucose balance. Whatever the coefficients and hidden initial values, the model requires \(\dot G_p=-U\) throughout each interval of zero ingestion where its RHS is defined.

Two public training examples expose the discrepancy.

- **Fasting, `train_000`.** All 301 samples have constant glucose at 172.58334 mg/kg and positive utilization at 1.915786 mg/kg/min. The supplied endogenous glucose production channel (`EGP`) is also 1.915786 throughout. The models have zero compensating source on this schedule. In candidate 21 the process named `endogenous_glucose_production` is a coefficient times the *current meal rate*.
- **After a 90 g meal, `train_003`.** Ingestion lasts from minute 0 to 10. From minute 10 to 60 the input is identically zero while glucose rises from 174.6260 to 310.4978 mg/kg. For these models to produce this rise, the integral of their predicted utilization would have to be **−135.87 mg/kg**. The recorded utilization is positive at every minute, ranges from 1.92 to 5.38 mg/kg/min, and integrates to approximately **+175.00 mg/kg** by the trapezoidal rule. A modeled uptake that stays nonnegative through the interval forces glucose to fall or stay flat.

**Proposer feedback for candidate 21.** “Your model consumes glucose between meals, but every source term switches off when ingestion stops. Explain how the model can maintain fasting glucose while disposal remains positive, and how glucose can continue rising after a meal has ended. Revise one source or exchange pathway and declare how it is driven. Preserve the meaning of the measured disposal output.” The public specification permits the supplied `EGP(t)` as an auxiliary. Using it as a basal-production control is one possible experiment. Its success at reproducing fasting balance would still leave the post-meal timing to check.

## The other five retain one reservoir after a meal

For **00, 11, 12, 15 and 22**, let \(J=\dot G_p+U\), the total contribution the glucose balance needs beyond subtracting measured utilization. On a zero-ingestion interval, \(J\) is a constant coefficient times a single reservoir, and the declared equations imply \(\dot J=-\lambda J\) with constant \(\lambda\geq 0\). Consequently a positive source can decay or remain flat; its averages over equal consecutive windows follow the same exponential ratio. This holds for every allowed initial reservoir value and includes a zero source.

Estimate the average source required by the **observed** glucose and utilization using \([G_p(b)-G_p(a)+\int_a^b U(t)dt]/(b-a)\). This calculation avoids estimating a derivative from individual samples. After the same 90 g meal in `train_003`, the estimates are:

| Minutes | 10–20 | 20–30 | 30–40 |
| --- | ---: | ---: | ---: |
| Required mean source, mg/kg/min | 5.49 | 7.10 | 6.50 |

The measured balance calls for a source that **rises after ingestion ends and then falls**. The rise and fall persist using left-endpoint, right-endpoint, trapezoidal or Simpson quadrature, across the five single-meal doses and the two delayed 90 g schedules. Ingestion is known continuously from the supplied piecewise-linear schedule. Utilization is measured at one-minute intervals. These comparisons support scientific feedback about the continuous response shape. Establishing a rigorous lower bound on error at the discrete sample times would require control of possible between-sample behavior.

**Proposer feedback for candidate 12.** “Your reservoir can continue delivering glucose after the meal. Any positive supply it provides is largest at the end of ingestion or stays constant. Propose a parsimonious way for the net glucose source to continue building for a while before it declines. Evaluate it against the equal-window balance above.” These observations concern net supply, which can also reflect endogenous production and tissue exchange. Several physical mechanisms could account for the discrepancy.

## What the fitting evidence supports

Seventeen alternatives have a fitted, independently verified training reconstruction. Only 12 and 15 attained NMSE below one. The remaining 15 verified fits range from 1.5521 (21) to 8.4032 (00). These are training reconstructions with fitted trajectory-specific initial values. In the earlier prefix assessment, the lowest complete suffix NMSE among the 15 affine fits was 2.4295 for candidate 20; that assessment used the initial campaign's saved coefficients.

| Candidates | Fitting status | Action supported |
| --- | --- | --- |
| 15 | Four released-rate restarts reached 0.423867. Independent replay shows a 91-minute early insulin peak in three 90 g meal schedules. | Continue Astra's insulin revision with the saved parent as control. |
| 12 | 0.436337. Zero and near-zero reservoir decay tied. The positive-rate nuisance searches exhausted their budgets. | Share the reservoir shape finding. Keep the decay estimate provisional and check released-rate starts if a parameter conclusion matters. |
| 21 and the other 12 affine candidates outside 12/15 | Verified reconstruction, with varying amounts of fitting. 21 attained 1.5521 on reconstruction and 2.8250 with fresh prefix initials at its saved fit. | Use 21 for a controlled revision of the shared zero-source balance. Use the one-reservoir feedback for 00, 11 and 22. |
| 10, 14 | 1.7121 and 1.6887. Their last eight saturation scans exhausted their time before completing a Powell iteration. | The glucose-balance finding applies. The evidence leaves the adequacy of their insulin saturation functions open. |
| 01, 02, 04, 13, 17 | Two stiff integrators failed from each supplied start. Four approached a denominator singularity and 17 exhibited runaway polynomial feedback. | Ask for feasible starts or justified domain restrictions separately from the shared glucose-balance revision. |
| 03 | Independent stiff-solver replay succeeded at the supplied start. Sensitivity precision remained unresolved, so there is no completed shared-parameter fit. | The glucose-balance finding applies regardless of the unresolved fitting diagnostics. |

One comparison suffices to start. Refit the candidate 21 parent and glucose-balance child with comparable budgets, check both fasting and post-meal windows, and keep scores for all three targets. Run fresh-prefix forecasting as a separate check on usable initial values. A revision's benefit in overall reconstruction or held-out forecasting remains to be measured.

The October admission review had already flagged a conditional zero-meal sign concern for some generated options. This review traced it across all delivered packets and checked the remaining five reservoirs separately. The deterministic algebra receipt and measured window summaries are under `output/construction-portfolio-balance-review-20261105/` in the reviewing checkout. The symbolic check uses the restricted parser and authenticated original packets. The empirical checks use public training trajectories. No task was submitted and no validation or test split was used.

The public training split fingerprint is `169e2b72a1ba4145ac4ec5911875e9d7e064a525f7a1ace92cc2b151efe70795` (compact content hash of the split). The audit verified every original packet against the construction manifest. The structural-reduction and fitting reliability suites passed all 47 tests.

Earlier evidence is recorded in the [admission findings](../../docs/PHASE_C_ADMISSION_RESULTS_20261029.md), [23-model recovery review](2026-10-30-construction-portfolio-recovery-results.md), [targeted parameter studies](2026-11-03-construction-portfolio-targeted-results.md) and [candidate 15 result](2026-11-05-construction-meal-reliability-results.md).

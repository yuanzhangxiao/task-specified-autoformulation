# Dalla rescue results and fixed-parameter intervention inspection

The six ACES rescue tasks completed. Refitting substantially improves R4, and
Brief-only R9 is a usable predictive illustration on the existing canonical
intervention probes. These results do **not** yet establish the stronger claim
that fitted mechanism compliance plus pruning recovered a physiologically
correct model that dominates the comparators.

## Verification and scope

Input: the user's uploaded `models.json`, exported from
`dalla-demonstration-rescue-v1`. All six result seals, selected request hashes,
lowered candidate hashes, initialization-plan hashes, parameter inventories and
training/validation content hashes were verified. Every retained model was
replayed on all 16 training and four validation trajectories. The largest
absolute discrepancy from an exported NMSE was below 4e-15.

No fitting or LLM call was made during this inspection. There were 120 original
development rollouts and 63 exploratory intervention rollouts: six rescued
models and the three existing canonical comparators, each on the same seven
previously defined cases. The two reference variants remain separate. Canonical
reference arrays were imported unchanged; perturbed references were regenerated
with the existing `perturbed_b1` simulator, same physical parameters and probe
definitions, and checked with both Radau and DOP853. Each reference's auxiliaries
come from its own reference trajectory. Hard models receive only their original
permitted auxiliary, Gt; easy models receive their original four auxiliaries.

The interventions change **initial tissue glucose Gt**, holding initial plasma
glucose Gp and other physical states fixed, or split a 60 g meal into two 30 g
meals at minutes 60 and 90. These are the existing post-hoc diagnostic cases,
not untouched benchmark test data and not direct insulin interventions. Models
and parameters were frozen before this evaluation. All cases are reported.

Artifacts: `artifacts/dalla-rescue-inspection-2026-09-24/`. The frozen plan contains
full source identities, fitted models, public data and reference settings.
`summary.json` contains every metric; `equation-review.json` contains fitted
equations; `curves.csv` contains all intervention curves;
`canonical_development_curves.csv` includes every canonical training/validation
trajectory for Brief R9 and the three original comparators. Both absolute and
matched-control plots are supplied. The standalone inspection and plotting
scripts are included with the analysis artifacts, not installed in the pipeline.

## Numerical rescue

| Retained endpoint | Original seed train / validation | Final train / validation | Selected pruning arm |
| --- | --- | --- | --- |
| Brief canonical R9 | 0.01913 / 0.01776 | **0.01841 / 0.01833** | Pruned |
| Full perturbed R4, original start | 36.22871 / 37.86737 | **0.08545 / 0.06756** | Unchanged control |
| Full perturbed R4, R13 start | 0.20475 / 0.33832 | **0.05336 / 0.08592** | Pruned |
| Full perturbed R13 | 0.04235 / 0.05933 | **0.04235 / 0.05933** | Pruned |
| Full hard parent R14 | 0.23258 / 0.30355 | **0.21076 / 0.31063** | Pruned |
| Full hard corrected delay | 0.33964 / 0.83520 | **0.16321 / 0.20421** | Pruned |

R4's poor original result was clearly not the best achievable fit of that
structure. This confirms the value of revisiting its numerical estimation; it
does not identify the precise cause of the earlier default-like coefficients.
The corrected hard delay also helped relative to the matched parent. Conversely,
R13 changed negligibly, and R9's slightly better training fit came with a slightly
worse original validation score. Those qualifications should remain visible.

## What pruning actually did

- **R9:** deleted the meal-input term from w, leaving an autonomous decaying
  transient. It retained both latent states, all 12 parameters, and the other
  meal filter. This is not removal of a whole latent state.
- **R4, original start:** the attempted child removed the excretion term and
  disconnected state M, but the unchanged refit performed better. Thus its large
  improvement should be attributed to refitting, not accepted pruning.
- **R4 from R13 and R13:** removed the excretion term and disconnected M. The
  training excretion channel is identically zero in this perturbed cell, so that
  coefficient was not identified by these data. Deleting it is a local predictive
  simplification, not proof excretion is physiologically unnecessary.
- **Both hard models:** deleted meal forcing from I, leaving a very fast
  autonomous transient. The remaining fitted time constants are approximately
  4e-8 and 2e-8 minutes, with initial I values around 3.5e7. They are not credible
  physiological insulin delays merely because the state is named I.

The pruning/control scores differ negligibly for R9, R13 and the hard pair.
This run does not demonstrate a substantial generalization gain caused by pruning.

## Canonical R9 versus the existing comparators

R9's absolute intervention NMSE ranges from **0.00834 to 0.02531** over all seven
cases. Its original training trajectory NMSEs range from 0.00176 to 0.09570;
validation trajectory NMSEs range from 0.00920 to 0.02899. Thus a good pooled score
does not mean every training curve is equally good.

The table below measures intervention-response error separately from absolute
trajectory error. For intervention i and its matched control c, it is
sum[(prediction_i - prediction_c) - (reference_i - reference_c)]^2 divided by
sum[reference_i - reference_c]^2. Lower is better; a zero predicted response has
error 1. These are not the ordinary free-rollout NMSEs.

| Model | Fasting Gt -20% | Fasting Gt +20% | Split meal |
| --- | ---: | ---: | ---: |
| **Brief R9, rescued** | **0.156** | **0.264** | **0.118** |
| Sol, no additional latent states | 0.901 | 1.543 | 0.075 |
| Sol, three latent states | 0.206 | 0.246 | 0.366 |
| No specification | 1.000 | 1.000 | 1.135 |

R9 reduces the exaggerated Gt response of Sol's no-additional-latent model, and
it responds to Gt when no-specification does not. For -20%/+20% Gt, reference
peak absolute changes are 12.40/11.09 mg/kg, versus 16.53/17.07 for R9 and
31.97/36.39 for Sol without additional latent states. Sol's three-latent model
has the closer peak magnitudes, 13.27/11.73, though response timing also matters.

There is no uniform winner. On these two fasting cases, the three-latent Sol
model has **lower absolute NMSE** (0.00173/0.00347) than R9 (0.00834/0.01565).
On split meals R9 has lower absolute NMSE than that Sol model (0.01340 versus
0.02679), while the no-additional-latent Sol model has slightly better response
error than R9. Absolute curves must accompany response-difference curves.

### Why R9 is still not a recovered physiological equation

After mapping the anonymous channels back for inspection, its fitted target
equation is approximately

    Gp' = -0.43559 meal - 1.91155 EGP + 2.94045 Uii + 145.98558 E
          + 0.11644 Gt - 0.086633 Gp + W + H,
    W' = -0.14021 W,
    H' = -0.018639 H + 0.10123 meal.

Here W and H are constant rescalings of the saved latent states, used only to
make the explanation readable; simulations used the original exact vector.
Its positive Gt coupling is much less excessive than Sol's coefficient 0.97352,
which helps explain the improved Gt response. But its negative production term,
positive consumption/excretion terms and negative direct meal term are not the
reference physiological balance. Uii is constant during training, so its fitted
coefficient can act as an offset instead of an identified consumption effect.
Large latent values alone are also not proof of failure: latent state scales
can trade against gains. In R9 the large initial w and small gain have a finite
product of about -3.44.

These were real-valued historical coefficient declarations. Rescue refitting
preserved their domains; it did not add physical sign constraints. The public
graph checks establish a causal input-to-target path, not correctness of these
fitted partial effects. Passing them cannot be described as full scientific
mechanism recovery.

## Full and hard outcomes

The three perturbed Full models fit the split-meal trajectory reasonably well:
absolute NMSEs are 0.03764, **0.00472**, and **0.00744** for R4, R4 from R13, and
R13. However, they fail the perturbed initial-Gt probes: fasting response errors
range from 14.4 to 59.2. The perturbed reference has nonlinear exchange and a
small, sometimes opposite-direction response in these cases; the saved models
retain linear exchange. They also retain negative EGP and positive effective
Uii terms. Low absolute error on a small reference response should not hide this.

The corrected hard model improves split-meal NMSE from 0.19691 to 0.11408 and
response error from 0.865 to 0.370. But its fasting absolute NMSE remains around
0.15-0.21, and its fast I transient remains. This is a useful repair result,
not a strong positive trajectory example.

## Numerical health and next decision

Initializer worker exit -11 was reported on ACES as well as previously locally.
It is therefore not established to be a laptop-only problem. Fallback fitting
still returned the finite, independently reproduced endpoints above. All six
selected public results have `native_optimizer_converged: null`; absence of
budget exhaustion must not be interpreted as certified convergence.

Use **Brief R9** if the figure's claim is limited to a reasonably accurate model
that avoids some failures of the existing comparators across these exploratory
conditions. Do not label it a fully recovered mechanism or attribute its benefit
to pruning. For the stronger desired claim, none of these six yet qualifies.
Another unconstrained refit alone does not address the observed sign/structural
defects. Any next constrained or revised experiment should derive obligations
from the public task, freeze them before scoring, and keep these already inspected
probe results out of fitting and selection. It would be a new development run,
not a retrospective correction to the benchmark scores.

Verification: all six endpoints and 63 intervention rollouts completed; exact
resume preserved every recorded JSON byte. The rescue regression suite passed
13 tests. Whole-repository Ruff still reports the 37 existing issues in unrelated
`analysis/claude/` files. No production implementation or benchmark changed.

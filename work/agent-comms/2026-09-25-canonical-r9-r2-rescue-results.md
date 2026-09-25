# Canonical R9/R2 rescue: repaired equations and frozen intervention replay

All four uploaded endpoints are valid, readable, and independently replayable.
R2's repair is useful: it substantially improves training, validation and all
seven exploratory intervention trajectories relative to its equally budgeted
unchanged control. R9's repair worsens all seven absolute intervention NMSEs.
These results still do not establish a uniformly superior, fully recovered
mechanistic model.

Source campaign plan:
`f7af8b474cb83760b842639465fbdce786ac9c29128850ea7243b17c2b9bf9f6`.
Replay plan:
`bd4f12f04cf502bb8538bab48e723d3a1c8e87f3d296456b36381349dc19328d`.

## Outcome

| Endpoint | Train NMSE | Validation NMSE | Split-meal NMSE | Split-minus-single RSE |
|---|---:|---:|---:|---:|
| R9 repaired | 0.04155 | 0.05846 | 0.04676 | 0.21655 |
| R9 unchanged, refitted control | 0.01841 | 0.01833 | 0.01340 | 0.11846 |
| R2 repaired | 0.04332 | 0.05486 | 0.04643 | 0.22223 |
| R2 unchanged, refitted control | 0.21856 | 0.39694 | 0.17550 | 0.51627 |

R2 has 5.04-fold lower training error and 7.24-fold lower validation error.
The tissue-return gain in its unchanged control is effectively zero, because
the original equation subtracts a nonnegative tissue contribution. The repair
changes that outer operator to addition and the refit activates the pathway.
This is a concrete sign-repair example. It is not the originally desired
comparison of models with equally good training fits that diverge only under
intervention: the R2 control already fits training poorly.

For both repaired models pruning removes the renal-excretion term, whose fitted
contribution was negligible. R9's unchanged control instead loses the input
term in its fast latent state. Pruning acceptance shows near-equivalent
validation fit under a smaller structure; the large R2 improvement comes before
that negligible-term deletion. It should not be attributed to pruning alone.

## What the fitted equations actually do

The canonical physical plasma balance is

    Gp' = meal_appearance + EGP - Uii - E - 0.065 Gp + 0.079 Gt.

The benchmark supplies Gt, EGP, Uii and E under the T1-easy availability
contract. All replay comparisons remain conditional on these auxiliary
trajectories, which change consistently with the physical intervention.

For repaired R2 define B = 42.428596 A. Its equations reduce, to displayed
precision, to

    B'  = 0.0884091 u - 0.00493127 B,     B(0) = 2.315632
    Gp' = B + 0.136430 Gt - 0.0921557 Gp - 3.062918 Uii
          + 2.26e-27 EGP.

Here u is the supplied meal-event channel. The retained outer operators have
the intended directions. However, the fitted production contribution is
effectively zero, tissue exchange is stronger than the canonical reference,
utilization is multiplied by approximately three, and the meal contribution
decays on a roughly 203-minute time scale. Correct signs do not establish that
the complete physical mechanism or coefficients have been recovered.

For repaired R9 define B = 0.0932428 x and H = 1.17070e-9 w. The anonymous
channels are translated here for scientific interpretation only:

    B'  = 0.0932428 u - 0.00499376 B,     B(0) = 2.850950
    H'  = 1.17070e-9 u - 0.281886 H,     H(0) = -5.942094
    Gp' = B + H + 0.137723 Gt - 0.0952318 Gp - 2.915708 Uii
          + 1e-10 EGP + 1e-10 u.

Thus R9 and R2 have converged to rather similar conditional glucose dynamics:
a slow meal contribution, active tissue return, dissipation and utilization,
but effectively no production response. R9 adds a brief negative initial
transient. The enormous saved w(0) = -5.08e9 is multiplied by a tiny gain;
their product is -5.94. It is an awkward latent scaling/initialization, not
by itself proof of a huge observable state or an unstable trajectory. The
rescaled dynamics above expose the actual contribution.

All enforced retained sign domains and fitted bounds were reconstructed from
the original source packet and independently checked. Both repaired endpoints
pass. Neither model's fitted parameters reversed a fixed topology sign.
This audit certifies the specified outer terms, not active mechanistic
contributions, latent-state positivity, or global intervention monotonicity.

## Comparison with the matching Sol models

For canonical **named** T1-easy, repaired R2 beats Sol repetitions 0 and 2 on
the four meal-containing probes, but Sol repetition 1 has lower absolute NMSE
on all seven probes. In particular, split-meal NMSE is 0.04643 for R2 versus
0.01324 for Sol 1. R2's split-response RSE is better (0.22223 versus 0.4219),
which is a narrower observation. All three named Sol models respond extremely
accurately to the initial-Gt perturbations.

For canonical **obfuscated** T1-easy, repaired R9 has worse split-meal absolute
NMSE than all three Sol repetitions. For the initial-Gt response difference,
R9 improves on Sol 0/1 in all four fasting/meal and ±20% cases. Example fasting
response RSEs:

| Model | Initial Gt −20% | Initial Gt +20% |
|---|---:|---:|
| R9 repaired | 0.1261 | 0.1774 |
| Sol 0, no additional latent state | 0.9013 | 1.5427 |
| Sol 1, three latent states | 0.2058 | 0.2455 |
| Sol 2 | approximately 0 | approximately 0 |
| No specification | 1 | 1 |

Sol 2 must be retained: its fitted canonical tissue/plasma coefficients are
close to the reference and it responds very accurately to these perturbations.
It has higher pooled training NMSE (0.1623), but its initial-Gt behavior prevents
a blanket claim that the external baselines fail.

The strongest omission argument concerns no-specification. Its target and
latent equations have no dependency on the supplied tissue-glucose channel
or the other physiological auxiliary channels. With identical meal input,
initial observed Gp and fitted latent initial values, its prediction cannot
change when physical initial Gt changes. It predicts exactly zero intervention
effect. R9 responds in the correct direction and captures much of its time
course. Their pooled training errors are similar (0.04155 versus 0.04358), but
they are selected historical endpoints with different seeds, and R9 is assisted.
This supports an illustrative, conditional response argument, not a controlled
estimate of the autonomous specification ablation's average effect.

The absolute curves are essential alongside that argument. R9 still has
baseline drift and overshoot; differencing removes shared forecast bias.
For fasting Gt−20%, R9 absolute NMSE is 0.00932 versus no-specification 0.02511;
for Gt+20% the ranking reverses (0.01944 versus 0.00788). Absolute accuracy and
intervention sensitivity answer different questions and should both be shown.

## Recommendation

Use R2 as an explicitly assisted demonstration that a wrong topology sign can
disable a useful pathway and that sign-constrained refitting can restore it.
Use the R9/no-specification equations to illustrate why an omitted dependency
makes an intervention response impossible, accompanied by both absolute and
difference plots and the competitive Sol outcomes. Neither case establishes
complete mechanism recovery or uniform superiority on held-out trajectories.
The current results do not warrant another broad parameter-only retry: the
paired allocations already produced finite, reproducible endpoints, while
the fitted production/meal mechanisms remain approximate. Any further revision
would be a new exploratory development experiment, not a correction to the
frozen benchmark results.

The time-zero meal input-representation issue reported in the preceding R4
note remains relevant to pooled training scores. No input semantics or data
were modified here. All seven exploratory cases have either no meals or meals
after time zero.

## Artifacts and verification

Outputs: `artifacts/dalla-canonical-rescue-inspection-2026-09-25/`.
Plotting package: `transfers/dalla-canonical-rescue-curves-20260925.tar.gz`.
The package includes exact frozen models, complete metrics, and all 27 curves
per model for four rescue endpoints, 24 matching-cell external baselines and
the no-specification comparator (29 models total). Test metrics are excluded.
Named and obfuscated model cohorts are labelled separately; physical references
are canonical, not the perturbed R4 system.

- 108 new rollouts completed: 80 original development and 28 exploratory.
- All eight training/validation scores reproduced within 1.95e-15.
- All 108 NMSEs and 20 intervention-response RSEs independently recomputed.
- Source/result/request/parameter/data identities and both sign audits checked.
- Exact replay resume returned the identical sealed summary without new solves.
- Relevant regression tests: 23 passed. The 108 actual rollouts provide the
  relevant numerical smoke check; no local refitting was performed.
- `ruff check .` reports the same 37 existing findings under `analysis/claude`.
  They are unrelated and were left unchanged. No runtime implementation changed.

The source report records several collocation initialization failures and no
confirmed native optimizer convergence. Its `complete` statuses mean retained
finite endpoints, not verified optimal fits. Independent replay now verifies
their reported predictive scores; it does not establish optimizer optimality.
No fitting, live LLM calls, remote sessions, or new test-data access occurred.

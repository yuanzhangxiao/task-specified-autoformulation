# Assisted-model shortlist and equation-directed intervention design

Allowing assisted endpoints changes the preferred illustration. The strongest
currently supported missing-dependency comparison is **canonical-obfuscated T1:
assisted Brief-only R9 versus the actual no-specification seed-0 endpoint**.
Requiring an autonomous Full endpoint was unnecessarily restrictive for the
user's revised goal. It remains necessary to label the assistance and to avoid
claiming that an illustrative pair estimates the average effect of an ablation.

## What “strongest perturbed-obfuscated” meant

The earlier equation audit explicitly referred to Full versus Sol in
perturbed-obfuscated T1. Both Full seeds subsequently beat Sol repetitions 0/1
on all eight physical meal probes; Sol repetition 2 beat both Full seeds on all
eight. That is a robustness example with identifiable failures in two Sol
endpoints. It is not the leading Full-versus-ablation example. The local
matched no-specification/no-latent exports are canonical; they cannot be
relabeled as perturbed-cell ablations.

## Shortlist

| Candidate | Useful comparison | Present conclusion |
|---|---|---|
| Canonical-obfuscated assisted Brief-only R9 | Actual no-specification seed 0 | Best existing similar-training-fit dependency illustration |
| Canonical-named assisted Full R2 | Actual named no-latent seed 0 | Clear delayed-response contrast, but no-latent already fits training poorly |
| Perturbed-named R4/R13 rescues | Meal-spacing probes | Several good trajectories, but no matching actual ablations locally; sign and independent-Gt problems remain |
| Perturbed-obfuscated Full R12, both seeds | All three matching Sol repetitions | Stable positive meal pathways versus two defective Sol endpoints; strongest Sol remains better |

R9 and R2 repairs have similar active conditional plasma dynamics, including
positive tissue return. Their endogenous-production contributions are nearly
zero. They are useful approximate models, not recovered complete physiology.
R4's repaired observable equation has extreme fast relaxation and a very slow
meal state; correct outer signs alone do not make it the best demonstration.

## R9 versus no-specification: existing verified outcomes

Both are canonical-obfuscated T1 endpoints. R9 is seed 1 with assisted sign
repair, refitting and pruning; no-specification is seed 0. Parameters are frozen
throughout the reported rollouts.

| Quantity | Assisted R9 | No specification |
|---|---:|---:|
| Pooled training NMSE | 0.041554 | 0.043576 |
| Original validation NMSE | 0.058458 | 0.124052 |
| 60 g meal at minute 60, initial Gt -20%: absolute NMSE | 0.056585 | 0.121975 |
| Same meal, initial Gt +20%: absolute NMSE | 0.069325 | 0.074288 |
| Response RSE for -20% Gt, relative to the same meal at basal Gt | 0.121165 | 1.000000 |
| Response RSE for +20% Gt | 0.182250 | 1.000000 |

Unlike the previously emphasized fasting comparison, R9 has lower absolute
error on both meal-containing Gt perturbations. The gains are 2.16-fold and
1.07-fold, respectively: this is not uniformly dramatic separation. The
fasting +20% case reverses the absolute ranking and must remain in the report.

The no-specification target and all three latent equations depend on meal,
time and internal states, but no physiological auxiliary. Its fitted initial
states are constants and observed initial Gp. Therefore, with the same meal
schedule and initial Gp, changing physical initial Gt cannot change its
prediction. This is an exact dependency argument. R9 contains the active term

    Gp' = ... + 0.1377227572 Gt - 0.09523184357 Gp.

The canonical reference has positive tissue return, +0.079 Gt, so the fitted
term gives a justified direction of response, although its gain and the other
balance terms remain approximate. In these physical probes, the reference
regenerates all supplied auxiliary trajectories jointly. This is T1's
conditional prediction task, not an autonomous prediction of every auxiliary.

The initial-Gt intervention changes total initial glucose mass while holding
Gp and the other physical initial states fixed. It is a synthetic initial-state
probe, not the registered mass-conserving redistribution or a meal intervention.
T1's public task does not explicitly require Gt, so this illustrates a missing
dependency in a no-specification endpoint, not proof that it violates the
literal T1 requirement or that the specification alone caused the difference.

## Additional no-specification defects and their proposed probes

Expanding its saved fitted expression gives these direct target terms:

    Gp' = ... + 0.0263365690 t - 0.3276330705 u
                 + 0.0014872820 u^2 + 0.9568846115 z,
    z'  = 0.07627807075 u - 0.004799077484 z.

It has meal memory; it is not a memoryless model. However, its direct meal
contribution is negative for 0 < u < approximately 220.29 in the public pulse
units. The delayed positive z contribution can conceal an initial wrong-way
response. The explicit clock-time term also permits drift unrelated to a meal.

Read-only recomputation from the already saved 60 g / minute-60 and matched
fasting curves confirms an initial no-specification response minimum of
-11.6493 mg/kg at minute 61. The physical reference meal response there is
+0.01888 mg/kg; R9 is +5.28046 mg/kg. R9 avoids the dip but rises too quickly.
Whole-horizon meal-response RSEs are 0.045845 for R9 and 0.065025 for
no-specification. This supports a specific direction/shape observation, not
catastrophic failure. No new simulation was performed to obtain these values.

A bounded next diagnostic should freeze the following **before** new rollouts:

1. Initial-state sensitivity: Gt multipliers 0.6, 0.8, 1, 1.2, 1.4; identical
   initial Gp and all other physical states; both fasting and a 60 g meal at
   minute 60; horizon 300 minutes. This extends the existing symmetric +/-20%
   probe and tests the omitted argument directly. The range is a synthetic
   stress test, not a clinically validated intervention range.
2. Delayed meal after fasting: the same 60 g meal at minutes 60, 240 and 480,
   all observed through minute 780, plus a matched fasting control. This tests
   basal drift and elapsed-time dependence instead of selecting a pulse after
   seeing its result. Keep whole-horizon absolute errors; additionally compare
   the common 0–300 minute post-meal response window for all three schedules.
3. Report all cases for repaired R9 and no-specification, with the frozen R9
   unchanged/refitted control as a secondary comparator. Show both absolute
   trajectories and each model's matched-control response. Reuse the original
   meal input representation; regenerate physiological auxiliaries jointly.

These are predictions about discriminating conditions, not a claim that R9
will win the unrun cases. Preserve all endpoints and both intervention
directions. Any new results remain exploratory equation-directed stress tests,
separate from sealed benchmark test estimates.

## The no-latent arm needs a separate claim

The actual named no-latent equation is

    Gp' = 1.12821256 - 0.00669917911 Gp
          + (1.47474750 + 15.6042152 exp(-17.2340450 t)) u.

After a pulse ends, Gp must decay whenever it exceeds 168.410568. There is no
meal-derived state that can continue feeding the target. Assisted R2 has

    B' = 0.0884091 u - 0.00493127 B,
    Gp' = B + 0.136430 Gt - 0.0921557 Gp - 3.062918 Uii
          + 2.26e-27 EGP.

An isolated short pulse or two pulses with 15/30/60/120-minute spacing directly
tests the continuing post-meal drive. But no-latent's pooled training NMSE is
0.45712, versus R2's 0.04332. Changing the intervention cannot fix that premise.
Use this pair as an expressivity illustration, or first obtain a better
train-only fitted no-latent endpoint. Do not claim all three existing ablation
endpoints fit training equally well. The observed Gp is itself a state; this
argument concerns this fitted equation, not all models without extra latents.

## Sources and scope

- `artifacts/dalla-canonical-rescue-inspection-2026-09-25/comparison-metrics.json`
- `artifacts/t1-curves-round12-2026-09-22/models.json`
- `artifacts/t1-curves-round12-2026-09-22/summary.json`
- `artifacts/t1-intervention-probe-2026-09-22/replays/cell01_seed0_no_spec/`
- `artifacts/dalla-canonical-rescue-inspection-2026-09-25/replays/brief_canonical_r9_repaired/probes/`
- The prior rescue, sign audit and perturbed-meal comparison notes in this folder.

This review reads and recomputes statistics from existing local artifacts. No
new fit, rollout, model edit, LLM call, remote session or test-data access.
Only this communication note is added; no implementation changed.

Verification: the two meal/Gt rankings and unit no-specification response errors
were checked against the saved summary; the initial meal dip was recomputed
from saved trajectories. Six existing meal-probe tests pass. `ruff check .`
still reports the same 37 unrelated findings in `analysis/claude`; those files
were left unchanged. The full test suite was not rerun for this note.

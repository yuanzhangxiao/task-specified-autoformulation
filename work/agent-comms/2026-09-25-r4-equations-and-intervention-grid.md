# R4 equations, additional interventions, and R9/R2 follow-up

The user is correct: on the existing split-meal trajectory, Sol repetitions 0
and 1 have lower absolute NMSE than assisted R4. The respective NMSEs are
0.01203, 0.05228, and 0.05871. R4 wins the different question of predicting
the split-minus-single change: response RSE 0.22162 versus 0.92076 and 1.26068.
Neither metric substitutes for the other.

Among the previous seven perturbation cases, R4 never beats Sol 0 on absolute
NMSE. It beats Sol 1 only in fasting with initial tissue glucose increased 20%.
R4's response sensitivity to independently changed initial tissue glucose is
poor; that is not a suitable mechanism-recovery demonstration.

## Additional fixed grid

After inspecting equations, but before generating new outcomes, freeze total
meal mass at 30, 60, or 120 g and spacing at 0, 15, 30, or 60 minutes. The first
meal is at minute 60; spacing zero is a single meal, otherwise split the total
mass equally. All cases use basal initialization, the perturbed physical system,
a 300-minute horizon, and a one-minute sampling grid. The exact schedules are
absent from the released training and validation sets. Two of the 12 schedules
repeat earlier exploratory probes; the remaining schedules are new.

Replay five frozen endpoints on all cases: repaired R4, its refitted unchanged
control, and all three Sol repetitions. No parameters, model selection rules,
or benchmark data are changed. Regenerate the reference and all permitted
auxiliary trajectories jointly. This is a reference-informed exploratory
diagnostic, not an untouched benchmark test or an autonomous discovery result.

| Total g | Gap min | R4 repaired | R4 control | Sol 0 | Sol 1 |
|---:|---:|---:|---:|---:|---:|
| 30 | 0 | 0.10390 | 0.06372 | 0.04744 | 0.13613 |
| 30 | 15 | 0.09953 | 0.05796 | 0.03673 | 0.12968 |
| 30 | 30 | 0.09181 | 0.05258 | 0.03323 | 0.12385 |
| 30 | 60 | 0.07985 | 0.04605 | 0.03614 | 0.12261 |
| 60 | 0 | 0.07464 | 0.08674 | 0.01882 | 0.06366 |
| 60 | 15 | 0.06595 | 0.06719 | 0.00868 | 0.05358 |
| 60 | 30 | 0.05871 | 0.06042 | 0.01203 | 0.05228 |
| 60 | 60 | 0.05978 | 0.06260 | 0.02011 | 0.06680 |
| 120 | 0 | 0.14227 | 0.08338 | 0.08555 | 0.04327 |
| 120 | 15 | 0.19245 | 0.06982 | 0.22348 | 0.13042 |
| 120 | 30 | 0.20651 | 0.09306 | 0.29503 | 0.16657 |
| 120 | 60 | 0.18756 | 0.13009 | 0.25602 | 0.16170 |

These are absolute NMSEs, normalized using the original training scale
34.47316059348893. Sol 2 is included in the machine-readable results; its
absolute NMSE ranges from 9.38 to 162.03 and it does not supply a competitive
matched-training comparison.

R4 beats Sol 0 on 3/12 absolute trajectories and Sol 1 on 5/12, but beats both
on 0/12. Its unchanged refitted control also outperforms repaired R4 for every
30 g and 120 g absolute trajectory. Thus the grid does not establish a general
benefit of sign repair for prediction.

R4 has lower split-minus-single response RSE than both Sol 0 and Sol 1 in all
nine split cases. That is a consistent, narrower result. It is not uniformly
accurate in an absolute sense: R4 response RSE is 1.425 for 30 g/15 minutes,
worse than predicting no intervention effect. All response outcomes are in
the artifact README and metrics CSV.

## Fitted equation interpretation

Let u be the supplied meal-event channel, Gp plasma glucose, and Gt supplied
tissue glucose. R4's retained equations, rounded for readability, are

    X'  = 0.000845599 u - 0.000691988 X
    Y'  = 30.288665 X - 0.158843 Y
    Gp' = 5603.109 Y + 3759.737 Gt - 8402.397 Gp
          + 1.99e-15 EGP - 4.29e-10 Uii

The excretion term was pruned. The positive tissue contribution arises because
the assembly is `-k_Gt * Gt` and the fitted signed coefficient is negative;
the resulting contribution is positive. Initial X and Y are shared fitted
values 0.942878 and 164.620720, and initial Gp is observed.

Gp relaxes in approximately 0.000119 minutes (0.0071 seconds), so away from
the initial boundary layer it is approximately

    Gp ≈ 0.666847 Y + 0.447460 Gt.

Thus the model largely follows supplied tissue glucose plus a slowly changing
latent component. X has a roughly 1445-minute decay time and Y 6.30 minutes.
Latent scaling is arbitrary, but these time constants and the nearly absent
EGP/Uii contributions are substantive. A good response-difference curve does
not establish physiologically correct meal absorption or full mechanism recovery.

For Sol 0, with zero initial meal states M and A,

    M'  = u - 0.656975 M
    A'  = 0.656975 M - 0.0136628 A
    Gp' = EGP + 2.609817 × 0.0136628 A
          + 0.00207703 Gt - 0.00703260 Gp
          - (Uii + E) Gp / (Gp + 1e-6).

For Sol 1, with zero initial meal states M1 and M2,

    M1' = u - 0.0463068 M1
    M2' = 0.0463068 (M1 - M2)
    Gp' = EGP + 1.830969 × 0.0463068 M2 - Uii - E
          - 0.00899372 Gp + 0.00482145 Gt.

Both perturbed-cell Sol models have TWO latent states; they are not the earlier
canonical-cell no-latent/three-latent pair. Conditional on their forcing
channels, their plasma-glucose equations have approximate relaxation times of
142 and 111 minutes. This provides an equation-based reason for testing meal
spacing: R4 can follow changes in supplied tissue glucose more promptly, while
Sol has more lag. It predicts a possible difference in response timing, not
necessarily better absolute accuracy. The grid supports precisely that limited
interpretation.

The perturbed reference additionally uses nonlinear exchange laws:

    Jp→t = 0.065 Gp [1 + 0.35 tanh((Gp - Gpb)/(0.2 Gpb))]
    Jt→p = 0.079 Gt [1 - 0.25 tanh((Gt - Gtb)/(0.2 Gtb))].

All three candidates above replace this with constant exchange gains. Positive
fluxes do not imply positive local derivatives: the reference return-flux
derivative at basal Gt is -0.01975. This helps explain why extrapolating the
positive fitted tissue coupling to independently changed initial Gt is not a
reliable strategy. It is the derivative of this flux, not an eigenvalue of the
whole physical system.

## Input-representation issue in interpreting training errors

The physical generator applies meal mass as an exact stomach-state jump,
including at time zero (`dalla_man.py`, `simulate_dalla_man`). The common
continuous replay supplies `meal_event_g` through `PiecewiseLinearForcing`
(`fitting/simulation.py`, `trajectory_forcing`). On the released one-minute
grid, an isolated 90 g sample at time zero has area 45 under this interpolator;
the same 90 g sample at time 60 or 120 has area 90. The physical generator
applies the full 90 g in each case. This is an input/initialization semantics
confound, not evidence of a memory advantage.

The actual saved curves show the association clearly:

| Training schedule | R4 NMSE | Sol 0 NMSE | Sol 1 NMSE |
|---|---:|---:|---:|
| 90 g at 0 min | 0.0372 | 1.7442 | 0.7915 |
| 90 g at 60 min | 0.0319 | 0.0169 | 0.0090 |
| 90 g at 120 min | 0.0273 | 0.0127 | 0.0066 |

Sol starts its meal states at zero, whereas R4 has fitted nonzero latent starts.
This does not quantify how much of the score gap is caused by the representation
issue: no counterfactual input fix or refit was performed. It does mean that
the pooled training-score gap should not be attributed solely to mechanism
quality. All meals in the new grid occur after simulation start. Even then,
generic continuous pulses approximate the physical jump; they are not identical
input semantics. No benchmark or historical replay was changed in this work.

## R9 and R2

Only their saved starts and the new paired-rescue preparation packet are local;
new repaired/control fitted endpoints have not been supplied yet. The old R9
rescue previously achieved canonical-obfuscated training NMSE 0.01841 and split
NMSE 0.01340. Its split response RSE 0.11846 is nevertheless worse than that
cell's Sol no-latent repetition 0 (0.07488). This old rescued endpoint must not
be confused with the original R9 start used in the new sign-constrained run.

The original R9 start, with anonymous channels mapped to their physical roles,
has two parallel meal filters:

    w'  = u - 0.397687 w
    x'  = u - 0.0175829 x
    Gp' = -0.487219 u - 1.858307 EGP + 2.756999 Uii
          + 155.768590 E + 0.118856 Gt - 0.0876301 Gp
          + 0.0226426 w + 0.100126 x.

Initial w=-161.803 and x=17.083. The filter times are 2.51 and 56.87 minutes,
and plasma relaxation 11.41 minutes: potentially useful response scales, but
the direct meal/production/utilization/excretion signs are inappropriate for
their physical roles. Its original training accuracy is not proof of the
correct mechanism. The paired rescue constrains these six direct target signs,
without inserting ground-truth coefficients or changing the inner filter laws.

The original R2 start has one meal filter and the expected source/sink slots,
but incorrectly subtracts tissue return. Its fitted return gain is only
8.60e-22. Numerically the fitted equations reduce approximately to

    A'  = 0.00212458 u - 0.0361235 A
    Gp' = 44.54984 A + 0.338447 EGP - 0.00483042 Gp,

with utilization and excretion gains also effectively zero. Initial A=0.03717.
Correcting the tissue-return assembly sign and refitting is a reasonable next
experiment: it can make a previously unusable exchange path available. The
unchanged arm is essential to distinguish the sign change from extra fitting.
One absorption filter remains an approximation to the reference gut system;
we cannot promise the repaired endpoint will match intervention trajectories.

Retrieve top-level `models.json` and `summary.json` from the actual campaign
output, expected on ACES at
`/scratch/group/p.nairr260351.000/u.yx126462/dalla-canonical-rescue-v1`.
Then evaluate all four repaired/control endpoints on the existing seven-case
canonical suite, before defining any additional probes. Compare against
baselines from the corresponding canonical cell, not against perturbed R4's
baselines. No remote sessions were used in this inspection.

## Artifacts and verification

Outputs: `artifacts/r4-equation-probes-2026-09-25/`.
The sealed plan, exact equations/parameters, complete metrics, plotting curves,
and PNG/SVG figure are available there. All 60 rollouts completed. All 60
absolute NMSEs and 45 response RSEs were independently recomputed from arrays;
two reference solvers agree within 1.2e-8 on every generated channel.

The replay script's exact-source resume correctly refused a subsequent retry
after an unrelated untracked `rebuttal/initialization_recovery.py` appeared in
the shared source tree. Excluding that newly added file reproduces the frozen
source hash exactly; no existing runtime file changed. The saved plan and
outputs were not overwritten, and no solves were repeated. This is not claimed
as a successful whole-script exact-resume test. Existing cache records and
sealed hashes remain readable and verified.

Only this analysis note is committed. No implementation or dataset changes,
fitting, LLM calls, or new test-data access occurred. Runtime tests were not
rerun for an artifact-only analysis; numerical verification is recorded in
`verification.json` and `time-zero-input-audit.json`.

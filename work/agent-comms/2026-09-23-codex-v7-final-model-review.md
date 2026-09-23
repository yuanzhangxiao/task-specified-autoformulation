# Completed v7 Dalla Man inspection, rounds 14-17

## Conclusion

The completed run still supplies no fitted model that combines a good training
fit with the required intervention behavior. Rounds 16-17 make three additional
selections, one of which is a numerical tie in practical terms. All T2 models
still lack a meal/glucose-driven insulin source. The strongest further disposal
improvement remains a fitted initial transient, not a recovered meal response.

These are six Full lineages on T1-hard, T2-easy, and T2-hard. This run contains
neither new Brief-only models nor new T1-easy models.

## Scope, completeness, and numerical verification

Archive: `dalla-v7-final-20260923-123653.tar.gz`.
Plan: `ae50b5e87ded15a1e0752e8ebeaeb5ca0e9e53eb4e7435b49eda4aa8b38d0f98`.
ACES root:
`/scratch/group/p.nairr260351.000/u.yx126462/dalla-response-feedback-r14-v1`.

All 24 planned result records are present: six imported round-14 checkpoints
and 18 complete round-15/16/17 visits. The manifest records all rounds submitted.
The archive contains 64 JSON files, including 44 verified content seals.

All 24 distinct request/parameter records reproduce their lowered candidate
hash and parameter-name set. This includes every retained and unretained fitted
trial, without an NMSE cutoff. Twelve records are new relative to the round-15
audit; repeated or renamed equations are not independent discoveries.

All twelve round-16/17 trials were replayed at their frozen fitted parameters
on all 16 training and four validation trajectories: 240 complete rollouts.
All per-target aggregate scores agree with saved scores within 2.0e-14.
Checkpoint resume performs zero new integrations. Previously verified round-14
and round-15 replay files supply unchanged final incumbents and comparisons.
This uses the production interpreter and solver, not an independent solver.

No new fitting, LLM calls, test-data access, benchmark edits, or remote session
was performed. Observed public training/validation trajectories suffice to
establish the missing-response problems described below.

## Final retained scores

T2 aggregate NMSE averages its target scores. It must not be read as an insulin
score or as evidence that every target improved.

| Task | R14 training / validation | Final training / validation | Change after R15 |
| --- | ---: | ---: | --- |
| T1-hard seed 0 | 0.232584 / 0.303552 | 0.232584 / 0.303552 | None |
| T1-hard seed 1 | 0.400073 / 0.490862 | 0.353788 / 0.398207 | None; retains R15 |
| T2-easy seed 0 | 0.898731 / 0.870820 | 0.898731 / 0.870820 | None |
| T2-easy seed 1 | 1.004513 / 0.969037 | 0.933320 / 0.911283 | R17 retained |
| T2-hard seed 0 | 0.921255 / 0.977849 | 0.921165 / 0.977849 | R16 retained; negligible validation difference |
| T2-hard seed 1 | 0.949987 / 0.973726 | 0.923174 / 0.969320 | R16 retained |

There are five accepted fitted-trial selections over the three visits. The two
T1/T2-easy seed-1 improvements at R15 were covered in the preceding report.

All six final endpoints have passing graph-requirement certificates, but those
certificates do not establish correct fitted signs or appreciable causal effects.
All reported native optimizer convergence values remain null. Budget status
varies across trials, so these fits do not establish their best possible scores.

## The twelve additional trials

| Task | Round | Trial training / validation | Retained? |
| --- | ---: | ---: | --- |
| T1-hard seed 0 | 16 | 0.235505 / 0.349404 | No |
| T1-hard seed 0 | 17 | 0.235505 / 0.349404 | No |
| T1-hard seed 1 | 16 | 18.829052 / 19.834311 | No |
| T1-hard seed 1 | 17 | 0.415428 / 0.680831 | No |
| T2-easy seed 0 | 16 | 0.894250 / 0.871940 | No |
| T2-easy seed 0 | 17 | 0.894191 / 0.872448 | No |
| T2-easy seed 1 | 16 | 1.932210 / 1.955107 | No |
| T2-easy seed 1 | 17 | 0.933320 / 0.911283 | Yes |
| T2-hard seed 0 | 16 | 0.921165 / 0.977849 | Yes |
| T2-hard seed 0 | 17 | 0.921047 / 0.989020 | No |
| T2-hard seed 1 | 16 | 0.923174 / 0.969320 | Yes |
| T2-hard seed 1 | 17 | 1.444328 / 1.542637 | No |

No promising fitted model is hidden among these rejected trials.

## Mechanism findings

### Insulin source remains missing

All 16 T2 request/parameter records in this archive (four imported parents and
twelve trials) have an observed insulin equation with source symbols restricted
to `I` and `insulin_pmol_per_kg_min`. No T2 structural proposal in any of the three
visits revises the I equation. They add downstream action states or meal terms
for glucose, leaving the endogenous insulin response unaddressed.

At fixed initial insulin and zero external infusion, changing the meal schedule
therefore cannot change predicted I. In T2-hard, both final models remain nearly
constant in I; their validation I NMSE is about 1.527. In T2-easy seed 1, I grows
exponentially instead of producing the observed meal-associated peak, and final
validation I NMSE is 1.246551. Parameter-only optimization cannot add the absent
driver to these equations.

### T2-easy seed 1 improves a transient, not meal-responsive disposal

The R17 retained model adds Z to the R15 X/Y disposal representation:

```text
dI/dt = 3.87816 * insulin_pmol_per_kg_min + I / 232.383
dX/dt = 1.19e-221 * I - X / 83.5931
dY/dt = 3.87816 * insulin_pmol_per_kg_min - Y / 69.5417
dZ/dt = 5.72e-32 * insulin_pmol_per_kg_min - Z / 32.5577
U = Uii + 622.243 * X + 0.173818 * Y + 2.08362 * Z
X(0) = 0.145944 + 1.66891e-6 * I(0)
Y(0) = -600.293
Z(0) = 6.90345
```

The new Z forcing coefficient and I-to-X coupling are numerically negligible.
For the illustrated zero-infusion schedules with I(0) approximately 25.5878,

```text
U(t) - Uii(t) ~= 90.84 * exp(-t / 83.59)
                 -104.34 * exp(-t / 69.54)
                 +14.38 * exp(-t / 32.56).
```

The hump is formed from fitted initial values. It does not shift with the meal.
On training_002 (60 g meal at minute 0) and validation_000 (75 g meal at minute
30), the predicted U curves differ by at most 4.4e-7, peak at minute 114 in both,
and attain 4.407 in both. The reference peaks are at minutes 111 and 137, with
heights 4.908 and 5.860. These schedules change both timing and amount, so this
comparison does not isolate one of those factors.

Final U NMSE is 0.795001 training / 0.820790 validation, compared with R15
0.808263 / 0.863591. Glucose remains poor, at 0.678459 / 0.666509. A visually
better U curve on one training trajectory is not recovery of the complete model.
This is a useful failure illustration, not evidence of our method's superiority.

### T1-hard adds or repeats components without obtaining a good fitted model

Seed 0 repeats the R15 meal-memory revision at R16, then renames the new state
from M to X at R17. All three trial score pairs are exactly identical.

There is a concrete declaration problem: each patch calls `par_004` a new time
constant, but that name already belongs to an existing nonnegative coefficient.
The saved declaration audit explicitly says the requested time-constant role
was ignored and the existing role inherited. No new equation parameter is
created. The fitted denominator stays 2.06e-19, invoking the runtime's 1e-12
division floor. Thus the intended delay is effectively instantaneous.

Seed 1 R16 adds a positive Gt contribution, which is a potentially reasonable
component in isolation. But its fitted Gp damping denominator collapses to
2.22e-16, and the actual model scores are about 19 rather than an accurate fit.
It is not a usable saved model. R17 adds a state called Gt_dyn whose fitted
equation and initializer exactly duplicate the existing M state:
`dM/dt = dGt_dyn/dt = 0.1 * meal - state/30`, both initially zero. It supplies
another positive source, despite the prose describing a delayed sink. The
trial worsens validation to 0.681. The retained model remains R15, with its
abrupt meal response and validation NMSE 0.398.

### T2-hard: tiny selection and a small glucose improvement

Seed 0 R16 adds a driven state D with `dD/dt = (infusion-D)/30`, D(0)=0, and
adds `0.1*D` to J. All supplied validation schedules have zero infusion, so this
new contribution is exactly zero there. The aggregate validation improvement is
only 3.3997e-11 (0.9778491472623487 to 0.9778491472283518). The selection code
compares the numeric validation score first and complexity only on an exact tie.
This selection should not be counted as meaningful validation progress.

Seed 1 R16 inserts an additional I-driven filter before X. Validation Gp NMSE
improves from 0.420284 to 0.411472, but I remains at 1.527168. It is a modest
glucose improvement without insulin recovery. Both R17 T2-hard trials worsen
validation and are not retained.

## Controller assessment and recommended next changes

The completed v7 run makes 21 recorded proposer requests and consumes 149,940
observed tokens. Seventeen visits commit a structural patch; one visit exhausts
three attempts and falls back to an incumbent refit. There are no delivery
failure visits, no missing results, and a maximum reported prompt input length
of 8,567 tokens. Compact feedback solved the observed delivery problem across
this pilot, but the scientific revision loop is still repeating the wrong type
of correction.

Before another unchanged continuation, prioritize:

1. Target-specific diagnosis: show that observed I has meal-associated peaks
   with zero infusion, while the current I equation cannot respond to meals.
   Ask for a supported correction to that response pathway, using public
   training evidence rather than prescribing private reference equations.
2. Parameter declaration clarity: preserve exact inheritance for existing
   parameters, but surface an occupied name declared as "new" with a different
   role. Require a fresh name if a new timescale is intended. Silently ignoring
   the declaration preserves the old parameter but defeats the proposed change.
3. Revision history and meaningful selection: recognize repeated or renamed
   failed patches, show their prior fit outcomes, and define a predeclared
   tolerance for practically tied validation scores with a preference for the
   simpler model. Do not tune the tolerance to make these particular results win.

No controller or fitting implementation was changed during this inspection.
No claim is made about statistical significance, global optimizer convergence,
or equivalence under every possible intervention.

## Artifacts and checks

Local directory: `artifacts/dalla-v7-final-review-2026-09-23/`.

- `verified-models.json`, `fitted-equations.md`: all 24 records and provenance.
- `replay-summary.json`: independent execution at the twelve saved trial vectors.
- `new-trial-curves.csv`: all 144,480 new trial target samples.
- `final-curves.csv`: all 72,240 target samples for the six final incumbents.
- `t2_easy_final_comparison.png` / `.svg`: R14/R15/R17 retained T2-easy seed-1 curves.
- `response-comparison.json`: the measured schedule-insensitivity facts above.

The relevant collector, multi-target profile, and replay tests pass: 28 passed.
Repository-wide Ruff reports the same 37 existing issues in unrelated,
untracked `analysis/claude/` scripts; those files were not changed. The only
tracked change is this report. Model archives, replay records, and plots remain
local generated artifacts and are not committed.

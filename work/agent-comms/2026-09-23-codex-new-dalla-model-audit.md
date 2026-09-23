# Inspection of the new Dalla Man model archive

## Conclusion

This archive supplies no new model combining a good training fit with a credible
mechanistic basis for the desired held-out intervention figure. The previous
T1-easy leads are unchanged. The new T1-hard candidates have moderate-to-poor
fits; every new T2 fitted record lacks an endogenous insulin-response pathway.
This conclusion concerns the downloaded fitted records, not the capability of
the method or the results of the pending response-feedback continuation.

## Scope and integrity

Source: `/Users/yuanzhangxiao/Downloads/dalla-model-review-20260923-072516.tar.gz`.
All 230 supplied JSON files have valid original artifact seals. Inspection kept
both retained endpoints and unretained fitted trials, without an NMSE cutoff.
The archive also includes non-Dalla cells and ablation arms in the old v5 run;
the fitted-equation census here concerns Full/Brief-only Dalla tasks.

| Campaign folder | Downloaded Full/Brief Dalla coverage |
| --- | --- |
| `t1_easy_v5` | Four T1-easy cells, both arms and seeds; rounds 12-13 complete, 15/16 records at round 14; **no round-15 results** |
| `mechanism_pilot` | T1-hard, T2-easy, T2-hard; two Full seeds, rounds 0-2 |
| `continuation_v2` | Same six lineages, imported round 2 and all rounds 3-14 |
| `response_r14` | Six imported round-14 checkpoints only; **no completed rounds 15-17** |

There are 123 distinct exported request/parameter records, of which 43 were
already in the previous 217-record inventory. The other 80 comprise 30 T1-hard,
24 T2-easy, and 26 T2-hard records. Repeated fits and similar equations are not
independent discoveries. One T1-easy round-14 trial lacks its fitted vector or
candidate identity; its retained parent was still inspected. Another T1-easy
round-14 checkpoint is absent entirely.

Every one of the 123 exported lowered-candidate hashes and parameter-name sets
reproduces. All 80 new models' state-equation expansions agree with the production
interpreter in 650 finite evaluations, including protected constant division.
Rounded equation displays retain the original division syntax; denominators
smaller than 1e-12 use the production signed floor during numerical checks.

## T1-hard: useful components, inadequate overall fit

Seed 0 has an active positive meal-memory filter with a roughly 35.7-minute time
constant and positive tissue-to-plasma coupling. Its retained training/validation
NMSE is 0.232584 / 0.303552. Later trials reach only 0.23258395 in training.
This is effectively a plateau, not meaningful progress toward the original
goal of an accurate training fit and discriminating intervention generalization.

The main fitted coefficients are approximately -0.008412 on Gp and +0.008412 on
Gt, versus the canonical direct exchange coefficients -0.065 and +0.079. This
comparison alone is not a complete test of a reduced model: other hidden fluxes
can alter effective coefficients. Actual rollouts show the practical problem:
too-early meal responses, incorrect tails, and pre-meal drift. An extra latent
state named I is not observed insulin and has an effectively instantaneous decay
at the denominator guard. Later added pathways often have negligible gains or
reproduce almost identical trajectories.

Seed 1 retains 0.400073 / 0.490862. Several latent relaxation times are extremely
long compared with the 300-minute horizon, and the glucose response is largely a
direct meal response with compensating latent terms. Its extra F state has zero
forcing and zero initial value, so it contributes identically zero.

## T2: a systematic missing pathway in all 50 new fitted records

In every new T2 record, the actual observed insulin state obeys

`dI/dt = a * insulin_pmol_per_kg_min + b * I`.

There is no meal, glucose, secretion state, or other modeled source driving I.
With zero infusion and the same initial I, changing meal timing or amount cannot
change predicted insulin. A finite-dimensional reduced model need not recover
all original insulin compartments, but it needs some route to generate the
meal-associated insulin peaks present in the training data.

Across the 50 records, six have a negative infusion gain, 23 have positive
insulin self-growth, and 19 have an absolute insulin self coefficient below
1e-8/min. These are fitted-record counts, not independent runs. All six final
incumbents nevertheless carry true public graph certificates: those certificates
do not establish correct fitted signs, active response magnitudes, or physiological
recovery.

For T2-easy seed 0, the retained equations include:

```text
dI/dt = -0.2976430975 * insulin_pmol_per_kg_min + 0.004850542713 * I
dX/dt = -0.000285091 * I - 0.0196081 * X
U = 5.54701 * Uii + 2.22217 * X
```

Insulin infusion lowers predicted I. Higher I drives lower X and hence lower U,
opposite to the intended insulin-dependent disposal effect. With no infusion,
the model produces exponential insulin growth instead of a meal-linked peak.

For T2-easy seed 1, the I-to-X coefficient is about 1.62e-30. Even accounting for
the downstream U gain of about 654, the dynamic insulin-to-disposal response is
negligible over the observation horizon. Its I equation also has positive
self-growth. A nonzero symbolic graph edge is not an appreciable fitted effect.

T2-hard has more plausible positive infusion gains (21.2172 and 20.3121, compared
with the reference 1/VI = 20). This is a useful recovered component. However,
its insulin self coefficients are approximately -3.39e-10 and -1.00e-10/min,
and secretion is absent. Predicted insulin stays almost constant without infusion.
The remaining delayed-action paths cannot compensate for the missing I response.

The current public T2 training set **does include nonzero insulin forcing**, with
a maximum of 0.35 pmol/kg/min. All four supplied validation schedules have zero
external insulin forcing. Thus these validation data check endogenous meal
responses; they do not establish performance on held-out infusion schedules.

## Independent fixed-parameter replay

All six round-14 retained models were replayed on all 16 training and four
validation trajectories, with their original public inputs, auxiliaries, and
causal initializations. All 120 rollouts completed. Per-target aggregate NMSEs
reproduce the stored values within 5.3e-15. Resume reuses every checkpoint with
zero new integration. These checks use the existing production interpreter and
solver settings; they are not an independent-solver accuracy certification.

| Model | Target | Training NMSE | Validation NMSE |
| --- | --- | ---: | ---: |
| T1-hard, seed 0 | Gp | 0.232584 | 0.303552 |
| T1-hard, seed 1 | Gp | 0.400073 | 0.490862 |
| T2-easy, seed 0 | Gp / I / U | 0.485640 / 1.335615 / 0.874938 | 0.612910 / 1.217140 / 0.782408 |
| T2-easy, seed 1 | Gp / I / U | 0.695913 / 1.326492 / 0.991136 | 0.667707 / 1.247285 / 0.992119 |
| T2-hard, seed 0 | Gp / I | 0.424054 / 1.418455 | 0.428531 / 1.527168 |
| T2-hard, seed 1 | Gp / I | 0.481420 / 1.418553 | 0.420284 / 1.527168 |

On validation_000, reference insulin peaks at 242.38 pmol/L at minute 128.
Both T2-hard models remain near 25.59 pmol/L. T2-easy produces increasing
exponentials, ending near 109.65 and 92.70 pmol/L at minute 300. These are
structural response failures, not missing-score or plotting artifacts.

## Next decision

The new response-feedback run has not produced a completed revision in this
snapshot. Inspect its next completed models for an active, meal/glucose-driven
insulin source, reasonable response signs, and non-negligible delayed action.
Response summaries should make the observed peak versus flat/exponential
prediction discrepancy explicit. A parameter-only continuation cannot add the
missing pathway to the present T2 equations. No model was edited or refitted
using this private-reference inspection, and no intervention was redesigned to
favor a selected model.

Actual T1-easy round-15 checkpoints still need locating or exporting before
claiming that those last models were assessed. The current archive does not
resolve that earlier retrieval gap. No T3/T4 fitted models are supplied.

Local audit outputs are in `artifacts/dalla-new-model-review-2026-09-23/`:
`verified-models.json`, `fitted-equations.md`, `active-equation-screen.json`,
`equation-verification.json`, `latest-models.json`, `replay-summary.json`,
`curves.csv`, and `latest_model_diagnostics.png`. The figure shows the same
training_002 and validation_000 schedules across both seeds. All replayed
trajectories and targets are exported, not just the illustrated panels.

## Verification and changes

No production implementation, benchmark, prompt, or fitted model was changed.
The tracked change is this audit report; generated audit records, equations,
trajectories, and plots remain local artifacts.

The relevant collector, multi-target fitting profile, and T1 curve replay tests
passed: **28 passed**. The 120 fixed-parameter replays and zero-integration resume
check provide the task-specific numerical smoke verification. Repository-wide
`ruff check .` reports 37 existing issues in unrelated, untracked
`analysis/claude/` scripts; those files were left unchanged.

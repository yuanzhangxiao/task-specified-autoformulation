# Fresh Dalla campaign: round-12 model inspection

The new constructor/revision/fitting integration ran, but this snapshot does not
yet supply the desired example of a well-fitted model with convincing held-out
mechanism behavior. T1-hard seed 0 captures a meal-shaped response imperfectly;
its nonzero latent initial conditions also produce a spurious no-meal excursion.
Every fitted T2 trial still lacks an active positive source that explains the
observed insulin increase when external insulin input is zero.

## Scope and provenance

- Input: `~/Downloads/model-review.jKjbM0.tar.gz`.
- Archive SHA256: `126d8d305481f6f07b9fae62599ea24329e027689464e7961144db7ae918396e`.
- Plan: `b86871edb8b937d6725b3c1e0a390fb8105dce44b50c60700c3704a2de92e006`.
- Protocol: `shared-multi-pruning-1`; six Full lineages, rounds 0–14 planned.
- All 78 supplied round self-seals and the plan self-seal match their contents.
  This checks artifact consistency, not unavailable original fit receipts or
  provider logs on ACES.
- Screened all 75 saved trial fits, including trials not retained. They contain
  62 distinct equation/declaration/initializer/parameter combinations under an
  exact structural comparison, and 50 distinct equation/declaration structures.
  These counts are not symbolic-equivalence classes or independent experiments.
- No fitting, model edits, LLM calls, test data, or private reference equations
  were used for this inspection. All response comparisons use the public
  training/validation observations in the supplied plan.

## Execution status

All six lineages have records through R12: 75 `complete` and three
`provider_request_failed`. The failures are T1-hard seeds 0/1 and T2-easy seed 0.
Each records one provider attempt lasting approximately 30 seconds, unknown token
usage, no fitted trial, and the unchanged incumbent. Their lineages are not closed.
The bundle does not contain `proposal.json`, call-cache errors, or server logs,
so timeout, proxy, server failure, and request rejection cannot be distinguished.

The dispatcher refuses to start another round after a provider/preflight failure.
Consequently these records block normal dispatch of R13; the twelve absent
R13/R14 task visits are unfinished work, not twelve scientific failures. Final
pruning is scheduled after R14 and is not represented in this snapshot. This is
an artifact/code interpretation, not a fresh scheduler query.

## Retained scores and model findings

| Cell | Seed | Origin | Train NMSE | Validation NMSE | Main finding |
|---|---:|---:|---:|---:|---|
| T1-hard | 0 | 9 | 0.204722 | 0.225688 | Meal-driven latent filters; weak tissue coupling and no plasma-state restoring term; spurious no-meal transient |
| T1-hard | 1 | 2 | 0.299748 | 0.408976 | Predominantly direct meal input; delayed component has negative fitted gain and a large negative fitted initial condition |
| T2-easy | 0 | 1 | 1.959029 | 2.006391 | Insulin rapidly decays; disposal does not track observed post-meal increase |
| T2-easy | 1 | 9 | 0.956755 | 0.981811 | Nearly constant insulin and disposal despite several nominal memory states |
| T2-hard | 0 | 6 | 0.945701 | 0.980073 | Shared meal process subtracts from insulin; fitted coupling nearly vanishes |
| T2-hard | 1 | 7 | 0.964097 | 1.006716 | Insulin-decay model and delayed disposal filters; no source for endogenous insulin rise |

T1-hard seed 0 also has R2 (train 0.149367, validation 0.234996) and an
unretained R7 trial (train 0.149277, validation 0.235809). These were included in
equation screening. R9 was legitimately preferred by aggregate validation;
that does not mean it is better on every trajectory.

## T2: an explicit public-data contradiction

Across all 51 T2 trial fits, the insulin observation is the state `I`. The
fitted equations have the form

\[
\dot I=a u_I-bI,
\quad\text{or}\quad
\dot I=a u_I-bI-c u_{\rm meal},
\]

with positive fitted decay and nonnegative fitted meal subtraction when present.
There is no positive endogenous source from glucose, meal processing, or another
modeled state. With zero external insulin forcing and nonnegative I/meal,
the modeled insulin cannot increase above its initial value. This describes the
actual fitted equations, not a general impossibility claim about every possible
parameterization or physiological model.

In public training trajectory `train_003`, external insulin input is identically
zero and a 90 g meal is supplied. Observed insulin rises from 25.5878 to 300.9364,
then returns toward baseline. For the retained T2-easy seed-1 model:

\[
\dot I=21.2176u_I-I/(1.3455\times10^7),
\]

so its saved training replay stays between 25.58724 and 25.58781. This failure
is already present in training; a specially designed held-out intervention is
not needed to reveal it. Fifteen of sixteen training trajectories and all four
validation trajectories have zero external insulin input. The sixteenth training
trajectory does include external insulin forcing; it must not be described as
absent from the entire dataset.

The saved response evidence already records the mismatch. Its worst insulin
example is `train_005`: observed peak 582.5303 versus predicted maximum 25.5878,
with external insulin forcing zero. The deterministic presentation policy keeps
the worst example for each target even at its smallest optional example count.
Thus these records indicate a failure to act on available evidence, rather than
establishing that the response packet omitted insulin. The actual sent request
must still be checked in the provider cache before claiming exact delivery.

No direct meal-to-insulin or glucose-to-insulin dependency is mandated by this
frozen public prompt. Do not add a ground-truth dependency as a hard public
requirement. A defensible next feedback packet can quote the observed rise,
zero external forcing, current predicted flat response, and current equation's
inability to generate that rise. Let the proposer infer a source and its form.

## Shared processes and fitted activity

The R0 gain-compilation records show that optional process construction was used
in four of six lineages; two used ordinary construction. In T2-hard seed 0,
`MealGlucoseInput` was declared a transfer with one shared gain, positive in Gp
and negative in I. The compiler records and assembles that proposal consistently.
It does not establish a scientifically coherent transfer between glucose mass
and insulin concentration. At R6 the shared gain is about 1.985e-31: the fit
suppresses the incorrect coupling, and another state supplies glucose's meal
response, while insulin's missing source remains unresolved.

Fitted coefficients must be interpreted together with initial values. In the
retained T2-easy seed-1 model,

\[
U=U_{ii}+1.75942\times10^{-7}Z,\qquad
\dot Z=(I-Z)/35817.3,\qquad Z(0)\approx1.34528\times10^7.
\]

The small gain does not make the term numerically negligible: its very large
initial state contributes about 2.37 to U. However, it is almost a constant
offset over the 300-minute horizon. In `train_003`, predicted U declines from
3.36690 to 3.34716 while the reference rises from 1.91579 to 6.86714. Several
other named insulin-related states/processes have no path to a target and are
potential pruning material. Removing them cannot supply the missing insulin
source.

## T1 diagnostic curves

Independently replayed R2 and R9 of T1-hard seed 0 on four existing trajectories:
`train_003`, `train_011`, `validation_000`, `validation_003`. Parameters and causal
initializers were fixed. The production restricted compiler/lowering reproduces
each saved lowered-candidate hash. DOP853, rtol 1e-7 and atol 1e-9, was used for
these diagnostic plots; the original fitted scores were not overwritten.
All eight replays completed. R2 on the two displayed training trajectories
also agrees with independent Radau replay to below 4e-11 mg/kg.

The meal trajectories have recognizable rises and falls, but peak timing,
amplitude and late levels remain wrong. More decisively, both models show
substantial transients without a meal. In the validation initial-state case,
the reference decreases toward equilibrium while both models first increase.
Both use shared fitted latent values, not trajectory-fitted validation initials.
Their glucose equations have effectively negligible supplied-tissue coupling and
no dependence on Gp itself; the latent initial transient substitutes for
baseline regulation. R9 has a tissue coefficient of about -5.50e-6.

Local diagnostic artifacts (generated, excluded from Git):

- `artifacts/fresh-dalla-review-20260924/t1-hard-replay.png`
- `artifacts/fresh-dalla-review-20260924/t1-hard-replay.pdf`
- `artifacts/fresh-dalla-review-20260924/curves.csv`
- `artifacts/fresh-dalla-review-20260924/replay-scores.json`
- `artifacts/fresh-dalla-review-20260924/all-fitted-equations.txt`

The four plotted cases are illustrative diagnoses, not a new aggregate endpoint
or a prospectively designed intervention suite. No favorable model or intervention
is claimed on their basis.

## Repetition and next step

T2-easy seed 0 repeats the same equations and fitted vector in R2/R3/R4/R6/R8/R10;
another rejected model repeats in R5/R9/R11. T1-hard seed 1 likewise repeats one
failed candidate in R4/R5/R7/R8. These are exact saved-equation/parameter repeats,
not a claim of identical raw LLM text or confirmed erroneous cache reuse.

Some genuine score improvements are refits: T2-easy seed 1 improves in R7–R9
after failed revisions use the incumbent fallback fit. Do not credit every
improvement to a new scientific mechanism or to shared-process construction.

Recommended order:

1. Retrieve R12 `proposal.json` error details and the proposer server log. Resolve
   the delivery failure with an auditable recovery; do not delete checkpoints or
   claim the current run finished.
2. Before another long run, test short public-data contradiction feedback and a
   compact record of unsuccessful prior proposals. Distinguish symbolic
   representational limits from unfinished numerical optimization. Keep the
   public prompt, numerical profile and train/validation/test boundary intact.
3. Evaluate a small matched continuation with that feedback. Inspect target-wise
   response recovery, model changes and repeat rates. The pending sign recheck
   is separate and cannot by itself fix the missing insulin source.
4. Finish the planned search/pruning only with clear recovery provenance. Pruning
   evaluates redundancy, not missing-mechanism discovery. Do not expand to T3/T4
   merely to compensate for these identifiable unresolved problems.

Verification accompanying this documentation-only change: 12 focused submission
and response-revision tests passed. `smoke_fresh_shared.py` passed with synthetic
joint-output fits, both pruning comparison arms, and exact resume. Whole-tree
Ruff still reports 37 pre-existing findings under unrelated `analysis/claude`.
No implementation files or running campaign artifacts were changed.

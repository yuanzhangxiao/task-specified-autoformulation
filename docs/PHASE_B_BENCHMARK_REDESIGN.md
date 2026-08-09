# Phase B benchmark and evaluation redesign

## Decision

The existing benchmark is useful for studying task-constrained one-step model
construction, but it is not sufficient for strong claims of autonomous ODE or
unique hidden-mechanism recovery. New expensive LLM experiments should wait
until the data and selection protocol below are frozen.

The Phase-B suite uses **two tiers only**. In every family, easy versus hard
measures scientific difficulty:

- For **Dalla Man and CSTR**, easy has a more focused mechanistic objective and
  richer task-relevant observations. Hard requires more coupled mechanisms from
  fewer observed channels under the same input schedules and trajectory count.
- For the **alien device**, easy similarly exposes a focused input-memory task
  with richer telemetry, while hard requires coupled persistent and nonlinear
  mechanisms from target-only telemetry under the same input schedules.

Every hard design must pass identifiability and excitation gates; hard means
difficult, not intentionally impossible.

Semantic information is an independent paired-control axis:

- Dalla Man and CSTR use named versus obfuscated representations.
- Alien device uses functional-description versus opaque-description
  representations. There is little risk of retrieving its procedurally
  generated equations; this comparison instead measures the value of task
  information when memorization is implausible.

Within each semantic pair, numerical trajectories and channel information are
identical. Across easy/hard tiers, mechanism burden and public channel roles
differ, while input schedules, trajectory counts, sampling, and noise are held
fixed.

There is no medium tier. Canonical versus perturbed dynamics is another
independent Dalla Man axis. The machine-readable freeze is
`configs/benchmarks/phase_b_suite_v1.json`; it remains separate from the
historical runtime registry until the new data and prompts pass validation.
Exact channel roles, split sizes, intervention schedules, and pre-release gates
are frozen in `docs/PHASE_B_EXACT_BENCHMARK_PROTOCOL.md`.

## Frozen suite matrix

| Family | Independent axes | Easy | Hard | Scientific role |
|---|---:|---|---|---|
| Dalla Man | T1--T4 × canonical/perturbed × named/obfuscated | Focused mechanism and richer observations | Coupled mechanisms and fewer observations | Mechanism depth, adaptation beyond memorization, and semantic-prior dependence |
| CSTR | named/obfuscated | Focused controlled-process mechanism and richer observations | Coupled reaction/exchange/control mechanisms and fewer observations | Generalization to a familiar non-biomedical ODE family |
| Alien device | functional/opaque semantic pair | Focused input-memory task and richer telemetry | Coupled persistent/nonlinear task and target-only telemetry | Task-information value without a retrieval confound |

The complete definition contains 40 cells before methods or seeds: 32 Dalla
Man, four CSTR, and four alien-device cells. This is the scientific factorial
definition, not a requirement that every method immediately receive the same
large seed count. Execution will be staged: inexpensive baselines and
identifiability checks first, sequential LLM seeds for headline cells, and
smaller paired samples for semantic controls where power permits.

### Claim levels

- T1 and T2 target task-sufficient recovery under deliberately varied meals,
  initial conditions, and insulin excitation. Hidden recovery is evaluated up
  to an equivalent representation or identifiable subspace.
- T3 and T4 target an identifiable regulatory subspace or task-compatible flux
  portrait. They do not claim unique recovery of every simulator coordinate.
- Perturbed Dalla Man cases test whether the method adapts equations to data
  rather than reproducing the canonical model suggested by semantic context.
- Named/obfuscated Dalla Man and CSTR pairs separately measure dependence on
  semantic priors without changing their numeric evidence.
- CSTR tests transfer outside glucose-insulin dynamics.
- Alien device tests whether a functional task description helps when a known
  domain template is unavailable and retrieval is not a plausible explanation.

Negative controls that are intentionally information-insufficient may be
reported separately, but they are not headline recovery cells and are not
called a harder tier of the primary benchmark.

## Why redesign is necessary

1. One-minute observed-state resets make persistence and weak autoregression
   artificially competitive.
2. Uninterrupted frozen rollout increases error by one to three orders of
   magnitude relative to reset-based intervention evaluation.
3. T2--T4 hard tiers ask for progressively richer mechanisms while exposing the
   same Gp/I trajectory and meal protocol.
4. Local initial-state sensitivity is numerically full-rank but dominated by
   one direction.
5. T3/T4 output-parameter sensitivity is practically rank-deficient at a
   `1e-3` relative threshold and extremely ill-conditioned.

## Minimum additional Dalla Man trajectory design

The following is a minimum diagnostic design, not a final power calculation.
All protocols must be generated before inspecting discovered-model test
performance.

### Shared meal-excitation block

| Block | Protocols | Purpose |
|---|---:|---|
| Dose response | 30, 60, 90, 120, 150 g at minute 0 | Identify nonlinearity and saturation |
| Timing | 90 g at minutes 0, 60, and 120 | Separate absolute time from event response |
| Multiple events | 60+30 g, 45+45 g, and 30+60 g at distinct gaps | Identify persistent meal compartments |
| Initial glucose | Basal, +10%, and -10% Gp with compensating tissue shifts | Test state dependence and autonomous recovery |
| Initial insulin | Basal, +15%, and -15% plasma/liver insulin | Excite delayed insulin action |

This gives 16 prespecified training trajectories, of which a fractional design
can be chosen only after sensitivity-based design analysis. Validation should
use four interpolating but unseen combinations. Test should contain at least six
extrapolating combinations, including delayed and multiple meals.

### Task-specific requirements

- **T1:** The shared meal block is likely sufficient to identify a
  task-sufficient absorption memory up to an input-output-equivalent state
  representation. Exact gastric-compartment identity should not be required.
- **T2:** Meal variation alone does not independently excite insulin-dependent
  disposal. Add controlled insulin perturbations or multiple initial insulin
  conditions. If the simulator cannot expose an insulin intervention, weaken
  the claim to recovery under naturally coupled meal responses.
- **T3:** Peripheral disposal and hepatic regulation are confounded by the same
  endogenous insulin response. Add orthogonal insulin/glucose perturbations
  such as an insulin pulse and glucose-clamp-like input, or provide one hepatic
  proxy in easier tiers. Otherwise report only an identifiable combined
  regulation subspace.
- **T4:** Recovering four internal fluxes from Gp/I alone is not presently a
  credible unique-identification task. Add flux-proxy observations and
  orthogonal interventions, or reclassify hard T4 as a mechanistic-coherence
  stress test evaluated by compatibility rather than exact hidden recovery.

## Evaluation redesign

Every selected model should be evaluated under three distinct protocols:

1. one-step causal prediction, retained as a diagnostic;
2. rolling-origin K-step-ahead forecasts at physically defined horizons such as
   5, 15, 30, 60, and 120 minutes, using causal latent-state estimates at each
   origin;
3. uninterrupted free simulation from unseen initial conditions and input
   schedules.

For CSTR and alien-device trajectories, whose time axes are not minutes, use
prespecified fractions (5%, 10%, 25%, 50%, and 100%) of the evaluation horizon
rather than incorrectly labeling their units as minutes.

Selection must use training/validation rollout losses at prespecified horizons.
Test rollout and private interventions remain unopened until selection is
frozen. A candidate that requires measured target forcing should be classified
as a one-step predictor and must not be presented as an autonomous ODE.

Recommended headline vector:

- free-rollout target MSE/NMSE;
- rise, peak, recovery, and equilibrium-stratified error;
- task-structural compliance;
- hidden-state/subspace recovery only where supported by identifiability;
- term count;
- completion and integration-failure rates.

## Search-objective consequence

A weighted sum of one-step validation error and judge score cannot repair the
current failure because both signals are evaluated on candidates trained for
reset-based prediction. First introduce rollout-aware fitting and validation.
Then compare validation-only, weighted-sum, epsilon-constrained, and Pareto
selection on the same cached or newly proposed structures.

A defensible objective family is:

\[
L = w_1 \widetilde{E}_{1\text{-step}}
  + w_2 \widetilde{E}_{30\text{-min}}
  + w_3 \widetilde{E}_{\text{free}}
  + w_4 (1-S_{\text{public}})
  + w_5 C,
\]

where every term is normalized using development data only, public structural
compliance is deterministic, and complexity is measured after pruning. Judge
score should initially remain an advisory or tie-breaking signal rather than a
large compensating term.

## Execution order

1. Freeze this benchmark redesign and sensitivity-selected protocol subset.
   The two-tier scientific contract is now frozen as Phase-B v1; numeric data
   generation and prompt authoring remain pending.
2. Add rollout-aware fitting/validation while preserving the historical mode.
3. Replay cached structures without LLM calls to measure numerical feasibility.
4. Run a small paired pilot on T1 and T3 using cached proposals or a low-cost
   model.
5. Only then launch sequential additional seeds or open-model ACCESS runs.

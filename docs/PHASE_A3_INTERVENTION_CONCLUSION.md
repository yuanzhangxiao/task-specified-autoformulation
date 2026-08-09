# Phase A3 intervention-performance conclusion

## Scope and leakage control

Phase A3 evaluates structures and parameters that were selected before the
private intervention suites were generated. No private trajectory, hidden
state, intervention outcome, or test metric was available to proposal,
fitting, pruning, judging, or selection. Models are never structurally changed
or refitted during intervention evaluation.

The final Phase A3 analysis contains 645 successful model/case evaluations:

- 172 evaluations of four core shifts on Benchmarks 5 and 6;
- 129 evaluations at additional one-, ten-, and twenty-percent noise levels;
- 344 evaluations of four shifts on original, perturbed, obfuscated-original,
  and obfuscated-perturbed Dalla Man B1.

The Dalla Man reference implementation is a side-effect-free extraction of the
private generator notebooks. Tests reproduce the original and perturbed
generator trajectories before any new intervention is evaluated.

## Metrics

Target NMSE uses the training-target scale. Qualitative response metrics score
the direction of the largest reference displacement, correlation of the
centered response shapes, and peak-time error divided by the evaluated
duration. Hidden alignment uses Hungarian matching between proposed persistent
latent states and private hidden trajectories. Pair error is the residual NMSE
of the best affine coordinate map, `1 - correlation^2`; missing private states
are penalized by one variance unit. Coverage reports the fraction of private
hidden coordinates that can be matched one-to-one.

The affine hidden score is invariant to latent naming, permutation, sign, and
scale. It measures trajectory recovery rather than semantic identification and
can still reward correlations caused by a common input. It is therefore a
necessary but not sufficient measure of mechanism recovery.

## Main observed-target findings

Full Autoformalism is competitive, but it does not dominate the frozen
intervention endpoints.

- On Benchmark 5's four core shifts, Full is better than LLM-feature-SINDy on
  matched cells (geometric Full/baseline NMSE ratio `0.432`, 95% bootstrap CI
  `[0.235, 0.781]`), but it is statistically indistinguishable from D3,
  No-judge, and No-latent.
- On Benchmark 6, Full modestly improves over D3 (`0.857 [0.751, 0.991]`) but
  is indistinguishable from LLM-feature-SINDy, No-judge, and No-latent.
- On semantic original and perturbed Dalla Man, none of Full's paired
  comparisons establishes a reliable advantage. Full, No-judge, No-latent,
  D3, and LLM-feature-SINDy are often close under the causal one-step protocol.
- Obfuscated-perturbed Dalla Man exposes a severe tail. Full seed 1 was already
  catastrophic in-distribution (`NMSE=515.6`) and remains catastrophic under
  all interventions; other Full seeds are usually accurate except under the
  parameter shift. This is a model-selection/fitting failure, not an
  intervention-time numerical failure, and it must remain in failure-aware
  reporting.
- Against LLM-feature-SINDy on obfuscated-perturbed Dalla Man, Full's geometric
  paired ratio is `9.73 [1.48, 91.6]`. The exact sign-flip p-value remains
  `0.065` because the effect is heterogeneous and the matched sample is small;
  this result is a warning rather than a stable ranking claim.

Increasing measurement noise produces approximately the same degradation
curve for every method. Full is slightly worse than LLM-feature-SINDy on the
noise sweep (B5 ratio `1.22 [1.07, 1.41]`; B6 `1.01 [1.01, 1.02]`). Under the
current causal protocol, noisy lagged target measurements are supplied to every
one-step prediction, so measurement error dominates differences in equation
structure.

## Hidden-state and qualitative findings

Observed accuracy overstates hidden-dynamics recovery.

- Benchmark 6 Full and No-judge selections contain no persistent latent state,
  so hidden recovery is undefined despite excellent target NMSE.
- Benchmark 5 Full models cover both private hidden coordinates, but mean
  aligned hidden NMSE ranges from `0.258` to `0.885` across shifts. Coverage
  alone therefore does not imply correct dynamics.
- Semantic original Dalla Man Full models cover about half of the four private
  B1 hidden coordinates and have mean aligned hidden NMSE about `0.69`.
- Semantic perturbed Dalla Man Full covers one quarter of the eight private
  coordinates with mean hidden NMSE about `0.81`.
- Obfuscated-perturbed Full covers only `15.6%` on average and has hidden NMSE
  about `0.92`. No-judge is similarly poor. Obfuscation removes semantic cues
  that otherwise help the LLM invent plausible latent compartments.

Response-direction accuracy is nearly saturated and should not be used as a
headline metric. Shape correlation and peak timing are more discriminating.
The particularly weak Benchmark 5 input-extrapolation shape/timing results show
that a small one-step target error can coexist with an incorrect global
response profile.

## Why Full is not consistently better

1. **Selection-target mismatch.** Search rewards in-distribution validation
   fit and an LLM rubric, whereas Phase A3 measures unseen interventions,
   parameter shifts, noise, and private latent trajectories.
2. **Teacher forcing permits shortcuts.** Lagged observed targets make
   persistence-like or algebraic models highly accurate without recovering the
   autonomous hidden mechanism.
3. **The judge is weakly grounded.** It can recognize plausible scientific
   narratives but cannot observe private structural truth. The earlier judge
   calibration study also found no reliable held-out benefit from reweighting
   its categories.
4. **Latent existence is not latent correctness.** Full often proposes states,
   but their trajectories cover only a small fraction of the true hidden
   system or align poorly.
5. **Semantic dependence remains.** The sharp deterioration and variance after
   obfuscation, especially under perturbation, indicate reliance on familiar
   labels and mechanisms rather than consistently data-grounded discovery.
6. **Frozen kinetic parameters are brittle.** Parameter-shift cases deliberately
   change the private simulator while holding discovered coefficients fixed.
   Structurally correct models should retain qualitative behavior, but exact
   predictive rankings also reflect parameter robustness.

## Improvement plan

The immediate improvements should reuse frozen candidates and cached LLM calls
before purchasing new generations.

1. Reselect cached candidates using the prespecified weighted validation,
   judge, and sparsity objective; report the Pareto set as a sensitivity
   analysis rather than tuning weights on private interventions.
2. Add deterministic public structural gates: task-required dynamic memory,
   causal paths from supplied drivers to targets, complete observation
   mappings, and bounded complexity. The LLM judge remains a secondary soft
   signal, not the structural validator.
3. Run Phase A5 scale-aware refitting on the same structures to separate bad
   equations from bad parameter estimates, especially for the catastrophic
   obfuscated-perturbed seed and parameter-shift cases.
4. Add blocked free-rollout validation and public training-only input/noise
   augmentation. This reduces reliance on lagged-target shortcuts without
   exposing private interventions.
5. Require task-relevant persistent latent memory when the specification calls
   for it, and penalize redundant or dynamically inactive latent states using
   training-only observability and sensitivity diagnostics.
6. After these zero- or low-LLM-cost studies, run a small confirmatory search on
   the hardest semantic/obfuscated paired benchmarks. Only then scale new LLM
   generations with ACCESS resources.

## Bottom line

Phase A3 does not support a claim that Full Autoformalism uniformly improves
intervention prediction. It supports a narrower and more informative claim:
the method is competitive on observed targets and occasionally improves over
adapted baselines, but current search and judging do not reliably recover
intervention-stable hidden dynamics. The strongest next contribution is to
make selection, validation, and latent-state requirements more causally
grounded, then confirm the change on the frozen private suites.

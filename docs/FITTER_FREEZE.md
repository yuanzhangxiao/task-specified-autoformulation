# Fitting protocol freeze — September 15, 2026

The user approved closing algorithm selection after the final saved-checkpoint
comparison. [The closeout](FITTER_CLOSEOUT_2026-09-14.md) records the evidence and
known limitations. This freeze governs the next full-pipeline work. It is an
operational decision, not a claim of reliable recovery for arbitrary models.

The subsequent user-authorized [public continuation pilot](PUBLIC_FIT_CONTINUATION.md)
adds one separately versioned 180-second sensitivity window to the selected
budget-limited public fit. It preserves the old experiment and full learned
initializer vector, caps cumulative allocations, and reports unresolved fitting
uncertainty before scientific redesign. It is not a general automatic extension
policy or a reopening of fitting-method comparisons.

## What is frozen

- Keep existing experiment roots pinned to their original code, inputs, starts,
  solver settings, budgets, and failure accounting. Resuming an old root cannot
  switch numerical algorithms or grant fresh consumed budgets.
- Prefer the established standard collocation followed by training-only
  feasibility screening and refinement where its adapter contract applies.
  Retain its existing exact-value directional-polling route for supported
  piecewise expressions and derivative failures. Do not promote alternating
  optimization, new Hessian strategies, or difficult-case multistart experiments.
- Keep scientific term signs and parameter roles, causal physical initial
  conditions, supplied-input interpolation, training-derived normalization,
  training-only parameter/start selection, and frozen-parameter validation.
  Collocation node guesses are not physical initial-condition overrides or
  reported trajectories.
- Preserve the best valid training evaluation across handoffs. Report native
  convergence, budget exhaustion, unsupported capabilities, numerical failure,
  retained finite points, replay agreement, and prediction quality separately.
- Keep validation/test labels out of initialization fitting and parameter/start
  selection. No test data or private reference equations, hidden boundaries,
  reference starts, or diagnostic recovery labels enter proposer/judge feedback.
- Keep the difficult reference case visible as unresolved within the tested
  budgets. No automatic new fitting comparison or relaxed success threshold
  follows this freeze.

## Actual integration status at the freeze

There is not yet one interchangeable fitter implementation for every candidate.
The following are existing, distinct execution paths; their differences must be
visible in any new pipeline manifest.

| Existing caller | Backend | Existing fitting budget and policy |
| --- | --- | --- |
| `repair_comparison.RepairComparisonConfig` | `fit_collocation_forward_sensitivity` | 120 s initializer; 180 s refinement; 240 evaluation ceiling; `rollout_or_observed` node guesses; 5 s warmup; `ftol=None`; `recovery_policy=feasible`; at most 10 screens with 10 s individual caps; Radau 1e-7/1e-9 |
| `prefit_matched_construction_v1.json` | General `fit_candidate` bounded rollout | One start; 240 optimizer evaluation ceiling; 300 s cooperative fitting deadline; Radau 1e-7/1e-9; derivative regression off; no collocation initializer |
| Current construction/requirement-only audits | No numerical fitting | A deterministic construction/requirement pass is not a fitted model or a scientific success |

The collocation adapter currently requires exactly target `v01`, open rollouts,
supported restricted expressions, global parameters, and its supported causal
initialization contract. In particular it is not a drop-in replacement for the
named multi-target construction cell. `SymbolicODE` can represent multiple
observation channels, but residual normalization, collocation and transfer
plumbing still require integration work. Do not rename outputs or silently
switch to another fitter to hide a capability gap.

The general construction fitter remains its historical control. Its support for
multiple targets does not establish equivalence to the collocation/refinement
route. The difficult-case recovery's 3,360-second refinement budget and private
reference initialization contract are diagnostic settings, not new pipeline
defaults.

## Completing the pipeline

The first integration milestone is now implemented in
[Public fitting handoff v1](PUBLIC_FITTING_HANDOFF.md): typed public requests,
deterministic initializer lowering, sealed data/provenance, explicit existing
backend profiles, a CPU CLI and separate execution/fit-quality evidence. It
does not add multi-target C+S support or modify the historical callers below.

Astra owns the fitter interface, numerical result interpretation, and CPU
handoff. Orion owns scientific construction, requirement repair and coordination
of the pre-fitting/controller path. They share the project checkout and commit
only their own files, coordinating source holds for provenance-sensitive tests.

The controller must export its accepted public model and causal initializer
through that versioned request, identify the numerical backend/settings explicitly,
and consume the result envelope without inventing unavailable metrics. Legacy
experiment callers remain unchanged. Capability failures are reported before
optimization; a poor fit must not become a scientific rejection or proof of
structural impossibility.

Multi-target support, if needed for the chosen pipeline pilot, is an adapter
integration milestone. It must preserve per-channel training scaling, residual
and Jacobian order, observed/latent boundary semantics, budgets and deterministic
resume. It is not permission for another optimization-method search. Verify it
on known-attainable synthetic controls, including target permutations and
different units, before scheduling a bounded public fitting handoff.

Then connect accepted construction/repair artifacts to CPU fitting and the
existing feedback/judge policy with exact candidate provenance. Distinguish a
working end-to-end execution from acceptable trajectory fit and scientific
requirements. Use ACES for the approved proposer work and Delta for numerical
work when a concrete public handoff is ready. Do not run the closed reference
comparison again.

No new cluster experiment is required merely to record this freeze. Integration
changes get their own version, local numerical smoke, full tests and explicit
user-run commands; they do not resume or relabel earlier comparisons.

# Feedback-rich incumbent refinement

The repaired proposer-finalist experiment evaluated one round-zero candidate per
condition. It established proposer transport, deterministic validity, public
target completeness, public mechanism compliance, and common-budget fitting,
but it did not test iterative improvement.

The refinement pilot is a separate development experiment. It keeps the
bounded nonlinear fitting backend fixed and compares two four-round search
policies:

1. `rich_exploratory` receives the rich feedback but may propose any novel
   candidate.
2. `rich_incumbent_refinement` receives the same feedback and must return a
   complete child of the active incumbent.

The arms use the same benchmark, seed, prompt, proposer and judge settings,
fitting budgets, and initial proposer request. The exploratory arm runs first
and populates the shared round-zero cache. The refinement arm treats a
round-zero cache miss as an error. Policy-specific instructions begin only in
round one.

## Rich feedback contract

For each eligible incumbent, `rich_v1` includes:

- the complete candidate structure: states, right-hand sides, algebraic
  processes, observation mappings, parameters, initial conditions,
  constraints, and mechanism tags;
- fitted parameter values and train/validation NMSE, including per-target
  errors;
- failed trajectories, optimizer messages, function evaluations, integration
  failures, bound saturation, and soft-constraint violations;
- deterministic public-target and public-mechanism predicate results;
- pruning results; and
- paired, orientation-normalized scientific assessments from the previous
  incumbent challenge, when available.

Rejected candidates are explicitly marked as context rather than eligible
parents. No test or private-reference result enters feedback.

## Refinement policy

The proposer returns a full candidate, not a patch. The policy requests the
smallest coherent set of edits supported by the evidence, but it does not cap
the number of edits. It asks the proposer to preserve already validated
components unless changing them is necessary for a stated repair. Runtime
binds the returned lineage to the active incumbent deterministically.

This policy addresses proposal strategy only. It does not yet impose a model
class that is linear in fitted weights, and it does not change the numerical
optimizer. A linear-combination proposer plus profiled or exact-derivative
weight fitting remains a separate milestone so the effects of proposal policy
and fitting method can be identified independently.

## Pilot endpoints

The analysis reports endpoints separately rather than constructing a weighted
winner: completion and valid-round coverage, incumbent replacements,
train/validation NMSE, target completeness, mechanism compliance, fit retry
activation, model complexity, LLM calls/tokens/latency, and allocated CPU/GPU
hours. The experiment is development-only and leaves test and private data
closed.

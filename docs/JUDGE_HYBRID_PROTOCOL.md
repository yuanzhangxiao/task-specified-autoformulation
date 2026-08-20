# Hybrid scientific judge protocol

## Status and boundary

The hybrid protocol is calibration-only. It does not change production search,
candidate ranking, benchmark prompts, or finalized benchmark data. It evaluates
the canonical executable candidate after behavior-preserving repair and records
the raw proposal repair ledger separately.

Completely unreferenced processes and unused parameters may be removed before
judging only when removal cannot alter executable behavior. Retained components
that lack a path to requested targets remain in the candidate and are certified
to the judge. Repair frequency is a proposer-contract diagnostic, not a
scientific model score.

## Authority and provenance

The runtime extracts task-required mechanism bullets only from the public prompt
and freezes them as `ScientificRequirement` records. Absence of the explicit
public marker creates no inferred requirement. Requirement sources are
`domain_expert`, `benchmark`, `proposer`, and `runtime`; enforcement is `hard`,
`soft`, or `descriptive`.

Mechanism tags attached to candidate states and processes become proposer-owned
`ProposerClaim` records. They cannot create a task requirement or earn task
coverage merely by existing. They are evaluated only for scientific support,
connectivity, and consistency.

## Deterministic facts

For both candidates, the runtime constructs the same directed expression graph
and certifies:

- paths from declared task inputs to requested targets;
- paths from retained states and processes to requested targets;
- equation-level drivers of latent states;
- candidate-owned mechanism claims and their component subjects.

The runtime owns absolute pass/fail results for task-input reachability, claimed
component reachability, latent incoming pathways, and latent target influence.
These facts contain no fit data, trajectory data, hidden mechanism, mutation
label, or scientific verdict.

## LLM absolute assessments

The judge sees Candidate A and Candidate B together in randomized order but
answers each requested predicate separately for each candidate with `pass`,
`fail`, `indeterminate`, or `not_applicable`. Required mechanisms are always
applicable. Every answer cites candidate or certified-fact evidence.

For every public task requirement, the judge answers:

1. whether an identifiable candidate structure represents the requirement;
2. whether that representation has scientifically relevant target influence.

Candidate-wide semantic questions assess source roles, sink roles, semantic
flux duplication, conflicting claims, accumulator justification, meaningful
delays, appropriate saturation, and support for proposer-owned claims.

The provider must return exactly the runtime-requested criterion/subject keys.
Post-schema validation rejects missing, duplicate, or invented units.

## Direct comparative residual

The same blinded call also asks which candidate is better on three irreducibly
relative properties:

- parsimony while satisfying the public task;
- fewer unsupported scientific assumptions;
- mechanistic interpretability.

These answers use `candidate_a`, `candidate_b`, `tie`, or `indeterminate`. They
are never converted into absolute pass/fail claims. Candidate order is reversed
and repeated during calibration.

## Generic conjunctions

The runtime, not the LLM, instantiates generic group templates. Instances come
only from public requirements and submitted candidate structure; hidden ground
truth never creates a group. The initial templates are:

- one group per public user/benchmark requirement;
- task-input connectivity;
- proposer-claim integrity;
- balance semantics;
- latent-state validity;
- delay and saturation claim validity.

Atomic predicates inside a group are combined with logical AND. For example,
a public requirement is complete only when it is both represented and
functionally connected. Atomic predicates are not also added as independent
bonuses.

The candidate's conjunctive score is the weighted fraction of complete,
determined, applicable groups. The partial score is the weighted atomic pass
fraction and is used only as a small tiebreak signal:

```text
shaped = (conjunctive + epsilon * partial) / (1 + epsilon)
```

Hard public requirements additionally produce an explicit eligibility status.
One candidate satisfying all hard requirements defeats a candidate with a
known hard-requirement failure. Missing information remains indeterminate and
reduces reported coverage.

The final pair decision is exploratory and calibration-owned:

```text
decision(A, B) = shaped(A) - shaped(B)
               + lambda * comparative_residual(A, B)
```

The partial weight, comparative weight, and tie threshold are recorded CLI
parameters. They must be chosen on calibration training data and frozen before
held-out evaluation. They are not production defaults.

## Question-level truth

`build_hybrid_judge_label_template.py` creates one label record per pair.
Deterministic items are filled from the canonical graph. Semantic absolute and
direct comparative labels are `unlabeled` until a domain expert reviews the
public prompt and both blinded candidates. Mutation names are not used to
fabricate semantic answers. Each reviewed label contains a rationale and label
source.

The analyzer reports:

- absolute verdict accuracy;
- exact paired absolute accuracy;
- direct comparative question accuracy;
- absolute-only, comparative-only, and combined preference accuracy;
- pair-aggregated accuracy;
- order consistency;
- reviewed-label coverage and criterion-specific results.

An overall baseline/mutation preference is insufficient for atomic evaluation:
a globally degraded candidate may genuinely improve one narrow predicate.

## Reproducibility and safety

Every call remains content-addressed, cached, logged, order-reversed,
checkpointable, and shard-resumable. Schema/format failures may receive bounded
contract-only repair attempts. No repair prompt contains a scientific answer,
fit result, hidden model, or expected label. The LLM never emits an aggregate
score or overall winner.

The initial experiment retains five repetitions in both orders. Existing output
supports offline ablations of absolute-only, comparative-only, and combined
decisions before any production integration.

`build_hybrid_judge_pairs.py` preserves the original frozen pairs as controls
and adds a retained disconnected pathway for every old repaired-away
disconnection. The retained variant uses a driven, relaxing latent module whose
components survive canonicalization but have no target path. This tests the
production judge boundary without confusing a harmless discarded declaration
with an executable scientific defect.

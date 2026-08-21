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

`build_hybrid_judge_label_template.py` creates one versioned label record per
pair without requiring domain-expert review at this protocol-engineering stage.
Deterministic items are filled from the canonical graph. Each controlled
mutation type has an explicit, typed contract containing only the semantic and
comparative consequences guaranteed by its construction. For example, a
duplicated-flux mutation certifies failure of the no-duplication predicate, but
does not claim to establish whether an unrelated saturation law is appropriate.

Questions not certified by either the runtime or that mutation's contract stay
`unlabeled` with source `not_scored_by_mutation_contract`. They remain in the
judge request for protocol diagnostics but are excluded from question-level
accuracy. Canonically identical candidates receive comparative ties. Unknown
mutation types fail label generation instead of inheriting guessed labels.
Every scored label includes its exact runtime or mutation-contract provenance
and rationale. Gold labels, mutation types, and baseline identities are never
sent to the judge.

The analyzer reports:

- deterministic runtime consistency as an integrity check, never as LLM
  accuracy;
- mutation-targeted LLM absolute verdict accuracy;
- exact paired semantic accuracy when both sides are contract-labeled;
- direct comparative question accuracy;
- absolute-only, comparative-only, and combined preference accuracy;
- pair-aggregated accuracy;
- order consistency;
- certified-label coverage and criterion-specific results.

An overall baseline/mutation preference is insufficient for atomic evaluation:
a globally degraded candidate may genuinely improve one narrow predicate. This
calibration therefore reports both the broad pair-level test and the narrower
certified atomic subset. It does not claim full domain-expert validation of every
scientific rubric question.

## Reproducibility and safety

Every call remains content-addressed, cached, logged, order-reversed,
checkpointable, and shard-resumable. Schema/format failures may receive bounded
contract-only repair attempts. No repair prompt contains a scientific answer,
fit result, hidden model, or expected label. The LLM never emits an aggregate
score or overall winner.

If all bounded provider attempts fail, the runner writes a versioned terminal
record to `hybrid_judge_failures.jsonl` and continues with the next logical call.
It does not parse hidden reasoning or impute a score. Resume treats successful
CSV rows and terminal failure records as disjoint completed outcomes. Shard merge
requires their union to cover the planned calls. Analysis reports scientific
accuracy conditional on a valid structured response, structured-response success
rate, and end-to-end accuracy that counts a terminal response failure as an
incorrect judge decision.

Each shard also stores an immutable `hybrid_judge_run_manifest.json` containing
the pair-file digest, selected pair identifiers, sharding rule, model settings,
seeds, repetition count, and scoring configuration. Resume fails closed if the
current configuration differs or if saved outcome keys do not belong to the
selected shard. This prevents an expired shell environment from silently mixing
an older calibration pair set into an existing output directory.

For seeded Ollama calls, provider attempt 1 uses the registered repetition seed.
If that attempt fails, repair attempt `n` uses `seed + n - 1`. This deterministic
fallback prevents an empty-content response from reproducing forever under the
same seed while preserving the original seed for every ordinary successful call.
The actual attempt number and sampling seed are stored with the raw response;
all attempts still count as one logical judge call.

If Ollama returns reasoning but leaves `message.content` empty, subsequent repair
attempts use Ollama's OpenAI-compatible chat endpoint with JSON mode and
`reasoning_effort: none`, a strict `json_schema` response format, and an explicit
final-content instruction. If that endpoint wraps the object in prose, the runtime
accepts it only when exactly one embedded object independently satisfies the full
Pydantic schema. Expected-unit post-validation remains mandatory, so this recovery
changes only the provider emission channel and does not weaken the trusted boundary
or infer an answer from hidden reasoning.

The initial experiment retains five repetitions in both orders. Existing output
supports offline ablations of absolute-only, comparative-only, and combined
decisions before any production integration.

`build_hybrid_judge_pairs.py` preserves the original frozen pairs as controls
and adds a retained disconnected pathway for every old repaired-away
disconnection. The retained variant uses a driven, relaxing latent module whose
components survive canonicalization but have no target path. This tests the
production judge boundary without confusing a harmless discarded declaration
with an executable scientific defect.

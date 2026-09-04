# Staged candidate representation

This milestone separates scientific model construction into two immutable
proposal artifacts before producing the existing executable `CandidateModel`.
It adds an opt-in staged proposer path without changing the existing search
controller, mechanism scoring, or fitting backend.

## Representation layers

`TopologyCandidate` owns the dynamic interaction graph:

- observed and latent dynamic states;
- instantaneous algebraic process nodes;
- public forcing symbols;
- signed interaction hyperedges, including their source set, target, and
  mechanism annotations;
- direct mappings from generated nodes to public observation channels.

Every dynamic state must have at least one derivative interaction and every
algebraic process must have at least one defining interaction. Dynamic feedback
loops are valid. A cycle made only of instantaneous algebraic processes is
rejected at this stage because it has no evaluation order.

`FunctionalCandidate` references one topology artifact by its SHA-256
commitment. It assigns exactly one restricted expression to every interaction
and carries executable parameter, initial-condition, and constraint metadata.
An expression must reference exactly the non-parameter sources declared on its
interaction. It therefore cannot silently add or remove an edge.

`expand_staged_candidate` verifies the commitment and source bindings, combines
signed interactions into state equations and algebraic expressions, and creates
one ordinary `CandidateModel`. The result is passed through the existing
restricted parser and complete deterministic candidate validator. Staging does
not create a route around the execution safety boundary.

## Compact provider contracts

The LLM does not emit the complete internal artifacts. The
`ProposedTopologyCandidate` contract omits equations, functions, parameters,
units, external-symbol declarations, and observed/latent labels. The runtime
derives external symbols from the public validation context and marks a state
observed only when a public target maps directly to it. This prevents invented
data access and the state-label error seen in earlier complete-model proposals.

The `ProposedFunctionalCandidate` contract receives the exact committed
topology. It emits one restricted expression per interaction, parameter names
and qualitative roles, and latent initial values. Parameter scope, domain, and
all numerical ranges are runtime-owned. Deterministic enrichment creates the
full `FunctionalCandidate` before ordinary staged expansion and validation.

`CachedLLMClient.propose_topology` and `propose_functions` use distinct roles,
schemas, request hashes, cache entries, and append-only call records. The
opt-in `StagedProposer` orchestrator checkpoints each validated stage by a
run-scoped input hash. Repeating the same operation resumes without another
provider call. Passing a fixed topology performs a function-only refinement;
the topology proposer is skipped and the exact commitment is preserved. Passing
an incumbent topology instead asks for a feedback-localized graph revision and
then regenerates functions against the new commitment.

## Feedback routing

`route_proposer_feedback` converts typed public evidence into a bounded,
priority-ordered record:

- public target and graph-mechanism failures go to topology construction;
- expression-contract and annotation issues go to functional construction;
- fit, integration, and the worst public validation target go to the function
  stage as numerical evidence;
- scientific-judge requirements and actionable edits are retained for an
  integrated repair stage.

Each provider call receives only its stage view. The topology proposer does not
need to sift through optimizer diagnostics, and the function proposer does not
receive already-resolved graph obligations. An integrated repair view remains
available for a later localized revision action.

## Two different uses of hashes

The topology commitment hashes the exact validated topology artifact. It makes
the handoff to the functional stage immutable and changes if that artifact is
edited.

After expansion, the existing `CandidateIdentity` supplies name-invariant
topology, functional, and executable fingerprints. Those fingerprints support
scientific duplicate detection and failure-aware reuse. A topology commitment
and a name-invariant topology fingerprint are deliberately not interchangeable:
the former identifies an artifact; the latter identifies a scientific
equivalence projection.

## Current boundary and next steps

The staged schemas, compact provider schemas, deterministic expansion, feedback
router, and checkpointed two-call proposer are available as library APIs and
exported JSON Schemas. Existing search remains on the complete-candidate path,
so pending or frozen experiments are unaffected. The staged path has not yet
been used for a two-benchmark LLM search.

The next independent milestones are:

1. report graph-mechanism compliance separately from annotation compliance
   (implemented; graph compliance is the primary scientific signal);
2. add the profiled affine-weight fitter for models with latent basis dynamics
   (implemented as `profiled_latent_basis_linear_ridge`);
3. run a cached topology/function transport smoke, then a two-benchmark,
   three-seed public fitting pilot;
4. connect accepted staged candidates to localized revision actions;
5. compare staged beam search with a small-budget tree-search policy only after
   the underlying actions and scores are stable.

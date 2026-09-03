# Staged candidate representation

This milestone separates scientific model construction into two immutable
proposal artifacts before producing the existing executable `CandidateModel`.
It does not yet change the search controller, proposer prompts, mechanism
scoring, or fitting backend.

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

The staged schemas and deterministic expansion are available as library APIs
and exported JSON Schemas. Existing search remains on the complete-candidate
path, so pending or frozen experiments are unaffected.

The next independent milestones are:

1. report graph-mechanism compliance separately from annotation compliance;
2. add the profiled affine-weight fitter for models with latent basis dynamics;
3. route stage-specific feedback into topology and function revision actions;
4. compare staged beam search with a small-budget tree-search policy only after
   the underlying actions and scores are stable.

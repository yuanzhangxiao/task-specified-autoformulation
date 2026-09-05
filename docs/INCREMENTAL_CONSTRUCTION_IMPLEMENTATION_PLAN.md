# Incremental construction and conditional beam implementation plan

## Objective

Replace monolithic topology/function regeneration with small typed actions while
retaining the existing canonical candidate, deterministic validator, fitter,
scientific judge, and final evaluation boundary.  A beam branch is a coupled
partial or complete `(topology, functions)` model.  Functions are generated and
scored only under the exact topology to which they belong.

## Invariants

1. Public targets, inputs, auxiliaries, requirements, and causal availability
   remain runtime-owned.
2. Provider output is an untrusted action transaction, not a complete canonical
   candidate.
3. Unmentioned model components are preserved by default.
4. Topology and functional content receive separate canonical hashes.
5. A failed function closes only that functional child; it does not invalidate
   the parent topology unless every permitted functional expansion is exhausted
   by a genuine structural incompatibility.
6. Numerical fitting and validation scores are attached only to complete
   topology/function branches.
7. Test and private information never enters construction, fitting selection,
   or feedback.
8. Every accepted action, repair, rejection, hash, and beam decision is
   checkpointed and deterministically resumable.

## Milestone 1 — action schemas and pure compiler

Status: implemented behind a non-live additive API.

- Add a compact construction-intent decision containing only the public
  requirement/target anchors and the kind of repair or construction task.
- Add typed topology actions for nodes, interactions, direct measurements, and
  target mappings.
- Add typed functional actions for localized interaction functions and latent
  initializers.
- Maintain incomplete topology and functional drafts in the runtime.
- Apply actions without silently deleting or changing unmentioned structure.
- Canonicalize drafts, compute content hashes, and expose a checkpointable
  transposition table.
- Check topology/function compatibility before executable expansion.
- Provide a conditional beam selector that retains at most a configured number
  of functions per topology and reserves one viable child per topology before
  globally filling the beam.

Gate: unit tests cover successful construction, incomplete construction,
topology/function incompatibility, non-cascading deletion, order-independent
hashing, duplicate collapse, and topology-diverse beam selection.

## Milestone 2 — provider calls and action checkpoints

Status: implemented behind a non-live additive API.

- Add separate mechanism-intent, topology-action, and functional-action LLM
  calls with strict schemas.
- Build their prompts from a runtime-owned public problem contract and routed
  feedback.
- Cache every call by public contract, parent draft hash, action schema,
  provider settings, and retry context.
- Run lossless representation normalization before bounded provider retry.
- Checkpoint action transactions and their deterministic applications rather
  than only the resulting complete candidate.
- Persist transposition records so resume cannot regenerate an equivalent
  branch under reordered actions. Alpha-equivalent renaming detection remains
  a later semantic-deduplication extension; exact runtime identifiers are still
  meaningful because feedback and localized actions address them directly.

Gate: mocked integration tests prove checkpoint/resume and verify that a retry
cannot alter the parent draft or access test/private data.

## Milestone 3 — conditional topology/function beam

Status: implemented as a checkpointed deterministic controller; live search
integration remains in Milestone 4.

- Introduce a construction branch record with topology draft, topology
  commitment, zero or more functional drafts, completion state, and routed
  failures.
- Expand each surviving topology conditionally with up to `B_F` functional
  children.
- Reject incompatible functional children before fitting while retaining the
  topology branch.
- Fit and score only complete compatible children.
- Select a global beam of size `B`, capped at `B_F` children per topology and
  using a one-child-per-topology diversity reservation when capacity permits.
- Delay topology pruning until at least one bounded functional expansion has
  been attempted, except for deterministic topology-contract failures.

Gate: offline end-to-end tests include two topologies with multiple compatible
and incompatible functions and demonstrate deterministic selection under
resume.

## Milestone 4 — routed revision and bounded search policy

- Route target-path and graph-mechanism failures to topology actions.
- Route expression, integration, parameter, and worst-target fit failures to
  functional actions.
- Route cross-cutting judge feedback through an integrated-repair intent.
- Distinguish transport/schema failure, restricted-grammar failure, numerical
  failure, and exhausted structural compatibility.
- Preserve failed actions and failed complete branches as bounded negative
  memory without exposing provider internals.
- Start with beam search; do not introduce MCTS until the action state and
  reward routing are empirically stable.

Gate: a one-seed public smoke produces at least one complete branch, preserves
all action/beam/accounting records, and resumes without new provider calls.

## Milestone 5 — cluster pilots

1. Run local and Delta CPU tests for schemas, compiler, fitting, summaries, and
   deterministic resume.
2. Run one ACES H100 seed for provider transport and action quality.
3. Inspect the exact action sequence, compatibility failures, fitted models,
   feedback routing, tokens, latency, and GPU time.
4. Freeze the two-benchmark/three-seed development plan only after the one-seed
   gate passes.
5. Run the same frozen six tasks on ACES; use Delta for CPU fitting/evaluation
   shards where the artifact handoff is hash-verified.

This pilot remains development-only and keeps test trajectories closed.

## Later MCTS migration

The action drafts become MCTS states, typed transactions become edges, and
complete fitted models provide leaf rewards.  The deterministic compiler and
transposition table remain unchanged.  MCTS is therefore a controller change,
not another candidate-schema redesign.

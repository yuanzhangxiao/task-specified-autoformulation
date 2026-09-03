# Candidate identity and public target roles

This document defines the first representation milestone for staged model
construction. It does not change the candidate execution schema or fitting
backend. It adds separate identities for different scientific decisions and a
publicly justified check of whether a target is a dynamic state or an
instantaneous process.

## Three candidate identities

Each validated candidate receives three SHA-256 fingerprints computed from a
canonical, proposer-name-invariant projection:

- `topology_sha256` records typed declarations, signed additive dependency
  hyperedges, target mappings, initialization modes, and constraint kinds. It
  ignores operators, numeric constants, and fitted parameters.
- `functional_sha256` adds expression operators, fixed constants, and parameter
  placement while excluding optimizer bounds and initialization ranges.
- `executable_sha256` also includes bounds, initialization ranges, fixed initial
  values, and bounded-constraint values that can change a numerical run.

Public input and target names remain anchored. Proposer-owned state, process,
and parameter names are replaced by iteratively refined structural labels.
Commutative addition and multiplication are canonicalized. Therefore, merely
renaming a latent state or parameter does not create a new candidate.

These digests identify canonical summaries; they do not replace the candidate
payload. The color-refinement procedure is designed for practical duplicate
detection and is not claimed to be a complete graph-isomorphism algorithm.

## Failure-aware duplicate handling

The search controller records a failure class with every rejection and applies
the identity level appropriate to that failure:

- parseable deterministic schema, public-target, and other structural failures
  blacklist the functional identity;
- numerical-fit failures blacklist only the executable identity.

Consequently, a numerical failure may be retried with revised bounds or initial
conditions without claiming scientific novelty. An identical executable retry
is still rejected before consuming fitting time. Failed optimizer outputs and
sentinel losses are marked invalid and are not presented to the proposer as
estimated parameters or measured NMSEs.

Old checkpoints remain readable. A failed stored fit is interpreted as a
numerical failure when the checkpoint predates explicit failure classes.
Candidates whose expressions are not syntactically parseable have no canonical
identity and remain available only as ordinary recent-rejection feedback.

## Public target representation

Version 2 public target contracts may declare one of three representations:

- `dynamic_state`: the target is a stored quantity represented by an ODE state;
- `instantaneous_process`: the target is a rate or algebraic process;
- `unspecified`: the public prompt does not justify either claim.

The contract generator uses only explicit public descriptions. For example,
"plasma glucose mass" and "stored-quantity target" justify a dynamic state,
whereas "utilization rate" and "removal-rate target" justify an instantaneous
process. It does not inspect private equations or infer roles from hidden
channel semantics.

The runtime resolves identity aliases in observation mappings. A target mapping
that ultimately reaches a declared state is dynamic; an algebraic mapping or a
mapping to a non-alias process is instantaneous. A specified mismatch is a hard
public-target failure. Version 1 contracts continue to behave exactly as
before, and the version 2 bundle is stored separately.

This catches the general modeling error of integrating a quantity that the
public task defines as an instantaneous rate. It is not a glucose-specific rule
and does not assume a fixed list of biological mechanisms.

## Next representation milestones

The following work remains deliberately separate so its effects can be tested
independently:

1. immutable topology and functional candidate schemas for staged proposal;
2. graph-mechanism and annotation-compliance outputs as separate metrics;
3. profiled affine-weight fitting with latent basis or collocation variables;
4. routed local-revision actions and, after those primitives are stable, a
   small-budget tree-search policy.

# Phase C construction: dependent initializers and observation-prefix roots

Updated 2026-10-19. Implements the construction side of
`PHASE_C_OBSERVATION_PREFIX_INITIALIZATION.md` as an explicit opt-in policy.
Historical construction, saved results, benchmark prompts, and fitting policies
retain their existing behavior.

## Decisions and stage boundaries

At the initialization portion of **topology construction**, the proposer declares
one record per latent differential state. It chooses either an independent value
and its scope, or a dependent map and its state/public initial dependencies.

| Declaration | Meaning |
| --- | --- |
| `value`, `shared_training` | One unknown initial constant learned on training and reused |
| `value`, `trajectory` | One independent root per trajectory, inferred from an allowed prefix on new trajectories |
| `map` | A calculated initial value with shared coefficients and explicit dependencies |
| `known` | A fixed preparation value already established by the runtime, with provenance |

Observed differential states retain their first measured value. They cannot be
converted to free roots to avoid a model/data conflict. A dependent map is never
another independent fitted coordinate. For example,

    z1_j(0) = c,       z2_j(0) = a*z1_j(0) + b*y_j(0)

has shared unknowns c, a, b. Changing the first rule to a trajectory root replaces
c by xi_j; it does not make a or b trajectory-specific.

During **interaction/initialization formula construction**, the accepted graph is
fixed. The proposer supplies a qualitative role for each root and an analytic
expression plus local parameter roles for each map. Runtime-assigned `state_1`,
`public_1`, etc. keep state initials distinct from measured channels. Expressions
use the restricted parser. The runtime chooses numerical starting values and
lowers to Orion's `latent-initialization-graph-1` and `prefix-fit-problem-1`.

## The window is a fitting hyperparameter

Construction requires explicit authorization for observation-prefix roots, but
does **not** require a window duration. The proposer never chooses that duration.
A contract may carry a concrete `PrefixPolicy` or leave it unresolved. Fitting
experiments may select it using training/validation data; the complete policy and
post-window scoring horizon must be frozen before test evaluation. A missing
window policy prevents a forecast claim, not a structural dependency declaration.

## Validation and evidence

The dependency validator checks complete latent coverage, acyclicity, allowed
initial sources, fixed observed/known rules, scope authorization, and unchanged
state/channel/public-context identities. Rejected replies commit no partial graph.
Formula validation enforces the previously declared aliases and parameters.

Conditional evidence follows the full initializer dependency graph. Shared roots
and shared time-zero maps can still imply equal initials under matched permitted
readings. Any relevant upstream trajectory root makes that equality inference
unavailable. Such evidence is **deferred**, retaining its original witnesses and
assumptions; it is not counted as a scientific repair. Independent topology
diagnostics remain active. A prefix is not guaranteed to identify its roots.

The new staged-function result uses `scientific-staged-functions-3` and exports
the scoped M43 problem. Its top-level legacy `candidate` field is deliberately
empty: sending the compiled trajectory coordinates to the old shared-parameter
fitter would change their meaning. A compiled candidate remains available inside
the audited initialization result for inspection.

## Execution and limits

All requests use the existing cached transport, bounded attempts, retained costs,
and sealed checkpoints. Changed contracts, source declarations, code, or settings
require a fresh construction root. No fitting, model promotion, test access, or
automatic follow-up is part of this construction milestone.

The standalone `scripts/construction_initialization_graph.py` prepares a separate
dependency-only retest from saved initializer-choice cases, renders the prompts,
runs against a local server, and reports the results. This retest assesses whether
the proposer can use the new options; it does not run formula construction or
demonstrate improved prediction. Formula integration is verified separately with
synthetic cached-response tests against the actual M43 data structures.

The ACES bundle uses `start_initialization_graph_aces.sh` with `run`, `status`,
and `inspect`. It reuses the existing serving image and Python environment,
submits one H100 job, and permits at most two new calls per saved case. A confirmed
submission is not repeated; uncertain scheduler replies require inspection.
The initializer schemas are independent of numerical backends. The standalone
diagnostic still uses the existing shared construction client and its environment
dependencies; it does not install a smaller replacement environment.

Remaining limitations: acyclic explicit maps only; no implicit initial equations,
learned encoder, or posterior over ambiguous initials. Numerical backend support
is limited to Orion's opt-in M43 adapter. Full construction-to-fitting performance
and remote execution remain to be measured.

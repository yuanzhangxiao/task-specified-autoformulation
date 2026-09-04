# Staged proposer and runtime contract

This document records what the staged proposer actually sees and emits, what
the runtime derives, and which failures are repaired or retried. It describes
the opt-in staged path; the existing complete-candidate search remains a
separate path until staged search integration is complete.

The authoritative machine-readable provider schemas are:

- [`proposed-topology-candidate-v2.schema.json`](../schemas/proposed-topology-candidate-v2.schema.json)
- [`proposed-functional-candidate-v1.schema.json`](../schemas/proposed-functional-candidate-v1.schema.json)

The enriched handoff and final executable schemas are:

- [`topology-candidate-v2.schema.json`](../schemas/topology-candidate-v2.schema.json)
- [`functional-candidate-v1.schema.json`](../schemas/functional-candidate-v1.schema.json)
- [`candidate.schema.json`](../schemas/candidate.schema.json)

All schema objects reject undeclared fields. Identifiers must match
`^[A-Za-z][A-Za-z0-9_]*$`, strings and collections are bounded, and nonfinite
numbers are rejected.

## Public runtime context

Before either provider call, the runtime constructs a `ValidationContext` from
the public benchmark split:

```json
{
  "targets": ["y"],
  "lagged_targets": [],
  "auxiliaries": ["measured_x"],
  "external_inputs": ["u"],
  "fixed_covariates": [],
  "unavailable_observed_channels": [],
  "forbid_latent_states": false,
  "time_symbol": "t",
  "forcing_bounds": {}
}
```

Targets and auxiliaries are measured public data channels. External inputs,
fixed covariates, and permitted lagged targets are supplied forcing, not
modeled states. Unavailable channels are never exposed as usable symbols.

## Stage 1: topology proposal

The provider returns only this compact shape (fields with defaults may be
omitted):

```json
{
  "schema_version": "proposed-topology-candidate-2",
  "candidate_id": "candidate_1",
  "parent_candidate_id": null,
  "change_summary": "Add one latent mediator.",
  "states": [
    {"name": "x", "mechanisms": ["input_response"]},
    {"name": "z", "mechanisms": []}
  ],
  "processes": [
    {"name": "predicted_y", "mechanisms": []}
  ],
  "interactions": [
    {
      "interaction_id": "u_to_x",
      "target": "x",
      "target_kind": "state_derivative",
      "sources": ["u"],
      "polarity": "additive",
      "mechanisms": ["input_response"],
      "description": "Input drives x."
    },
    {
      "interaction_id": "x_to_y",
      "target": "predicted_y",
      "target_kind": "algebraic_process",
      "sources": ["x"],
      "polarity": "additive",
      "mechanisms": [],
      "description": "Measurement function."
    }
  ],
  "state_measurements": [
    {"state": "z", "channel": "measured_x", "unit": "unspecified"}
  ],
  "target_mappings": [
    {"channel": "y", "source": "predicted_y", "unit": "unspecified"}
  ]
}
```

The topology provider does **not** emit observed/latent labels, equations,
interaction functions, parameter declarations, parameter ranges, parameter
scopes, external-symbol declarations, or fitted numbers.

### Deterministic topology rules

The runtime enforces the following rules before the functional-form call:

1. State, process, interaction, measurement, and target identifiers are unique
   in their respective namespaces. State and process names cannot collide.
2. Every interaction source must be a generated state/process or an available
   public forcing/time symbol. Referenced external symbols are derived locally;
   the provider cannot declare extra data access.
3. A `state_derivative` interaction must target a declared state. An
   `algebraic_process` interaction must target a declared process.
4. Every state has at least one derivative interaction and every process has at
   least one defining interaction.
5. Dynamic feedback loops are allowed. A cycle containing only instantaneous
   algebraic processes is rejected because it has no evaluation order.
6. `target_mappings` must cover every public target exactly once and no other
   channel. Its source must be a generated state or process.
7. `state_measurements` may use only supplied auxiliary channels and declared
   states. Inputs and covariates remain forcing rather than measurements.
8. Effective observability is derived from identity measurement, not from a
   provider label:
   - `{"channel":"y", "source":"x"}` directly observes state `x`;
   - `{"state":"x", "channel":"measured_x"}` directly observes `x`;
   - mapping `y` from an algebraic process does not observe that process's
     internal states.
9. A state cannot be bound to conflicting direct measurement channels.
10. If the explicit no-latent ablation is active, any state without a direct
    identity measurement is rejected.

The runtime then adds units/descriptions, derived external symbols, derived
state kinds, and a canonical topology commitment SHA-256. The SHA-256 is a
content identity for immutable handoff; it is not a scientific score.

`polarity` is the outer operator used when terms are assembled. For example, a
subtractive interaction with function `k * (x - x0)` becomes
`-(k * (x - x0))`. It does not assert that `(x - x0)` is nonnegative.

## Stage 2: functional-form proposal

The provider receives the immutable enriched topology and returns:

```json
{
  "schema_version": "proposed-functional-candidate-1",
  "candidate_id": "candidate_1_functions",
  "parent_candidate_id": "candidate_1",
  "change_summary": "Assign functions to the committed graph.",
  "interaction_functions": [
    {"interaction_id": "u_to_x", "expression": "gain * u"},
    {"interaction_id": "x_to_y", "expression": "theta * x"}
  ],
  "parameters": [
    {"name": "gain", "role": "nonnegative_coefficient"},
    {"name": "theta", "role": "scale"}
  ],
  "latent_initials": [
    {"state": "x", "initial": {"fixed_value": 0.0, "expression": null}}
  ]
}
```

The currently accepted parameter roles and runtime-derived domains are:

| Role | Derived domain |
|---|---|
| `coefficient` | real |
| `nonnegative_coefficient` | nonnegative |
| `rate`, `time_constant`, `scale`, `positive_shape` | positive |
| `offset`, `shape` | real |

The provider supplies no magnitude ranges or scopes. The runtime sets parameter
scope to global and owns numerical coordinates, starts, bounds when justified,
and fitted values.

### Deterministic functional and expansion rules

1. The functional artifact must bind every committed interaction exactly once,
   with no added interaction.
2. Each restricted expression must use exactly the non-parameter source set of
   its interaction. A function cannot silently add or remove a graph edge.
3. Expressions pass the existing AST-based restricted grammar; arbitrary code,
   attribute access, indexing, and undeclared functions/symbols are rejected.
4. Every used parameter is declared exactly once. Its domain is derived from
   its role.
5. Because topology owns the outer plus/minus operator, a scalar edge weight in
   a staged interaction cannot use the signed `coefficient` role. It must use a
   suitable nonnegative/positive role. Signed offsets and shapes remain valid.
   This is a coefficient-domain check, not a proof of global monotonicity.
6. The functional topology commitment must exactly match the validated
   topology artifact.
7. `latent_initials` currently supplies exactly one fixed or safe analytic
   initializer for every runtime-derived latent state, and none for observed
   states. Directly measured state initials are bound to their public channel.
8. The runtime assembles interaction terms by polarity, generates state RHSs
   and processes, adds target and auxiliary observation mappings, and runs the
   ordinary complete `CandidateValidator` again.
9. The complete validator checks namespace closure, expression safety,
   initialization causality, channel availability, parameter usage/domain,
   algebraic evaluation order, and public forcing bounds. Numerical fitting is
   reached only after these checks pass.

Point 7 is the current implementation, not the final latent-initial strategy.
Fitting or encoding latent initial conditions from training data without
leaking validation/test outcomes remains a separate numerical-design milestone.

## Observability examples

Direct measurement of a modeled state:

```text
dx/dt = f(x, z, u)
dz/dt = g(z, u)
y <- x
```

Here `x` is observed through target channel `y`; `z` is latent.

Parameterized observation function:

```text
dx/dt = f(x, u)
predicted_y = theta * x
y <- predicted_y
```

Here `x` remains latent and `theta` is fitted. The algebraic process is the
target source, so the target does not reveal `x`.

A target state may depend on a latent state:

```text
dx/dt = f(x, u)
dz/dt = g(z, x, u)
z_data <- z
```

Here `z` is observed and `x` is latent. Target status and state observability
are therefore related only through a direct identity mapping, not by a rule
that every target-named variable is observed or every non-target variable is
latent.

## Repair, retry, cache, and feedback behavior

There are three distinct behaviors:

1. **Schema/provider retry.** Invalid JSON, strict-schema failures, and
   deterministic post-schema failures produce a sanitized typed diagnostic.
   The same logical call is retried within the configured attempt limit with a
   contract-repair instruction. Every provider attempt is counted and logged.
2. **Deterministic enrichment.** Runtime-owned metadata is added, but the staged
   path does not silently rewrite scientific nodes, edges, expressions, or
   mappings. Observed/latent metadata on historical complete-candidate payloads
   is canonicalized from public identity mappings before fitting and feedback.
   A bad scientific structure must be regenerated or revised by a later
   proposer action.
3. **Search feedback.** Once a candidate is executable, public evidence is
   routed by responsibility: target-contract and graph-mechanism failures to
   topology; expression/annotation failures plus fit, integration, and worst
   validation-target evidence to functional form; scientific-judge missing
   requirements and actionable edits to integrated repair. Each stage sees
   only its routed view.

Request hashes include provider, model, role, prompts, response schema,
provider options, validation context, and topology commitment where relevant.
Successful calls are cached, all calls are append-only logged, and validated
stages are checkpointed by a run-scoped input hash for deterministic resume.

## Current limits and historical scope

- The staged path has passed local construction/validation smokes but is not yet
  connected to the production search controller or evaluated in the planned
  two-benchmark, three-seed experiment.
- Its v1 topology prototype inferred observability from direct target mappings
  only. V2 corrects this by representing auxiliary state measurements
  separately. The established complete-candidate compiler already inferred
  effective observability from both target and auxiliary identity mappings, so
  this staged-v1 issue does not by itself explain all earlier search failures.
- A provider-declared `StateKind` remains in historical complete-candidate
  artifacts, but effective numerical observability is runtime-owned and is
  inferred from public identity mappings. At the runtime boundary the label is
  canonicalized as metadata, so a wrong label no longer determines the
  no-latent ablation verdict or downstream mechanism/complexity feedback.
- Polarity plus a positive scalar role does not prove a nonlinear interaction is
  sign-definite or monotone over its full domain. Such scientific properties
  require a separate certified constraint or evaluator.

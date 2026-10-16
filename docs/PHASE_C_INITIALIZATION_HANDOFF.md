# Initialization information at the topology boundary

This opt-in offline milestone implements typed initializer-dependency records and
qualifies matched-trajectory checks under an explicit runtime information contract.
It does not select formulas, change benchmark artifacts, fit a model, call a
proposer, or change historical construction policies. The prompt preview is the
proposed end-of-topology handoff; it is not yet a new live constructor stage.

## Where initialization already happens

The complete function-construction pipeline first constructs interaction laws.
`search/staged_function_runner.py` then handles latent initial conditions. Under
`causal_training`, `search/causal_initialization.py::construct_initializers` asks
for each latent state either a shared unknown value or a common causal map of the
allowed initial public measurements, inputs, auxiliaries, covariates and time.
Map coefficients are shared across trajectories and fitted on training. Directly
observed state initials are runtime-bound. The older function-stage path requests
one `LatentInitialReply` per latent state.

Recent Phase C topology-only experiments deliberately stop before this work.
Their prompts exclude initializer formulas, and their compatibility evidence has
no proposed initializer for newly created latent states. Thus unknown-initializer
deferrals in those results are expected; they are not evidence of successful repair.

The new handoff moves only **dependency information** earlier: after the whole
variable/topology draft exists, before candidate-specific compatibility review.
A later executable initializer must satisfy this declaration or explicitly revise
it and trigger revalidation. Formula/parameter construction is still later work;
that consumer integration and a live proposer confirmation are not implemented in
this offline milestone.

## Minimal record

A record names one differential state and one existing initial-rule mode:

- `shared_parameter`: a common unknown initial value, fitted once on training.
- `common_map`: a shared deterministic map of the listed initial public quantities.
- `unknown`: an explicit unresolved choice, never an acceptance shortcut.
- `observed_initial`: an existing frozen runtime rule, not a new proposer choice.

For example, `z(0)=g(a(0),u(0))` is represented by
`{"state":"z","rule":{"mode":"common_map","arguments":["a","u"]}}`.
This selects dependencies, not an executable function or numerical initial value.

Known runtime rules cannot be overridden. The response is a complete handoff:
missing states remain unresolved, duplicate state records are rejected, and records
for removed/algebraic states are stale. Arguments must be permitted recorded t0
quantities; generated-only states, hidden channels, future values and trajectory
identifiers are unavailable. The handoff receipt is bound to the exact topology
and training-evidence hashes. No initializer choice is inferred from a variable's
name or description.

## Preserving conditional trajectory evidence

The evaluator keeps two assessments separate:

1. **Declared-only:** known runtime rules plus explicit, validated handoff choices.
2. **Family-qualified:** for unresolved states, condition on every permitted initial
   quantity under the explicit shared, complete, recorded-t0 runtime contract.

The second is a mathematical family check, not a declaration that every state uses
every quantity. If two trajectories match on all permitted initial information,
then any shared deterministic initializer in that family must initialize them
identically. With identical complete forcing and unique shared dynamics, the
candidate subsystem must predict identical target trajectories. New latent states
or internal feedback alone cannot remove this restriction.

Different permitted initial values or forcing histories can remove this necessary
condition without establishing predictive accuracy. Unknown information scope,
unlisted prehistory, uncertain conditioning or missing topology cause abstention.
The original noise/error bounds, interpolation, shared-parameter and uniqueness
requirements remain in force. Exact reproduction and acceptable approximation are
reported separately; a small observation difference is not automatically a major
scientific failure.

Original evidence remains immutable. Derived conditioning has its own digest and
provenance. Conflict tracking retains the original data identity while recording
the initializer-hypothesis digest separately. Previous target/property conflicts
are retained as tracked conditional tests, not an exhaustive database of every
possible trajectory pair. No structural eligibility or historical result changes.

Saved paired witnesses are also rechecked against the current graph through their
original observation time. A newly added input that differs only later cannot
explain an earlier mismatch. Coordinates are validated and observations reread
from sealed evidence, rather than trusted from saved question text. Missing
assumptions or uncertain conditioning retain a deferred result; they do not clear
the previous conflict. A verified difference in permitted initial information or
forcing before the witness can remove that particular equality restriction.

## Offline commands

Use the repository environment, or the existing Jetstream construction venv:

```bash
export PYTHONPATH="$PWD:$PWD/src"
python scripts/audit_initialization_handoff.py \
  --source /path/to/saved-actionable-repair-v2 \
  --root /path/to/initialization-information-audit-v1 \
  --initial-information causal_runtime
```

`causal_runtime` explicitly reuses `causal-initializer-construction-2`'s initial
information restriction. Without that flag, the scope is unknown and the additional
family test is unavailable. Results include `summary.json`, per-model sealed
assessments, complete unexecuted requests, and `PROMPTS.html`. Repeating the same
command verifies identical outputs. Changed source/choices/policy require a new
output directory. No GPU or running vLLM server is required.

Explicit future proposer choices can be supplied with `--handoffs DIRECTORY`.
Each sealed `TASK.json` must contain `topology_sha256`, `evidence_sha256`, and
`handoff`; stale receipts are rejected. This is an offline consumer interface,
not permission to fabricate missing proposer responses.

## Reference expressivity audit is separate

The researcher-only reference audit counts physical additive ODE-term occurrences,
expands algebraic aliases for effective dependencies, and excludes fixed parameters
from the dynamic-argument count. It finds nonseparable joint laws in the full
Dalla Man generator, CSTR and the selected alien device; the basin rating laws are
unary in depth. Physical grouping changes denominators: a linear difference can
be split, while a product cannot generally become a direct sum of unary laws in
the same coordinates. Additional algebraic intermediates can change expressivity.

The Dalla full-generator count is not a mandated reduced candidate size; easy
supplied auxiliaries hide some joint laws from the target-facing equation.
Reference equations, counts and mechanism labels never enter the new proposer
requests or compatibility evidence. No minimum-arity quota is added.

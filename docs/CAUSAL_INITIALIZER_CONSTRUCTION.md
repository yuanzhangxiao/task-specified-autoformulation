# Causal initializer construction

This is milestone 2 of `PREFIT_MILESTONES.md`. The function constructor now has
an opt-in `initialization_policy="causal_training"`, versioned as
`scientific-staged-functions-2`. It replaces the old fixed-number initializer
call with a scientific policy choice. The existing fitter implementation and
shared-training initialization adapter are unchanged.

The runtime selects a latent state. The provider may return:

```json
{"initial":{"mode":"shared_value","role":"coefficient"}}
```

or, when the initial observations support different initial values:

```json
{"initial":{"mode":"causal_map","expression":"a+b*v01","parameters":[{"name":"a","role":"coefficient"},{"name":"b","role":"coefficient"}]}}
```

The map coefficients are fitted once on training and frozen for validation and
testing. Each trajectory evaluates the same map on its own initial public
observations, initial input/auxiliary values, numeric fixed covariates and time.
No trajectory ID, later measured target, future input schedule, generated state,
or undeclared equation parameter is an initializer feature. Directly observed
state initials are runtime-bound to their first measurements. The provider does
not emit optimizer guesses, parameter values or numerical ranges.

A shared value remains an explicit common-preparation hypothesis. A map is not
forced when public information cannot justify its form. Identical available
initial information produces identical deterministic initials. Arbitrary
unobserved preparation differences remain a limitation; independent training
initials or a distribution over latent states require a separate protocol.

## Construction and fitting handoff

`construct_initializers` validates every proposed state-local policy by applying
the existing restricted lowering adapter. Accepted siblings survive an invalid
later state. Bounded retries, consumed request costs, source/config fingerprints
and exact state digests survive resume. Repeated cached requests are deduplicated
in cost accounting. Missing or changed checkpoints/caches fail explicitly.

The function draft uses temporary zero boundaries solely to compile its equations
before this stage. No causal-training candidate is complete until every latent
boundary has a validated policy. A failed policy leaves no deployable placeholder
candidate. The resulting canonical candidate contains initializer parameters and
their actual initial expressions.

The result's `initialization` artifact records the original candidate/context,
the typed plan, canonical lowered candidate/context, runtime guesses and audit.
`compile_initialization_result` reconstructs the artifact against the frozen public
context and rejects inconsistent or incomplete data. There are two valid fitting
entry points:

- Compile the original `base_candidate` with `base_context`, then pass `plan` as
  `initialization_plan` to the existing fitter. The adapter lowers exactly once.
- Use `compile_initialization_result` to obtain the already lowered model and use
  its `guesses` as numerical starts. Do not apply the plan to that model again.

The legacy fixed-initial function handoff and older fitting campaign remain
separate. Do not drop the initialization artifact or pass a lowered map candidate
through a legacy loader that disables fitted-initialization expressions.

The six-topology freeze is exposed through
`configs/staged_causal_initialization_v1.json`; it pins
`scientific-staged-function-prefit-handoff-2`. Its prefit certificate reconstructs
the initializer against the supplied frozen public context. It retains the exact
source topology's serving identity; this is not permission to override an image
hash or replace the SIF. The later matched construction campaign will separately
pin its serving and fitting configuration.

## Bounded assignment normalization

Constructor protocol `causal-initializer-construction-2` accepts a harmless
assignment wrapper when its meaning is fixed by the selected initialization
stage. For a selected latent state `I_delay`, these forms have the same meaning:

```text
I + gain*meal_event_g
I_delay = I + gain*meal_event_g
I_delay_0 = I + gain*meal_event_g
```

The `_0` alias is accepted only when it is not already a state, process,
parameter, public channel, time symbol or map-local parameter. Runtime parses
the text without executing it, checks that it contains exactly one simple
assignment to the selected state or its available alias, and retains the RHS.
Parentheses and multiline grouping survive extraction. The original cached
response stays intact. The attempt records the exact supplied expression,
normalized RHS, target and `initial-assignment-normalization-1` policy, even if
subsequent RHS validation rejects the proposal. Canonical choices and plans use
the RHS; acceptance still requires the full restricted grammar, available-symbol,
parameter and domain checks.

Differential versus algebraic representation is selected earlier by topology.
This call supplies an initial value for an already selected dynamic state; it
does not define an algebraic identity for the entire trajectory. Derivative
assignments, other targets, chained assignments, multiple statements, augmented
assignments and indexed targets are not silently converted. Their diagnostics
identify the expected initial-value quantity and ask the proposer to preserve
the intended formula while correcting notation. There is no instruction to
simplify a rejected formula into an identity map.

This milestone applies only to the causal initializer constructor. Whole-equation
and per-interaction function responses have different meanings, including outer
sign assembly, and retain their existing contracts. The shared expression parser,
fitter, benchmark prompts and data are unchanged. Changed constructor protocol,
normalization policy and source hashes prevent resuming an older construction
checkpoint with these semantics. Previously frozen ACES results remain historical
results; a new acceptance rule does not retroactively change their success rate.

## Verification commands

From this commit's clean checkout and configured Python environment:

```bash
PYTHONPATH=src python -m pytest -q tests/test_causal_initialization_construction.py tests/test_staged_function_runner.py tests/test_staged_function_prefit_campaign.py tests/test_fitted_initialization.py
PYTHONPATH=src python scripts/smoke_causal_initialization.py
ruff check .
```

The smoke supplies `m = a+b*v01`, checks normalization on the first attempt,
and uses the prescribed synthetic scientific map with the real fitter. It
checks recovery of `1 + 2*v01(0)`, distinct initial values on unseen validation
initial observations, and exact experiment-stage resume with no repeated provider
or fit execution. It does not measure live proposer accuracy or benchmark recovery.

Verification at milestone commit `b14be41`: 80 focused tests passed; the real
fitter/resume smoke and Ruff passed. Full pytest reported 1,454 passed, 39 failures
from missing benchmark fixtures, and three optional skips. The 39 failure
identities match the pre-existing missing-fixture set; no data was copied or
modified to conceal them.

Verification for constructor protocol 2: 89 focused tests passed, and the full
suite passed 1,563 tests with three optional PyTorch skips. Ruff, the causal
initializer fitting/resume smoke, and the matched-construction smoke passed.
The tests cover matching assignments, preserved formulas and grouping, immutable
dynamics, exact cached responses and resumed costs, occupied aliases, derivative
and ambiguous assignments, and normalized RHS expressions that still fail
validation. This is an engineering check; no new live ACES experiment was run.

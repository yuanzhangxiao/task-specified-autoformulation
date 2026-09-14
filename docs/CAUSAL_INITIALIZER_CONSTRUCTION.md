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

## Verification commands

From this commit's clean checkout and configured Python environment:

```bash
PYTHONPATH=src python -m pytest -q tests/test_causal_initialization_construction.py tests/test_staged_function_runner.py tests/test_staged_function_prefit_campaign.py tests/test_fitted_initialization.py
PYTHONPATH=src python scripts/smoke_causal_initialization.py
ruff check .
```

The smoke uses a prescribed synthetic scientific map and the real fitter. It
checks recovery of `1 + 2*v01(0)`, distinct initial values on unseen validation
initial observations, and exact experiment-stage resume with no repeated provider
or fit execution. It does not measure live proposer accuracy or benchmark recovery.

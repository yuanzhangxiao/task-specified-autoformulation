# Optional investigation of additive restrictions

`fitting/interaction_dependency_advice.py` is an opt-in, read-only extension to
the [M37 packet](PHASE_C_FITTING_INTERACTION_ADAPTER.md) and
[M38 feedback](PHASE_C_FITTING_FEEDBACK_DRAFTS.md). It is not wired into a fitting
or construction runner. Existing numerical policies, model selection, prompts
and fitter defaults are unchanged. There are no additional numerical or LLM
calls.

The helper can ask a bounded investigative question when a verified training
discrepancy remains, M38 permits investigation using a matching retained-point
stationarity or restricted-output-optimum witness, and a displayed law affecting
that target separates at least two dynamic arguments additively. Accurate fits,
unverified rollouts, incomplete or improving searches, missing measured
discrepancies, and unresolved or already joint expressions yield no question.
A local stationary point or restricted fitted optimum is scoped to its candidate
and search domain; neither establishes global model adequacy or a missing
dependency.

The current evidence packet contains no controlled joint-effect test. Therefore
this helper asks whether training contrasts can support such an investigation;
it never diagnoses missing arguments, names a proposed new source, changes an
equation, or requests an arity quota. Parameter fitting, permitted initial rules,
limited excitation, measurement variation and existing state feedback remain
alternatives. Any eventual hypothesis needs the authoritative constructor
contract, a predicted observable change, a comparison against continued fitting
under predetermined equal budgets, and preservation of all targets and shared
consumers. Unsupported revisions retain the incumbent. The question limit bounds
the review, not the desired model complexity.

## API

Use the verified entry point with existing saved artifacts:

```python
from pathlib import Path
from autoformalism.fitting import interaction_dependency_advice as dependency

result = dependency.review(
    Path("/path/to/verified-diagnostics"),
    Path("/path/to/interaction-packet"),
    dependency.Policy(enabled=True, maximum_questions=1),
)
```

The default policy is disabled. `review` first calls M37's `completed` verifier,
which reconstructs and verifies the immutable packet against its diagnostic
source. The returned review records the source packet hash and
`source_verification="verified_snapshot"`. Altered evidence is rejected.
`draft(packet, policy)` is a deterministic pure projection for already verified
callers and unit tests; it explicitly records `source_verification="caller_required"`
and does not establish provenance. No publication, scheduling, transition or
automatic refit is included. A future controller must persist its chosen policy
and the verified review alongside its run before using a question in a request.

## RHS and initial information are different

For `dz/dt = f(z)` with `z(0) = s*y(0)`, the trajectory depends on the initial
reading of `y`; the RHS still has only `z` as a dynamic argument. The initial map
does not supply later readings `y(t)` to `f`. Initialization is inspected and
reported separately, consistent with `apply_initialization_plan` preserving RHS
and observation expressions. State dynamics are not substituted into other
states' RHS expressions.

Algebraic process aliases are expanded for argument inspection. For example,
`q = u-y` is additive, whereas `tanh(q)` with `q = u+y` is conservatively joint.
Fixed parameters and fixed covariates do not count as dynamic arguments:
`k*max(y-c, 0)/A` is unary in `y` when `c` and `A` are fixed. This is a bounded
syntactic inspection of current executable expressions, not a general symbolic
separability proof or a statement of which function family the constructor
allows. Nonlinear compositions and feedback can create joint behavior even when
individual components are unary. The packet does not carry the configured time
symbol, so unclassified symbols remain unresolved rather than being guessed as
time; this can conservatively withhold advice for explicitly time-dependent laws.

## Focused verification

```bash
.venv/bin/python -m pytest -q tests/test_interaction_dependency_advice.py
.venv/bin/ruff check src/autoformalism/fitting/interaction_dependency_advice.py \
  tests/test_interaction_dependency_advice.py
```

Tests cover verified read-only use and tampering, disabled/withheld/inapplicable
routes, scoped witnesses, citations, process composition, cycles and unsafe
expressions, fixed covariates, initial-time boundaries, and bounded questions
that preserve shared consumers. Synthetic mutations are used only to test the
pure projection and are never published as verified numerical evidence.

# Returned-construction fitting qualification

Implemented 21 October 2026. This is the first numerical qualification of all
eight complete packets returned by `saved-function-handoff-repair-1`. All eight
describe the same Dalla Man T2-easy public task: two topology seeds and their
accepted initializer proposals. They are not eight different benchmarks.

## Why EGP and Uii were blocked

The benchmark expressly supplies its auxiliary trajectories over the observation
horizon. The older observation-prefix adapter only authorized future external
inputs and fixed covariates. Its refusal of `EGP` and `Uii` was an interface
limitation, not a scientific rejection or a prohibition in this public task.

`PrefixPolicy.future_auxiliaries` now explicitly authorizes such trajectories.
It is separate from `future_inputs`; channel roles do not change. The loader
verifies the public authorization, channel roles, exact training observations,
historical accepted replies and fitting packets before freezing the experiment.
No benchmark prompt or data are edited. An empty auxiliary allowlist preserves
the previous policy, serialized payloads and their hashes.

Prefix inference receives auxiliary samples only up to the cutoff. Forecasting
receives the authorized full auxiliary schedules, without suffix target values.
This produces a forecast conditional on supplied auxiliary observations. It is
not an autonomous forecast of those auxiliary quantities or evidence that they
would be available in a different deployment scenario.

## Frozen population and numerical policy

- Retain all eight complete models, including topology seed 2 / proposal seed 2.
  No numerical or qualitative preselection is applied.
- Preserve equations, parameter domains, shared gains, parameter starts and
  initializer formulas/scopes. Scale redundancies remain visible.
- Use all 16 public training and four public validation trajectories.
- Fit shared dynamics/map parameters and any training trajectory roots on
  training data only, using the existing prefix execution kernel.
- Freeze shared parameters before validation. For the five root models,
  estimate only the trajectory-specific roots from the allowed early window.
  The three analytic-map models have no validation root search.
- Use a predeclared eight-minute observation window for this diagnostic and
  score every model strictly after that same cutoff. This is not an optimized
  duration or a claim that eight minutes is sufficient.
- Retain complete training reconstruction errors and post-cutoff reconstruction
  errors separately from held-out forecast errors. Training roots use the full
  training trajectory; training reconstruction is not a causal forecast.

This is an opt-in observation-prefix experiment authorized separately from the
historical benchmark protocol that prohibited fitting validation initials. Its
scores must be labeled accordingly. It does not revise or replace historical
benchmark scores. The window can be changed only in a new frozen experiment.

`configs/construction_fit_qualification_v1.json` freezes one declared start per
model. Training allows 300 oracle calls / 600 numerical-active seconds, with
separate independent verification. Each validation root inference allows 100
calls / 60 numerical-active seconds. Independent forecast and reconstruction
verification have their own limits. Finite inclusive deadlines also account for
setup and checkpoint overhead; cooperative timing cannot forcibly interrupt
every symbolic operation.

## Metrics and interpretation

NMSE uses training-channel standard deviations (floor `1e-12`), averaging squared
normalized errors across samples and targets. The report includes per-target
errors, validation coverage, numerical verification, conditional rank diagnostics,
median and unscaled MAD across available model scores. Failures stay in the
denominator and have no fabricated error value. The descriptive `poor_fit` label
means NMSE exceeds the configured threshold of 1; it never removes a model.

Numerical failures, incomplete operations and poor predictive fits remain
distinct. An unavailable score cannot establish an inadequate scientific model.
A bounded, single-start fit does not establish the best attainable fit. Successful
integration or agreement between integrators does not establish mechanism
compliance. Independent scientific assessment remains a separate next step.

No model is automatically discarded, promoted or revised. No LLM requests,
test-data access, hidden state labels or reference equations are used.

## Resume, integrity and execution

Preparation seals the complete public inputs, original packet hashes, policy,
source identity and numerical runtime. The ACES launcher separately binds the
selected Python/container and scripts. Every numerical operation writes a durable
reservation and progress before completion. Completed operations are reused.
An interrupted operation is reported with incomplete accounting and is not
silently restarted with a renewed budget. Reports and inspection do not fit.

The ACES launcher requests eight CPU tasks with at most four running concurrently,
one CPU and 8 GB per task, up to one hour. A report job runs after the array,
including failed tasks. There is no GPU request and no provider dependency.
Submission intents prevent duplicate submission after an uncertain scheduler
reply. The native Python preflight checks `ctypes`, scientific imports and the
real campaign import, with an ordered compiler/Python module fallback. An
explicit existing CPU container can be selected instead.

The original `0879f4d` launcher unnecessarily imported and pinned SymPy, which is
not used by this fitting path. The launch-only correction removes that check
and pins the actual CasADi dependency. A failure at the import preflight occurs
before preparation or scheduler submission; it starts no fitting jobs. Apply
the verified launcher patch only before the launch identity or submission intent
has been written. Models, source data, numerical policies and the campaign's
scientific source hash are unchanged by this script-only correction.

From the delivered pinned bundle on ACES:

```bash
bash run_fit.sh run
bash run_fit.sh status
bash run_fit.sh inspect
```

The default output root is
`/scratch/group/p.nairr260351.000/u.yx126462/phase-c-construction-fitting-v1`.
`SUMMARY.md`, `report/summary.json` and `inspection.tar.gz` retain all eight
models, operation outcomes and costs. A second `run` reuses the confirmed
submission; it does not create another array. An uncertain scheduler response
requires reconciliation of the saved intent and scheduler state before retry.

The independent CLI is `scripts/qualify_construction_fitting.py`, with `prepare`,
`verify`, `run --index`, `report`, `inspect` and synthetic `smoke` commands.

## Source record

- Returned archive SHA-256:
  `f324c78df63e64297a85a537eb0175c3064133711cc1b41b1e0b16e206658702`.
- Historical repair plan:
  `097374b30540bbf86fbb7088b9b070cc196c566b530328c0fb78418d381ea9dd`.
- Public cell: `phase_c_dalla_man_t2_canonical_named_easy_rates_v1`.
- Historical costs stay in source provenance and are not charged as new calls.

The delivered bundle contains the returned source archive and only the four
named public files: specification, proposer prompt, training and validation.
Generated data, inspection results and source bundles are not committed to Git.

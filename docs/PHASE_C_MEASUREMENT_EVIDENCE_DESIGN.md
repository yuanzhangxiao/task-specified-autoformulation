# Phase C measurement and residual evidence contract — proposed

Prepared 13 October 2026 after M33 submission. This is a design for subsequent
reviewed increments, not an implemented schema, changed fitter, experiment result,
or assessment of M33 completion. It implements the preparation scope in the
[Vega handoff](VEGA_FALLIBLE_SPECIFICATIONS_HANDOFF.md#phase-c-preparation-while-construction-remains-at-topology).

The first increment should preserve a public measurement manifest and its exact
relationship to training residuals. A later numerical increment can consume fixed,
positive, externally justified observation standard deviations. Neither increment
should infer measurement defects, revise specifications, or turn numerical rank
into calibrated confidence. The noise-aware design and controlled-study protocol
are separate companions; this document defines their data boundary.

## Existing interfaces and the gap

| Repository interface | Existing behavior to preserve or explicitly extend |
| --- | --- |
| `src/autoformalism/schemas/public_fitting.py`: `PublicTrajectory`, `PublicSplit`, `PublicFitRequest` | Finite aligned public arrays, increasing time, train/val names, `public-fit-1`; no units, missing readings, calibration, or stochastic observation model. |
| `src/autoformalism/fitting/public_fitting.py`: `pack_split`, `_lower`, `_bundle`, `prepare_fit` | Omits derivative labels, verifies public channels, lowers shared initializers, hashes actual content/settings/source; changed content cannot resume an old fit. |
| `src/autoformalism/fitting/initialization.py`: `apply_initialization_plan` | Directly observed states start at the first observation. Latent values/maps use shared training parameters. This is not propagation of first-reading measurement error. |
| `src/autoformalism/fitting/confidence_checks.py`: `TrainingProblem`, `Rollouts.evaluate` | Training-only allowlist; residuals/Jacobians use `scale_for(train)` target scales. These scales are not measurement sigma. Only global coefficients and explicit shared initializer parameters are accepted. |
| `src/autoformalism/fitting/fitting_feedback.py`: `Binding`, `Residual`, `LocalSensitivity`, `FeedbackPacket` | Candidate/point-bound residual and joint coefficient/initializer evidence; `specification_verdict=not_assessed`, no confidence probability. |
| `src/autoformalism/fitting/staged_feedback.py`: `StageRecord`, `FeedbackPacket`, `training_problem` | Independently checkpointed output/sensitivity stages, missing-probe reasons and costs, qualified piecewise scope. |
| `src/autoformalism/fitting/portable_feedback.py`: `freeze_plan`, `verify_packet`, `load_snapshot` | Exact executed perturbations/artifact identity; restricted numerical-summary tolerances and subspace-aware checks; no regenerated experimental direction. |
| `src/autoformalism/fitting/feedback_supplement.py`: `Identity`, `SupplementPlan`, `SupplementResult` | Explicit incremental allowance, successful-operation reuse, original failures/costs retained. |
| `src/autoformalism/schemas/residual_feedback.py`: `ResidualEvidence`; `src/autoformalism/search/residual_evidence.py`: `build_residual_evidence`, `validate_residual_evidence` | Training-only measured mismatches, deterministic details/omission counts, exact sample indices and explicit interpretation limits; normalization uses training variability. |

The basin public prompt `configs/detention_prompts/coupled.md` specifies minutes,
depth/flow units, linear input interpolation, initial readings, open rollouts and
no validation-specific initial fitting. Those semantics are evidence to retain;
the proposed noisy-data profile cannot silently change that finalized prompt.
The generic one-step defaults described in `PIPELINE_DESIGN.md` are not the open
rollout policy of this opt-in Phase C handoff.

## Proposed closed record vocabulary

The following is documentation-level Pydantic-style notation, not runnable code.
All records inherit the behavior of `schemas/base.py:StrictSchema`, plus deep
immutability through tuples/closed records rather than mutable nested dictionaries.
`ID` is a bounded identifier, `Hash` is 64 lowercase SHA256 hex characters, `F` is
finite float, `Positive` is finite and greater than zero. Enums are closed; extra
keys are rejected recursively. All evidence references resolve inside an
allowlisted public training/acquisition bundle, never an arbitrary URL or path.

```python
class EvidenceSource(StrictRecord):
    id: ID
    artifact_sha256: Hash
    kind: Literal["public_preparation", "acquisition", "calibration", "train_replicates"]

class Assumption(StrictRecord):
    id: ID
    topic: Literal["noise", "independence", "time", "input", "calibration", "initial"]
    status: Literal["known", "estimated", "unavailable"]
    sources: tuple[ID, ...]         # references EvidenceSource.id
    method_id: ID | None           # registered estimator/protocol, never executable text
    estimate_sample_ids: tuple[ID, ...]
    reason: str                   # bounded public explanation; unavailable requires one

class PublicRawArtifact(StrictRecord):
    sha256: Hash                   # exact bytes; no ingestion-time rewriting
    media_type: Literal["text/csv", "application/json"]
    acquisition_source: ID
    parser_protocol: ID

class Reading(StrictRecord):
    id: ID                         # stable within manifest, not an array position alone
    artifact_sha256: Hash
    source_row: int                # nonnegative, binds exact source record
    trajectory: ID
    channel: ID
    clock_value: F
    raw_value: F | None
    condition: Literal["present", "missing", "left_censored", "right_censored", "interval_censored"]
    censor_lower: F | None
    censor_upper: F | None          # bounds in raw channel units

class Channel(StrictRecord):
    id: ID
    role: Literal["target", "auxiliary", "external_input", "fixed_covariate"]
    raw_unit: str
    analysis_unit: str
    gain: F                       # nonzero; analysis_value = gain * raw_value + offset
    offset: F
    calibration_assumption: ID

class TimeBase(StrictRecord):
    trajectory: ID
    raw_unit: str
    analysis_unit: str
    origin: F
    factor: Positive              # t = factor * (clock_value - origin)
    assumption: ID

class ForcingRule(StrictRecord):
    channel: ID
    interpolation: Literal["piecewise_linear", "left_hold", "pulse_events"]
    knot_reading_ids: tuple[ID, ...]
    pulse_convention: Literal["rate_half_open", "instantaneous_amount"] | None
    assumption: ID

class TrainingView(StrictRecord):
    protocol: Literal["measurement-training-view-1"]
    split: Literal["train"]
    manifest_sha256: Hash
    ordered_reading_ids: tuple[ID, ...]
    analysis_times: tuple[F, ...]
    analysis_values: tuple[F | None, ...]
    included: tuple[bool, ...]
    exclusion_reason: tuple[Literal["none", "missing", "acquisition_excluded"], ...]
    mask_policy_sha256: Hash

class DiagonalNoise(StrictRecord):
    protocol: Literal["known-diagonal-gaussian-1"]
    reading_ids: tuple[ID, ...]     # exactly the included target readings, in view order
    sigma: tuple[Positive, ...]    # observation SD in analysis units; never solver tolerance
    scale_assumption: ID
    independence_assumption: ID

class ExactBoundary(StrictRecord):
    trajectory: ID
    value: F                     # state units, from public preparation
    preparation_source: ID       # evidence independent of a noisy target reading

class InitialRule(StrictRecord):
    state: ID
    mode: Literal["exact_boundary", "shared_value", "shared_exact_causal_map", "unresolved"]
    boundary_reading_ids: tuple[ID, ...]
    exact_boundaries: tuple[ExactBoundary, ...]
    shared_coordinate: ID | None
    expression: str | None         # restricted causal grammar, never Python
    map_coordinates: tuple[ID, ...]
    assumption: ID
    uncertainty: Literal["exact", "jointly_estimated", "unavailable"]

class JointCoordinate(StrictRecord):
    name: ID
    role: Literal["physical", "initial", "observation_nuisance", "input_nuisance"]
    scope: Literal["global_training"]
    unit: str
    numerical_scale: Positive
    lower: F | None
    upper: F | None
    bounds_source: ID
    prior: Literal["none"]         # a later prior family needs a new objective protocol

class Objective(StrictRecord):
    protocol: Literal["known-diagonal-gaussian-objective-1"]
    estimation_split: Literal["train"]
    view_sha256: Hash
    noise_sha256: Hash
    initial_rules_sha256: Hash
    coordinates_sha256: Hash
    loss: Literal["half_sum_squared_standardized_residuals"]
    aggregation: Literal["one_term_per_included_target_reading"]
    additional_weights: Literal["none"]
    regularization: Literal["none"]

class MeasurementFitEnvelope(StrictRecord):
    protocol: Literal["public-fit-measurement-2"]
    base_request_sha256: Hash      # exact public-fit-1 request retained separately
    manifest_sha256: Hash          # sources, raw artifacts/readings, channel/time/forcing metadata
    training_view_sha256: Hash
    objective_sha256: Hash
    joint_coordinates_sha256: Hash
    numerical_policy_sha256: Hash
    source_sha256: Hash
```

Each hash resolves to its typed record in the sealed bundle; the manifest also
contains `Assumption`, `InitialRule`, and `ForcingRule` tuples. References must
resolve before computation. A present reading requires a value and no censor
bounds; a missing reading requires null value/bounds; censoring requires its finite
ordered bound(s). Only sanitized public raw files may enter this bundle:
mixed files containing private labels or held-out rows require an upstream public
export, not a fitter-side filter. Private generator truth, fault labels, clean
companion traces, test/validation metrics, credentials and unrestricted metadata
are absent from all fitter and proposer schemas. Acquisition text is data, not
instructions. A proposer gets a compact allowlisted projection, not raw files.

## Validation and first numerical capability

The manifest can record more than the first numerical profile can consume.
Malformed input is rejected as invalid; a valid unsupported contract returns
`capability_unsupported` before any fitting allowance is spent.

1. Preserve original bytes/readings. Calibration, clock conversion and masks
   create a new derived view with a parent digest; they never overwrite raw data.
   IDs are unique and trace every included residual back to one exact reading.
   Source hashes, parser version and conversion order are part of identity.
2. Align view columns exactly. All included target values are finite; missing
   readings have null values and fixed exclusion reasons. Masks are frozen before
   candidate fitting, with counts/reasons retained. A residual-selected removal
   is a new explicitly reviewed objective, not a resume or automatic outlier fix.
   Every target/trajectory needs an included reading; no empty objective.
3. First numerical scope: a common increasing time grid within each trajectory,
   at most one reading per channel/time, deterministic open ODE rollouts, declared
   complete exact forcing/covariates, known affine calibration and exact clocks.
   Missing target samples may be omitted from residuals without interpolation;
   missing forcing or necessary initial inputs is unsupported. Censoring, repeated
   same-time readings, covariance across readings, uncertain clocks/calibration/
   forcing, pulse-event execution and process noise are unsupported. Initially
   support the existing piecewise-linear forcing semantics only; recording another
   rule never authorizes reinterpreting it as linear samples.
4. Every included target reading has one positive sigma. Constant per-channel
   scales are expanded before hashing. `known` means justified and fixed under
   the declared protocol, not logically certain. External finite calibration
   estimates remain `estimated`; residual estimates and unavailable scales cannot
   enter `known-diagonal-gaussian-1`. Independence is an explicit assumption with
   its evidence/limits; it is not inferred from diagonal storage. No arbitrary
   epsilon substitutes for zero sigma, and no automatic inflation follows poor fit.
5. Keep raw units, target normalization, observation sigma and numerical coordinate
   scales distinct; raw-unit sigma converts by `abs(gain)`. For included reading i,
   retain `r_i = prediction_i - y_i` and
   `z_i = r_i / sigma_i`; optimize `Q = 0.5 * sum(z_i**2)`. With fixed Gaussian
   sigma, this differs from the negative log likelihood by a recorded constant.
   Optional legacy NMSE is descriptive, uses its own frozen training-scale digest,
   and is never the noisy-data objective or a noise estimate. Unequal sample counts
   contribute unequal information under this objective; no hidden target balancing.

## Uncertain initial conditions and joint evidence

An uncertain first observation must not become an exact initial state just because
the current runtime hard-binds it. `exact_boundary` needs public preparation
evidence establishing an exact value under the profile. For noisy initial readings,
the first supported extension should fit a shared true initial value when the
preparation explicitly supports a common state across trajectories, or use a
shared restricted causal map of exact public boundary inputs/covariates. The
noisy initial target reading enters the observation likelihood exactly once;
there is no duplicate initial-value penalty. Existing initializer expressions
still pass the restricted parser and causal namespace checks.
Each state has one rule: exact mode supplies one boundary for every trajectory
with unique, complete trajectory IDs; shared-value mode supplies one declared
initial coordinate; map mode supplies an expression using exactly its declared
coordinates plus allowed exact initial inputs; unresolved mode supplies none of
these. Other modes have empty `exact_boundaries`. Exact preparations can differ
between trajectories; a shared map can also return different initials from their
exact covariates without introducing separately fitted trajectory coordinates.
Coordinate names/roles must match the rule, and the complete rule tuple is hashed.

This requires an explicit new lowering/simulation path for direct observed states;
it is not achieved by supplying an old `LatentInitializationPlan` or changing an
observation expression to evade direct-state detection. If a request requires
independently fitted trajectory-specific initials, or a map uses a noisy boundary
observation, return
`INITIAL_UNCERTAINTY_UNSUPPORTED`. Per-trajectory latent initial inference and
propagating correlated boundary errors are deferred. Retaining a legacy conditional
replay is allowed, but it is labeled conditional on measured initials and cannot
claim that their uncertainty was assessed. No held-out initial fitting is added.

Compute joint whitened sensitivity with ordered coordinates `(physical, initial)`
and their frozen numerical scales. Store all columns, inactive columns, rank,
weak subspace and bounds; parameter-only rank with initials frozen is not a joint
claim. Observation/input nuisance roles are reserved in the record, but any such
coordinate is unsupported by the first profile. A later profile must include
eligible nuisance quantities jointly, preserve calibration bounds/priors, and
charge their search/profile cost. Freezing an uncertain gain cannot establish
physical-parameter confidence; for `y = s*a*u`, only the product may be informed.

## Residual facts, interpretation and identity

Add a separate `training-measurement-residual-evidence-1` record, with closed
fields: envelope hash, candidate/lowered-candidate hash, retained joint-vector
hash, replay/solver-policy hash, ordered reading IDs, raw residuals, standardized
residuals, full row counts and deterministic detail-selection policy. Failed
rollouts produce unavailable fields and reason codes, never zero residuals or
optimizer failure penalties masquerading as observations. Partial numerical
stages retain their identities, costs and completed facts.

Report raw and standardized bias/RMS/maxima by target and trajectory, plus exact
IDs for selected large residuals. A separate optional descriptive record may store
lagged residual products with time gaps and pair counts, replicate contrasts only
when declared replicate IDs exist, and pulse-aligned windows only when events are
publicly recorded. Missing facts say unavailable; sampled input values alone do
not establish pulse timing. These quantities describe residual patterns at one
retained point, not established correlation, defective sensors or false mechanisms.

Keep interpretation in a separate closed record: `specification_verdict` and
`data_fault_verdict` are both `not_assessed`; `evidence_level` is
`descriptive_at_retained_point` or `qualified_local_sensitivity`; assumptions and
unavailable capabilities are referenced by ID; `confidence_probability` is null.
Full rank, Gaussian weighting, local stationarity, solver agreement and M32
piecewise qualification alone do not justify a coverage level. Calibrated intervals
would require a named later procedure, assumptions, nuisance treatment, boundary/
nonlinearity assessment and independent coverage qualification; a profile curve
without these remains descriptive. No noiseless near-zero-NMSE gate is reused.

Hash the envelope, all resolved records, retained coordinates, objective, replay
policy, diagnostic policy and source before work. Changing sigma, masks, units,
calibration, input semantics, initial policy or loss creates a new experiment.
Reuse saved outputs only when model, vector, times, input/initial semantics and
solver accuracy match; recomputing residuals for a new objective creates separately
identified evidence and transfers no old stationarity/optimum claim. Follow M33's
exact perturbation-plan and operation-ledger rules; supplemental budgets never
erase failed calls or refresh the original allowance.

## Migration and acceptance tests for a later implementation

Leave `public-fit-1`, `fitting-feedback-1/2`, historical freezes and M33 source
snapshots byte-for-byte unchanged. A new wrapper references an old request rather
than appending default fields that alter its hash. Read-only legacy import marks
absent measurement assumptions `unavailable`; it never invents unit sigma or calls
training SD a noise model. No production default or finalized benchmark changes.

Implement/verify one increment at a time: (1) strict manifest/view validation and
read-only residual provenance; (2) the opt-in fixed-sigma objective and supported
shared-initial adapter; (3) separately reviewed statistical qualification. Tests
must cover exact sigma-weighted arithmetic and units; unchanged legacy packets;
missing-target masks and empty-objective rejection; unsupported censor/covariance/
forcing cases; sigma zero/nonfinite/estimated rejection; noisy-first-reading
rejection on the legacy path; joint initializer columns and no double counting;
changed objective refusing resume; tampered/reordered IDs and nested private-field
rejection; and interrupted-stage cost preservation without new allowance.

Extend the boundaries exercised by `tests/test_public_fitting.py`,
`tests/test_fitting_feedback.py`, `tests/test_residual_evidence.py`,
`tests/test_staged_feedback.py`, `tests/test_portable_feedback.py`, and
`tests/test_feedback_supplement.py`. Use small isolated synthetic fixtures with
separate evaluator truth, never edited benchmark data. CLI schema/inspection
smokes must run without optimization; numerical smokes belong to increment 2.
This document supplies no newly qualified noisy fit, uncertainty interval, data
fault attribution, or broadened initializer capability.

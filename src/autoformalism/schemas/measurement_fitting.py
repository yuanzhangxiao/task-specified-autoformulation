"""Opt-in, training-only known-noise contract with immutable raw provenance.

This wrapper does not change public-fit-1. Its explicit all-state preparation
rules replace legacy measured-initial conditioning only in the measurement path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from autoformalism.fitting.coordinates import NumericalCoordinates
from autoformalism.schemas.base import FiniteFloat, Identifier, StrictSchema
from autoformalism.schemas.public_fitting import (
    PublicFitRequest,
    PublicSplit,
    PublicTrajectory,
    Sha256,
)

Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
BoundedText = Annotated[str, Field(min_length=1, max_length=2000)]
Unit = Annotated[str, Field(min_length=1, max_length=128)]


class _FrozenDict(dict):
    """JSON-compatible mapping whose ordinary mutation operations are disabled."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("measurement mappings are immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable
    __ior__ = _immutable

    def __deepcopy__(self, memo):
        return self


def _freeze(value):
    """Detach and freeze nested legacy mappings without changing serialization."""
    if isinstance(value, BaseModel):
        return value.model_copy(
            update={
                name: _freeze(getattr(value, name)) for name in type(value).model_fields
            }
        )
    if isinstance(value, Mapping):
        return _FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


class EvidenceSource(StrictSchema):
    """An allowlisted public artifact identity, never an executable path or URL."""

    id: Identifier
    artifact_sha256: Sha256
    kind: Literal[
        "public_preparation", "acquisition", "calibration", "train_replicates"
    ]


class Assumption(StrictSchema):
    """Explicit provenance and limits of an externally supplied known quantity."""

    id: Identifier
    topic: Literal["noise", "independence", "time", "input", "calibration", "initial"]
    status: Literal["known", "estimated", "unavailable"]
    sources: tuple[Identifier, ...] = Field(min_length=1, max_length=64)
    reason: BoundedText


class PublicRawArtifact(StrictSchema):
    """Identity of sanitized public bytes; no raw file is rewritten by fitting."""

    sha256: Sha256
    media_type: Literal["text/csv", "application/json"]
    acquisition_source: Identifier
    parser_protocol: Identifier


class Reading(StrictSchema):
    """One original acquisition record; missing values remain explicit nulls."""

    id: Identifier
    artifact_sha256: Sha256
    source_row: int = Field(ge=0, strict=True)
    channel: Identifier
    clock_value: FiniteFloat
    raw_value: FiniteFloat | None
    condition: Literal["present", "missing"]

    @model_validator(mode="after")
    def value_matches_condition(self):
        if (self.condition == "present") != (self.raw_value is not None):
            raise ValueError(
                "present readings require a value; missing readings require null"
            )
        return self


class Channel(StrictSchema):
    """Known affine conversion: analysis_value = gain * raw_value + offset."""

    id: Identifier
    role: Literal["target", "auxiliary", "external_input", "fixed_covariate"]
    raw_unit: Unit
    analysis_unit: Unit
    gain: FiniteFloat
    offset: FiniteFloat
    calibration_assumption: Identifier
    exact_input_assumption: Identifier | None = None


class TimeBase(StrictSchema):
    """Exact clock conversion: analysis_time = factor * (clock_value - origin)."""

    trajectory: Identifier
    raw_unit: Unit
    analysis_unit: Unit
    origin: FiniteFloat
    factor: Positive
    assumption: Identifier


class ForcingRule(StrictSchema):
    """Exact knots and the only forcing interpolation supported in this profile."""

    trajectory: Identifier
    channel: Identifier
    interpolation: Literal["piecewise_linear"] = "piecewise_linear"
    knot_reading_ids: tuple[Identifier, ...] = Field(min_length=2)
    assumption: Identifier


class DiagonalNoise(StrictSchema):
    """Fixed, known, positive observation standard deviations in analysis units."""

    protocol: Literal["known-diagonal-gaussian-1"] = "known-diagonal-gaussian-1"
    units: Literal["analysis"] = "analysis"
    reading_ids: tuple[Identifier, ...] = Field(min_length=1)
    sigma: tuple[Positive, ...] = Field(min_length=1)
    scale_assumption: Identifier
    independence_assumption: Identifier

    @model_validator(mode="after")
    def aligned(self):
        if len(self.reading_ids) != len(self.sigma):
            raise ValueError("noise reading IDs and sigma lengths differ")
        if len(set(self.reading_ids)) != len(self.reading_ids):
            raise ValueError("duplicate noise reading identifier")
        return self


class MeasurementManifest(StrictSchema):
    """Closed public provenance; unsupported stochastic capabilities are absent."""

    protocol: Literal["measurement-manifest-1"] = "measurement-manifest-1"
    sources: tuple[EvidenceSource, ...] = Field(min_length=1, max_length=256)
    assumptions: tuple[Assumption, ...] = Field(min_length=1, max_length=256)
    artifacts: tuple[PublicRawArtifact, ...] = Field(min_length=1, max_length=256)
    channels: tuple[Channel, ...] = Field(min_length=1, max_length=256)
    time_bases: tuple[TimeBase, ...] = Field(min_length=1)
    forcing_rules: tuple[ForcingRule, ...] = ()
    noise: DiagonalNoise


class MeasurementTrajectory(StrictSchema):
    """Aligned raw public readings on one common acquisition clock grid."""

    trajectory_id: Identifier
    clock_values: tuple[FiniteFloat, ...] = Field(min_length=2)
    readings: tuple[Reading, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def increasing(self):
        if any(b <= a for a, b in pairwise(self.clock_values)):
            raise ValueError("clock values must be strictly increasing")
        return self


class MeasurementTraining(StrictSchema):
    """Only training acquisition rows; there is no validation/test payload slot."""

    protocol: Literal["measurement-training-1"] = "measurement-training-1"
    split: Literal["train"] = "train"
    mask_policy: Literal["include_present_targets"] = "include_present_targets"
    rows: tuple[MeasurementTrajectory, ...] = Field(min_length=1)


class ExactBoundary(StrictSchema):
    """State-unit boundary from preparation, independent of noisy target readings."""

    trajectory: Identifier
    value: FiniteFloat
    preparation_source: Identifier


class InitialRule(StrictSchema):
    """One exact preparation or jointly fitted shared value for every state."""

    state: Identifier
    mode: Literal["exact_boundary", "shared_value"]
    exact_boundaries: tuple[ExactBoundary, ...] = ()
    shared_coordinate: Identifier | None = None
    assumption: Identifier

    @model_validator(mode="after")
    def coherent(self):
        if self.mode == "exact_boundary":
            if not self.exact_boundaries or self.shared_coordinate is not None:
                raise ValueError(
                    "exact initial rules require boundaries and no coordinate"
                )
        elif self.exact_boundaries or self.shared_coordinate is None:
            raise ValueError(
                "shared initial rules require one coordinate and no boundaries"
            )
        return self


def _unique(items, key: str, label: str) -> dict:
    values = {getattr(item, key): item for item in items}
    if len(values) != len(items):
        raise ValueError(f"duplicate {label}")
    return values


class MeasurementProblem(StrictSchema):
    """Versioned wrapper whose complete serialized content binds the objective.

    The legacy request remains byte-equivalent under canonical serialization.
    All-state rules are authoritative on this opt-in path; no legacy initializer
    plan, nuisance parameter, causal initial map or held-out row is accepted.
    """

    protocol: Literal["public-fit-measurement-1"] = "public-fit-measurement-1"
    request: PublicFitRequest
    manifest: MeasurementManifest
    training: MeasurementTraining
    initial_rules: tuple[InitialRule, ...] = Field(min_length=1, max_length=64)
    coordinates: NumericalCoordinates
    start: Mapping[Identifier, FiniteFloat]
    incumbent: Mapping[Identifier, FiniteFloat]

    @model_validator(mode="after")
    def validate_contract(self):
        _validate_contract(self)
        for name in ("request", "coordinates", "start", "incumbent"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        return self


def _validate_contract(problem: MeasurementProblem) -> None:
    """Resolve every reference and reject unsupported capabilities before fitting."""
    manifest, request = problem.manifest, problem.request
    sources = _unique(manifest.sources, "id", "evidence source")
    assumptions = _unique(manifest.assumptions, "id", "assumption")
    artifacts = _unique(manifest.artifacts, "sha256", "raw artifact")
    channels = _unique(manifest.channels, "id", "channel")
    clocks = _unique(manifest.time_bases, "trajectory", "time base")
    rows = _unique(problem.training.rows, "trajectory_id", "trajectory")
    for assumption in assumptions.values():
        if len(set(assumption.sources)) != len(assumption.sources):
            raise ValueError("duplicate assumption evidence source")
        if set(assumption.sources) - set(sources):
            raise ValueError("unresolved assumption evidence source")
    for artifact in artifacts.values():
        source = sources.get(artifact.acquisition_source)
        if source is None or source.kind != "acquisition":
            raise ValueError("raw artifact requires an acquisition evidence source")

    def known(name: str, topic: str) -> Assumption:
        assumption = assumptions.get(name)
        if assumption is None or assumption.topic != topic:
            raise ValueError(f"unresolved {topic} assumption")
        if assumption.status != "known":
            raise ValueError(
                f"{topic.upper()}_UNSUPPORTED: requires a known assumption"
            )
        return assumption

    expected = {
        channel: role
        for role, names in (
            ("target", request.context.targets),
            ("auxiliary", request.context.auxiliaries),
            ("external_input", request.context.external_inputs),
            ("fixed_covariate", request.context.fixed_covariates),
        )
        for channel in names
    }
    if {name: channel.role for name, channel in channels.items()} != expected:
        raise ValueError("measurement channels differ from public context")
    if set(clocks) != set(rows):
        raise ValueError("time bases must cover exactly the training trajectories")
    for channel in channels.values():
        if channel.gain == 0:
            raise ValueError("calibration gain must be nonzero")
        known(channel.calibration_assumption, "calibration")
        if channel.role != "target":
            if channel.exact_input_assumption is None:
                raise ValueError(
                    "INPUT_UNSUPPORTED: all public forcing/covariates must be exact"
                )
            known(channel.exact_input_assumption, "input")
        elif channel.exact_input_assumption is not None:
            raise ValueError("noisy targets cannot be declared exact forcing")
    for clock in clocks.values():
        known(clock.assumption, "time")
    known(manifest.noise.scale_assumption, "noise")
    known(manifest.noise.independence_assumption, "independence")
    _validate_readings(problem, channels, clocks, artifacts, known)
    _validate_initials(problem, rows, sources, known)


def _validate_readings(problem, channels, clocks, artifacts, known) -> None:
    """Bind raw records, fixed masks, exact forcing and sigma to one ordered view."""
    seen_ids, seen_sources, included = set(), set(), []
    expected_forcing = {}
    targets = problem.request.context.targets
    for row in problem.training.rows:
        indexed = {}
        for reading in row.readings:
            if reading.id in seen_ids:
                raise ValueError("duplicate reading identifier")
            seen_ids.add(reading.id)
            provenance = (reading.artifact_sha256, reading.source_row, reading.channel)
            if provenance in seen_sources:
                raise ValueError("duplicate raw source record for channel")
            seen_sources.add(provenance)
            if reading.artifact_sha256 not in artifacts:
                raise ValueError("unresolved reading raw artifact")
            if (
                reading.channel not in channels
                or reading.clock_value not in row.clock_values
            ):
                raise ValueError(
                    "reading channel/time is not in the declared training grid"
                )
            key = (reading.channel, reading.clock_value)
            if key in indexed:
                raise ValueError("repeated same-time reading is unsupported")
            indexed[key] = reading
            if reading.raw_value is not None:
                channel = channels[reading.channel]
                if not math.isfinite(channel.gain * reading.raw_value + channel.offset):
                    raise ValueError("nonfinite calibrated reading")
        clock = clocks[row.trajectory_id]
        times = [clock.factor * (value - clock.origin) for value in row.clock_values]
        if any(not math.isfinite(value) for value in times):
            raise ValueError("nonfinite converted clock")
        if any(b <= a for a, b in pairwise(times)):
            raise ValueError("converted clock must remain strictly increasing")
        expected_keys = set()
        for name, channel in channels.items():
            values = (
                row.clock_values[:1]
                if channel.role == "fixed_covariate"
                else row.clock_values
            )
            readings = []
            for value in values:
                key = (name, value)
                expected_keys.add(key)
                if key not in indexed:
                    raise ValueError(
                        "every channel/time needs an explicit reading or missing record"
                    )
                reading = indexed[key]
                if channel.role != "target" and reading.condition != "present":
                    raise ValueError("INPUT_UNSUPPORTED: missing forcing/covariates")
                readings.append(reading)
            if channel.role == "target" and not any(
                r.condition == "present" for r in readings
            ):
                raise ValueError("every target/trajectory requires an included reading")
            if channel.role in {"auxiliary", "external_input"}:
                expected_forcing[(row.trajectory_id, name)] = tuple(
                    r.id for r in readings
                )
        if set(indexed) != expected_keys:
            raise ValueError("unexpected fixed-covariate or channel reading")
        included.extend(
            indexed[(name, value)].id
            for value in row.clock_values
            for name in targets
            if indexed[(name, value)].condition == "present"
        )
    forcing = {}
    for rule in problem.manifest.forcing_rules:
        key = (rule.trajectory, rule.channel)
        if key in forcing:
            raise ValueError("duplicate forcing rule")
        known(rule.assumption, "input")
        forcing[key] = rule.knot_reading_ids
    if forcing != expected_forcing:
        raise ValueError(
            "forcing rules must match all exact public knots in clock order"
        )
    if tuple(included) != problem.manifest.noise.reading_ids:
        raise ValueError(
            "sigma must align exactly with included targets "
            "in trajectory/time/channel order"
        )


def _validate_initials(problem, rows, sources, known) -> None:
    """Require explicit all-state preparation and complete shared parameter axes."""
    request = problem.request
    if request.initialization_plan.rules:
        raise ValueError(
            "INITIAL_UNCERTAINTY_UNSUPPORTED: legacy initial plan must be empty"
        )
    if request.context.lagged_targets or request.context.fitted_initialization:
        raise ValueError("measurement path requires unlowered open-rollout context")
    parameters = _unique(request.base_candidate.parameters, "name", "parameter")
    if any(parameter.scope.value != "global" for parameter in parameters.values()):
        raise ValueError(
            "INITIAL_UNCERTAINTY_UNSUPPORTED: only shared global parameters"
        )
    rules = _unique(problem.initial_rules, "state", "initial rule")
    states = {state.name for state in request.base_candidate.states}
    if set(rules) != states:
        raise ValueError("initial rules must cover every state exactly once")
    if set(problem.coordinates.states) != states:
        raise ValueError("state coordinate symbols differ from model")
    for mapping in (problem.start, problem.incumbent, problem.coordinates.parameters):
        if set(mapping) != set(parameters):
            raise ValueError(
                "joint parameter coordinates/start/incumbent must be complete"
            )
    for rule in rules.values():
        assumption = known(rule.assumption, "initial")
        if not any(
            sources[source].kind == "public_preparation"
            for source in assumption.sources
        ):
            raise ValueError("initial rule requires public preparation evidence")
        if rule.mode == "shared_value":
            if rule.shared_coordinate not in parameters:
                raise ValueError(
                    "shared initial coordinate must be a declared global parameter"
                )
        else:
            boundaries = _unique(rule.exact_boundaries, "trajectory", "exact boundary")
            if set(boundaries) != set(rows):
                raise ValueError(
                    "exact boundaries must cover every training trajectory"
                )
            for boundary in boundaries.values():
                source = sources.get(boundary.preparation_source)
                if source is None or source.kind != "public_preparation":
                    raise ValueError(
                        "exact boundary requires independent public preparation"
                    )
                if boundary.preparation_source not in assumption.sources:
                    raise ValueError(
                        "exact boundary source must support its initial assumption"
                    )


class MeasurementLayout(StrictSchema):
    """Flat immutable view in trajectory/time/channel order, including missing slots."""

    reading_ids: tuple[Identifier, ...]
    trajectory_ids: tuple[Identifier, ...]
    channels: tuple[Identifier, ...]
    analysis_times: tuple[FiniteFloat, ...]
    values: tuple[FiniteFloat | None, ...]
    sigma: tuple[Positive | None, ...]
    included: tuple[bool, ...]


def analysis_layout(
    problem: MeasurementProblem, channels: Sequence[str] | None = None
) -> MeasurementLayout:
    """Derive observations and known sigma without replacing missing source values."""
    ordered = tuple(problem.request.context.targets if channels is None else channels)
    if set(ordered) != set(problem.request.context.targets) or len(set(ordered)) != len(
        ordered
    ):
        raise ValueError("layout channels must be exactly the declared targets")
    metadata = {channel.id: channel for channel in problem.manifest.channels}
    clocks = {clock.trajectory: clock for clock in problem.manifest.time_bases}
    noise = dict(
        zip(
            problem.manifest.noise.reading_ids,
            problem.manifest.noise.sigma,
            strict=True,
        )
    )
    result = {name: [] for name in MeasurementLayout.model_fields}
    for row in problem.training.rows:
        indexed = {
            (reading.channel, reading.clock_value): reading for reading in row.readings
        }
        clock = clocks[row.trajectory_id]
        for value in row.clock_values:
            for channel_id in ordered:
                reading = indexed[(channel_id, value)]
                channel = metadata[channel_id]
                result["reading_ids"].append(reading.id)
                result["trajectory_ids"].append(row.trajectory_id)
                result["channels"].append(channel_id)
                result["analysis_times"].append(clock.factor * (value - clock.origin))
                result["values"].append(
                    None
                    if reading.raw_value is None
                    else channel.gain * reading.raw_value + channel.offset
                )
                result["sigma"].append(noise.get(reading.id))
                result["included"].append(reading.condition == "present")
    return MeasurementLayout.model_validate(result)


def analysis_training(problem: MeasurementProblem) -> PublicSplit:
    """Build numerical arrays; missing placeholders require layout masks and initials.

    Zero placeholders only keep the legacy finite trajectory container usable.
    They are not measurements, may never enter residuals, and may never initialize
    a state. Consumers must use analysis_layout and explicit initial_values.
    """
    from autoformalism.fitting.public_fitting import content_sha256

    channels = {channel.id: channel for channel in problem.manifest.channels}
    clocks = {clock.trajectory: clock for clock in problem.manifest.time_bases}
    rows = []
    for row in problem.training.rows:
        indexed = {
            (reading.channel, reading.clock_value): reading for reading in row.readings
        }
        payload = {
            "targets": {},
            "auxiliaries": {},
            "external_inputs": {},
            "fixed_covariates": {},
        }
        for name, channel in channels.items():
            values = (
                row.clock_values[:1]
                if channel.role == "fixed_covariate"
                else row.clock_values
            )
            converted = tuple(
                0.0
                if indexed[(name, time)].raw_value is None
                else channel.gain * indexed[(name, time)].raw_value + channel.offset
                for time in values
            )
            key = {
                "target": "targets",
                "auxiliary": "auxiliaries",
                "external_input": "external_inputs",
                "fixed_covariate": "fixed_covariates",
            }[channel.role]
            payload[key][name] = (
                converted[0] if channel.role == "fixed_covariate" else converted
            )
        clock = clocks[row.trajectory_id]
        rows.append(
            PublicTrajectory(
                trajectory_id=row.trajectory_id,
                time=tuple(
                    clock.factor * (value - clock.origin) for value in row.clock_values
                ),
                **payload,
            )
        )
    return PublicSplit(
        name="train", fingerprint=content_sha256(problem), rows=tuple(rows)
    )


def initial_values(
    problem: MeasurementProblem, trajectory_id: str, parameters: Mapping[str, float]
) -> dict[str, float]:
    """Return every state's explicit boundary without reading target observations."""
    if trajectory_id not in {row.trajectory_id for row in problem.training.rows}:
        raise ValueError("unknown training trajectory")
    if set(parameters) != set(problem.start) or any(
        not math.isfinite(value) for value in parameters.values()
    ):
        raise ValueError(
            "initial evaluation requires a complete finite parameter vector"
        )
    return {
        rule.state: (
            parameters[rule.shared_coordinate]
            if rule.mode == "shared_value"
            else next(
                boundary.value
                for boundary in rule.exact_boundaries
                if boundary.trajectory == trajectory_id
            )
        )
        for rule in problem.initial_rules
    }

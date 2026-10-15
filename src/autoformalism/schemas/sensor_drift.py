"""Opt-in shared observation drift, separate from immutable physical equations."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from autoformalism.fitting.coordinates import AffineAxis, NumericalCoordinates
from autoformalism.schemas.base import FiniteFloat, Identifier, StrictSchema
from autoformalism.schemas.measurement_fitting import (
    BoundedText,
    MeasurementProblem,
    Unit,
    _freeze,
)


class SensorDrift(StrictSchema):
    """One bounded additive slope shared by a target channel's trajectories.

    The prediction contribution is slope * (analysis_time - time_origin).
    Sources justify testing this hypothesis; they do not establish a data fault.
    """

    name: Identifier = "sensor_drift"
    channel: Identifier
    lower: FiniteFloat = Field(strict=True)
    upper: FiniteFloat = Field(strict=True)
    start: FiniteFloat = Field(strict=True)
    incumbent: FiniteFloat = Field(strict=True)
    scale: FiniteFloat = Field(gt=0, strict=True)
    time_origin: FiniteFloat = Field(default=0.0, strict=True)
    unit: Unit
    sources: tuple[Identifier, ...] = Field(min_length=1, max_length=64)
    rationale: BoundedText

    @model_validator(mode="after")
    def bounded(self):
        """Reject degenerate domains, out-of-domain starts and duplicate sources."""
        if self.lower >= self.upper:
            raise ValueError("sensor drift bounds must be strictly ordered")
        if not all(
            self.lower <= value <= self.upper for value in (self.start, self.incumbent)
        ):
            raise ValueError("sensor drift start and incumbent must lie within bounds")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("duplicate sensor drift source")
        return self


class DriftProblem(StrictSchema):
    """Versioned observation extension; the base measurement contract is unchanged."""

    protocol: Literal["public-fit-sensor-drift-1"] = "public-fit-sensor-drift-1"
    base: MeasurementProblem
    drift: SensorDrift

    @model_validator(mode="after")
    def validate_contract(self):
        """Resolve provenance, physical units and namespaces before numerical work."""
        base = MeasurementProblem.model_validate(self.base.model_dump(mode="json"))
        object.__setattr__(self, "base", base)
        object.__setattr__(
            self,
            "drift",
            SensorDrift.model_validate(self.drift.model_dump(mode="json")),
        )
        candidate = base.request.base_candidate
        occupied = (
            {
                item.name
                for group in (
                    candidate.parameters,
                    candidate.states,
                    candidate.processes,
                )
                for item in group
            }
            | {channel.id for channel in base.manifest.channels}
            | {"t"}
        )
        if self.drift.name in occupied:
            raise ValueError(
                "sensor drift name collides with a physical or channel symbol"
            )
        channels = {channel.id: channel for channel in base.manifest.channels}
        channel = channels.get(self.drift.channel)
        if channel is None or channel.role != "target":
            raise ValueError("sensor drift requires a declared target channel")
        sources = {source.id for source in base.manifest.sources}
        if not set(self.drift.sources) <= sources:
            raise ValueError("unresolved sensor drift provenance source")
        time_units = {clock.analysis_unit for clock in base.manifest.time_bases}
        if len(time_units) != 1:
            raise ValueError("shared sensor drift requires common analysis time units")
        expected = f"{channel.analysis_unit}/{next(iter(time_units))}"
        if self.drift.unit != expected:
            raise ValueError(f"sensor drift unit must be {expected!r}")
        return self

    @property
    def request(self):
        """Expose the original request without adding the nuisance to its candidate."""
        return self.base.request

    @property
    def manifest(self):
        """Expose immutable acquisition and calibration provenance."""
        return self.base.manifest

    @property
    def training(self):
        """Expose the original raw readings without correction or replacement."""
        return self.base.training

    @property
    def initial_rules(self):
        """Keep exact/shared physical preparation independent of observation drift."""
        return self.base.initial_rules

    @property
    def start(self):
        """Extend the frozen numerical start with the separately declared nuisance."""
        return _freeze(dict(self.base.start) | {self.drift.name: self.drift.start})

    @property
    def incumbent(self):
        """Extend the incumbent without changing any physical parameter value."""
        return _freeze(
            dict(self.base.incumbent) | {self.drift.name: self.drift.incumbent}
        )

    @property
    def coordinates(self) -> NumericalCoordinates:
        """Append a numerical drift scale; this is not a prior or a noise scale."""
        original = self.base.coordinates
        return _freeze(
            NumericalCoordinates(
                parameters=dict(original.parameters)
                | {self.drift.name: AffineAxis(center=0.0, scale=self.drift.scale)},
                states=original.states,
                provenance=dict(original.provenance)
                | {self.drift.name: "predeclared observation nuisance coordinate"},
            )
        )

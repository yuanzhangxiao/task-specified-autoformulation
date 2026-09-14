"""Bounded descriptive evidence from public training observations only."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from autoformalism.data import DatasetSplit, SplitName
from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.staged_topology import content_hash


class EvidenceSettings(StrictSchema):
    """Frozen presentation limits and descriptive, non-statistical tolerances."""

    maximum_trajectories: int = Field(default=6, ge=1, le=12)
    maximum_events: int = Field(default=4, ge=1, le=12)
    maximum_samples: int = Field(default=7, ge=2, le=16)
    absolute_tolerance: FiniteFloat = Field(default=1e-8, ge=0)
    relative_tolerance: FiniteFloat = Field(default=0.01, ge=0, le=1)


class Sample(StrictSchema):
    """An exact measured sample, never an interpolated extremum."""

    index: int = Field(ge=0)
    time: FiniteFloat
    value: FiniteFloat


class ChannelEvidence(StrictSchema):
    """Descriptive shape and explicit truncation of a sampled channel."""

    initial: Sample
    final: Sample
    minimum: Sample
    maximum: Sample
    rms: FiniteFloat
    tolerance: FiniteFloat
    sampled_path: tuple[Sample, ...]
    turning_points: tuple[Sample, ...]
    turning_point_count: int = Field(ge=0)
    omitted_turning_points: int = Field(ge=0)


class InputChange(StrictSchema):
    """Sample-bracketed input change and coincident measured target differences."""

    before: Sample
    after: Sample
    target_change_in_same_bracket: dict[str, FiniteFloat]


class InputEvidence(StrictSchema):
    """Sampled input variation; continuous ramps may produce many changes."""

    initial: Sample
    final: Sample
    maximum_absolute_value: FiniteFloat
    tolerance: FiniteFloat
    change_count: int = Field(ge=0)
    omitted_changes: int = Field(ge=0)
    changes: tuple[InputChange, ...]


class TrajectoryEvidence(StrictSchema):
    """Provenance identifies training evidence, not a prediction feature."""

    trajectory_id: str
    sample_count: int = Field(ge=1)
    minimum_time_step: FiniteFloat | None
    maximum_time_step: FiniteFloat | None
    targets: dict[str, ChannelEvidence]
    inputs: dict[str, InputEvidence]
    zero_sampled_inputs_with_changing_targets: tuple[str, ...]


class TrainingEvidence(StrictSchema):
    """An immutable training-only packet with an independently checkable digest."""

    protocol: Literal["prefit-training-evidence-1"] = "prefit-training-evidence-1"
    split: Literal["train"] = "train"
    settings: EvidenceSettings
    context: ValidationContext
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trajectory_count: int = Field(ge=1)
    omitted_trajectories: int = Field(ge=0)
    initial_target_ranges: dict[str, tuple[FiniteFloat, FiniteFloat]]
    trajectories: tuple[TrajectoryEvidence, ...]
    interpretation: tuple[str, ...]
    packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


INTERPRETATION = (
    "Training observations only. Trajectory IDs locate evidence "
    "and are not model features.",
    "Tolerances are descriptive thresholds, not measurement-error estimates "
    "or confidence probabilities.",
    "Extrema and turning points refer to sampled values; unresolved "
    "between-sample behavior remains unknown.",
    "Input changes are bracketed by adjacent samples. Coincident output "
    "differences do not establish causation or a response lag.",
    "Zero sampled input with varying output does not identify a hidden state "
    "or prove autonomous oscillation.",
    "Initial variability does not establish observability or identify latent "
    "initial values. Prediction uses only permitted initial information and inputs.",
    "The packet describes data; it does not add mandatory variables, topology, "
    "polarity, or functional forms to the public requirements.",
)


def _indices(count: int, limit: int) -> list[int]:
    """Cover the beginning and end deterministically when truncation is needed."""
    return np.unique(np.linspace(0, count - 1, min(count, limit), dtype=int)).tolist()


def build_training_evidence(
    train: DatasetSplit,
    context: ValidationContext,
    settings: EvidenceSettings | None = None,
) -> TrainingEvidence:
    """Reject held-out splits before accessing trajectories; hash actual public data."""
    if not isinstance(train, DatasetSplit) or train.name is not SplitName.TRAIN:
        raise ValueError("training evidence requires the train split only")
    settings = settings or EvidenceSettings()
    trajectories = sorted(train.trajectories, key=lambda t: t.trajectory_id)
    if not trajectories or len({t.trajectory_id for t in trajectories}) != len(
        trajectories
    ):
        raise ValueError(
            "training trajectories must be nonempty and uniquely identified"
        )
    source = []
    for trajectory in trajectories:
        times = trajectory.time
        if (
            len(times) < 1
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError("training times must be finite and strictly increasing")
        if set(trajectory.targets) != set(context.targets):
            raise ValueError("training targets differ from the public context")
        if set(trajectory.external_inputs) != set(context.external_inputs):
            raise ValueError("training inputs differ from the public context")
        channels = {**trajectory.targets, **trajectory.external_inputs}
        if any(
            v.shape != times.shape or not np.isfinite(v).all()
            for v in channels.values()
        ):
            raise ValueError("training channels must be finite and aligned")
        source.append(
            {
                "id": trajectory.trajectory_id,
                "time": times.tolist(),
                "targets": {k: v.tolist() for k, v in trajectory.targets.items()},
                "inputs": {
                    k: v.tolist() for k, v in trajectory.external_inputs.items()
                },
            }
        )
    selected = []
    for index in _indices(len(trajectories), settings.maximum_trajectories):
        trajectory = trajectories[index]
        times = trajectory.time

        def sample(values: np.ndarray, i: int, times: np.ndarray = times) -> Sample:
            return Sample(index=int(i), time=float(times[i]), value=float(values[i]))

        def tolerance(values: np.ndarray) -> float:
            return float(
                settings.absolute_tolerance
                + settings.relative_tolerance * np.ptp(values)
            )

        targets = {}
        for name, values in sorted(trajectory.targets.items()):
            tol = tolerance(values)
            changes = np.diff(values)
            signs = np.where(np.abs(changes) > tol, np.sign(changes), 0)
            nonzero = np.flatnonzero(signs)
            # Across a sampled plateau, report the last sample before reversal.
            turns = [
                int(b)
                for a, b in pairwise(nonzero)
                if signs[a] != signs[b]
            ]
            kept = (
                [turns[i] for i in _indices(len(turns), settings.maximum_events)]
                if turns
                else []
            )
            targets[name] = ChannelEvidence(
                initial=sample(values, 0),
                final=sample(values, len(values) - 1),
                minimum=sample(values, int(np.argmin(values))),
                maximum=sample(values, int(np.argmax(values))),
                rms=float(np.linalg.norm(values / np.sqrt(len(values)))),
                tolerance=tol,
                sampled_path=tuple(
                    sample(values, i)
                    for i in _indices(len(values), settings.maximum_samples)
                ),
                turning_points=tuple(sample(values, i) for i in kept),
                turning_point_count=len(turns),
                omitted_turning_points=len(turns) - len(kept),
            )
        inputs = {}
        for name, values in sorted(trajectory.external_inputs.items()):
            indices = (
                np.flatnonzero(np.abs(np.diff(values)) > tolerance(values)) + 1
            ).tolist()
            kept = (
                [indices[i] for i in _indices(len(indices), settings.maximum_events)]
                if indices
                else []
            )
            inputs[name] = InputEvidence(
                initial=sample(values, 0),
                final=sample(values, len(values) - 1),
                maximum_absolute_value=float(np.max(np.abs(values))),
                tolerance=tolerance(values),
                change_count=len(indices),
                omitted_changes=len(indices) - len(kept),
                changes=tuple(
                    InputChange(
                        before=sample(values, i - 1),
                        after=sample(values, i),
                        target_change_in_same_bracket={
                            k: float(v[i] - v[i - 1])
                            for k, v in trajectory.targets.items()
                        },
                    )
                    for i in kept
                ),
            )
        dt = np.diff(times)
        all_zero = bool(inputs) and all(
            v.maximum_absolute_value <= settings.absolute_tolerance
            for v in inputs.values()
        )
        selected.append(
            TrajectoryEvidence(
                trajectory_id=trajectory.trajectory_id,
                sample_count=len(times),
                minimum_time_step=float(dt.min()) if len(dt) else None,
                maximum_time_step=float(dt.max()) if len(dt) else None,
                targets=targets,
                inputs=inputs,
                zero_sampled_inputs_with_changing_targets=tuple(
                    k
                    for k, v in targets.items()
                    if all_zero and v.maximum.value - v.minimum.value > v.tolerance
                ),
            )
        )
    payload = {
        "protocol": "prefit-training-evidence-1",
        "split": "train",
        "settings": settings.model_dump(mode="json"),
        "context": context.model_dump(mode="json"),
        "source_sha256": content_hash(source),
        "trajectory_count": len(trajectories),
        "omitted_trajectories": len(trajectories) - len(selected),
        "initial_target_ranges": {
            name: [
                min(float(t.targets[name][0]) for t in trajectories),
                max(float(t.targets[name][0]) for t in trajectories),
            ]
            for name in sorted(context.targets)
        },
        "trajectories": [t.model_dump(mode="json") for t in selected],
        "interpretation": list(INTERPRETATION),
    }
    return TrainingEvidence.model_validate(
        {**payload, "packet_sha256": content_hash(payload)}
    )


def validate_training_evidence(
    packet: TrainingEvidence, context: ValidationContext
) -> dict:
    """Verify the immutable presentation boundary before exposing it to a provider."""
    payload = packet.model_dump(mode="json")
    digest = payload.pop("packet_sha256")
    if digest != content_hash(payload) or packet.context != context:
        raise ValueError("training evidence digest or public context differs")
    return {**payload, "packet_sha256": digest}


def freeze_training_evidence(path: Path, packet: TrainingEvidence) -> None:
    """Reject stale content even when an upstream split fingerprint was reused."""
    payload = validate_training_evidence(packet, packet.context)
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("frozen training evidence differs; use a new output root")
    atomic_json(path, payload)


def evidence_brief(
    brief: dict, context: ValidationContext, packet: TrainingEvidence | None
) -> dict:
    """Keep finalized scientific prose intact and attach separate observed evidence."""
    if packet is None:
        return brief
    return {
        **brief,
        "training_observations": validate_training_evidence(packet, context),
    }

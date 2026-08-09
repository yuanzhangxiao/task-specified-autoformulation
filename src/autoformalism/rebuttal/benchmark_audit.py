"""Model-independent diagnostics for benchmark predictability and excitation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from autoformalism.data.models import DatasetSplit, Trajectory

ShortcutName = Literal["persistence", "constant", "ar", "arx"]


@dataclass(frozen=True)
class ShortcutMetrics:
    """Per-sample errors for one shortcut and forecast horizon."""

    model: ShortcutName
    horizon: int
    raw_mse: float
    normalized_mse: float
    event_raw_mse: float | None
    event_normalized_mse: float | None
    event_sample_fraction: float
    sample_count: int


@dataclass(frozen=True)
class ExcitationMetrics:
    """Descriptive evidence about target motion and external excitation."""

    target: str
    sample_count: int
    trajectory_count: int
    target_variance: float
    increment_variance_ratio: float
    active_input_fraction: float
    input_change_fraction: float
    distinct_input_levels: int
    event_window_fraction: float


@dataclass(frozen=True)
class ResponsePhaseMetrics:
    """Shortcut error within a reference-defined response phase."""

    model: ShortcutName
    horizon: int
    phase: Literal["rise", "peak", "recovery", "near_equilibrium"]
    raw_mse: float
    normalized_mse: float
    sample_count: int


@dataclass(frozen=True)
class _LinearShortcut:
    intercept: float
    target_coefficient: float
    input_coefficients: NDArray[np.float64]


def audit_shortcuts(
    train: DatasetSplit,
    evaluation: DatasetSplit,
    target: str,
    horizons: tuple[int, ...] = (1, 5, 10, 30),
    *,
    event_radius: int = 30,
    scale_floor: float = 1e-8,
) -> tuple[ShortcutMetrics, ...]:
    """Evaluate shortcut forecasts fitted only on the training split.

    At horizon ``h``, each prediction starts from a measured target and advances
    recursively for ``h`` samples. Larger horizons therefore expose the decay of
    persistence-like shortcuts without fitting on validation or test data.
    """

    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("forecast horizons must be positive")
    if event_radius < 0:
        raise ValueError("event_radius must be nonnegative")
    train_values = _target_values(train, target)
    scale = max(float(np.var(train_values)), scale_floor)
    constant = float(np.mean(train_values))
    ar = _fit_linear_shortcut(train, target, include_inputs=False)
    arx = _fit_linear_shortcut(train, target, include_inputs=True)
    records: list[ShortcutMetrics] = []
    for horizon in horizons:
        for name in ("persistence", "constant", "ar", "arx"):
            errors: list[float] = []
            event_errors: list[float] = []
            for trajectory in evaluation.trajectories:
                values = _trajectory_target(trajectory, target)
                event_mask = _event_mask(trajectory, event_radius)
                for start in range(0, len(values) - horizon):
                    stop = start + horizon
                    prediction = _forecast(
                        name, trajectory, values, start, stop, constant, ar, arx
                    )
                    squared_error = float((prediction - values[stop]) ** 2)
                    errors.append(squared_error)
                    if bool(np.any(event_mask[start + 1 : stop + 1])):
                        event_errors.append(squared_error)
            raw = float(np.mean(errors)) if errors else float("nan")
            event_raw = float(np.mean(event_errors)) if event_errors else None
            records.append(
                ShortcutMetrics(
                    model=name,
                    horizon=horizon,
                    raw_mse=raw,
                    normalized_mse=raw / scale,
                    event_raw_mse=event_raw,
                    event_normalized_mse=(
                        event_raw / scale if event_raw is not None else None
                    ),
                    event_sample_fraction=(
                        len(event_errors) / len(errors) if errors else 0.0
                    ),
                    sample_count=len(errors),
                )
            )
    return tuple(records)


def audit_excitation(
    split: DatasetSplit,
    target: str,
    *,
    event_radius: int = 30,
    active_tolerance: float = 1e-12,
) -> ExcitationMetrics:
    """Summarize target motion and externally supplied input variation."""

    values = _target_values(split, target)
    increments = np.concatenate(
        [np.diff(_trajectory_target(item, target)) for item in split.trajectories]
    )
    target_variance = float(np.var(values))
    input_values: list[NDArray[np.float64]] = []
    input_changes: list[NDArray[np.bool_]] = []
    event_masks: list[NDArray[np.bool_]] = []
    for trajectory in split.trajectories:
        matrix = _input_matrix(trajectory)
        if matrix.shape[1]:
            input_values.append(matrix.reshape(-1))
            input_changes.append(
                np.any(np.abs(np.diff(matrix, axis=0)) > active_tolerance, axis=1)
            )
        event_masks.append(_event_mask(trajectory, event_radius))
    flattened = np.concatenate(input_values) if input_values else np.zeros(0)
    changes = (
        np.concatenate(input_changes) if input_changes else np.zeros(0, dtype=bool)
    )
    masks = np.concatenate(event_masks)
    return ExcitationMetrics(
        target=target,
        sample_count=len(values),
        trajectory_count=len(split.trajectories),
        target_variance=target_variance,
        increment_variance_ratio=(
            float(np.var(increments)) / max(target_variance, 1e-12)
            if len(increments)
            else 0.0
        ),
        active_input_fraction=(
            float(np.mean(np.abs(flattened) > active_tolerance))
            if len(flattened)
            else 0.0
        ),
        input_change_fraction=float(np.mean(changes)) if len(changes) else 0.0,
        distinct_input_levels=(
            len(np.unique(np.round(flattened, decimals=10))) if len(flattened) else 0
        ),
        event_window_fraction=float(np.mean(masks)) if len(masks) else 0.0,
    )


def audit_response_phases(
    train: DatasetSplit,
    evaluation: DatasetSplit,
    target: str,
    horizons: tuple[int, ...] = (1, 5, 10, 30),
    *,
    scale_floor: float = 1e-8,
) -> tuple[ResponsePhaseMetrics, ...]:
    """Evaluate shortcuts during rise, recovery, and near-equilibrium phases.

    Phases are defined from the held-out reference trajectory solely for
    post-selection scoring. They are never inputs to fitting or forecasting.
    """

    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("forecast horizons must be positive")
    train_values = _target_values(train, target)
    scale = max(float(np.var(train_values)), scale_floor)
    constant = float(np.mean(train_values))
    ar = _fit_linear_shortcut(train, target, include_inputs=False)
    arx = _fit_linear_shortcut(train, target, include_inputs=True)
    buckets: dict[tuple[ShortcutName, int, str], list[float]] = {}
    for trajectory in evaluation.trajectories:
        values = _trajectory_target(trajectory, target)
        phases = _response_phase_labels(values)
        for horizon in horizons:
            for start in range(0, len(values) - horizon):
                stop = start + horizon
                phase = phases[stop]
                for name in ("persistence", "constant", "ar", "arx"):
                    prediction = _forecast(
                        name, trajectory, values, start, stop, constant, ar, arx
                    )
                    buckets.setdefault((name, horizon, phase), []).append(
                        float((prediction - values[stop]) ** 2)
                    )
    records: list[ResponsePhaseMetrics] = []
    for (model, horizon, phase), errors in sorted(buckets.items()):
        raw = float(np.mean(errors))
        records.append(
            ResponsePhaseMetrics(
                model=model,
                horizon=horizon,
                phase=phase,  # type: ignore[arg-type]
                raw_mse=raw,
                normalized_mse=raw / scale,
                sample_count=len(errors),
            )
        )
    return tuple(records)


def downsample_split(split: DatasetSplit, stride: int) -> DatasetSplit:
    """Return a deterministic stride-downsampled split for diagnostic use."""

    if stride < 1:
        raise ValueError("downsampling stride must be positive")
    trajectories = tuple(
        Trajectory(
            trajectory_id=item.trajectory_id,
            time=np.array(item.time[::stride], copy=True),
            targets={
                name: np.array(values[::stride], copy=True)
                for name, values in item.targets.items()
            },
            auxiliaries={
                name: np.array(values[::stride], copy=True)
                for name, values in item.auxiliaries.items()
            },
            external_inputs={
                name: np.array(values[::stride], copy=True)
                for name, values in item.external_inputs.items()
            },
            fixed_covariates=dict(item.fixed_covariates),
            derivatives={
                name: np.array(values[::stride], copy=True)
                for name, values in item.derivatives.items()
            },
        )
        for item in split.trajectories
    )
    return DatasetSplit(
        name=split.name,
        trajectories=trajectories,
        fingerprint=f"{split.fingerprint}:stride={stride}",
    )


def _fit_linear_shortcut(
    split: DatasetSplit,
    target: str,
    *,
    include_inputs: bool,
) -> _LinearShortcut:
    rows: list[NDArray[np.float64]] = []
    responses: list[NDArray[np.float64]] = []
    for trajectory in split.trajectories:
        values = _trajectory_target(trajectory, target)
        inputs = _input_matrix(trajectory)
        columns = [np.ones(len(values) - 1), values[:-1]]
        if include_inputs:
            columns.extend(inputs[:-1, index] for index in range(inputs.shape[1]))
        rows.append(np.column_stack(columns))
        responses.append(values[1:])
    coefficients, *_ = np.linalg.lstsq(
        np.vstack(rows), np.concatenate(responses), rcond=None
    )
    return _LinearShortcut(
        intercept=float(coefficients[0]),
        target_coefficient=float(coefficients[1]),
        input_coefficients=np.asarray(coefficients[2:], dtype=np.float64),
    )


def _forecast(
    name: ShortcutName,
    trajectory: Trajectory,
    values: NDArray[np.float64],
    start: int,
    stop: int,
    constant: float,
    ar: _LinearShortcut,
    arx: _LinearShortcut,
) -> float:
    if name == "constant":
        return constant
    prediction = float(values[start])
    if name == "persistence":
        return prediction
    model = arx if name == "arx" else ar
    inputs = _input_matrix(trajectory)
    for index in range(start, stop):
        prediction = model.intercept + model.target_coefficient * prediction
        if len(model.input_coefficients):
            prediction += float(model.input_coefficients @ inputs[index])
    return prediction


def _event_mask(trajectory: Trajectory, radius: int) -> NDArray[np.bool_]:
    inputs = _input_matrix(trajectory)
    mask = np.zeros(len(trajectory.time), dtype=bool)
    if not inputs.shape[1]:
        return mask
    active = np.any(np.abs(inputs) > 1e-12, axis=1)
    changed = np.zeros(len(mask), dtype=bool)
    changed[1:] = np.any(np.abs(np.diff(inputs, axis=0)) > 1e-12, axis=1)
    for index in np.flatnonzero(active | changed):
        mask[max(0, index - radius) : min(len(mask), index + radius + 1)] = True
    return mask


def _response_phase_labels(values: NDArray[np.float64]) -> NDArray[np.str_]:
    baseline = float(values[0])
    displacement = values - baseline
    peak = int(np.argmax(np.abs(displacement)))
    amplitude = max(float(np.max(np.abs(displacement))), 1e-12)
    labels = np.full(len(values), "near_equilibrium", dtype="U16")
    active = np.abs(displacement) >= 0.1 * amplitude
    indices = np.arange(len(values))
    peak_radius = max(1, round(0.02 * len(values)))
    labels[(indices < peak - peak_radius) & active] = "rise"
    labels[(np.abs(indices - peak) <= peak_radius) & active] = "peak"
    labels[(indices > peak + peak_radius) & active] = "recovery"
    labels[~active] = "near_equilibrium"
    return labels


def _input_matrix(trajectory: Trajectory) -> NDArray[np.float64]:
    if not trajectory.external_inputs:
        return np.empty((len(trajectory.time), 0), dtype=np.float64)
    numeric: list[NDArray[np.float64]] = []
    for name in sorted(trajectory.external_inputs):
        try:
            values = np.asarray(trajectory.external_inputs[name], dtype=np.float64)
        except (TypeError, ValueError):
            # Public JSON event schedules accompany a separate numeric forcing
            # channel in the affected benchmark. They are metadata, not an ARX
            # regressor, and must not be coerced to arbitrary numeric values.
            continue
        if np.all(np.isfinite(values)):
            numeric.append(values)
    if not numeric:
        return np.empty((len(trajectory.time), 0), dtype=np.float64)
    return np.column_stack(numeric)


def _target_values(split: DatasetSplit, target: str) -> NDArray[np.float64]:
    return np.concatenate(
        [_trajectory_target(trajectory, target) for trajectory in split.trajectories]
    )


def _trajectory_target(
    trajectory: Trajectory,
    target: str,
) -> NDArray[np.float64]:
    try:
        return trajectory.targets[target]
    except KeyError as exc:
        raise ValueError(f"unknown target {target!r}") from exc

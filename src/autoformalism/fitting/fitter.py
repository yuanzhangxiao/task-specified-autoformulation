"""Bounded multistart least-squares fitting for compiled ODE candidates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from autoformalism.data import DatasetSplit, TrainingScaler, Trajectory
from autoformalism.expressions import CompiledModel
from autoformalism.fitting.models import (
    EvaluationMetrics,
    FailureCounter,
    FitConfig,
    FitResult,
    OptimizationDiagnostic,
)
from autoformalism.fitting.simulation import simulate_trajectory
from autoformalism.schemas import ParameterScope


@dataclass(frozen=True)
class _Variable:
    name: str
    lower: float
    upper: float
    start_lower: float
    start_upper: float


@dataclass(frozen=True)
class _Decoded:
    parameters: Mapping[str, float]
    global_initials: Mapping[str, float]
    trajectory_initials: Mapping[str, Mapping[str, float]]


def fit_candidate(
    model: CompiledModel,
    training: DatasetSplit,
    validation: DatasetSplit,
    config: FitConfig | None = None,
) -> FitResult:
    """Fit global quantities on train, then evaluate train and validation."""
    settings = config or FitConfig()
    _validate_splits(model, training, validation)
    scaler = TrainingScaler().fit(training)
    target_scales = {
        channel: scaler.scales[f"target:{channel}"].standard_deviation
        for channel in model.validated.context.targets
    }
    variables = _training_variables(model, training)
    lower = np.asarray([item.lower for item in variables])
    upper = np.asarray([item.upper for item in variables])
    rng = np.random.default_rng(settings.random_seed)
    starts = _starts(variables, settings.number_of_starts, rng)
    residual_size = _residual_size(training, model)
    outcomes: list[tuple[object, FailureCounter]] = []

    def decode(values: NDArray[np.float64]) -> _Decoded:
        return _decode_training(model, training, variables, values)

    for start in starts:
        counter = FailureCounter()
        result = least_squares(
            _residual_function(
                model,
                training.trajectories,
                decode,
                target_scales,
                settings,
                counter,
                residual_size,
            ),
            start,
            bounds=(lower, upper),
            max_nfev=settings.maximum_function_evaluations,
        )
        outcomes.append((result, counter))

    best_index = min(
        range(len(outcomes)),
        key=lambda index: float(outcomes[index][0].cost),
    )
    best = outcomes[best_index][0]
    decoded = decode(best.x)
    diagnostics = tuple(
        _diagnostic(index, result, counter, variables, settings)
        for index, (result, counter) in enumerate(outcomes)
    )
    train_metrics = _evaluate(
        model,
        training,
        decoded.parameters,
        decoded.global_initials,
        decoded.trajectory_initials,
        target_scales,
        settings,
    )
    validation_initials = _fit_validation_initials(
        model,
        validation,
        decoded.parameters,
        decoded.global_initials,
        target_scales,
        settings,
    )
    validation_metrics = _evaluate(
        model,
        validation,
        decoded.parameters,
        decoded.global_initials,
        validation_initials,
        target_scales,
        settings,
    )
    succeeded = bool(
        bool(best.success)
        and not train_metrics.failed_trajectories
        and np.isfinite(train_metrics.normalized_mse)
    )
    return FitResult(
        success=succeeded,
        global_parameters=decoded.parameters,
        global_initial_conditions=decoded.global_initials,
        training_trajectory_initial_conditions=decoded.trajectory_initials,
        validation_trajectory_initial_conditions=validation_initials,
        training_metrics=train_metrics,
        validation_metrics=validation_metrics,
        diagnostics=diagnostics,
        best_start_index=best_index,
        target_scales=target_scales,
        message=None if succeeded else "best fit contains numerical failures",
    )


def evaluate_fitted_candidate(
    model: CompiledModel,
    split: DatasetSplit,
    *,
    global_parameters: Mapping[str, float],
    global_initial_conditions: Mapping[str, float],
    target_scales: Mapping[str, float],
    config: FitConfig | None = None,
    fit_trajectory_initial_conditions: bool = True,
) -> tuple[Mapping[str, Mapping[str, float]], EvaluationMetrics]:
    """Evaluate frozen globals with fitted or target-free local initials."""
    settings = config or FitConfig()
    local_initials = (
        _fit_validation_initials(
            model,
            split,
            global_parameters,
            global_initial_conditions,
            target_scales,
            settings,
        )
        if fit_trajectory_initial_conditions
        else _midpoint_trajectory_initials(model, split)
    )
    metrics = _evaluate(
        model,
        split,
        global_parameters,
        global_initial_conditions,
        local_initials,
        target_scales,
        settings,
    )
    return local_initials, metrics


def _midpoint_trajectory_initials(
    model: CompiledModel,
    split: DatasetSplit,
) -> Mapping[str, Mapping[str, float]]:
    local = {
        item.state: (
            item.initialization_range.lower + item.initialization_range.upper
        )
        / 2.0
        for item in model.validated.candidate.initial_conditions
        if item.scope is ParameterScope.TRAJECTORY_SPECIFIC
    }
    return {
        trajectory.trajectory_id: dict(local) for trajectory in split.trajectories
    }


def _validate_splits(
    model: CompiledModel,
    training: DatasetSplit,
    validation: DatasetSplit,
) -> None:
    from autoformalism.data import SplitName

    if training.name is not SplitName.TRAIN:
        raise ValueError("fitting requires the training split")
    if validation.name is not SplitName.VALIDATION:
        raise ValueError("validation evaluation requires the validation split")
    if not training.trajectories:
        raise ValueError("training split has no trajectories")
    if not validation.trajectories:
        raise ValueError("validation split has no trajectories")
    for parameter in model.validated.candidate.parameters:
        if parameter.scope is not ParameterScope.GLOBAL:
            raise ValueError(
                "trajectory-specific model parameters are not supported; "
                "only latent initial conditions may be trajectory-specific"
            )
    targets = set(model.validated.context.targets)
    for split in (training, validation):
        identifiers: set[str] = set()
        for trajectory in split.trajectories:
            if trajectory.trajectory_id in identifiers:
                raise ValueError(
                    f"duplicate trajectory id in {split.name.value}: "
                    f"{trajectory.trajectory_id}"
                )
            identifiers.add(trajectory.trajectory_id)
            if set(trajectory.targets) != targets:
                raise ValueError(
                    f"trajectory {trajectory.trajectory_id} target mismatch"
                )


def _training_variables(
    model: CompiledModel,
    split: DatasetSplit,
) -> tuple[_Variable, ...]:
    result: list[_Variable] = []
    for parameter in model.validated.candidate.parameters:
        result.append(
            _Variable(
                f"parameter:{parameter.name}",
                parameter.bounds.lower,
                parameter.bounds.upper,
                parameter.initialization_range.lower,
                parameter.initialization_range.upper,
            )
        )
    initials = {
        item.state: item for item in model.validated.candidate.initial_conditions
    }
    for state in model.state_names:
        initial = initials[state]
        value_range = initial.initialization_range
        if initial.scope is ParameterScope.GLOBAL:
            result.append(
                _Variable(
                    f"initial:{state}",
                    value_range.lower,
                    value_range.upper,
                    value_range.lower,
                    value_range.upper,
                )
            )
    for trajectory in split.trajectories:
        for state in model.state_names:
            initial = initials[state]
            if initial.scope is ParameterScope.TRAJECTORY_SPECIFIC:
                value_range = initial.initialization_range
                result.append(
                    _Variable(
                        f"initial:{trajectory.trajectory_id}:{state}",
                        value_range.lower,
                        value_range.upper,
                        value_range.lower,
                        value_range.upper,
                    )
                )
    return tuple(result)


def _starts(
    variables: Sequence[_Variable],
    count: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], ...]:
    lower = np.asarray([item.start_lower for item in variables])
    upper = np.asarray([item.start_upper for item in variables])
    midpoint = (lower + upper) / 2.0
    return (midpoint, *(rng.uniform(lower, upper) for _ in range(count - 1)))


def _decode_training(
    model: CompiledModel,
    split: DatasetSplit,
    variables: Sequence[_Variable],
    values: NDArray[np.float64],
) -> _Decoded:
    named = dict(zip((item.name for item in variables), values, strict=True))
    parameters = {
        parameter.name: float(named[f"parameter:{parameter.name}"])
        for parameter in model.validated.candidate.parameters
    }
    initials = {
        item.state: item for item in model.validated.candidate.initial_conditions
    }
    global_initials = {
        state: float(named[f"initial:{state}"])
        for state in model.state_names
        if initials[state].scope is ParameterScope.GLOBAL
    }
    trajectory_initials = {
        trajectory.trajectory_id: {
            state: float(named[f"initial:{trajectory.trajectory_id}:{state}"])
            for state in model.state_names
            if initials[state].scope is ParameterScope.TRAJECTORY_SPECIFIC
        }
        for trajectory in split.trajectories
    }
    return _Decoded(parameters, global_initials, trajectory_initials)


def _residual_function(
    model: CompiledModel,
    trajectories: Sequence[Trajectory],
    decode: Callable[[NDArray[np.float64]], _Decoded],
    scales: Mapping[str, float],
    config: FitConfig,
    failures: FailureCounter,
    residual_size: int,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    def residual(values: NDArray[np.float64]) -> NDArray[np.float64]:
        decoded = decode(values)
        pieces: list[np.ndarray] = []
        for trajectory in trajectories:
            initial = {
                **decoded.global_initials,
                **decoded.trajectory_initials.get(trajectory.trajectory_id, {}),
            }
            simulation = simulate_trajectory(
                model, trajectory, decoded.parameters, initial, config
            )
            if not simulation.success:
                failures.count += 1
                return np.full(residual_size, config.failure_penalty)
            for channel in model.validated.context.targets:
                pieces.append(
                    (simulation.predictions[channel] - trajectory.targets[channel])
                    / scales[channel]
                )
        return np.concatenate(pieces)

    return residual


def _residual_size(split: DatasetSplit, model: CompiledModel) -> int:
    return sum(
        trajectory.number_of_rows * len(model.validated.context.targets)
        for trajectory in split.trajectories
    )


def _diagnostic(
    index: int,
    result: object,
    failures: FailureCounter,
    variables: Sequence[_Variable],
    config: FitConfig,
) -> OptimizationDiagnostic:
    lower_hits: list[str] = []
    upper_hits: list[str] = []
    for variable, value in zip(variables, result.x, strict=True):
        width = variable.upper - variable.lower
        tolerance = config.bound_tolerance * max(1.0, width)
        if value - variable.lower <= tolerance:
            lower_hits.append(variable.name)
        if variable.upper - value <= tolerance:
            upper_hits.append(variable.name)
    return OptimizationDiagnostic(
        start_index=index,
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        cost=float(result.cost),
        function_evaluations=int(result.nfev),
        integration_failures=failures.count,
        parameters_at_lower_bound=tuple(lower_hits),
        parameters_at_upper_bound=tuple(upper_hits),
    )


def _fit_validation_initials(
    model: CompiledModel,
    validation: DatasetSplit,
    parameters: Mapping[str, float],
    global_initials: Mapping[str, float],
    scales: Mapping[str, float],
    config: FitConfig,
) -> Mapping[str, Mapping[str, float]]:
    local_specs = [
        item
        for item in model.validated.candidate.initial_conditions
        if item.scope is ParameterScope.TRAJECTORY_SPECIFIC
    ]
    if not local_specs:
        return {trajectory.trajectory_id: {} for trajectory in validation.trajectories}
    result: dict[str, Mapping[str, float]] = {}
    rng = np.random.default_rng(config.random_seed + 1)
    for trajectory in validation.trajectories:
        variables = tuple(
            _Variable(
                f"initial:{trajectory.trajectory_id}:{item.state}",
                item.initialization_range.lower,
                item.initialization_range.upper,
                item.initialization_range.lower,
                item.initialization_range.upper,
            )
            for item in local_specs
        )
        lower = np.asarray([item.lower for item in variables])
        upper = np.asarray([item.upper for item in variables])

        def residual(
            values: NDArray[np.float64],
            current_trajectory: Trajectory = trajectory,
        ) -> NDArray[np.float64]:
            local = dict(
                zip((item.state for item in local_specs), values, strict=True)
            )
            simulation = simulate_trajectory(
                model,
                current_trajectory,
                parameters,
                {**global_initials, **local},
                config,
            )
            if not simulation.success:
                return np.full(
                    current_trajectory.number_of_rows
                    * len(model.validated.context.targets),
                    config.failure_penalty,
                )
            return np.concatenate(
                [
                    (
                        simulation.predictions[channel]
                        - current_trajectory.targets[channel]
                    )
                    / scales[channel]
                    for channel in model.validated.context.targets
                ]
            )

        fits = [
            least_squares(
                residual,
                start,
                bounds=(lower, upper),
                max_nfev=config.maximum_function_evaluations,
            )
            for start in _starts(variables, config.number_of_starts, rng)
        ]
        best = min(fits, key=lambda fit: float(fit.cost))
        result[trajectory.trajectory_id] = dict(
            zip((item.state for item in local_specs), best.x, strict=True)
        )
    return result


def _evaluate(
    model: CompiledModel,
    split: DatasetSplit,
    parameters: Mapping[str, float],
    global_initials: Mapping[str, float],
    trajectory_initials: Mapping[str, Mapping[str, float]],
    scales: Mapping[str, float],
    config: FitConfig,
) -> EvaluationMetrics:
    squared: dict[str, list[np.ndarray]] = {
        channel: [] for channel in model.validated.context.targets
    }
    failed: list[str] = []
    for trajectory in split.trajectories:
        simulation = simulate_trajectory(
            model,
            trajectory,
            parameters,
            {
                **global_initials,
                **trajectory_initials.get(trajectory.trajectory_id, {}),
            },
            config,
        )
        if not simulation.success:
            failed.append(trajectory.trajectory_id)
            for channel in squared:
                squared[channel].append(
                    np.full(trajectory.number_of_rows, config.failure_penalty**2)
                )
            continue
        for channel in squared:
            normalized = (
                simulation.predictions[channel] - trajectory.targets[channel]
            ) / scales[channel]
            squared[channel].append(normalized**2)
    per_target = {
        channel: float(np.mean(np.concatenate(values)))
        for channel, values in squared.items()
    }
    aggregate = float(
        np.mean(np.concatenate([np.concatenate(values) for values in squared.values()]))
    )
    return EvaluationMetrics(aggregate, per_target, tuple(failed))

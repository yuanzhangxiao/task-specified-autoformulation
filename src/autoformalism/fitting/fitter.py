"""Bounded multistart least-squares fitting for compiled ODE candidates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import monotonic

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, least_squares

from autoformalism.data import DatasetSplit, TrainingScaler, Trajectory
from autoformalism.expressions import CompiledModel, RuntimeExpressionError
from autoformalism.fitting.models import (
    EvaluationMetrics,
    FailureCounter,
    FitConfig,
    FitResult,
    OptimizationDiagnostic,
)
from autoformalism.fitting.simulation import simulate_trajectory, trajectory_forcing
from autoformalism.schemas import InitialConditionSpec, ParameterScope


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
    *,
    initial_global_parameters: Mapping[str, float] | None = None,
) -> FitResult:
    """Fit global quantities on train, then evaluate train and validation."""
    settings = config or FitConfig()
    deadline = (
        None
        if settings.maximum_wall_time_seconds is None
        else monotonic() + settings.maximum_wall_time_seconds
    )
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
    starts = _starts(
        variables,
        settings.number_of_starts,
        rng,
        preferred_parameters=initial_global_parameters,
    )
    residual_size = _residual_size(training, model)
    outcomes: list[tuple[object, FailureCounter]] = []
    derivative_backend = settings.allow_derivative_regression and (
        _can_use_derivative_regression(model, training)
    )

    def decode(values: NDArray[np.float64]) -> _Decoded:
        return _decode_training(model, training, variables, values)

    for start in starts:
        counter = FailureCounter()
        residual = (
            _derivative_residual_function(
                model,
                training.trajectories,
                decode,
                settings,
                counter,
                deadline,
            )
            if derivative_backend
            else _residual_function(
                model,
                training.trajectories,
                decode,
                target_scales,
                settings,
                counter,
                residual_size,
                deadline,
            )
        )
        try:
            if len(variables):
                result = least_squares(
                    residual,
                    start,
                    bounds=(lower, upper),
                    max_nfev=settings.maximum_function_evaluations,
                )
            else:
                values = residual(start)
                result = OptimizeResult(
                    x=start,
                    success=True,
                    status=1,
                    message="no free parameters",
                    cost=float(0.5 * np.dot(values, values)),
                    nfev=1,
                )
        except TimeoutError as exc:
            counter.record(str(exc))
            result = OptimizeResult(
                x=start,
                success=False,
                status=-2,
                message=str(exc),
                cost=0.5 * residual_size * settings.failure_penalty**2,
                nfev=0,
            )
        outcomes.append((result, counter))
        if int(result.status) == -2:
            break

    best_index = min(
        range(len(outcomes)),
        key=lambda index: float(outcomes[index][0].cost),
    )
    best = outcomes[best_index][0]
    decoded = decode(best.x)
    diagnostics = tuple(
        _diagnostic(
            index,
            result,
            counter,
            variables,
            settings,
            backend=(
                "derivative_regression"
                if derivative_backend
                else "rollout_least_squares"
            ),
        )
        for index, (result, counter) in enumerate(outcomes)
    )
    if int(best.status) == -2:
        return _timeout_fit_result(
            model,
            training,
            validation,
            decoded,
            diagnostics,
            best_index,
            target_scales,
            settings.failure_penalty,
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
    evaluation_budget_reached = int(best.status) == 0
    succeeded = bool(
        (bool(best.success) or evaluation_budget_reached)
        and np.isfinite(float(best.cost))
        and np.isfinite(best.x).all()
        and not train_metrics.failed_trajectories
        and not validation_metrics.failed_trajectories
        and np.isfinite(train_metrics.normalized_mse)
        and np.isfinite(validation_metrics.normalized_mse)
    )
    if succeeded and evaluation_budget_reached:
        message = (
            "optimizer evaluation budget reached before convergence; "
            "finite fitted candidate retained"
        )
    elif succeeded:
        message = None
    else:
        message = "best fit contains numerical failures or invalid optimizer state"
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
        message=message,
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
    if model.validated.context.lagged_targets:
        return {trajectory.trajectory_id: {} for trajectory in split.trajectories}
    local = {
        item.state: (
            item.initialization_range.lower + item.initialization_range.upper
        )
        / 2.0
        for item in model.validated.candidate.initial_conditions
        if item.scope is ParameterScope.TRAJECTORY_SPECIFIC
        and item.state not in model.observed_state_channels
        and item.initialization_range is not None
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
    if model.validated.context.lagged_targets:
        return tuple(result)
    for state in model.state_names:
        if state in model.observed_state_channels:
            continue
        initial = initials[state]
        value_range = initial.initialization_range
        if value_range is None:
            # Fixed and analytic initializers are evaluated by the simulator;
            # only explicit ranges introduce optimization variables.
            continue
        if (
            initial.scope is ParameterScope.GLOBAL
            and value_range.lower < value_range.upper
        ):
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
            if state in model.observed_state_channels:
                continue
            initial = initials[state]
            if initial.scope is ParameterScope.TRAJECTORY_SPECIFIC:
                value_range = initial.initialization_range
                if value_range is None:
                    continue
                if value_range.lower == value_range.upper:
                    continue
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
    *,
    preferred_parameters: Mapping[str, float] | None = None,
) -> tuple[NDArray[np.float64], ...]:
    lower = np.asarray([item.start_lower for item in variables])
    upper = np.asarray([item.start_upper for item in variables])
    midpoint = (lower + upper) / 2.0
    if preferred_parameters is not None:
        declared = {
            item.name.removeprefix("parameter:")
            for item in variables
            if item.name.startswith("parameter:")
        }
        unknown = sorted(set(preferred_parameters) - declared)
        if unknown:
            raise ValueError(f"warm start contains unknown parameters: {unknown}")
        midpoint = midpoint.copy()
        for index, variable in enumerate(variables):
            if not variable.name.startswith("parameter:"):
                continue
            name = variable.name.removeprefix("parameter:")
            if name not in preferred_parameters:
                continue
            value = float(preferred_parameters[name])
            if not np.isfinite(value):
                raise ValueError(f"warm-start parameter {name} must be finite")
            if value < variable.lower or value > variable.upper:
                raise ValueError(
                    f"warm-start parameter {name}={value} is outside "
                    f"[{variable.lower}, {variable.upper}]"
                )
            midpoint[index] = value
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
    if model.validated.context.lagged_targets:
        return _Decoded(
            parameters,
            {},
            {trajectory.trajectory_id: {} for trajectory in split.trajectories},
        )
    initials = {
        item.state: item for item in model.validated.candidate.initial_conditions
    }
    global_initials = {
        state: (
            value_range.lower
            if value_range.lower == value_range.upper
            else float(named[f"initial:{state}"])
        )
        for state in model.state_names
        if initials[state].scope is ParameterScope.GLOBAL
        and state not in model.observed_state_channels
        and (value_range := initials[state].initialization_range) is not None
    }
    trajectory_initials = {
        trajectory.trajectory_id: {
            state: (
                value_range.lower
                if value_range.lower == value_range.upper
                else float(named[f"initial:{trajectory.trajectory_id}:{state}"])
            )
            for state in model.state_names
            if initials[state].scope is ParameterScope.TRAJECTORY_SPECIFIC
            and state not in model.observed_state_channels
            and (value_range := initials[state].initialization_range) is not None
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
    deadline: float | None,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    def residual(values: NDArray[np.float64]) -> NDArray[np.float64]:
        _check_deadline(deadline)
        decoded = decode(values)
        pieces: list[np.ndarray] = []
        for trajectory in trajectories:
            initial = {
                **decoded.global_initials,
                **decoded.trajectory_initials.get(trajectory.trajectory_id, {}),
            }
            simulation = simulate_trajectory(
                model,
                trajectory,
                decoded.parameters,
                initial,
                config,
                deadline=deadline,
            )
            if not simulation.success:
                failures.record(simulation.message)
                return np.full(residual_size, config.failure_penalty)
            for channel in model.validated.context.targets:
                start = 1 if model.validated.context.lagged_targets else 0
                pieces.append(
                    (
                        simulation.predictions[channel][start:]
                        - trajectory.targets[channel][start:]
                    )
                    / scales[channel]
                )
        return np.concatenate(pieces)

    return residual


def _can_use_derivative_regression(
    model: CompiledModel,
    training: DatasetSplit,
) -> bool:
    """Return whether every state and derivative is directly observed."""
    observed = model.observed_state_channels
    if set(observed) != set(model.state_names):
        return False
    return all(
        all(channel in trajectory.derivatives for channel in observed.values())
        for trajectory in training.trajectories
    )


def _derivative_residual_function(
    model: CompiledModel,
    trajectories: Sequence[Trajectory],
    decode: Callable[[NDArray[np.float64]], _Decoded],
    config: FitConfig,
    failures: FailureCounter,
    deadline: float | None,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Build a training-only RHS regression residual without ODE integration."""
    observed = model.observed_state_channels
    scales = {
        state: max(
            float(
                np.std(
                    np.concatenate(
                        [
                            trajectory.derivatives[channel]
                            for trajectory in trajectories
                        ]
                    )
                )
            ),
            1e-8,
        )
        for state, channel in observed.items()
    }
    skipped = 1 if model.validated.context.lagged_targets else 0
    residual_size = sum(
        (trajectory.number_of_rows - skipped) * len(model.state_names)
        for trajectory in trajectories
    )

    def residual(values: NDArray[np.float64]) -> NDArray[np.float64]:
        _check_deadline(deadline)
        decoded = decode(values)
        pieces: list[float] = []
        try:
            for trajectory in trajectories:
                for index in range(skipped, trajectory.number_of_rows):
                    _check_deadline(deadline)
                    forcing = trajectory_forcing(
                        model,
                        trajectory,
                        causal_index=max(0, index - 1),
                    )
                    state = np.asarray(
                        [
                            (
                                trajectory.targets[channel][index]
                                if channel in trajectory.targets
                                else trajectory.auxiliaries[channel][index]
                            )
                            for channel in (
                                observed[name] for name in model.state_names
                            )
                        ],
                        dtype=float,
                    )
                    predicted = model.rhs(
                        float(trajectory.time[index]),
                        state,
                        decoded.parameters,
                        forcing,
                    )
                    for state_index, state_name in enumerate(model.state_names):
                        channel = observed[state_name]
                        pieces.append(
                            (
                                float(predicted[state_index])
                                - float(trajectory.derivatives[channel][index])
                            )
                            / scales[state_name]
                        )
        except (
            ArithmeticError,
            RuntimeError,
            RuntimeExpressionError,
            TypeError,
            ValueError,
        ) as exc:
            failures.record(str(exc))
            return np.full(residual_size, config.failure_penalty)
        return np.asarray(pieces, dtype=float)

    return residual


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and monotonic() >= deadline:
        raise TimeoutError("fitting wall-clock limit reached")


def _timeout_fit_result(
    model: CompiledModel,
    training: DatasetSplit,
    validation: DatasetSplit,
    decoded: _Decoded,
    diagnostics: tuple[OptimizationDiagnostic, ...],
    best_index: int,
    target_scales: Mapping[str, float],
    failure_penalty: float,
) -> FitResult:
    """Return a checkpointable candidate failure instead of raising on timeout."""
    penalty_mse = failure_penalty**2
    per_target = dict.fromkeys(model.validated.context.targets, penalty_mse)
    return FitResult(
        success=False,
        global_parameters=decoded.parameters,
        global_initial_conditions=decoded.global_initials,
        training_trajectory_initial_conditions=decoded.trajectory_initials,
        validation_trajectory_initial_conditions={
            trajectory.trajectory_id: {} for trajectory in validation.trajectories
        },
        training_metrics=EvaluationMetrics(
            penalty_mse,
            per_target,
            tuple(trajectory.trajectory_id for trajectory in training.trajectories),
        ),
        validation_metrics=EvaluationMetrics(
            penalty_mse,
            per_target,
            tuple(trajectory.trajectory_id for trajectory in validation.trajectories),
        ),
        diagnostics=diagnostics,
        best_start_index=best_index,
        target_scales=target_scales,
        message="fitting wall-clock limit reached; candidate rejected",
    )


def _residual_size(split: DatasetSplit, model: CompiledModel) -> int:
    skipped = 1 if model.validated.context.lagged_targets else 0
    return sum(
        (trajectory.number_of_rows - skipped)
        * len(model.validated.context.targets)
        for trajectory in split.trajectories
    )


def _diagnostic(
    index: int,
    result: object,
    failures: FailureCounter,
    variables: Sequence[_Variable],
    config: FitConfig,
    *,
    backend: str = "rollout_least_squares",
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
        backend=backend,
        integration_failure_messages=tuple(failures.messages),
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
    """Initialize held-out latent states without fitting held-out targets."""
    if model.validated.context.lagged_targets:
        return {
            trajectory.trajectory_id: {} for trajectory in validation.trajectories
        }
    local_specs = [
        item
        for item in model.validated.candidate.initial_conditions
        if item.scope is ParameterScope.TRAJECTORY_SPECIFIC
        and item.state not in model.observed_state_channels
        and item.initialization_range is not None
    ]
    if not model.validated.context.lagged_targets:
        return _fit_open_loop_validation_initials(
            model,
            validation,
            parameters,
            global_initials,
            scales,
            config,
            local_specs,
        )
    midpoint_values = {
        item.state: (
            item.initialization_range.lower + item.initialization_range.upper
        )
        / 2.0
        for item in local_specs
    }
    return {
        trajectory.trajectory_id: dict(midpoint_values)
        for trajectory in validation.trajectories
    }


def _fit_open_loop_validation_initials(
    model: CompiledModel,
    validation: DatasetSplit,
    parameters: Mapping[str, float],
    global_initials: Mapping[str, float],
    scales: Mapping[str, float],
    config: FitConfig,
    local_specs: Sequence[InitialConditionSpec],
) -> Mapping[str, Mapping[str, float]]:
    """Retain legacy open-loop behavior for isolated numerical unit tests."""
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
                    np.full(
                        trajectory.number_of_rows
                        - (1 if model.validated.context.lagged_targets else 0),
                        config.failure_penalty**2,
                    )
                )
            continue
        for channel in squared:
            start = 1 if model.validated.context.lagged_targets else 0
            normalized = (
                simulation.predictions[channel][start:]
                - trajectory.targets[channel][start:]
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

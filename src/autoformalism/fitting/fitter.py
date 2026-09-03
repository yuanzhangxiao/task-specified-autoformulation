"""Bounded multistart least-squares fitting for compiled ODE candidates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import monotonic

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult, least_squares, lsq_linear

from autoformalism.data import (
    DatasetSplit,
    DerivativeProvenance,
    TrainingScaler,
    Trajectory,
)
from autoformalism.expressions import (
    CompiledModel,
    ReciprocalParameterTransformation,
    RuntimeExpressionError,
    validate_fixed_latent_basis_parameterization,
    validate_gmm_parameterization,
    validate_profiled_latent_basis_parameterization,
)
from autoformalism.fitting.models import (
    EvaluationMetrics,
    ExactDerivativeFitError,
    FailureCounter,
    FitConfig,
    FitResult,
    OptimizationDiagnostic,
)
from autoformalism.fitting.simulation import simulate_trajectory, trajectory_forcing
from autoformalism.schemas import (
    ConstraintEnforcement,
    ConstraintKind,
    ConstraintSpec,
    InitialConditionSpec,
    ParameterScope,
    ParameterSpec,
)


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


@dataclass(frozen=True)
class _ProfiledOutcome:
    result: OptimizeResult
    parameters: Mapping[str, float]
    failures: FailureCounter


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
    if settings.parameter_fit_strategy == "exact_derivative_linear_ridge":
        return _fit_exact_derivative_linear_ridge(
            model,
            training,
            validation,
            settings,
            target_scales,
            deadline,
        )
    if settings.parameter_fit_strategy == "fixed_latent_basis_linear_ridge":
        return _fit_fixed_latent_basis_linear_ridge(
            model,
            training,
            validation,
            settings,
            target_scales,
            deadline,
        )
    if settings.parameter_fit_strategy == "profiled_latent_basis_linear_ridge":
        return _fit_profiled_latent_basis_linear_ridge(
            model,
            training,
            validation,
            settings,
            target_scales,
            deadline,
            initial_global_parameters,
        )
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


def _fit_exact_derivative_linear_ridge(
    model: CompiledModel,
    training: DatasetSplit,
    validation: DatasetSplit,
    config: FitConfig,
    target_scales: Mapping[str, float],
    deadline: float | None,
) -> FitResult:
    """Fit graph-edge weights once by Eq. (11), then score causal rollouts.

    This is deliberately an oracle first milestone. Every modeled state and its
    exact derivative must be supplied. Validation derivatives are never used.
    Nonlinear basis-shape constants belong in the proposer-owned expression as
    numeric literals; only affine graph weights are solved here.
    """
    if not config.allow_derivative_regression:
        raise ExactDerivativeFitError(
            "exact_derivative_linear_ridge requires derivative regression"
        )
    validate_gmm_parameterization(model.validated)
    observed = model.direct_state_observation_channels
    missing_states = sorted(set(model.state_names) - set(observed))
    if missing_states:
        raise ExactDerivativeFitError(
            "exact derivative fitting requires every state to be directly "
            f"observed; missing={missing_states}"
        )
    for trajectory in training.trajectories:
        if trajectory.derivative_provenance is not DerivativeProvenance.EXACT:
            raise ExactDerivativeFitError(
                "exact derivative fitting refuses non-exact derivative labels: "
                f"{trajectory.trajectory_id} has "
                f"{trajectory.derivative_provenance.value!r} provenance"
            )
        missing = sorted(set(observed.values()) - set(trajectory.derivatives))
        if missing:
            raise ExactDerivativeFitError(
                f"trajectory {trajectory.trajectory_id} lacks exact derivatives: "
                f"{missing}"
            )
    parameter_specs = tuple(model.validated.candidate.parameters)
    parameter_names = tuple(item.name for item in parameter_specs)
    variables = tuple(
        _Variable(
            f"parameter:{item.name}",
            item.bounds.lower,
            item.bounds.upper,
            item.initialization_range.lower,
            item.initialization_range.upper,
        )
        for item in parameter_specs
    )
    anchor = {
        item.name: (item.bounds.lower + item.bounds.upper) / 2.0
        for item in parameter_specs
    }
    rows: list[NDArray[np.float64]] = []
    labels: list[float] = []
    for trajectory in training.trajectories:
        skipped = 1 if model.validated.context.lagged_targets else 0
        for index in range(skipped, trajectory.number_of_rows):
            _check_deadline(deadline)
            forcing = trajectory_forcing(
                model,
                trajectory,
                causal_index=max(0, index - 1),
            )
            state = np.asarray(
                [
                    _observed_value(trajectory, observed[name], index)
                    for name in model.state_names
                ],
                dtype=float,
            )
            time = float(trajectory.time[index])
            anchor_rhs = model.rhs(time, state, anchor, forcing)
            design = np.empty((len(model.state_names), len(parameter_names)))
            for parameter_index, spec in enumerate(parameter_specs):
                probe = dict(anchor)
                probe[spec.name] = float(spec.bounds.lower)
                delta = probe[spec.name] - anchor[spec.name]
                if delta == 0.0:  # schema forbids degenerate bounds
                    raise AssertionError("parameter probe has zero displacement")
                probe_rhs = model.rhs(time, state, probe, forcing)
                design[:, parameter_index] = (probe_rhs - anchor_rhs) / delta
            intercept = anchor_rhs - design @ np.asarray(
                [anchor[name] for name in parameter_names], dtype=float
            )
            for state_index, state_name in enumerate(model.state_names):
                channel = observed[state_name]
                rows.append(design[state_index].copy())
                labels.append(
                    float(trajectory.derivatives[channel][index])
                    - float(intercept[state_index])
                )

    matrix = (
        np.vstack(rows)
        if rows
        else np.empty((0, len(parameter_names)), dtype=float)
    )
    response = np.asarray(labels, dtype=float)
    if parameter_names:
        gram = matrix.T @ matrix
        rhs = matrix.T @ response
        regularized = gram + config.derivative_ridge_regularization * np.eye(
            len(parameter_names)
        )
        try:
            values = np.linalg.solve(regularized, rhs)
        except np.linalg.LinAlgError:
            values = np.linalg.lstsq(regularized, rhs, rcond=None)[0]
    else:
        values = np.empty(0, dtype=float)
    if not np.isfinite(values).all():
        raise ExactDerivativeFitError("closed-form ridge solution is nonfinite")

    bounds_error = [
        f"{spec.name}={value:.12g} outside "
        f"[{spec.bounds.lower:.12g}, {spec.bounds.upper:.12g}]"
        for spec, value in zip(parameter_specs, values, strict=True)
        if value < spec.bounds.lower or value > spec.bounds.upper
    ]
    if bounds_error:
        raise ExactDerivativeFitError(
            "unconstrained Eq. (11) solution violates proposer bounds; "
            + "; ".join(bounds_error)
        )

    parameters = dict(zip(parameter_names, values, strict=True))
    training_initials = _direct_observed_initials(model, training, observed)
    decoded = _Decoded(parameters, {}, training_initials)
    residual = matrix @ values - response
    result = OptimizeResult(
        x=values,
        success=True,
        status=1,
        message="closed-form exact-derivative ridge solution",
        cost=float(0.5 * np.dot(residual, residual)),
        nfev=1,
    )
    diagnostic = _diagnostic(
        0,
        result,
        FailureCounter(),
        variables,
        config,
        backend="exact_derivative_linear_ridge",
    )
    train_metrics = _evaluate(
        model,
        training,
        decoded.parameters,
        decoded.global_initials,
        decoded.trajectory_initials,
        target_scales,
        config,
    )
    validation_initials = _direct_observed_initials(model, validation, observed)
    validation_metrics = _evaluate(
        model,
        validation,
        decoded.parameters,
        decoded.global_initials,
        validation_initials,
        target_scales,
        config,
    )
    succeeded = bool(
        not train_metrics.failed_trajectories
        and not validation_metrics.failed_trajectories
        and np.isfinite(train_metrics.normalized_mse)
        and np.isfinite(validation_metrics.normalized_mse)
    )
    return FitResult(
        success=succeeded,
        global_parameters=decoded.parameters,
        global_initial_conditions=decoded.global_initials,
        training_trajectory_initial_conditions=decoded.trajectory_initials,
        validation_trajectory_initial_conditions=validation_initials,
        training_metrics=train_metrics,
        validation_metrics=validation_metrics,
        diagnostics=(diagnostic,),
        best_start_index=0,
        target_scales=target_scales,
        message=(
            None
            if succeeded
            else "closed-form parameters produced an invalid rollout"
        ),
    )


def _fit_fixed_latent_basis_linear_ridge(
    model: CompiledModel,
    training: DatasetSplit,
    validation: DatasetSplit,
    config: FitConfig,
    target_scales: Mapping[str, float],
    deadline: float | None,
) -> FitResult:
    """Fit affine weights while generating, rather than revealing, latent states.

    Exact derivatives are required only for directly observed dynamic states.
    The proposer owns the parameter-free latent dynamics and fixed latent
    initialization. During training, measured observed-state paths drive that
    fixed subsystem; no latent trajectory or latent derivative is supplied.
    The fitted model is then scored by the ordinary causal rollout evaluator.
    """
    if not config.allow_derivative_regression:
        raise ExactDerivativeFitError(
            "fixed_latent_basis_linear_ridge requires derivative regression"
        )
    report = validate_fixed_latent_basis_parameterization(model.validated)
    direct = model.direct_state_observation_channels
    observed_names = {
        item.name
        for item in model.validated.candidate.states
        if item.kind.value == "observed"
    }
    latent_names = {
        item.name
        for item in model.validated.candidate.states
        if item.kind.value == "latent"
    }
    missing_observed = sorted(observed_names - set(direct))
    mislabeled_latent = sorted(latent_names & set(direct))
    if missing_observed or mislabeled_latent:
        raise ExactDerivativeFitError(
            "fixed latent-basis fitting requires identity observations for all "
            "observed states and no direct observation of latent states; "
            f"missing_observed={missing_observed}, "
            f"directly_observed_latent={mislabeled_latent}"
        )
    for trajectory in training.trajectories:
        if trajectory.derivative_provenance is not DerivativeProvenance.EXACT:
            raise ExactDerivativeFitError(
                "fixed latent-basis fitting refuses non-exact observed "
                f"derivatives: {trajectory.trajectory_id} has "
                f"{trajectory.derivative_provenance.value!r} provenance"
            )
        missing = sorted(set(direct.values()) - set(trajectory.derivatives))
        if missing:
            raise ExactDerivativeFitError(
                f"trajectory {trajectory.trajectory_id} lacks exact observed "
                f"derivatives: {missing}"
            )

    parameter_specs = tuple(model.validated.candidate.parameters)
    parameter_names = report.parameter_names
    variables = tuple(
        _Variable(
            f"parameter:{item.name}",
            item.bounds.lower,
            item.bounds.upper,
            item.initialization_range.lower,
            item.initialization_range.upper,
        )
        for item in parameter_specs
    )
    anchor = {
        item.name: (item.bounds.lower + item.bounds.upper) / 2.0
        for item in parameter_specs
    }
    rows: list[NDArray[np.float64]] = []
    labels: list[float] = []
    for trajectory in training.trajectories:
        state_values = _conditioned_latent_state_matrix(
            model,
            trajectory,
            anchor,
            config,
            deadline,
        )
        skipped = 1 if model.validated.context.lagged_targets else 0
        for index in range(skipped, trajectory.number_of_rows):
            _check_deadline(deadline)
            forcing = trajectory_forcing(
                model,
                trajectory,
                causal_index=max(0, index - 1),
            )
            state = state_values[:, index]
            time = float(trajectory.time[index])
            anchor_rhs = model.rhs(time, state, anchor, forcing)
            design = np.empty((len(model.state_names), len(parameter_names)))
            for parameter_index, spec in enumerate(parameter_specs):
                probe = dict(anchor)
                probe[spec.name] = float(spec.bounds.lower)
                delta = probe[spec.name] - anchor[spec.name]
                if delta == 0.0:
                    raise AssertionError("parameter probe has zero displacement")
                probe_rhs = model.rhs(time, state, probe, forcing)
                design[:, parameter_index] = (probe_rhs - anchor_rhs) / delta
            intercept = anchor_rhs - design @ np.asarray(
                [anchor[name] for name in parameter_names], dtype=float
            )
            for state_name in sorted(observed_names):
                state_index = model.state_names.index(state_name)
                channel = direct[state_name]
                rows.append(design[state_index].copy())
                labels.append(
                    float(trajectory.derivatives[channel][index])
                    - float(intercept[state_index])
                )

    matrix = (
        np.vstack(rows)
        if rows
        else np.empty((0, len(parameter_names)), dtype=float)
    )
    response = np.asarray(labels, dtype=float)
    if parameter_names:
        regularized = matrix.T @ matrix + (
            config.derivative_ridge_regularization
            * np.eye(len(parameter_names))
        )
        rhs = matrix.T @ response
        try:
            values = np.linalg.solve(regularized, rhs)
        except np.linalg.LinAlgError:
            values = np.linalg.lstsq(regularized, rhs, rcond=None)[0]
    else:
        values = np.empty(0, dtype=float)
    if not np.isfinite(values).all():
        raise ExactDerivativeFitError("fixed latent-basis solution is nonfinite")
    bounds_error = [
        f"{spec.name}={value:.12g} outside "
        f"[{spec.bounds.lower:.12g}, {spec.bounds.upper:.12g}]"
        for spec, value in zip(parameter_specs, values, strict=True)
        if value < spec.bounds.lower or value > spec.bounds.upper
    ]
    if bounds_error:
        raise ExactDerivativeFitError(
            "unconstrained fixed latent-basis solution violates proposer bounds; "
            + "; ".join(bounds_error)
        )

    parameters = dict(zip(parameter_names, values, strict=True))
    training_initials = _direct_observed_initials(model, training, direct)
    validation_initials = _direct_observed_initials(model, validation, direct)
    residual = matrix @ values - response
    result = OptimizeResult(
        x=values,
        success=True,
        status=1,
        message="fixed latent-basis exact-derivative ridge solution",
        cost=float(0.5 * np.dot(residual, residual)),
        nfev=1,
    )
    diagnostic = _diagnostic(
        0,
        result,
        FailureCounter(),
        variables,
        config,
        backend="fixed_latent_basis_linear_ridge",
    )
    train_metrics = _evaluate(
        model,
        training,
        parameters,
        {},
        training_initials,
        target_scales,
        config,
    )
    validation_metrics = _evaluate(
        model,
        validation,
        parameters,
        {},
        validation_initials,
        target_scales,
        config,
    )
    succeeded = bool(
        not train_metrics.failed_trajectories
        and not validation_metrics.failed_trajectories
        and np.isfinite(train_metrics.normalized_mse)
        and np.isfinite(validation_metrics.normalized_mse)
    )
    return FitResult(
        success=succeeded,
        global_parameters=parameters,
        global_initial_conditions={},
        training_trajectory_initial_conditions=training_initials,
        validation_trajectory_initial_conditions=validation_initials,
        training_metrics=train_metrics,
        validation_metrics=validation_metrics,
        diagnostics=(diagnostic,),
        best_start_index=0,
        target_scales=target_scales,
        message=(
            None
            if succeeded
            else "fixed latent-basis parameters produced an invalid rollout"
        ),
    )


def _fit_profiled_latent_basis_linear_ridge(
    model: CompiledModel,
    training: DatasetSplit,
    validation: DatasetSplit,
    config: FitConfig,
    target_scales: Mapping[str, float],
    deadline: float | None,
    initial_global_parameters: Mapping[str, float] | None,
) -> FitResult:
    """Profile affine weights inside a small nonlinear latent-shape search.

    Exact derivatives are supplied only for directly observed states. For each
    outer latent-shape vector, the candidate's latent subsystem is integrated
    while conditioning on measured observed paths, then all remaining affine
    RHS weights are solved by bounded ridge least squares. Validation continues
    to use a causal rollout of the complete fitted model.
    """
    if not config.allow_derivative_regression:
        raise ExactDerivativeFitError(
            "profiled_latent_basis_linear_ridge requires derivative regression"
        )
    report = validate_profiled_latent_basis_parameterization(model.validated)
    direct, observed_names = _validate_partial_exact_derivative_contract(
        model,
        training,
        backend_label="profiled latent-basis fitting",
    )
    derivative_scales = {
        state_name: max(
            float(
                np.std(
                    np.concatenate(
                        [
                            trajectory.derivatives[direct[state_name]]
                            for trajectory in training.trajectories
                        ]
                    )
                )
            ),
            1e-8,
        )
        for state_name in observed_names
    }
    specs_by_name = {
        item.name: item for item in model.validated.candidate.parameters
    }
    outer_specs = tuple(
        specs_by_name[name] for name in report.latent_shape_parameter_names
    )
    inner_specs = tuple(
        specs_by_name[name] for name in report.affine_parameter_names
    )
    active_transformations = (
        report.reciprocal_transformations
        if config.use_certified_reciprocal_coordinates
        else ()
    )
    reciprocal_by_name = {
        item.parameter_name: item for item in active_transformations
    }
    outer_variables = tuple(
        _profiled_outer_variable(item, reciprocal_by_name.get(item.name))
        for item in outer_specs
    )
    all_variables = tuple(
        _Variable(
            f"parameter:{item.name}",
            item.bounds.lower,
            item.bounds.upper,
            item.initialization_range.lower,
            item.initialization_range.upper,
        )
        for item in model.validated.candidate.parameters
    )
    lower = np.asarray([item.lower for item in outer_variables], dtype=float)
    upper = np.asarray([item.upper for item in outer_variables], dtype=float)
    preferred_outer = _profiled_outer_warm_start(
        initial_global_parameters,
        report.latent_shape_parameter_names,
        reciprocal_by_name,
    )
    starts = _starts(
        outer_variables,
        config.number_of_starts,
        np.random.default_rng(config.random_seed),
        preferred_parameters=preferred_outer,
    )
    residual_size = sum(
        (
            trajectory.number_of_rows
            - (1 if model.validated.context.lagged_targets else 0)
        )
        * len(observed_names)
        for trajectory in training.trajectories
    ) + len(inner_specs)
    outcomes: list[_ProfiledOutcome] = []

    for start in starts:
        failures = FailureCounter()

        def residual(
            outer_values: NDArray[np.float64],
            counter: FailureCounter = failures,
        ) -> NDArray[np.float64]:
            try:
                _, values = _profile_affine_latent_basis_weights(
                    model,
                    training,
                    config,
                    direct,
                    observed_names,
                    derivative_scales,
                    outer_specs,
                    inner_specs,
                    outer_values,
                    reciprocal_by_name,
                    deadline,
                )
                return values
            except (
                ExactDerivativeFitError,
                RuntimeExpressionError,
                FloatingPointError,
                OverflowError,
                ValueError,
            ) as exc:
                counter.record(str(exc))
                return np.full(residual_size, config.failure_penalty, dtype=float)

        try:
            outer_result = least_squares(
                residual,
                start,
                bounds=(lower, upper),
                max_nfev=config.maximum_function_evaluations,
            )
        except TimeoutError as exc:
            failures.record(str(exc))
            outer_result = OptimizeResult(
                x=start,
                success=False,
                status=-2,
                message=str(exc),
                cost=0.5 * residual_size * config.failure_penalty**2,
                nfev=0,
            )
        try:
            parameters, profiled_residual = _profile_affine_latent_basis_weights(
                model,
                training,
                config,
                direct,
                observed_names,
                derivative_scales,
                outer_specs,
                inner_specs,
                np.asarray(outer_result.x, dtype=float),
                reciprocal_by_name,
                deadline,
            )
            cost = float(0.5 * np.dot(profiled_residual, profiled_residual))
        except (
            ExactDerivativeFitError,
            RuntimeExpressionError,
            FloatingPointError,
            OverflowError,
            ValueError,
            TimeoutError,
        ) as exc:
            failures.record(str(exc))
            parameters = {
                **_decode_profiled_outer_parameters(
                    outer_specs,
                    np.asarray(outer_result.x, dtype=float),
                    reciprocal_by_name,
                ),
                **{
                    item.name: (item.bounds.lower + item.bounds.upper) / 2.0
                    for item in inner_specs
                },
            }
            cost = 0.5 * residual_size * config.failure_penalty**2
            outer_result.success = False
            if isinstance(exc, TimeoutError):
                outer_result.status = -2
                outer_result.message = str(exc)
        full_values = np.asarray(
            [parameters[item.name] for item in model.validated.candidate.parameters],
            dtype=float,
        )
        result = OptimizeResult(
            x=full_values,
            success=bool(outer_result.success),
            status=int(outer_result.status),
            message=str(outer_result.message),
            cost=cost,
            nfev=int(outer_result.nfev),
        )
        outcomes.append(_ProfiledOutcome(result, parameters, failures))
        if int(result.status) == -2:
            break

    best_index = min(
        range(len(outcomes)), key=lambda index: float(outcomes[index].result.cost)
    )
    best = outcomes[best_index]
    training_initials = _direct_observed_initials(model, training, direct)
    validation_initials = _direct_observed_initials(model, validation, direct)
    diagnostics = tuple(
        _diagnostic(
            index,
            outcome.result,
            outcome.failures,
            all_variables,
            config,
            backend="profiled_latent_basis_linear_ridge",
            certified_parameter_transformations=tuple(
                f"{item.coordinate_name}=1/{item.parameter_name}"
                for item in active_transformations
            ),
        )
        for index, outcome in enumerate(outcomes)
    )
    decoded = _Decoded(best.parameters, {}, training_initials)
    if int(best.result.status) == -2:
        return _timeout_fit_result(
            model,
            training,
            validation,
            decoded,
            diagnostics,
            best_index,
            target_scales,
            config.failure_penalty,
        )
    train_metrics = _evaluate(
        model,
        training,
        best.parameters,
        {},
        training_initials,
        target_scales,
        config,
    )
    validation_metrics = _evaluate(
        model,
        validation,
        best.parameters,
        {},
        validation_initials,
        target_scales,
        config,
    )
    evaluation_budget_reached = int(best.result.status) == 0
    succeeded = bool(
        (bool(best.result.success) or evaluation_budget_reached)
        and np.isfinite(float(best.result.cost))
        and np.isfinite(best.result.x).all()
        and not train_metrics.failed_trajectories
        and not validation_metrics.failed_trajectories
        and np.isfinite(train_metrics.normalized_mse)
        and np.isfinite(validation_metrics.normalized_mse)
    )
    return FitResult(
        success=succeeded,
        global_parameters=best.parameters,
        global_initial_conditions={},
        training_trajectory_initial_conditions=training_initials,
        validation_trajectory_initial_conditions=validation_initials,
        training_metrics=train_metrics,
        validation_metrics=validation_metrics,
        diagnostics=diagnostics,
        best_start_index=best_index,
        target_scales=target_scales,
        message=(
            "outer optimizer evaluation budget reached; finite profiled fit retained"
            if succeeded and evaluation_budget_reached
            else None
            if succeeded
            else "profiled latent-basis fit is numerically invalid"
        ),
    )


def _profile_affine_latent_basis_weights(
    model: CompiledModel,
    training: DatasetSplit,
    config: FitConfig,
    direct: Mapping[str, str],
    observed_names: frozenset[str],
    derivative_scales: Mapping[str, float],
    outer_specs: Sequence[ParameterSpec],
    inner_specs: Sequence[ParameterSpec],
    outer_values: NDArray[np.float64],
    reciprocal_by_name: Mapping[str, ReciprocalParameterTransformation],
    deadline: float | None,
) -> tuple[Mapping[str, float], NDArray[np.float64]]:
    """Solve bounded affine weights conditional on one latent-shape vector."""
    outer = _decode_profiled_outer_parameters(
        outer_specs,
        outer_values,
        reciprocal_by_name,
    )
    inner_anchor = {
        item.name: (item.bounds.lower + item.bounds.upper) / 2.0
        for item in inner_specs
    }
    anchor = {**outer, **inner_anchor}
    inner_names = tuple(item.name for item in inner_specs)
    rows: list[NDArray[np.float64]] = []
    labels: list[float] = []
    for trajectory in training.trajectories:
        state_values = _conditioned_latent_state_matrix(
            model,
            trajectory,
            anchor,
            config,
            deadline,
        )
        skipped = 1 if model.validated.context.lagged_targets else 0
        for index in range(skipped, trajectory.number_of_rows):
            _check_deadline(deadline)
            forcing = trajectory_forcing(
                model,
                trajectory,
                causal_index=max(0, index - 1),
            )
            state = state_values[:, index]
            time = float(trajectory.time[index])
            anchor_rhs = model.rhs(time, state, anchor, forcing)
            design = np.empty((len(model.state_names), len(inner_names)))
            for parameter_index, spec in enumerate(inner_specs):
                probe = dict(anchor)
                probe[spec.name] = float(spec.bounds.lower)
                delta = probe[spec.name] - anchor[spec.name]
                if delta == 0.0:
                    raise AssertionError("parameter probe has zero displacement")
                probe_rhs = model.rhs(time, state, probe, forcing)
                design[:, parameter_index] = (probe_rhs - anchor_rhs) / delta
            intercept = anchor_rhs - design @ np.asarray(
                [inner_anchor[name] for name in inner_names], dtype=float
            )
            for state_name in sorted(observed_names):
                state_index = model.state_names.index(state_name)
                channel = direct[state_name]
                scale = derivative_scales[state_name]
                rows.append(design[state_index].copy() / scale)
                labels.append(
                    (
                        float(trajectory.derivatives[channel][index])
                        - float(intercept[state_index])
                    )
                    / scale
                )
    matrix = np.vstack(rows)
    response = np.asarray(labels, dtype=float)
    if not np.isfinite(matrix).all() or not np.isfinite(response).all():
        raise ExactDerivativeFitError("profiled design matrix is nonfinite")
    regularization = config.derivative_ridge_regularization
    if regularization:
        matrix_for_solve = np.vstack(
            (matrix, np.sqrt(regularization) * np.eye(len(inner_names)))
        )
        response_for_solve = np.concatenate(
            (response, np.zeros(len(inner_names), dtype=float))
        )
    else:
        matrix_for_solve = matrix
        response_for_solve = response
    solution = lsq_linear(
        matrix_for_solve,
        response_for_solve,
        bounds=(
            np.asarray([item.bounds.lower for item in inner_specs], dtype=float),
            np.asarray([item.bounds.upper for item in inner_specs], dtype=float),
        ),
    )
    if not solution.success or not np.isfinite(solution.x).all():
        raise ExactDerivativeFitError(
            "bounded profiled affine solve failed: " + str(solution.message)
        )
    parameters = {**outer, **dict(zip(inner_names, solution.x, strict=True))}
    residual = np.concatenate(
        (
            matrix @ solution.x - response,
            np.sqrt(regularization) * np.asarray(solution.x, dtype=float),
        )
    )
    return parameters, residual


def _profiled_outer_variable(
    spec: ParameterSpec,
    transformation: ReciprocalParameterTransformation | None,
) -> _Variable:
    """Represent one outer parameter in its certified optimizer coordinate."""
    if transformation is None:
        return _Variable(
            f"parameter:{spec.name}",
            spec.bounds.lower,
            spec.bounds.upper,
            spec.initialization_range.lower,
            spec.initialization_range.upper,
        )
    return _Variable(
        f"parameter:{spec.name}",
        transformation.coordinate_lower,
        transformation.coordinate_upper,
        transformation.coordinate_start_lower,
        transformation.coordinate_start_upper,
    )


def _profiled_outer_warm_start(
    initial_parameters: Mapping[str, float] | None,
    outer_names: Sequence[str],
    reciprocal_by_name: Mapping[str, ReciprocalParameterTransformation],
) -> Mapping[str, float] | None:
    """Map an original-parameter warm start into outer optimizer coordinates."""
    if initial_parameters is None:
        return None
    result: dict[str, float] = {}
    for name in outer_names:
        if name not in initial_parameters:
            continue
        value = float(initial_parameters[name])
        result[name] = 1.0 / value if name in reciprocal_by_name else value
    return result


def _decode_profiled_outer_parameters(
    specs: Sequence[ParameterSpec],
    coordinate_values: NDArray[np.float64],
    reciprocal_by_name: Mapping[str, ReciprocalParameterTransformation],
) -> dict[str, float]:
    """Map optimizer coordinates back to the candidate's declared parameters."""
    parameters: dict[str, float] = {}
    for spec, value in zip(specs, coordinate_values, strict=True):
        parameters[spec.name] = (
            1.0 / float(value)
            if spec.name in reciprocal_by_name
            else float(value)
        )
    return parameters


def _validate_partial_exact_derivative_contract(
    model: CompiledModel,
    training: DatasetSplit,
    *,
    backend_label: str,
) -> tuple[Mapping[str, str], frozenset[str]]:
    """Validate public observations needed by partial exact-derivative fits."""
    direct = model.direct_state_observation_channels
    observed_names = frozenset(
        item.name
        for item in model.validated.candidate.states
        if item.kind.value == "observed"
    )
    latent_names = {
        item.name
        for item in model.validated.candidate.states
        if item.kind.value == "latent"
    }
    missing_observed = sorted(observed_names - set(direct))
    mislabeled_latent = sorted(latent_names & set(direct))
    if missing_observed or mislabeled_latent:
        raise ExactDerivativeFitError(
            f"{backend_label} requires identity observations for all observed "
            "states and no direct observation of latent states; "
            f"missing_observed={missing_observed}, "
            f"directly_observed_latent={mislabeled_latent}"
        )
    for trajectory in training.trajectories:
        if trajectory.derivative_provenance is not DerivativeProvenance.EXACT:
            raise ExactDerivativeFitError(
                f"{backend_label} refuses non-exact observed derivatives: "
                f"{trajectory.trajectory_id} has "
                f"{trajectory.derivative_provenance.value!r} provenance"
            )
        missing = sorted(set(direct.values()) - set(trajectory.derivatives))
        if missing:
            raise ExactDerivativeFitError(
                f"trajectory {trajectory.trajectory_id} lacks exact observed "
                f"derivatives: {missing}"
            )
    return direct, observed_names


def _conditioned_latent_state_matrix(
    model: CompiledModel,
    trajectory: Trajectory,
    parameters: Mapping[str, float],
    config: FitConfig,
    deadline: float | None,
) -> NDArray[np.float64]:
    """Generate latent basis paths while conditioning on observed train paths."""
    direct = model.direct_state_observation_channels
    latent_indices = tuple(
        index
        for index, state in enumerate(model.validated.candidate.states)
        if state.kind.value == "latent"
    )
    state_values = np.empty(
        (len(model.state_names), trajectory.number_of_rows), dtype=float
    )
    for state_name, channel in direct.items():
        state_values[model.state_names.index(state_name)] = np.asarray(
            [
                _observed_value(trajectory, channel, index)
                for index in range(trajectory.number_of_rows)
            ],
            dtype=float,
        )
    if not latent_indices:
        return state_values

    known = {
        **{name: float(values[0]) for name, values in trajectory.targets.items()},
        **{
            name: float(values[0])
            for name, values in trajectory.auxiliaries.items()
        },
        **{
            name: float(values[0])
            for name, values in trajectory.external_inputs.items()
        },
        **{
            name: float(value)
            for name, value in trajectory.fixed_covariates.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        },
    }
    latent_initial: list[float] = []
    for index in latent_indices:
        state_name = model.state_names[index]
        value = model.initial_condition_value(state_name, known)
        if value is None:
            raise ExactDerivativeFitError(
                f"fixed latent initial is unavailable: {state_name}"
            )
        latent_initial.append(float(value))
        known[state_name] = float(value)
    state_values[np.asarray(latent_indices), 0] = np.asarray(latent_initial)

    time = np.asarray(trajectory.time, dtype=float)
    latent = np.asarray(latent_initial, dtype=float)
    for interval in range(trajectory.number_of_rows - 1):
        _check_deadline(deadline)
        start = float(time[interval])
        end = float(time[interval + 1])
        forcing = trajectory_forcing(model, trajectory, causal_index=interval)

        def latent_rhs(
            current_time: float,
            latent_state: NDArray[np.float64],
            interval_forcing=forcing,
        ):
            full_state = np.empty(len(model.state_names), dtype=float)
            for state_name, channel in direct.items():
                state_index = model.state_names.index(state_name)
                series = (
                    trajectory.targets[channel]
                    if channel in trajectory.targets
                    else trajectory.auxiliaries[channel]
                )
                full_state[state_index] = float(
                    np.interp(current_time, time, series)
                )
            full_state[np.asarray(latent_indices)] = latent_state
            return model.rhs(
                current_time, full_state, parameters, interval_forcing
            )[np.asarray(latent_indices)]

        if config.integration_backend == "fixed_rk4":
            step = (end - start) / config.fixed_step_substeps
            current = start
            updated = latent.copy()
            for _ in range(config.fixed_step_substeps):
                k1 = latent_rhs(current, updated)
                k2 = latent_rhs(current + step / 2.0, updated + step * k1 / 2.0)
                k3 = latent_rhs(current + step / 2.0, updated + step * k2 / 2.0)
                k4 = latent_rhs(current + step, updated + step * k3)
                updated = updated + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                current += step
            latent = updated
        else:
            solution = solve_ivp(
                latent_rhs,
                (start, end),
                latent,
                t_eval=[end],
                method=config.integration_method,
                rtol=config.relative_tolerance,
                atol=config.absolute_tolerance,
            )
            if not solution.success:
                raise ExactDerivativeFitError(
                    "fixed latent-basis integration failed: "
                    f"{solution.message}"
                )
            latent = np.asarray(solution.y[:, -1], dtype=float)
        if not np.isfinite(latent).all():
            raise ExactDerivativeFitError(
                "fixed latent-basis integration produced nonfinite values"
            )
        state_values[np.asarray(latent_indices), interval + 1] = latent
    return state_values


def _observed_value(
    trajectory: Trajectory,
    channel: str,
    index: int,
) -> float:
    if channel in trajectory.targets:
        return float(trajectory.targets[channel][index])
    if channel in trajectory.auxiliaries:
        return float(trajectory.auxiliaries[channel][index])
    raise ExactDerivativeFitError(
        f"directly observed state channel is unavailable: {channel}"
    )


def _direct_observed_initials(
    model: CompiledModel,
    split: DatasetSplit,
    observed: Mapping[str, str],
) -> Mapping[str, Mapping[str, float]]:
    """Use measured initial states directly; these are data, not fit variables."""
    return {
        trajectory.trajectory_id: {
            state_name: _observed_value(trajectory, channel, 0)
            for state_name, channel in observed.items()
            if state_name not in model.observed_state_channels
        }
        for trajectory in split.trajectories
    }


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
    if settings.parameter_fit_strategy in {
        "exact_derivative_linear_ridge",
        "fixed_latent_basis_linear_ridge",
        "profiled_latent_basis_linear_ridge",
    }:
        observed = model.direct_state_observation_channels
        if (
            settings.parameter_fit_strategy == "exact_derivative_linear_ridge"
            and set(observed) != set(model.state_names)
        ):
            raise ExactDerivativeFitError(
                "exact derivative evaluation requires every state to be "
                "directly observed"
            )
        local_initials = _direct_observed_initials(model, split, observed)
    else:
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
                    _bounded_normalized_residual(
                        simulation.predictions[channel][start:],
                        trajectory.targets[channel][start:],
                        scales[channel],
                        config.failure_penalty,
                    )
                )
            assert simulation.states is not None
            soft = _soft_constraint_residuals(model, simulation.states)
            if soft.size:
                pieces.append(
                    np.sqrt(config.soft_constraint_penalty_weight) * soft
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
        return _bounded_residuals(
            np.asarray(pieces, dtype=float), config.failure_penalty
        )

    return residual


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and monotonic() >= deadline:
        raise TimeoutError("fitting wall-clock limit reached")


def _bounded_normalized_residual(
    predicted: NDArray[np.float64],
    observed: NDArray[np.float64],
    scale: float,
    limit: float,
) -> NDArray[np.float64]:
    """Return finite normalized residuals bounded for stable least squares."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        residual = (predicted - observed) / scale
    return _bounded_residuals(residual, limit)


def _bounded_residuals(
    residual: NDArray[np.float64], limit: float
) -> NDArray[np.float64]:
    """Replace nonfinite residuals and clip magnitudes before squaring."""
    finite = np.nan_to_num(
        residual,
        nan=limit,
        posinf=limit,
        neginf=-limit,
    )
    return np.clip(finite, -limit, limit)


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


def _supported_soft_state_constraints(
    model: CompiledModel,
) -> tuple[ConstraintSpec, ...]:
    """Return soft pointwise state constraints supported by the fitter."""
    return tuple(
        constraint
        for constraint in model.validated.candidate.constraints
        if constraint.enforcement is ConstraintEnforcement.SOFT
        and constraint.subject in model.state_names
        and constraint.kind
        in {
            ConstraintKind.NONNEGATIVE,
            ConstraintKind.POSITIVE,
            ConstraintKind.BOUNDED,
        }
    )


def _soft_constraint_series(
    model: CompiledModel,
    states: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Return normalized nonnegative violation magnitudes by constraint."""
    by_state = {
        name: states[index]
        for index, name in enumerate(model.state_names)
    }
    series: dict[str, NDArray[np.float64]] = {}
    for index, constraint in enumerate(_supported_soft_state_constraints(model)):
        values = by_state[constraint.subject]
        maximum = float(np.max(np.abs(values)))
        rms = (
            0.0
            if maximum == 0.0
            else maximum * float(np.sqrt(np.mean((values / maximum) ** 2)))
        )
        scale = max(1.0, rms)
        if constraint.kind is ConstraintKind.NONNEGATIVE:
            violation = np.maximum(-values, 0.0)
        elif constraint.kind is ConstraintKind.POSITIVE:
            violation = np.maximum(1e-12 - values, 0.0)
        else:
            assert constraint.bounds is not None
            violation = np.maximum(constraint.bounds.lower - values, 0.0)
            violation += np.maximum(values - constraint.bounds.upper, 0.0)
        key = f"{index}:{constraint.subject}:{constraint.kind.value}"
        series[key] = violation / scale
    return series


def _soft_constraint_residuals(
    model: CompiledModel,
    states: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Flatten soft-constraint penalties into a fixed-size residual block."""
    if states is None:
        return np.empty(0, dtype=float)
    series = _soft_constraint_series(model, states)
    if not series:
        return np.empty(0, dtype=float)
    return np.concatenate(list(series.values()))


def _residual_size(split: DatasetSplit, model: CompiledModel) -> int:
    skipped = 1 if model.validated.context.lagged_targets else 0
    target_residuals = sum(
        (trajectory.number_of_rows - skipped)
        * len(model.validated.context.targets)
        for trajectory in split.trajectories
    )
    soft_constraints = len(_supported_soft_state_constraints(model))
    return target_residuals + sum(
        trajectory.number_of_rows * soft_constraints
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
    certified_parameter_transformations: tuple[str, ...] = (),
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
        certified_parameter_transformations=certified_parameter_transformations,
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
    soft_violations: dict[str, list[np.ndarray]] = {}
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
        assert simulation.states is not None
        for key, values in _soft_constraint_series(
            model, simulation.states
        ).items():
            soft_violations.setdefault(key, []).append(values)
    per_target = {
        channel: float(np.mean(np.concatenate(values)))
        for channel, values in squared.items()
    }
    aggregate = float(
        np.mean(np.concatenate([np.concatenate(values) for values in squared.values()]))
    )
    violation_summary = {
        key: {
            "maximum_normalized_violation": float(np.max(combined)),
            "mean_normalized_violation": float(np.mean(combined)),
            "violating_fraction": float(np.mean(combined > 0.0)),
        }
        for key, pieces in soft_violations.items()
        if (combined := np.concatenate(pieces)).size
    }
    return EvaluationMetrics(
        aggregate,
        per_target,
        tuple(failed),
        violation_summary,
    )

"""Opt-in joint training initials and prefix-only latent-root inference.

Identity-observed states use their noiseless first readings. Only independent
latent roots vary by trajectory; dependent graph nodes are never free coordinates.
The caller owns durable operation journals. These numerical functions expose
started-call checkpoints and never read held-out suffix targets during inference.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from itertools import pairwise
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from autoformalism.data import Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import fitter, recovery_numerics
from autoformalism.fitting.initialization_graph import (
    InitializationGraph,
    apply_initialization_graph,
)
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.public_fitting import content_sha256, unpack_split
from autoformalism.fitting.sensitivity_probe import SymbolicODE, symbolic_rollout
from autoformalism.schemas.base import FiniteFloat, Identifier, StrictSchema
from autoformalism.schemas.candidate import CandidateModel
from autoformalism.schemas.public_fitting import PublicSplit, PublicTrajectory

Checkpoint = Callable[[dict], None]


class PrefixFitProblem(StrictSchema):
    """Serializable model boundary with no validation/test data or legacy plan."""

    protocol: Literal["prefix-fit-problem-1"] = "prefix-fit-problem-1"
    base_candidate: CandidateModel
    context: ValidationContext
    graph: InitializationGraph
    parameter_guesses: Mapping[Identifier, FiniteFloat] = Field(default_factory=dict)


class PrefixPolicy(StrictSchema):
    """Predeclared elapsed prefix duration and genuinely known future inputs."""

    duration: FiniteFloat = Field(ge=0)
    future_inputs: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def unique_inputs(self):
        if len(self.future_inputs) != len(set(self.future_inputs)):
            raise ValueError("duplicate future input")
        return self


class NumericalPolicy(StrictSchema):
    """Frozen search and independent numerical verification allowances."""

    maximum_calls: int = Field(default=100, ge=1, le=10000)
    seconds: FiniteFloat = Field(default=30, gt=0, le=3600)
    verification_seconds: FiniteFloat = Field(default=10, gt=0, le=300)
    rtol: FiniteFloat = Field(default=1e-9, gt=0, le=1e-5)
    atol: FiniteFloat = Field(default=1e-11, gt=0, le=1e-7)
    solver_tolerance: FiniteFloat = Field(default=1e-6, gt=0, le=1e-3)


class PrefixView(StrictSchema):
    """Physically restricted observed prefix; no suffix fields are accepted."""

    trajectory_id: str = Field(min_length=1)
    policy: PrefixPolicy
    time: tuple[FiniteFloat, ...] = Field(min_length=1)
    targets: Mapping[Identifier, tuple[FiniteFloat, ...]]
    initial_auxiliaries: Mapping[Identifier, FiniteFloat] = Field(default_factory=dict)
    initial_inputs: Mapping[Identifier, FiniteFloat] = Field(default_factory=dict)
    external_inputs: Mapping[Identifier, tuple[FiniteFloat, ...]] = Field(
        default_factory=dict
    )
    fixed_covariates: Mapping[Identifier, FiniteFloat] = Field(default_factory=dict)

    @model_validator(mode="after")
    def prefix_arrays(self):
        _check_arrays(self.time, self.targets, self.external_inputs)
        if not np.isfinite(self.cutoff):
            raise ValueError("prefix cutoff must be finite")
        if self.time[-1] > self.cutoff:
            raise ValueError("prefix contains observations after cutoff")
        if set(self.external_inputs) - set(self.policy.future_inputs):
            raise ValueError("prefix input is not in frozen future-input allowlist")
        for name, values in self.external_inputs.items():
            if self.initial_inputs.get(name) != values[0]:
                raise ValueError("prefix initial input differs from schedule")
        return self

    @property
    def cutoff(self) -> float:
        return self.time[0] + self.policy.duration


class ForecastSchedule(StrictSchema):
    """Target-free future schedule with only permitted initial observations."""

    trajectory_id: str = Field(min_length=1)
    policy: PrefixPolicy
    time: tuple[FiniteFloat, ...] = Field(min_length=2)
    initial_targets: Mapping[Identifier, FiniteFloat]
    initial_auxiliaries: Mapping[Identifier, FiniteFloat] = Field(default_factory=dict)
    initial_inputs: Mapping[Identifier, FiniteFloat] = Field(default_factory=dict)
    external_inputs: Mapping[Identifier, tuple[FiniteFloat, ...]] = Field(
        default_factory=dict
    )
    fixed_covariates: Mapping[Identifier, FiniteFloat] = Field(default_factory=dict)

    @model_validator(mode="after")
    def forecast_arrays(self):
        _check_arrays(self.time, self.external_inputs)
        if not np.isfinite(self.cutoff):
            raise ValueError("forecast cutoff must be finite")
        if set(self.external_inputs) - set(self.policy.future_inputs):
            raise ValueError("future input is not in frozen allowlist")
        for name, values in self.external_inputs.items():
            if self.initial_inputs.get(name) != values[0]:
                raise ValueError("forecast initial input differs from schedule")
        return self

    @property
    def cutoff(self) -> float:
        return self.time[0] + self.policy.duration


def _check_arrays(time, *groups) -> None:
    if any(right <= left for left, right in pairwise(time)):
        raise ValueError("time must be strictly increasing")
    if any(len(values) != len(time) for group in groups for values in group.values()):
        raise ValueError("channel length differs from time")


def _initial_fields(row: PublicTrajectory) -> dict:
    return {
        "trajectory_id": row.trajectory_id,
        "initial_auxiliaries": {
            name: values[0] for name, values in row.auxiliaries.items()
        },
        "initial_inputs": {
            name: values[0] for name, values in row.external_inputs.items()
        },
        "fixed_covariates": dict(row.fixed_covariates),
    }


def make_prefix(row: PublicTrajectory, policy: PrefixPolicy) -> PrefixView:
    """Copy only observed samples at or before the predeclared cutoff."""
    if set(policy.future_inputs) - set(row.external_inputs):
        raise ValueError("future-input allowlist is not supplied external input")
    count = sum(t <= row.time[0] + policy.duration for t in row.time)
    return PrefixView(
        **_initial_fields(row),
        policy=policy,
        time=row.time[:count],
        targets={name: values[:count] for name, values in row.targets.items()},
        external_inputs={
            name: row.external_inputs[name][:count] for name in policy.future_inputs
        },
    )


def make_forecast_schedule(
    row: PublicTrajectory, policy: PrefixPolicy
) -> ForecastSchedule:
    """Drop all future target/auxiliary readings at the forecast boundary."""
    if set(policy.future_inputs) - set(row.external_inputs):
        raise ValueError("future-input allowlist is not supplied external input")
    return ForecastSchedule(
        **_initial_fields(row),
        policy=policy,
        time=row.time,
        initial_targets={name: values[0] for name, values in row.targets.items()},
        external_inputs={
            name: row.external_inputs[name] for name in policy.future_inputs
        },
    )


def _boundary(view: PrefixView | ForecastSchedule) -> dict:
    return {
        "trajectory_id": view.trajectory_id,
        "time": view.time[0],
        "policy": view.policy.model_dump(mode="json"),
        "initial_targets": dict(view.initial_targets)
        if isinstance(view, ForecastSchedule)
        else {name: values[0] for name, values in view.targets.items()},
        "initial_auxiliaries": dict(view.initial_auxiliaries),
        "initial_inputs": dict(view.initial_inputs),
        "fixed_covariates": dict(view.fixed_covariates),
    }


def _prefix_forcing(view: PrefixView | ForecastSchedule, end: float) -> dict:
    """Bind every supplied input knot used while estimating the roots."""
    count = sum(time <= end for time in view.time)
    return {
        "time": view.time[:count],
        "external_inputs": {
            name: values[:count] for name, values in view.external_inputs.items()
        },
    }


def _trajectory(view: PrefixView | ForecastSchedule) -> Trajectory:
    """Reconstruct the numerical row without introducing any suffix readings."""
    count = len(view.time)
    targets = (
        {name: np.full(count, value) for name, value in view.initial_targets.items()}
        if isinstance(view, ForecastSchedule)
        else {name: np.asarray(values, float) for name, values in view.targets.items()}
    )
    inputs = {
        name: np.full(count, value) for name, value in view.initial_inputs.items()
    }
    inputs.update(
        {
            name: np.asarray(values, float)
            for name, values in view.external_inputs.items()
        }
    )
    return Trajectory(
        view.trajectory_id,
        np.asarray(view.time, float),
        targets,
        {
            name: np.full(count, value)
            for name, value in view.initial_auxiliaries.items()
        },
        inputs,
        dict(view.fixed_covariates),
        {},
    )


def _compile(problem: PrefixFitProblem, policy: NumericalPolicy):
    if not isinstance(problem, PrefixFitProblem):
        raise TypeError("typed prefix fit problem required")
    if problem.context.fitted_initialization or problem.context.lagged_targets:
        raise ValueError("prefix fitting requires an unlowered free-rollout context")
    if any(p.scope.value != "global" for p in problem.base_candidate.parameters):
        raise ValueError("dynamics parameters must be shared")
    base = compile_candidate(problem.base_candidate, problem.context)
    model, guesses, audit = apply_initialization_graph(base, problem.graph)
    if set(problem.parameter_guesses) != set(base.parameter_names):
        raise ValueError("explicit guesses must cover exactly dynamics parameters")
    guesses.update(problem.parameter_guesses)
    system = SymbolicODE(model)
    settings = FitConfig(
        integration_method="DOP853",
        relative_tolerance=policy.rtol,
        absolute_tolerance=policy.atol,
        allow_derivative_regression=False,
    )
    variables = [
        fitter._parameter_variable(model, spec, settings)
        for spec in model.validated.candidate.parameters
    ]
    return system, guesses, audit, settings, variables


def _physical_vector(system, variables, parameters: Mapping[str, float]) -> np.ndarray:
    """Validate the complete frozen-and-free vector before symbolic execution."""
    if set(parameters) != set(system.names):
        raise ValueError("physical parameter names differ from compiled model")
    vector = np.asarray([parameters[name] for name in system.names], float)
    lower = np.asarray([variable.lower for variable in variables])
    upper = np.asarray([variable.upper for variable in variables])
    if (
        not np.isfinite(vector).all()
        or np.any(vector < lower)
        or np.any(vector > upper)
    ):
        raise ValueError("physical parameters outside declared domain")
    return vector


def _validate_row(system: SymbolicODE, row: Trajectory) -> None:
    context = system.model.validated.context
    if set(row.targets) != set(context.targets):
        raise ValueError("trajectory targets differ from declared targets")
    for role in ("auxiliaries", "external_inputs", "fixed_covariates"):
        if set(getattr(row, role)) - set(getattr(context, role)):
            raise ValueError(f"undeclared trajectory {role}")


def _validate_future(system: SymbolicODE, view: PrefixView | ForecastSchedule) -> None:
    context = system.model.validated.context
    if set(view.policy.future_inputs) - set(context.external_inputs):
        raise ValueError("future forcing must be explicitly declared external input")
    allowed = set(view.policy.future_inputs) | set(context.fixed_covariates)
    forbidden = set(system.model.validated.forcing_symbols) - allowed
    if forbidden:
        raise ValueError(
            f"future target/auxiliary or unavailable input forcing: {sorted(forbidden)}"
        )
    missing = set(system.model.validated.forcing_symbols) & set(context.external_inputs)
    if missing - set(view.external_inputs):
        raise ValueError("future forcing schedule is incomplete")


class TrainingOracle:
    """Joint shared coefficients and independent per-row latent root coordinates."""

    def __init__(
        self, problem: PrefixFitProblem, training: PublicSplit, policy: NumericalPolicy
    ):
        if training.name != "train":
            raise ValueError("joint fitting accepts training only")
        self.problem = problem
        self.system, guesses, self.audit, self.settings, variables = _compile(
            problem, policy
        )
        self.rows = unpack_split(training).trajectories
        for row in self.rows:
            _validate_row(self.system, row)
        local = self.system.names
        self.roots = tuple(self.audit["trajectory_parameter_names"])
        self.shared = tuple(name for name in local if name not in self.roots)
        self.names = self.shared + tuple(
            f"trajectory:{row.trajectory_id}:{name}"
            for row in self.rows
            for name in self.roots
        )
        self.indices = {
            row.trajectory_id: np.asarray(
                [
                    self.names.index(
                        name
                        if name in self.shared
                        else f"trajectory:{row.trajectory_id}:{name}"
                    )
                    for name in local
                ],
                int,
            )
            for row in self.rows
        }
        specs = dict(zip(local, variables, strict=True))
        repeated = list(self.shared) + [name for _ in self.rows for name in self.roots]
        self.lower = np.asarray([specs[name].lower for name in repeated])
        self.upper = np.asarray([specs[name].upper for name in repeated])
        self.start = dict(
            zip(self.names, [guesses[name] for name in repeated], strict=True)
        )
        self.units = np.asarray([max(abs(value), 1.0) for value in self.start.values()])
        self.scales = {
            channel: max(
                float(
                    np.std(np.concatenate([row.targets[channel] for row in self.rows]))
                ),
                1e-12,
            )
            for channel in self.system.channels
        }
        self.vector(self.start)

    def vector(self, values: Mapping[str, float]) -> np.ndarray:
        if set(values) != set(self.names):
            raise ValueError("incomplete optimizer parameter vector")
        vector = np.asarray([values[name] for name in self.names], float)
        if (
            not np.isfinite(vector).all()
            or np.any(vector < self.lower)
            or np.any(vector > self.upper)
        ):
            raise ValueError("optimizer parameters outside declared domain")
        return vector

    def parameters(self, vector: np.ndarray) -> dict:
        return dict(zip(self.names, vector.tolist(), strict=True))

    def local_vector(self, vector: np.ndarray, row: Trajectory) -> np.ndarray:
        return vector[self.indices[row.trajectory_id]]

    def evaluate(self, vector, deadline, *, jacobian=True, method=None):
        """Stack full training sensitivities, sharing only the declared columns."""
        self.vector(self.parameters(vector))
        settings = (
            self.settings.model_copy(update={"integration_method": method})
            if method
            else self.settings
        )
        scales = np.asarray([self.scales[name] for name in self.system.channels])
        residuals, derivatives = [], []
        for row in self.rows:
            values, derivative, _, _ = symbolic_rollout(
                self.system,
                row,
                self.local_vector(vector, row),
                settings,
                deadline,
                sensitivities=jacobian,
            )
            observed = np.column_stack(
                [row.targets[name] for name in self.system.channels]
            )
            residuals.append(((values - observed) / scales).ravel())
            if jacobian:
                local = (derivative / scales[None, :, None]).reshape(
                    values.size, len(self.system.names)
                )
                joint = np.zeros((len(local), len(self.names)))
                joint[:, self.indices[row.trajectory_id]] = local
                derivatives.append(joint)
        return np.concatenate(residuals), np.vstack(derivatives) if jacobian else None


class PrefixOracle(TrainingOracle):
    """Roots-only oracle with frozen shared parameters and training scales."""

    def __init__(
        self, problem, fit_result, prefix: PrefixView, policy: NumericalPolicy
    ):
        if not isinstance(prefix, PrefixView):
            raise TypeError("typed prefix-only view required")
        _validate_fit(problem, fit_result)
        self.problem = problem
        self.system, guesses, self.audit, self.settings, variables = _compile(
            problem, policy
        )
        _validate_future(self.system, prefix)
        self.rows = (_trajectory(prefix),)
        _validate_row(self.system, self.rows[0])
        self.roots = self.names = tuple(self.audit["trajectory_parameter_names"])
        self.shared = tuple(
            name for name in self.system.names if name not in self.roots
        )
        self.frozen = dict(fit_result["shared_parameters"])
        if set(self.frozen) != set(self.shared):
            raise ValueError("frozen shared parameter names differ")
        _physical_vector(self.system, variables, guesses | self.frozen)
        specs = dict(zip(self.system.names, variables, strict=True))
        self.lower = np.asarray([specs[name].lower for name in self.names])
        self.upper = np.asarray([specs[name].upper for name in self.names])
        self.start = {name: guesses[name] for name in self.names}
        self.units = np.asarray([max(abs(value), 1.0) for value in self.start.values()])
        self.scales = dict(fit_result["training_scales"])
        if set(self.scales) != set(self.system.channels) or any(
            not np.isfinite(value) or value <= 0 for value in self.scales.values()
        ):
            raise ValueError("invalid frozen training scales")
        self.vector(self.start)

    def local_vector(self, vector, row):
        values = self.frozen | self.parameters(vector)
        return np.asarray([values[name] for name in self.system.names])

    def evaluate(self, vector, deadline, *, jacobian=True, method=None):
        self.vector(self.parameters(vector))
        settings = (
            self.settings.model_copy(update={"integration_method": method})
            if method
            else self.settings
        )
        row = self.rows[0]
        values, derivative, _, _ = symbolic_rollout(
            self.system,
            row,
            self.local_vector(vector, row),
            settings,
            deadline,
            sensitivities=jacobian,
        )
        scales = np.asarray([self.scales[name] for name in self.system.channels])
        observed = np.column_stack([row.targets[name] for name in self.system.channels])
        indices = [self.system.names.index(name) for name in self.names]
        return ((values - observed) / scales).ravel(), (
            (derivative[:, :, indices] / scales[None, :, None]).reshape(
                -1, len(indices)
            )
            if jacobian and indices
            else np.empty((values.size, 0))
            if jacobian
            else None
        )


def _validate_fit(problem, result) -> None:
    if result.get("problem_sha256") != content_sha256(problem):
        raise ValueError("fit problem identity differs")
    if not result.get("fit_success"):
        raise ValueError(
            "prefix inference requires a numerically verified training fit"
        )


def _rank(derivative: np.ndarray, units: np.ndarray) -> dict:
    singular = np.linalg.svd(derivative * units, compute_uv=False)
    tolerance = (
        max(derivative.shape) * np.finfo(float).eps * float(singular[0])
        if len(singular)
        else 0.0
    )
    rank = int(np.sum(singular > tolerance))
    return {
        "rank": rank,
        "coordinates": len(units),
        "full_column_rank": rank == len(units),
        "singular_values": singular.tolist(),
        "tolerance": tolerance,
        "scope": "local numerical rank; no uniqueness or confidence claim",
    }


def _solve(oracle, policy: NumericalPolicy, checkpoint: Checkpoint | None) -> dict:
    save = checkpoint or (lambda value: None)
    if oracle.names:
        search = recovery_numerics.search(
            oracle,
            oracle.start,
            monotonic() + policy.seconds,
            save,
            calls=policy.maximum_calls,
            target=-1.0,
            telemetry=True,
            optimizer_scaling="guarded_relative",
            objective_multiplier=1.0,
        )
    else:
        search = {
            "best": {"parameters": {}, "training_nmse": None},
            "calls": 0,
            "completed_calls": 0,
            "stop_reason": "no_free_coordinates",
        }
    result = {
        "fit_success": False,
        "search": search,
        "parameters": None,
        "verification_calls_started": 0,
        "verification_calls_completed": 0,
    }
    if search["best"] is None:
        return result
    parameters = search["best"]["parameters"]
    vector = oracle.vector(parameters)
    deadline = monotonic() + policy.verification_seconds
    residuals, derivative = [], None
    try:
        for method in ("DOP853", "Radau"):
            result["verification_calls_started"] += 1
            save({"calls": search["calls"], "best": search["best"], **result})
            residual, jacobian = oracle.evaluate(
                vector, deadline, jacobian=method == "DOP853", method=method
            )
            residuals.append(residual)
            if jacobian is not None:
                derivative = jacobian
            result["verification_calls_completed"] += 1
        difference = float(np.max(np.abs(residuals[0] - residuals[1]), initial=0.0))
        verified = difference <= policy.solver_tolerance
        result.update(
            fit_success=verified,
            parameters=parameters if verified else None,
            normalized_mse=float(np.mean(residuals[0] ** 2)),
            solver_difference=difference,
            rank_diagnostic=_rank(derivative, oracle.units),
        )
    except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
        result["verification_error"] = str(error)
    save({"calls": search["calls"], "best": search["best"], **result})
    return result


def fit_training(
    problem, training: PublicSplit, policy: NumericalPolicy, checkpoint=None
) -> dict:
    """Fit shared dynamics/maps jointly with each training trajectory's roots."""
    oracle = TrainingOracle(problem, training, policy)
    result = _solve(oracle, policy, checkpoint)
    parameters = result["parameters"]
    realized = _realized_initials(oracle, parameters)
    return result | {
        "protocol": "prefix-training-fit-1",
        "problem_sha256": content_sha256(problem),
        "training_sha256": content_sha256(training),
        "training_scales": oracle.scales,
        "shared_parameters": {name: parameters[name] for name in oracle.shared}
        if parameters is not None
        else None,
        "training_roots": {
            row.trajectory_id: {
                name: parameters[f"trajectory:{row.trajectory_id}:{name}"]
                for name in oracle.roots
            }
            for row in oracle.rows
        }
        if parameters is not None
        else None,
        "initialization_audit": oracle.audit,
        "realized_initial_states": realized,
        "policy": policy.model_dump(mode="json"),
        "selection_data": "train_only",
    }


def infer_prefix(
    problem, fit_result, prefix: PrefixView, policy: NumericalPolicy, checkpoint=None
) -> dict:
    """Infer only independent roots; shared coefficients remain frozen."""
    oracle = PrefixOracle(problem, fit_result, prefix, policy)
    result = _solve(oracle, policy, checkpoint)
    return result | {
        "protocol": "prefix-inference-1",
        "problem_sha256": content_sha256(problem),
        "fit_sha256": content_sha256(fit_result),
        "prefix_sha256": content_sha256(prefix),
        "boundary_sha256": content_sha256(_boundary(prefix)),
        "prefix_end_time": prefix.time[-1],
        "prefix_forcing_sha256": content_sha256(
            _prefix_forcing(prefix, prefix.time[-1])
        ),
        "cutoff": prefix.cutoff,
        "shared_parameters": dict(oracle.frozen),
        "training_scales": dict(oracle.scales),
        "roots": result["parameters"],
        "realized_initial_states": _realized_initials(oracle, result["parameters"]),
        "policy": policy.model_dump(mode="json"),
    }


def _realized_initials(oracle, parameters) -> dict | None:
    """Report every dependent state after evaluating the declared initial DAG."""
    if parameters is None:
        return None
    vector = oracle.vector(parameters)
    return {
        row.trajectory_id: dict(
            zip(
                oracle.system.model.state_names,
                oracle.system.initial_for(
                    row, oracle.local_vector(vector, row)
                ).tolist(),
                strict=True,
            )
        )
        for row in oracle.rows
    }


def forecast(
    problem,
    fit_result,
    inference,
    schedule: ForecastSchedule,
    policy: NumericalPolicy,
    checkpoint: Checkpoint | None = None,
) -> dict:
    """Free rollout from the original boundary, without future observed resets."""
    if not isinstance(schedule, ForecastSchedule):
        raise TypeError("typed target-free forecast schedule required")
    _validate_fit(problem, fit_result)
    if not inference.get("fit_success") or inference.get(
        "fit_sha256"
    ) != content_sha256(fit_result):
        raise ValueError("verified inference from this frozen fit required")
    if inference.get("problem_sha256") != content_sha256(problem) or inference.get(
        "boundary_sha256"
    ) != content_sha256(_boundary(schedule)):
        raise ValueError("forecast initial boundary or policy differs from inference")
    if inference.get("prefix_forcing_sha256") != content_sha256(
        _prefix_forcing(schedule, inference["prefix_end_time"])
    ):
        raise ValueError("forecast prefix time/input history differs from inference")
    system, _, audit, settings, variables = _compile(problem, policy)
    _validate_future(system, schedule)
    row = _trajectory(schedule)
    _validate_row(system, row)
    if set(inference["roots"]) != set(audit["trajectory_parameter_names"]):
        raise ValueError("inferred root coordinates differ")
    parameters = dict(fit_result["shared_parameters"]) | dict(inference["roots"])
    vector = _physical_vector(system, variables, parameters)
    result = {
        "protocol": "prefix-forecast-1",
        "trajectory_id": row.trajectory_id,
        "time": row.time.tolist(),
        "cutoff": schedule.cutoff,
        "forecast_success": False,
        "predictions": {},
        "states": {},
        "solver_counters": {},
        "verification_calls_started": 0,
        "verification_calls_completed": 0,
        "inference_sha256": content_sha256(inference),
        "schedule_sha256": content_sha256(schedule),
    }
    save = checkpoint or (lambda value: None)
    deadline = monotonic() + policy.verification_seconds
    solutions = []
    try:
        for method in ("DOP853", "Radau"):
            result["verification_calls_started"] += 1
            save(result.copy())
            values, _, states, counters = symbolic_rollout(
                system,
                row,
                vector,
                settings.model_copy(update={"integration_method": method}),
                deadline,
            )
            solutions.append((values, states))
            result["solver_counters"][method] = counters
            result["verification_calls_completed"] += 1
        scales = np.asarray(
            [fit_result["training_scales"][name] for name in system.channels]
        )
        difference = float(np.max(np.abs((solutions[0][0] - solutions[1][0]) / scales)))
        result["solver_difference"] = difference
        if difference <= policy.solver_tolerance:
            values, states = solutions[0]
            result.update(
                forecast_success=True,
                predictions={
                    name: values[:, index].tolist()
                    for index, name in enumerate(system.channels)
                },
                states={
                    name: states[:, index].tolist()
                    for index, name in enumerate(system.model.state_names)
                },
            )
        else:
            result["verification_error"] = "independent forecast integrators disagree"
    except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
        result["verification_error"] = str(error)
    save(result.copy())
    return result


def score_forecast(
    predictions: dict,
    heldout: PublicTrajectory,
    cutoff: float,
    scales: Mapping[str, float],
) -> dict:
    """Evaluator-only strict future score; no score is returned for an empty horizon."""
    if (
        predictions["trajectory_id"] != heldout.trajectory_id
        or tuple(predictions["time"]) != heldout.time
    ):
        raise ValueError("forecast evaluation trajectory/time differs")
    if not predictions.get("forecast_success"):
        return {"available": False, "normalized_mse": None, "sample_count": 0}
    if (
        predictions["cutoff"] != cutoff
        or set(predictions["predictions"]) != set(heldout.targets)
        or set(scales) != set(heldout.targets)
    ):
        raise ValueError("forecast evaluation cutoff/channels differ")
    mask = np.asarray(heldout.time) > cutoff
    if not mask.any():
        return {"available": False, "normalized_mse": None, "sample_count": 0}
    scores = {}
    for name, target in heldout.targets.items():
        scale = scales[name]
        values = np.asarray(predictions["predictions"][name], float)
        if (
            not np.isfinite(scale)
            or scale <= 0
            or len(values) != len(target)
            or not np.isfinite(values).all()
        ):
            raise ValueError("invalid forecast values or frozen scale")
        scores[name] = float(
            np.mean(((values[mask] - np.asarray(target)[mask]) / scale) ** 2)
        )
    return {
        "available": True,
        "normalized_mse": float(np.mean(list(scores.values()))),
        "per_target_normalized_mse": scores,
        "sample_count": int(mask.sum()),
        "scoring_rule": "strictly_after_prefix_cutoff",
    }

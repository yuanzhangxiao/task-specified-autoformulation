"""Replay native D3 increments without an ODE reinterpretation or refitting."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from time import monotonic
from types import SimpleNamespace

import numpy as np

from autoformalism.baselines.d3_native import (
    _evaluate,
    _ordered_processes,
    _target_scales,
    validate_native_candidate,
)
from autoformalism.data import DatasetSplit, SplitName, TrainingScaler, Trajectory
from autoformalism.expressions import RestrictedParser
from autoformalism.schemas import CandidateModel

PROTOCOL = "d3-native-phase-b-rollout-1"

# The existing restricted AST interpreter accepts a numerical backend. These
# operations preserve native unguarded arithmetic; the ODE compiler's protected
# division would change the saved model. PyTorch is needed for fitting only.
_NUMPY = SimpleNamespace(
    tensor=lambda value, dtype=None: np.asarray(value, dtype=dtype),
    float64=np.float64,
    abs=np.abs,
    exp=np.exp,
    log=np.log,
    sigmoid=lambda value: np.exp(-np.logaddexp(0.0, -value)),
    sqrt=np.sqrt,
    tanh=np.tanh,
    minimum=np.minimum,
    maximum=np.maximum,
    nn=SimpleNamespace(
        functional=SimpleNamespace(
            softplus=lambda value: np.where(
                value > 20.0, value, np.logaddexp(0.0, value)
            )
        )
    ),
)


@dataclass(frozen=True)
class NativeMap:
    """Parsed native increment expressions and exact, finite saved coefficients."""

    states: tuple[str, ...]
    parameters: dict[str, float]
    processes: tuple[tuple[str, ast.AST], ...]
    equations: tuple[ast.AST, ...]
    bounds_violations: tuple[dict, ...]

    @classmethod
    def build(
        cls,
        candidate: CandidateModel,
        parameters: dict[str, float],
        observed: tuple[str, ...],
        inputs: tuple[str, ...],
    ) -> NativeMap:
        """Reject malformed models, but audit native Adam's unenforced bounds."""
        validate_native_candidate(candidate, observed, inputs)
        names = {item.name for item in candidate.parameters}
        if set(parameters) != names:
            raise ValueError("require exactly every saved global parameter")
        if not all(
            isinstance(value, (float, int))
            and not isinstance(value, bool)
            and np.isfinite(value)
            for value in parameters.values()
        ):
            raise ValueError("saved parameters must be finite")
        parser = RestrictedParser()
        available = {*observed, *inputs, *names, "t"}
        processes = tuple(
            (item.name, parser.parse(item.expression, location=item.name).tree.body)
            for item in _ordered_processes(candidate, available)
        )
        rhs = {item.state: item.rhs for item in candidate.state_equations}
        states = tuple(item.name for item in candidate.states)
        violations = tuple(
            {
                "parameter": item.name,
                "value": float(parameters[item.name]),
                "declared_bounds": item.bounds.model_dump(),
            }
            for item in candidate.parameters
            if item.bounds is not None
            and not item.bounds.lower <= parameters[item.name] <= item.bounds.upper
        )
        return cls(
            states,
            dict(parameters),
            processes,
            tuple(
                parser.parse(rhs[state], location=state).tree.body for state in states
            ),
            violations,
        )

    def increment(self, environment: dict) -> np.ndarray:
        """Evaluate only restricted ASTs; never reinterpret increments as rates."""
        environment = {**environment, **self.parameters}
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            for name, node in self.processes:
                environment[name] = _evaluate(node, environment, _NUMPY)
            values = np.asarray(
                [_evaluate(node, environment, _NUMPY) for node in self.equations],
                dtype=float,
            )
        if not np.isfinite(values).all():
            raise ValueError("native increment produced nonfinite values")
        return values


def predict(
    model: NativeMap,
    trajectory: Trajectory,
    *,
    teacher_forced: bool,
    seconds: float = 120.0,
) -> np.ndarray:
    """Use x_next=x+f; only the one-step diagnostic reads later target states."""
    deadline = monotonic() + seconds
    measured = {**trajectory.targets, **trajectory.auxiliaries}
    state = np.asarray([measured[name][0] for name in model.states], dtype=float)
    predictions = [state.copy()]
    for index in range(len(trajectory.time) - 1):
        if monotonic() >= deadline:
            raise TimeoutError("native rollout wall-clock limit reached")
        state = state.copy()
        for position, name in enumerate(model.states):
            if teacher_forced or name in trajectory.auxiliaries:
                state[position] = measured[name][index]
        environment = dict(zip(model.states, state, strict=True))
        environment["t"] = float(trajectory.time[index])
        environment.update(trajectory.fixed_covariates)
        environment.update(
            {name: values[index] for name, values in trajectory.external_inputs.items()}
        )
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            state = state + model.increment(environment)
        if not np.isfinite(state).all():
            raise ValueError("native rollout produced nonfinite states")
        predictions.append(state.copy())
    return np.asarray(predictions)


def _metrics(
    model: NativeMap,
    validation: DatasetSplit,
    scales: dict[str, float],
    *,
    teacher_forced: bool,
    seconds: float,
) -> dict:
    """Keep failures in the denominator and never average partial trajectories."""
    trajectories = []
    squared: dict[str, list[np.ndarray]] = {name: [] for name in scales}
    for trajectory in validation.trajectories:
        row = {"trajectory_id": trajectory.trajectory_id}
        try:
            values = predict(
                model, trajectory, teacher_forced=teacher_forced, seconds=seconds
            )
            offset = 1 if teacher_forced else 0
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                errors = {
                    name: np.square(
                        (
                            values[offset:, model.states.index(name)]
                            - trajectory.targets[name][offset:]
                        )
                        / scale
                    )
                    for name, scale in scales.items()
                }
            if any(not np.isfinite(values).all() for values in errors.values()):
                raise ValueError("nonfinite normalized residuals")
            row.update(
                success=True,
                per_target_normalized_mse={
                    name: finite_mean(values) for name, values in errors.items()
                },
            )
            for name, values in errors.items():
                squared[name].append(values)
        except (ArithmeticError, ValueError, TypeError, TimeoutError) as exc:
            row.update(success=False, message=f"{type(exc).__name__}: {exc}")
        trajectories.append(row)
    complete = all(row["success"] for row in trajectories)
    per_target = (
        {
            name: finite_mean(np.concatenate(values))
            for name, values in squared.items()
        }
        if complete
        else {}
    )
    return {
        "status": "complete" if complete else "rollout_failed",
        "normalized_mse": finite_mean(list(per_target.values()))
        if complete
        else None,
        "per_target_normalized_mse": per_target,
        "normalization_scales": scales,
        "trajectories": trajectories,
        "sample_policy": "exclude_initial" if teacher_forced else "include_initial",
    }


def finite_mean(values) -> float:
    """Average finite nonnegative errors without overflowing their sum."""
    values = np.asarray(values, dtype=float)
    maximum = float(np.max(values))
    return float(np.mean(values / maximum)) * maximum if maximum else 0.0


def evaluate_validation(
    candidate: CandidateModel,
    parameters: dict[str, float],
    train: DatasetSplit,
    validation: DatasetSplit,
    *,
    seconds: float = 120.0,
    saved_one_step_nmse: float | None = None,
) -> dict:
    """Compare native one-step and Phase-B target-recursive validation only."""
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("require a positive finite trajectory time limit")
    if train.name is not SplitName.TRAIN or validation.name is not SplitName.VALIDATION:
        raise ValueError("require TRAIN and VALIDATION splits, never TEST")
    if not train.trajectories or not validation.trajectories:
        raise ValueError("require nonempty development trajectories")
    first = train.trajectories[0]
    targets = tuple(first.targets)
    observed = (*targets, *first.auxiliaries)
    inputs = (*first.external_inputs, *first.fixed_covariates)
    for trajectory in (*train.trajectories, *validation.trajectories):
        if len(trajectory.time) < 2:
            raise ValueError("require at least two samples per trajectory")
        if (
            set(trajectory.targets) != set(targets)
            or set(trajectory.auxiliaries) != set(first.auxiliaries)
            or set(trajectory.external_inputs) != set(first.external_inputs)
            or set(trajectory.fixed_covariates) != set(first.fixed_covariates)
        ):
            raise ValueError("development channel identities differ")
    model = NativeMap.build(candidate, parameters, observed, inputs)
    native = _metrics(
        model,
        validation,
        _target_scales(train, targets),
        teacher_forced=True,
        seconds=seconds,
    )
    fitted_scales = TrainingScaler().fit(train).scales
    common_scales = {
        name: float(fitted_scales[f"target:{name}"].standard_deviation)
        for name in targets
    }
    recursive = _metrics(
        model, validation, common_scales, teacher_forced=False, seconds=seconds
    )
    measured = native["normalized_mse"]
    matches = (
        None
        if saved_one_step_nmse is None
        else bool(
            measured is not None
            and np.isclose(measured, saved_one_step_nmse, rtol=1e-6, atol=1e-10)
        )
    )
    return {
        "protocol": PROTOCOL,
        "state_update": "x_next = x + f(x, supplied_inputs); no dt multiplier",
        "native_one_step": native,
        "phase_b_rollout": recursive,
        "saved_one_step_nmse": saved_one_step_nmse,
        "saved_one_step_matches": matches,
        "bounds_violations": list(model.bounds_violations),
        "saved_parameter_values_preserved": True,
        "target_states_reset_after_initial": False,
        "supplied_auxiliaries_used_at_each_step": True,
        "continuous_time_requirement_certified": False,
        "scientific_validity_assessed": False,
        "test_data_opened": False,
        "parameter_refit_applied": False,
    }

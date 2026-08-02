"""Safe expression execution with D3's native Adam/Euler fitting protocol."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from autoformalism.data import DatasetSplit, Trajectory
from autoformalism.expressions import RestrictedParser
from autoformalism.schemas import CandidateModel, ParameterScope, StateKind


@dataclass(frozen=True)
class NativeD3Fit:
    """Fitted parameters and leakage-safe one-step metrics for one candidate."""

    parameters: dict[str, float]
    training_mse: float
    validation_mse: float
    epochs_completed: int
    target_scales: dict[str, float]


class NativeD3Error(RuntimeError):
    """Raised when a candidate cannot use the native D3 fitting protocol."""


def observed_state_names(candidate: CandidateModel) -> tuple[str, ...]:
    """Return the candidate's ordered observed-state skeleton."""
    return tuple(state.name for state in candidate.states)


def validate_native_candidate(
    candidate: CandidateModel,
    observed_channels: tuple[str, ...],
    available_inputs: tuple[str, ...],
) -> None:
    """Validate the fixed observed-state D3 contract and restricted expressions."""
    states = observed_state_names(candidate)
    if set(states) != set(observed_channels):
        raise NativeD3Error(
            "D3 candidate states must be exactly the observed channels: "
            f"{observed_channels}"
        )
    if any(state.kind is not StateKind.OBSERVED for state in candidate.states):
        raise NativeD3Error("native D3 does not introduce latent states")
    if any(
        parameter.scope is not ParameterScope.GLOBAL
        for parameter in candidate.parameters
    ):
        raise NativeD3Error("native D3 parameters must be global")
    equations = {equation.state: equation.rhs for equation in candidate.state_equations}
    if set(equations) != set(observed_channels):
        raise NativeD3Error("native D3 requires one equation per observed channel")
    process_names = tuple(process.name for process in candidate.processes)
    declared = {
        *observed_channels,
        *available_inputs,
        *(parameter.name for parameter in candidate.parameters),
        *process_names,
        "t",
    }
    parser = RestrictedParser()
    available = declared - set(process_names)
    for process in _ordered_processes(candidate, available):
        available.add(process.name)
    for state, expression in equations.items():
        parsed = parser.parse(expression, location=f"state_equation:{state}")
        unknown = parsed.symbols - available
        if unknown:
            raise NativeD3Error(
                f"equation {state} uses unavailable symbols: {sorted(unknown)}"
            )


def fit_native_d3(
    candidate: CandidateModel,
    training: DatasetSplit,
    validation: DatasetSplit,
    *,
    targets: tuple[str, ...],
    seed: int,
    epochs: int = 2_000,
    learning_rate: float = 1e-2,
    validation_interval: int = 10,
    patience_checks: int = 100,
) -> NativeD3Fit:
    """Fit with Adam and teacher-forced one-step Euler updates like upstream D3."""
    torch = _import_torch()
    torch.manual_seed(seed)
    scales = _target_scales(training, targets)
    parameters = {
        item.name: torch.nn.Parameter(
            torch.tensor(
                (item.initialization_range.lower + item.initialization_range.upper)
                / 2.0,
                dtype=torch.float64,
            )
        )
        for item in candidate.parameters
    }
    if not parameters:
        train_mse = _target_mse(candidate, training, {}, targets, scales, torch)
        validation_mse = _target_mse(
            candidate, validation, {}, targets, scales, torch
        )
        return NativeD3Fit({}, train_mse, validation_mse, 0, scales)

    optimizer = torch.optim.Adam(
        list(parameters.values()), lr=learning_rate, weight_decay=0.0
    )
    best_state: dict[str, Any] | None = None
    best_validation = float("inf")
    stale_checks = 0
    epochs_completed = 0
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = _raw_state_loss(candidate, training, parameters, torch)
        if not bool(torch.isfinite(loss)):
            raise NativeD3Error("native D3 training loss became nonfinite")
        loss.backward()
        optimizer.step()
        epochs_completed = epoch + 1
        if epoch % validation_interval != 0:
            continue
        with torch.no_grad():
            validation_loss = float(
                _raw_state_loss(candidate, validation, parameters, torch).item()
            )
        if not np.isfinite(validation_loss):
            raise NativeD3Error("native D3 validation loss became nonfinite")
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {
                name: copy.deepcopy(value.detach())
                for name, value in parameters.items()
            }
            stale_checks = 0
        else:
            stale_checks += 1
            if stale_checks >= patience_checks:
                break
    if best_state is not None:
        for name, value in parameters.items():
            value.data.copy_(best_state[name])
    fitted = {name: float(value.detach().item()) for name, value in parameters.items()}
    return NativeD3Fit(
        fitted,
        _target_mse(candidate, training, fitted, targets, scales, torch),
        _target_mse(candidate, validation, fitted, targets, scales, torch),
        epochs_completed,
        scales,
    )


def evaluate_native_d3(
    candidate: CandidateModel,
    split: DatasetSplit,
    parameters: dict[str, float],
    targets: tuple[str, ...],
    target_scales: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Evaluate frozen one-step target predictions with training-fitted scales."""
    torch = _import_torch()
    per_target = _per_target_squared_errors(
        candidate, split, parameters, targets, target_scales, torch
    )
    metrics = {
        target: float(np.mean(np.concatenate(errors)))
        for target, errors in per_target.items()
    }
    return float(np.mean(list(metrics.values()))), metrics


def _raw_state_loss(candidate, split, parameters, torch):
    pieces = []
    for trajectory in split.trajectories:
        predictions, truth = _predict_trajectory(
            candidate, trajectory, parameters, torch
        )
        pieces.append(torch.mean((predictions - truth) ** 2))
    return torch.mean(torch.stack(pieces))


def _target_mse(candidate, split, parameters, targets, scales, torch) -> float:
    errors = _per_target_squared_errors(
        candidate, split, parameters, targets, scales, torch
    )
    return float(
        np.mean([np.mean(np.concatenate(values)) for values in errors.values()])
    )


def _per_target_squared_errors(
    candidate, split, parameters, targets, scales, torch
):
    states = observed_state_names(candidate)
    indexes = {name: states.index(name) for name in targets}
    output: dict[str, list[np.ndarray]] = {target: [] for target in targets}
    with torch.no_grad():
        for trajectory in split.trajectories:
            predictions, truth = _predict_trajectory(
                candidate, trajectory, parameters, torch
            )
            for target, index in indexes.items():
                residual = (
                    predictions[:, index].detach().cpu().numpy()
                    - truth[:, index].detach().cpu().numpy()
                ) / scales[target]
                output[target].append(residual**2)
    return output


def _target_scales(split: DatasetSplit, targets: tuple[str, ...]) -> dict[str, float]:
    return {
        target: max(
            float(
                np.std(
                    np.concatenate(
                        [
                            trajectory.targets[target]
                            for trajectory in split.trajectories
                        ]
                    )
                )
            ),
            1e-8,
        )
        for target in targets
    }


def _predict_trajectory(candidate, trajectory: Trajectory, parameters, torch):
    state_names = observed_state_names(candidate)
    state_columns = [
        trajectory.targets.get(name, trajectory.auxiliaries.get(name))
        for name in state_names
    ]
    if any(column is None for column in state_columns):
        raise NativeD3Error("trajectory is missing a declared observed state")
    states = torch.as_tensor(np.column_stack(state_columns), dtype=torch.float64)
    if len(states) < 2:
        raise NativeD3Error("native D3 requires at least two time samples")
    time = torch.as_tensor(trajectory.time, dtype=torch.float64)
    environment = {
        name: states[:-1, index] for index, name in enumerate(state_names)
    }
    environment["t"] = time[:-1]
    for name, values in trajectory.external_inputs.items():
        environment[name] = torch.as_tensor(values[:-1], dtype=torch.float64)
    for name, value in trajectory.fixed_covariates.items():
        environment[name] = torch.full_like(time[:-1], float(value))
    environment.update(
        {
            name: value
            if hasattr(value, "requires_grad")
            else torch.tensor(float(value), dtype=torch.float64)
            for name, value in parameters.items()
        }
    )
    parser = RestrictedParser()
    for process in _ordered_processes(candidate, set(environment)):
        tree = parser.parse(
            process.expression, location=f"process:{process.name}"
        ).tree.body
        environment[process.name] = _evaluate(tree, environment, torch)
    equations = {equation.state: equation.rhs for equation in candidate.state_equations}
    derivatives = torch.stack(
        [
            _evaluate(
                parser.parse(equations[name], location=f"state:{name}").tree.body,
                environment,
                torch,
            )
            for name in state_names
        ],
        dim=1,
    )
    # Upstream D3 calls this quantity ``dx_dt`` but advances by direct addition
    # without multiplying by elapsed time. Preserve that native one-slot update.
    predictions = states[:-1] + derivatives
    if not bool(torch.all(torch.isfinite(predictions))):
        raise NativeD3Error("native D3 prediction became nonfinite")
    return predictions, states[1:]


def _ordered_processes(candidate: CandidateModel, available: set[str]):
    """Topologically order algebraics without executing proposer-generated text."""
    parser = RestrictedParser()
    remaining = list(candidate.processes)
    ordered = []
    while remaining:
        ready = []
        for process in remaining:
            parsed = parser.parse(
                process.expression, location=f"process:{process.name}"
            )
            if parsed.symbols <= available:
                ready.append(process)
        if not ready:
            unresolved = {
                process.name: sorted(
                    parser.parse(
                        process.expression, location=f"process:{process.name}"
                    ).symbols
                    - available
                )
                for process in remaining
            }
            raise NativeD3Error(
                f"cyclic or unavailable algebraic dependencies: {unresolved}"
            )
        for process in ready:
            ordered.append(process)
            available.add(process.name)
            remaining.remove(process)
    return tuple(ordered)


def _evaluate(node: ast.AST, environment: dict[str, Any], torch):
    if isinstance(node, ast.Constant):
        return torch.tensor(float(node.value), dtype=torch.float64)
    if isinstance(node, ast.Name):
        return environment[node.id]
    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand, environment, torch)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left, environment, torch)
        right = _evaluate(node.right, environment, torch)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        values = [_evaluate(argument, environment, torch) for argument in node.args]
        functions = {
            "abs": torch.abs,
            "exp": torch.exp,
            "log": torch.log,
            "sigmoid": torch.sigmoid,
            "softplus": torch.nn.functional.softplus,
            "sqrt": torch.sqrt,
            "tanh": torch.tanh,
        }
        if node.func.id in functions:
            return functions[node.func.id](values[0])
        reducer = torch.minimum if node.func.id == "min" else torch.maximum
        result = values[0]
        for value in values[1:]:
            result = reducer(result, value)
        return result
    raise NativeD3Error(f"unsupported expression node: {type(node).__name__}")


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise NativeD3Error(
            "D3-native-no-tools requires the optional dependency: "
            "pip install -e '.[d3]'"
        ) from exc
    return torch

"""Diagnostic-only inspection of the production rollout least-squares objective."""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from autoformalism.data import DatasetSplit
from autoformalism.expressions import CompiledModel
from autoformalism.fitting import fitter
from autoformalism.fitting.models import FailureCounter, FitConfig
from autoformalism.rebuttal.fitter_diagnostic import (
    _finite_payload,
    _write_bytes,
    read_json,
    sha256,
    write_json,
)
from autoformalism.staged_topology import content_hash


class RolloutOracle:
    """Call the unchanged production residual and expose its actual evaluation cost."""

    def __init__(
        self,
        model: CompiledModel,
        training: DatasetSplit,
        scales: Mapping[str, float],
        config: FitConfig,
        directory: Path,
        deadline: float | None,
    ) -> None:
        self.variables = fitter._training_variables(model, training, config)
        if any(not v.name.startswith("parameter:") for v in self.variables):
            raise ValueError(
                "diagnosis requires fixed initial conditions and global parameters"
            )
        self.names = tuple(v.name.removeprefix("parameter:") for v in self.variables)
        self.lower = np.array([v.lower for v in self.variables])
        self.upper = np.array([v.upper for v in self.variables])
        self.failures = FailureCounter()
        self.raw = fitter._residual_function(
            model,
            training.trajectories,
            lambda x: fitter._decode_training(model, training, self.variables, x),
            scales,
            config,
            self.failures,
            fitter._residual_size(training, model),
            deadline,
        )
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.seconds = 0.0
        self.best: dict[str, Any] | None = None

    def vector(self, parameters: Mapping[str, float]) -> np.ndarray:
        """Use the same physical parameter ordering and domains as production."""
        if set(parameters) != set(self.names):
            raise ValueError("parameter vector differs from production layout")
        values = np.array([parameters[name] for name in self.names], dtype=float)
        if (
            not np.isfinite(values).all()
            or np.any(values < self.lower)
            or np.any(values > self.upper)
        ):
            raise ValueError(
                "starting vector lies outside the unchanged parameter domain"
            )
        return values

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """Log physical residual calls, including derivative probes and failures."""
        self.calls += 1
        before = self.failures.count
        started = monotonic()
        record = {
            "call": self.calls,
            "parameters": dict(zip(self.names, map(float, values), strict=True)),
        }
        try:
            residual = self.raw(values)
        except Exception as error:
            record.update(
                status="raised", error=str(error), exception=type(error).__name__
            )
            raise
        else:
            cost = float(0.5 * np.dot(residual, residual))
            record.update(
                cost=cost,
                residual_norm=float(np.linalg.norm(residual)),
                residual_count=len(residual),
                status="evaluated",
            )
            if (
                self.failures.count == before
                and np.isfinite(cost)
                and (self.best is None or cost < self.best["cost"])
            ):
                self.best = {
                    "parameters": record["parameters"],
                    "cost": cost,
                    "call": self.calls,
                }
                write_json(self.directory / "best_evaluated.json", self.best)
            return residual
        finally:
            seconds = monotonic() - started
            self.seconds += seconds
            record.update(
                seconds=seconds, integration_failures=self.failures.count - before
            )
            write_json(
                self.directory / f"{self.calls:06d}.json", _finite_payload(record)
            )


def matrix_report(matrix: np.ndarray, residual: np.ndarray) -> dict[str, Any]:
    """Report raw physical-coordinate sensitivity without claiming identifiability."""
    singular = np.linalg.svd(matrix, compute_uv=False)
    gradient = matrix.T @ residual
    return _finite_payload(
        {
            "column_norms": np.linalg.norm(matrix, axis=0).tolist(),
            "singular_values": singular.tolist(),
            "matrix_rank": int(np.linalg.matrix_rank(matrix)),
            "condition_number": float(singular[0] / singular[-1])
            if singular[-1] > 0
            else None,
            "gradient": gradient.tolist(),
            "gradient_inf_norm": float(np.linalg.norm(gradient, ord=np.inf)),
        }
    )


def inspect_steps(
    oracle: RolloutOracle,
    anchor: Mapping[str, float],
    steps: tuple[float, ...],
    cache: Path,
    identity: str,
) -> dict[str, Any]:
    """Compare forward/central derivative estimates with auditable cached rollouts."""
    cache.mkdir(parents=True, exist_ok=True)
    x = oracle.vector(anchor)

    def evaluate(key: str, point: np.ndarray) -> np.ndarray:
        entry = cache / f"{key}.json"
        array = cache / f"{key}.npz"
        expected = content_hash([identity, key, point.tolist()])
        if entry.exists():
            record = read_json(entry)
            if record["identity"] != expected or record["array_sha256"] != sha256(
                array
            ):
                raise ValueError("profile cache differs from frozen point")
            with np.load(array, allow_pickle=False) as payload:
                return payload["residual"]
        before = oracle.failures.count
        residual = oracle(point)
        if oracle.failures.count != before:
            raise ValueError("profile point produced an integration failure")
        buffer = io.BytesIO()
        np.savez_compressed(buffer, residual=residual)
        _write_bytes(array, buffer.getvalue())
        write_json(
            entry,
            {
                "identity": expected,
                "array_sha256": sha256(array),
                "parameters": point.tolist(),
                "cost": float(0.5 * residual @ residual),
            },
        )
        return residual

    base = evaluate("base", x)
    repeated = evaluate("repeat", x)
    estimates = []
    matrices = []
    for index, step in enumerate(steps):
        forward = []
        central = []
        actual_steps = []
        for column in range(len(x)):
            amount = step * max(1.0, abs(x[column]))
            plus, minus = x.copy(), x.copy()
            plus[column] += amount
            minus[column] -= amount
            if (
                plus[column] > oracle.upper[column]
                or minus[column] < oracle.lower[column]
            ):
                raise ValueError(
                    "symmetric profile step leaves frozen parameter domain"
                )
            positive = evaluate(f"h{index}_p{column}_plus", plus)
            negative = evaluate(f"h{index}_p{column}_minus", minus)
            forward.append((positive - base) / amount)
            central.append((positive - negative) / (2 * amount))
            actual_steps.append(amount)
        fwd, ctr = np.column_stack(forward), np.column_stack(central)
        matrices.append(ctr)
        estimates.append(
            {
                "step_factor": step,
                "actual_absolute_steps": actual_steps,
                "forward": matrix_report(fwd, base),
                "central": matrix_report(ctr, base),
                "forward_central_relative_difference": float(
                    np.linalg.norm(fwd - ctr) / max(np.linalg.norm(ctr), 1e-300)
                ),
            }
        )
    comparisons = []
    for index in range(1, len(matrices)):
        left, right = matrices[index - 1], matrices[index]
        comparisons.append(
            {
                "left_step": steps[index - 1],
                "right_step": steps[index],
                "central_relative_difference": float(
                    np.linalg.norm(left - right) / max(np.linalg.norm(right), 1e-300)
                ),
            }
        )
    return {
        "parameters": dict(anchor),
        "parameter_order": oracle.names,
        "base_cost": float(0.5 * base @ base),
        "residual_count": len(base),
        "exact_repeat_residual_difference": float(np.linalg.norm(repeated - base)),
        "steps": estimates,
        "adjacent_steps": comparisons,
    }


def instrumented_fit(
    oracle: RolloutOracle,
    start: Mapping[str, float],
    *,
    diff_step: float | None,
    max_nfev: int,
    optimizer: Callable[..., Any] = least_squares,
) -> dict[str, Any]:
    """Run the production least-squares call with one explicit diagnostic override."""
    x = oracle.vector(start)
    iterations = []

    def callback(intermediate_result):
        values = np.asarray(intermediate_result.x)
        entry = {
            "iteration": len(iterations),
            "parameters": dict(zip(oracle.names, map(float, values), strict=True)),
            "cost": float(intermediate_result.cost),
            "nfev": int(intermediate_result.nfev),
            "distance_from_start": float(np.linalg.norm(values - x)),
        }
        iterations.append(entry)
        write_json(oracle.directory / f"iteration-{len(iterations):04d}.json", entry)

    started = monotonic()
    try:
        result = optimizer(
            oracle,
            x,
            bounds=(oracle.lower, oracle.upper),
            max_nfev=max_nfev,
            diff_step=diff_step,
            callback=callback,
        )
    except TimeoutError:
        selected = oracle.best
        report = {
            "optimizer_success": False,
            "optimizer_status": -2,
            "message": "fitting wall-clock limit reached",
            "selection": "best_finite_evaluation_after_timeout",
            "parameters": selected["parameters"] if selected else None,
            "cost": selected["cost"] if selected else None,
            "nfev": None,
            "njev": None,
            "optimality": None,
            "jacobian": None,
        }
    else:
        report = {
            "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status),
            "message": str(result.message),
            "selection": "optimizer_return",
            "parameters": dict(zip(oracle.names, map(float, result.x), strict=True)),
            "cost": float(result.cost),
            "nfev": int(result.nfev),
            "njev": int(result.njev) if result.njev is not None else None,
            "optimality": float(result.optimality),
            "gradient": np.asarray(result.grad).tolist(),
            "active_mask": np.asarray(result.active_mask).tolist(),
            "jacobian": matrix_report(np.asarray(result.jac), np.asarray(result.fun)),
        }
    return _finite_payload(
        {
            **report,
            "actual_residual_calls": oracle.calls,
            "residual_seconds": oracle.seconds,
            "fit_seconds": monotonic() - started,
            "integration_failures": oracle.failures.count,
            "integration_failure_messages": oracle.failures.messages,
            "best_evaluated": oracle.best,
            "iterations": iterations,
            "initial_parameters": dict(start),
            "domain": {
                name: {"lower": float(lo), "upper": float(hi)}
                for name, lo, hi in zip(
                    oracle.names, oracle.lower, oracle.upper, strict=True
                )
            },
        }
    )

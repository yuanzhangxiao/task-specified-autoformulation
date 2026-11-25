"""Fit frozen equations with hard nonnegativity constraints on training samples."""

from __future__ import annotations

import platform
from functools import lru_cache
from importlib.metadata import version
from time import monotonic

import numpy as np
from scipy.optimize import minimize

from autoformalism.research.trajectory_function_fitting import (
    _effective_sources,
    _metrics,
    _residual,
    _Rollout,
    _rows,
    _targets,
)
from autoformalism.staged_topology import content_hash

TOLERANCE = 1e-10


class _ConstrainedObjective:
    """Share each charged rollout between the objective and all constraint rows."""

    def __init__(self, graph, formula, rows, observed, names, constrained, deadline):
        self.names = names
        self.target_idx = [names.index(name) for name in constrained]
        self.observed = observed
        magnitudes = np.maximum(np.max(abs(observed), axis=1), 1e-8)
        self.scales = np.maximum(
            (observed / magnitudes[:, None]).std(axis=1) * magnitudes, 1e-8
        )
        self.coarse = _Rollout(graph, formula, names, 2)
        self.fine = _Rollout(graph, formula, names, 4)
        self.coarse_records = [self.coarse.record(row) for row in rows]
        self.fine_records = [self.fine.record(row) for row in rows]
        self.coarse_calls = self.fine_calls = self.failed = self.coarse_feasible = 0
        self.fine_feasible = self.unresolved = 0
        self.deadline = deadline
        self.limit = 0
        self.best_coarse_loss = float("inf")
        self.best_result = None
        self.invalid = np.r_[1e20, np.full(len(rows) * len(constrained), -1e10)]

    @property
    def calls(self) -> int:
        return self.coarse_calls + self.fine_calls

    def _charge(self, fine: bool) -> None:
        if self.calls >= self.limit or monotonic() >= self.deadline:
            raise TimeoutError("wall or rollout evaluation budget reached")
        if fine:
            self.fine_calls += 1
        else:
            self.coarse_calls += 1

    def evaluate(self, parameters: np.ndarray) -> np.ndarray:
        """Return pooled NMSE and scaled minimum for each training event and target."""
        self._charge(False)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                outputs = self.coarse.run(self.coarse_records, parameters)
                residual = _residual(outputs, self.observed, self.scales)
                loss = float(np.mean(residual**2))
                minima = np.stack([y.min(axis=1) for y, _ in outputs])
                bounds = minima[:, self.target_idx]
                constraints = bounds / self.scales[self.target_idx]
        except ArithmeticError:
            self.failed += 1
            return self.invalid
        if np.all(bounds >= -TOLERANCE):
            self.coarse_feasible += 1
            if loss < self.best_coarse_loss and self.calls < self.limit:
                self._verify(parameters, outputs, loss, minima)
        return np.r_[loss, constraints.ravel()]

    def _verify(self, parameters, coarse_outputs, loss, coarse_minima):
        """Retain only a finite, nonnegative and resolution-verified shared vector."""
        self._charge(True)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                fine = self.fine.run(self.fine_records, parameters)
                gap = float(
                    max(
                        np.max(abs(a - b) / self.scales[:, None])
                        for (a, _), (b, _) in zip(coarse_outputs, fine, strict=True)
                    )
                )
                metrics = _metrics(fine, self.observed, self.scales, self.names)
        except ArithmeticError:
            self.failed += 1
            return
        if gap > 0.01:
            self.unresolved += 1
            return
        if any(metrics["negative_samples"][self.names[i]] for i in self.target_idx):
            return
        self.fine_feasible += 1
        self.best_coarse_loss = loss
        self.best_result = {
            **metrics,
            "parameters": dict(
                zip(self.coarse.parameters, parameters.tolist(), strict=True)
            ),
            "maximum_scaled_grid_difference": gap,
            "minimum_coarse_prediction": dict(
                zip(self.names, coarse_minima.min(axis=0).tolist(), strict=True)
            ),
            "training_nmse": metrics["pooled_nmse"],
            "status": "fitted",
        }


def _optimize_start(problem, start, scale):
    """Pass explicit inequalities to SLSQP and share one finite-difference sweep."""

    @lru_cache(maxsize=len(start) + 2)
    def values(key: bytes) -> np.ndarray:
        delta = np.frombuffer(key, dtype=np.float64)
        return problem.evaluate(start + scale * delta)

    @lru_cache(maxsize=1)
    def jac(key: bytes) -> np.ndarray:
        x = np.frombuffer(key, dtype=np.float64)
        base = values(key)
        cols = []
        for i in range(len(x)):
            step = 1e-6 * max(1.0, abs(x[i]))
            shifted = x.copy()
            shifted[i] += step
            cols.append((values(shifted.tobytes()) - base) / (shifted[i] - x[i]))
        return np.column_stack(cols)

    x = np.zeros_like(start)
    if len(x) == 0:
        values(x.tobytes())
        return "fixed model evaluated"
    result = minimize(
        lambda v: values(v.tobytes())[0],
        x,
        method="SLSQP",
        jac=lambda v: jac(v.tobytes())[0],
        constraints=[
            {
                "type": "ineq",
                "fun": lambda v: values(v.tobytes())[1:],
                "jac": lambda v: jac(v.tobytes())[1:],
            }
        ],
        options={"ftol": 1e-9, "maxiter": problem.limit},
    )
    return str(result.message)


def fit_nonnegative(
    graph: dict,
    formula: dict,
    train: list,
    *,
    seed: int = 0,
    max_evaluations: int = 100,
    wall_seconds: float = 60.0,
    starts: int = 1,
    targets: list[str] | None = None,
) -> dict:
    """Constrain sampled training predictions, without changing equations or initials.

    Each coarse or fine complete training rollout costs one evaluation. Both
    objective and inequalities reuse the same predictions. Eligibility requires
    nonnegativity on both grids and the original finer-grid agreement threshold.
    """
    if (
        starts < 1
        or max_evaluations < starts
        or not np.isfinite(wall_seconds)
        or wall_seconds <= 0
    ):
        raise ValueError(
            "Require a positive budget with at least one evaluation per start"
        )
    started = monotonic()
    rows = _rows(train, "train")
    names = sorted(rows[0].targets)
    constrained = names if targets is None else list(targets)
    if (
        not constrained
        or len(set(constrained)) != len(constrained)
        or set(constrained) - set(names)
    ):
        raise ValueError("Expected distinct observed nonnegative targets")
    problem = _ConstrainedObjective(
        graph,
        formula,
        rows,
        _targets(rows, names),
        names,
        constrained,
        started + wall_seconds,
    )
    guess = np.array(
        [formula["parameters"][n] for n in problem.coarse.parameters], dtype=float
    )
    scale = np.maximum(abs(guess), 0.01)
    rng = np.random.default_rng(seed)
    reasons = []
    for index in range(starts):
        problem.limit = (index + 1) * max_evaluations // starts
        start = (
            guess if index == 0 else guess + rng.normal(size=len(guess)) * 0.25 * scale
        )
        try:
            reason = _optimize_start(problem, start, scale)
        except TimeoutError as error:
            reason = str(error)
        except (ArithmeticError, np.linalg.LinAlgError) as error:
            reason = f"Optimizer numerical failure: {error}"
        reasons.append(reason)
        if monotonic() >= problem.deadline:
            break
    result = problem.best_result or {
        "status": "fit_failed",
        "training_nmse": None,
        "parameters": None,
        "reason": "No feasible resolution-verified training vector within budget",
    }
    if result["parameters"] is not None:
        result["effective_sources"] = _effective_sources(
            graph, formula, result["parameters"]
        )
    return {
        **result,
        "graph_sha256": content_hash(graph),
        "formula_sha256": content_hash(formula),
        "evaluations": problem.calls,
        "failed_rollouts": problem.failed,
        "coarse_evaluations": problem.coarse_calls,
        "fine_evaluations": problem.fine_calls,
        "coarse_feasible_evaluations": problem.coarse_feasible,
        "fine_verified_feasible_evaluations": problem.fine_feasible,
        "resolution_rejections": problem.unresolved,
        "scales": dict(zip(names, problem.scales.tolist(), strict=True)),
        "constrained_targets": constrained,
        "nonnegativity_tolerance": TOLERANCE,
        "substeps": 2,
        "seconds": monotonic() - started,
        "budget": {
            "seed": seed,
            "max_evaluations": max_evaluations,
            "starts": starts,
            "wall_seconds": wall_seconds,
        },
        "starts_attempted": len(reasons),
        "stop_reasons": reasons,
        "runtime_versions": {
            "python": platform.python_version(),
            **{n: version(n) for n in ("casadi", "numpy", "scipy")},
        },
    }

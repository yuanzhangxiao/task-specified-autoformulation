"""Bounded Gaussian objective-rise support for certified affine observations.

This diagnostic consumes a retained fit and public training acquisition only.
It reoptimizes every other coordinate at each target value. Support thresholds
are declared objective rises, without a probability or coverage interpretation.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from time import monotonic, process_time
from typing import Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.optimize import brentq, lsq_linear

from autoformalism.fitting import robust_measurement
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.measurement_fitting import MeasurementProblem
from autoformalism.schemas.sensor_drift import DriftProblem

PROTOCOL = "bounded-affine-gaussian-practical-profile-1"


class Policy(StrictSchema):
    """Frozen thresholds, numerical conventions and independent work allowance."""

    protocol: Literal["practical-profile-policy-1"] = "practical-profile-policy-1"
    thresholds: tuple[FiniteFloat, ...] = (0.5, 2.0)
    grid_points: int = Field(default=41, ge=3, le=1001, strict=True)
    maximum_calls: int = Field(default=400, ge=1, le=10000, strict=True)
    max_inner_iterations: int = Field(default=200, ge=1, le=10000, strict=True)
    kkt_tolerance: FiniteFloat = Field(default=1e-8, gt=0, le=1e-4)
    objective_absolute_tolerance: FiniteFloat = Field(default=1e-6, gt=0, le=1e-3)
    objective_relative_tolerance: FiniteFloat = Field(default=1e-8, gt=0, le=1e-4)
    residual_relative_tolerance: FiniteFloat = Field(default=1e-7, gt=0, le=1e-4)
    root_relative_tolerance: FiniteFloat = Field(default=1e-10, gt=0, le=1e-5)
    rank_relative_tolerance: FiniteFloat = Field(default=1e-10, gt=0, le=1e-5)
    active_bound_relative_tolerance: FiniteFloat = Field(default=1e-8, gt=0, le=1e-4)

    @model_validator(mode="after")
    def ordered_thresholds(self):
        if (
            not self.thresholds
            or len(self.thresholds) > 10
            or any(value <= 0 for value in self.thresholds)
            or tuple(sorted(set(self.thresholds))) != self.thresholds
        ):
            raise ValueError("support thresholds must be positive and strictly ordered")
        return self


POLICY = Policy().model_dump(mode="json")


class ExecutionPolicy(StrictSchema):
    """Opt in to M40 execution changes without changing M39 numerical policy."""

    protocol: Literal["practical-profile-execution-2"] = "practical-profile-execution-2"
    mode: Literal["separate-exact-target-caches"] = "separate-exact-target-caches"


EXECUTION_POLICY = ExecutionPolicy().model_dump(mode="json")


class _SaveFailure(RuntimeError):
    """Stop further numerical work when a durable progress callback fails."""


class _Timing:
    """Disjoint numerical/callback timers, including exceptions and final saves."""

    def __init__(self, wall: float, cpu: float):
        self.wall, self.cpu = wall, cpu
        self.kinds: dict[str, dict] = {}
        self.save_wall = self.save_cpu = 0.0
        self.save_calls = self.save_failures = 0

    def numerical(self, kind: str, function):
        record = self.kinds.setdefault(
            kind,
            {
                "started": 0,
                "completed": 0,
                "failed": 0,
                "wall_seconds": 0.0,
                "cpu_seconds": 0.0,
            },
        )
        record["started"] += 1
        wall, cpu = monotonic(), process_time()
        try:
            value = function()
        except BaseException:
            record["failed"] += 1
            raise
        else:
            record["completed"] += 1
            return value
        finally:
            record["wall_seconds"] += monotonic() - wall
            record["cpu_seconds"] += process_time() - cpu

    def callback(self, function):
        self.save_calls += 1
        wall, cpu = monotonic(), process_time()
        try:
            function()
        except Exception as error:
            self.save_failures += 1
            raise _SaveFailure(
                f"practical profile save callback failed: {error}"
            ) from error
        finally:
            self.save_wall += monotonic() - wall
            self.save_cpu += process_time() - cpu

    def snapshot(self) -> dict:
        wall, cpu = monotonic() - self.wall, process_time() - self.cpu
        numerical_wall = sum(row["wall_seconds"] for row in self.kinds.values())
        numerical_cpu = sum(row["cpu_seconds"] for row in self.kinds.values())
        other_wall = max(0.0, wall - numerical_wall - self.save_wall)
        other_cpu = max(0.0, cpu - numerical_cpu - self.save_cpu)
        return {
            "protocol": "practical-profile-timing-1",
            "numerical_wall_seconds": numerical_wall,
            "numerical_cpu_seconds": numerical_cpu,
            "save_callback_wall_seconds": self.save_wall,
            "save_callback_cpu_seconds": self.save_cpu,
            "orchestration_wall_seconds": other_wall,
            "orchestration_cpu_seconds": other_cpu,
            "total_wall_seconds": wall,
            "total_cpu_seconds": cpu,
            "unattributed_non_cpu_wall_seconds": max(0.0, wall - cpu),
            "orchestration_non_cpu_wall_seconds": max(0.0, other_wall - other_cpu),
            "save_callback_calls": self.save_calls,
            "save_callback_failures": self.save_failures,
            "numerical_operations_started": sum(
                row["started"] for row in self.kinds.values()
            ),
            "numerical_operations_completed": sum(
                row["completed"] for row in self.kinds.values()
            ),
            "numerical_operations_failed": sum(
                row["failed"] for row in self.kinds.values()
            ),
            "numerical_by_kind": deepcopy(self.kinds),
            "scope": (
                "process CPU and monotonic wall; disjoint timers include failures. "
                "Wall minus CPU cannot distinguish scheduler wait from blocking. "
                "Progress snapshots exclude the currently executing save callback."
            ),
        }


class _SolveFailure(ValueError):
    """Preserve known inner work even when the optimizer cannot qualify a node."""

    def __init__(self, message: str, iterations: int):
        super().__init__(message)
        self.iterations = iterations


def _active(values, lower, upper, names, policy: Policy) -> list[dict]:
    """List all active coordinate faces under a declared domain-relative rule."""
    tolerance = policy.active_bound_relative_tolerance * (upper - lower)
    result = []
    for i, name in enumerate(names):
        for side, bound in (("lower", lower), ("upper", upper)):
            if abs(values[i] - bound[i]) <= tolerance[i]:
                result.append({"name": name, "side": side, "value": float(bound[i])})
    return result


def _solve(design, offset, lower, upper, policy: Policy) -> tuple[np.ndarray, dict]:
    """Solve a convex box-constrained LS problem and independently check KKT.

    Unit-norm columns make the KKT convention independent of parameter units.
    Rank deficiency permits a global optimum, but does not establish uniqueness.
    """
    columns = np.linalg.norm(design, axis=0)
    columns = np.where(columns > 0, columns, 1.0)
    normalized = design / columns
    lo, hi = lower * columns, upper * columns
    if not np.isfinite(np.r_[lo, hi, normalized.ravel(), offset]).all():
        raise ValueError("nonfinite normalized affine problem")
    if design.shape[1]:
        solved = lsq_linear(
            normalized,
            -offset,
            bounds=(lo, hi),
            method="bvls",
            tol=1e-13,
            max_iter=policy.max_inner_iterations,
        )
        if not solved.success:
            raise _SolveFailure(
                f"bounded affine solve did not converge: {solved.message}",
                int(solved.nit),
            )
        values = np.clip(solved.x / columns, lower, upper)
        iterations = int(solved.nit)
    else:
        values, iterations = np.empty(0), 0
    residual = design @ values + offset
    gradient = normalized.T @ residual
    at_lower = (values - lower) <= policy.active_bound_relative_tolerance * (
        upper - lower
    )
    at_upper = (upper - values) <= policy.active_bound_relative_tolerance * (
        upper - lower
    )
    violation = np.where(at_lower, np.minimum(gradient, 0), gradient)
    violation = np.where(at_upper, np.maximum(violation, 0), violation)
    kkt = float(np.max(np.abs(violation), initial=0))
    tolerance = policy.kkt_tolerance * max(1.0, float(np.linalg.norm(residual)))
    objective = float(residual @ residual / 2)
    if not np.isfinite(np.r_[values, residual, gradient, objective]).all():
        raise _SolveFailure("nonfinite bounded affine solution", iterations)
    if kkt > tolerance:
        raise _SolveFailure(
            f"bounded affine KKT check failed: {kkt} > {tolerance}", iterations
        )
    return values, {
        "objective": objective,
        "kkt_violation": kkt,
        "kkt_tolerance": tolerance,
        "iterations": iterations,
        "solver": "column-normalized BVLS with independent KKT",
    }


def _curvature(design, target: int, policy: Policy) -> dict:
    """Project in whitened reading space, keeping raw parameter units."""
    column = design[:, target]
    nuisance = np.delete(design, target, axis=1)
    scales = np.linalg.norm(nuisance, axis=0)
    normalized = nuisance / np.where(scales > 0, scales, 1)
    left, singular, _ = np.linalg.svd(normalized, full_matrices=False)
    tolerance = policy.rank_relative_tolerance * max(singular, default=0.0)
    rank = int(np.sum(singular > tolerance))
    basis = left[:, :rank]
    conditional = column - basis @ (basis.T @ column)
    raw = float(conditional @ conditional)
    frozen = float(column @ column)
    conditional_tolerance = policy.rank_relative_tolerance * np.linalg.norm(column)
    conditional_rank = int(np.linalg.norm(conditional) > conditional_tolerance)
    joint_scale = np.linalg.norm(design, axis=0)
    joint_singular = np.linalg.svd(
        design / np.where(joint_scale > 0, joint_scale, 1), compute_uv=False
    )
    joint_rank = int(
        np.sum(joint_singular > policy.rank_relative_tolerance * joint_singular[0])
    )
    return {
        "conditional_information": raw if conditional_rank else 0.0,
        "raw_conditional_information": raw,
        "frozen_information": frozen,
        "conditional_rank": conditional_rank,
        "nuisance_rank": rank,
        "joint_rank": joint_rank,
        "joint_dimension": design.shape[1],
        "conditional_column_rank_tolerance": float(conditional_tolerance),
        "nuisance_normalized_singular_values": singular.tolist(),
        "joint_normalized_singular_values": joint_singular.tolist(),
        "units": "Gaussian NLL curvature per squared raw target parameter unit",
        "scope": (
            "unconstrained data information; "
            "nuisance bounds add no measurement information"
        ),
        "projection": "a - Q_nuisance @ (Q_nuisance.T @ a)",
    }


def _endpoint_identity(problem, endpoint: dict) -> None:
    """Reject endpoints detached from this acquisition or verified Gaussian loss."""
    if not isinstance(endpoint, dict):
        raise ValueError("full retained fit endpoint required")
    identity = endpoint.get("identity", {})
    verification = endpoint.get("verification", {})
    if not all(isinstance(value, dict) for value in (identity, verification)):
        raise ValueError("malformed retained identity or verification")
    if identity.get("problem_sha256") != content_sha256(
        problem.model_dump(mode="json")
    ):
        raise ValueError("retained endpoint problem identity differs")
    if identity.get("base_request_sha256") != content_sha256(
        problem.request.model_dump(mode="json")
    ):
        raise ValueError("retained endpoint candidate identity differs")
    if endpoint.get("status") != "complete":
        raise ValueError("retained endpoint is not verified complete")
    if verification.get("passed") is not True or verification.get(
        "parameters"
    ) != endpoint.get("parameters"):
        raise ValueError("retained endpoint verification is absent or detached")
    expected = (
        ("sensor-drift-fitting-execution-1", "sensor-drift-fit-result-1")
        if isinstance(problem, DriftProblem)
        else (
            (
                "robust-measurement-fitting-execution-1",
                "robust-measurement-fit-result-1",
            )
            if "objective_definition" in identity
            else ("measurement-fitting-execution-1", "measurement-fit-result-1")
        )
    )
    if (identity.get("protocol"), endpoint.get("protocol")) != expected:
        raise ValueError("retained endpoint execution/result protocol differs")
    if "objective_definition" in identity:
        loss = robust_measurement.Loss.model_validate(identity["objective_definition"])
        if loss.kind != "gaussian" or any(
            record.get("objective_definition") != identity["objective_definition"]
            for record in (endpoint, verification)
        ):
            raise ValueError(
                "practical profile requires the same Gaussian objective throughout"
            )


def evaluate(
    problem: DriftProblem | MeasurementProblem,
    endpoint: dict,
    *,
    target: str = "g",
    policy: Policy | None = None,
    deadline: float,
    save: Callable[[dict], None] | None = None,
    execution: ExecutionPolicy | None = None,
) -> dict:
    """Qualify a retained optimum, profile nuisances, and locate bounded support.

    Every expensive operation publishes a start and terminal callback before the
    next operation. Completed nodes survive a partial result; incomplete support
    brackets never produce a width. The caller owns durable journal persistence.
    """
    invocation_wall, invocation_cpu = monotonic(), process_time()
    if execution is not None:
        if not isinstance(execution, ExecutionPolicy):
            raise TypeError("typed practical profile execution policy required")
        execution = ExecutionPolicy.model_validate(execution.model_dump(mode="json"))
    timing = _Timing(invocation_wall, invocation_cpu) if execution is not None else None
    if not isinstance(problem, (MeasurementProblem, DriftProblem)):
        raise TypeError("typed public measurement or sensor drift problem required")
    problem = type(problem).model_validate(problem.model_dump(mode="json"))
    policy = (
        Policy()
        if policy is None
        else Policy.model_validate(policy.model_dump(mode="json"))
    )
    if not np.isfinite(deadline):
        raise ValueError("finite monotonic deadline required")
    started = monotonic()
    result = {
        "protocol": PROTOCOL,
        "policy": policy.model_dump(mode="json"),
        "problem_sha256": content_sha256(problem.model_dump(mode="json")),
        "endpoint_sha256": content_sha256(endpoint),
        "target": target,
        "status": "unqualified",
        "reason": None,
        "endpoint_qualification": None,
        "calls": 0,
        "completed_calls": 0,
        "solve_calls": 0,
        "rollout_calls": 0,
        "failed_calls": 0,
        "inner_iterations": 0,
        "inner_iterations_complete": True,
        "nodes_completed": 0,
        "nodes": [],
        "supports": [],
        "curvature": None,
        "confidence_probability": None,
        "reference_values_used": False,
        "test_data_opened": False,
        "validation_used_for_selection": False,
        "interpretation": (
            "bounded Gaussian objective-rise support; no coverage probability"
        ),
    }
    if execution is not None:
        result.update(
            execution=execution.model_dump(mode="json"),
            frozen_calls=0,
            frozen_nodes_completed=0,
            frozen_nodes=[],
            cache_reuses={"profiled": 0, "frozen": 0},
            journal_persistence_complete=True,
        )

    def persist(event):
        if save is None or (execution and not result["journal_persistence_complete"]):
            return
        snapshot = {
            key: result[key]
            for key in (
                "calls",
                "completed_calls",
                "solve_calls",
                "rollout_calls",
                "failed_calls",
                "inner_iterations",
                "inner_iterations_complete",
                "nodes_completed",
            )
        } | {"event": deepcopy(event)}
        if timing is None:
            save(snapshot)
        else:
            snapshot.update(
                execution=result["execution"],
                timing=timing.snapshot(),
                frozen_calls=result["frozen_calls"],
                frozen_nodes_completed=result["frozen_nodes_completed"],
                cache_reuses=dict(result["cache_reuses"]),
            )
            try:
                timing.callback(lambda: save(snapshot))
            except _SaveFailure:
                result["journal_persistence_complete"] = False
                raise

    def numerical(kind, function):
        return function() if timing is None else timing.numerical(kind, function)

    def check():
        if monotonic() >= deadline:
            raise TimeoutError("practical profile wall allowance exhausted")
        if result["calls"] >= policy.maximum_calls:
            raise TimeoutError("practical profile call allowance exhausted")

    def call(phase, function, *, point=None, kind="solve"):
        check()
        result["calls"] += 1
        result[f"{kind}_calls"] += 1
        event = {"phase": phase, "point": point, "status": "started"}
        if execution is None:
            persist(event)
        iterations_recorded = False
        numerical_entered = False
        try:
            if execution is not None:
                persist(event)
                if monotonic() >= deadline:
                    raise TimeoutError(
                        "practical profile allowance exhausted during save"
                    )
            numerical_entered = True
            value = numerical(kind, function)
            if kind == "solve":
                result["inner_iterations"] += value[1]["iterations"]
                iterations_recorded = True
            if monotonic() >= deadline:
                raise TimeoutError(
                    "practical profile operation exceeded wall allowance"
                )
        except (ValueError, RuntimeError, TimeoutError, np.linalg.LinAlgError) as error:
            if kind == "solve" and isinstance(error, _SolveFailure):
                result["inner_iterations"] += error.iterations
            elif kind == "solve" and numerical_entered and not iterations_recorded:
                result["inner_iterations_complete"] = False
            result["completed_calls"] += 1
            result["failed_calls"] += 1
            persist(event | {"status": "failed", "reason": str(error)})
            raise
        result["completed_calls"] += 1
        persist(event | {"status": "completed"})
        return value

    if execution is None:
        persist({"phase": "diagnostic", "status": "started"})
    qualified = False
    try:
        if execution is not None:
            persist({"phase": "diagnostic", "status": "started"})
        check()
        _endpoint_identity(problem, endpoint)
        optimizer, profile, route = numerical(
            "setup",
            lambda: robust_measurement.build(
                problem, robust_measurement.Loss(kind="gaussian")
            ),
        )
        oracle = optimizer.base
        if (
            route["selected"] != "affine_profile"
            or profile is None
            or tuple(profile.outer)
            or set(profile.inner) != set(range(len(oracle.names)))
            or profile.audit.get("propagation") != "affine-source-matrix-exponential-1"
        ):
            raise ValueError(
                "complete measured output is not certified globally affine "
                "in every unknown"
            )
        result["affine_certificate"] = deepcopy(profile.audit)
        names = list(oracle.names)
        if target not in names:
            raise ValueError("profile target is not an estimated coordinate")
        target_index = names.index(target)
        nuisance = [i for i in range(len(names)) if i != target_index]
        nuisance_names = [names[i] for i in nuisance]
        lower, upper = np.asarray(oracle.lower), np.asarray(oracle.upper)
        if (
            not np.isfinite(np.r_[lower, upper, oracle.units]).all()
            or np.any(lower >= upper)
            or np.any(oracle.units <= 0)
        ):
            raise ValueError("finite strict domains and positive scales are required")
        vector = oracle.vector(endpoint["parameters"])
        residual, design = call(
            "affine_design", lambda: oracle.evaluate(vector, deadline), kind="rollout"
        )
        if (
            design.shape != (len(residual), len(names))
            or not len(residual)
            or not np.isfinite(np.r_[residual, design.ravel()]).all()
        ):
            raise ValueError("invalid certified affine residual or design")
        stored = np.asarray(
            endpoint["verification"].get("standardized_residuals"), float
        )
        if (
            stored.shape != residual.shape
            or not np.isfinite(stored).all()
            or np.linalg.norm(stored - residual)
            > policy.residual_relative_tolerance * max(1.0, np.linalg.norm(residual))
        ):
            raise ValueError(
                "retained residuals do not reproduce at the bound endpoint"
            )
        offset = residual - design @ vector
        optimum, audit = call(
            "global_optimum", lambda: _solve(design, offset, lower, upper, policy)
        )
        retained_objective = float(residual @ residual / 2)
        tolerance = (
            policy.objective_absolute_tolerance
            + policy.objective_relative_tolerance
            * max(1.0, retained_objective, audit["objective"])
        )
        saved_objectives = [
            endpoint.get("objective_half_sum_squares"),
            endpoint["verification"].get("primary_half_sum_squares"),
        ]
        if "objective_definition" in endpoint["identity"]:
            saved_objectives.extend(
                [
                    endpoint.get("objective_value"),
                    endpoint["verification"].get("primary_objective_value"),
                ]
            )
        if any(
            value is None
            or not np.isfinite(value)
            or abs(value - retained_objective) > tolerance
            for value in saved_objectives
        ):
            raise ValueError(
                "retained objective does not match its reproduced residuals"
            )
        difference = retained_objective - audit["objective"]
        result["endpoint_qualification"] = {
            "passed": abs(difference) <= tolerance,
            "retained_objective": retained_objective,
            "global_objective": audit["objective"],
            "objective_gap": difference,
            "objective_tolerance": tolerance,
            "global_parameters": oracle.parameters(optimum),
            "solver_audit": audit,
            "scope": "global convex optimum on declared parameter box",
        }
        if abs(difference) > tolerance:
            raise ValueError(
                "retained endpoint is not a global own-loss optimum "
                "within declared tolerance"
            )
        qualified = True
        result["curvature"] = numerical(
            "curvature", lambda: _curvature(design, target_index, policy)
        )
        result["parameter_order"], result["nuisance_names"] = names, nuisance_names
        result["retained_parameters"] = dict(endpoint["parameters"])
        result["domain"] = [float(lower[target_index]), float(upper[target_index])]
        result["retained_active_bounds"] = _active(vector, lower, upper, names, policy)
        center = float(vector[target_index])
        base_objective = audit["objective"]
        result["objective_rise_reference"] = (
            "independently certified bounded global Gaussian optimum"
        )
        cache, frozen_cache = {}, {}

        def frozen_values(value: float) -> dict:
            """Evaluate the retained-nuisance slice; never imply an inner optimum."""
            frozen = vector.copy()
            frozen[target_index] = value
            frozen_residual = design @ frozen + offset
            objective = float(frozen_residual @ frozen_residual / 2)
            if not np.isfinite(objective) or objective < base_objective - tolerance:
                raise ValueError(
                    "frozen slice contradicts the certified global optimum"
                )
            return {
                "value": value,
                "mode": "frozen",
                "parameters": oracle.parameters(frozen),
                "frozen_objective": objective,
                "frozen_rise": max(0.0, objective - base_objective),
            }

        def node(value: float, phase: str = "grid") -> dict:
            value = float(value)
            if value in cache:
                if execution:
                    if monotonic() >= deadline:
                        raise TimeoutError("practical profile wall allowance exhausted")
                    result["cache_reuses"]["profiled"] += 1
                return cache[value]
            nuisance_values, inner = call(
                phase,
                lambda: _solve(
                    design[:, nuisance],
                    offset + design[:, target_index] * value,
                    lower[nuisance],
                    upper[nuisance],
                    policy,
                ),
                point=value,
            )
            fitted = vector.copy()
            fitted[target_index], fitted[nuisance] = value, nuisance_values
            if execution:
                if value in frozen_cache:
                    result["cache_reuses"]["frozen"] += 1
                    frozen_entry = frozen_cache[value]
                else:
                    frozen_entry = numerical(
                        "frozen_with_profile", lambda: frozen_values(value)
                    )
                frozen_objective = frozen_entry["frozen_objective"]
            else:
                frozen = vector.copy()
                frozen[target_index] = value
                frozen_residual = design @ frozen + offset
                frozen_objective = float(frozen_residual @ frozen_residual / 2)
            if min(inner["objective"], frozen_objective) < base_objective - tolerance:
                raise ValueError(
                    "profile node contradicts the certified global optimum"
                )
            entry = {
                "value": value,
                "phase": phase,
                "parameters": oracle.parameters(fitted),
                "profiled_objective": inner["objective"],
                "profiled_rise": max(0.0, inner["objective"] - base_objective),
                "frozen_objective": frozen_objective,
                "frozen_rise": max(0.0, frozen_objective - base_objective),
                "nuisance_active_bounds": _active(
                    nuisance_values,
                    lower[nuisance],
                    upper[nuisance],
                    nuisance_names,
                    policy,
                ),
                "solver_audit": inner,
            }
            if inner["objective"] > frozen_objective + tolerance:
                raise ValueError(
                    "profiled nuisance objective exceeds feasible frozen slice"
                )
            cache[value] = entry
            if execution:
                # This same-point arithmetic was already performed above. It is
                # evidence only for the frozen curve, not a second inner solve.
                frozen_cache[value] = frozen_entry
            result["nodes"].append(entry)
            result["nodes_completed"] += 1
            persist({"phase": phase, "status": "node_completed", "node": entry})
            return entry

        def curve_node(value: float, mode: str, phase: str = "grid") -> dict:
            """Cache exact floating target values separately for the two curves."""
            if execution is None or mode == "profiled":
                return node(value, phase)
            value = float(value)
            if value in frozen_cache:
                if monotonic() >= deadline:
                    raise TimeoutError("practical profile wall allowance exhausted")
                result["cache_reuses"]["frozen"] += 1
                return frozen_cache[value]
            entry = call(
                phase + "_frozen",
                lambda: frozen_values(value),
                point=value,
                kind="frozen",
            ) | {"phase": phase}
            frozen_cache[value] = entry
            result["frozen_nodes"].append(entry)
            result["frozen_nodes_completed"] += 1
            persist({"phase": phase, "status": "frozen_node_completed", "node": entry})
            return entry

        # The retained point is first so an interrupted grid never silently uses
        # a nearby point as its minimum; all declared grid nodes must finish.
        node(center)
        grid = np.unique(
            np.r_[np.linspace(*result["domain"], policy.grid_points), center]
        )
        result["grid"] = grid.tolist()
        for value in grid:
            node(value)

        def support_end(bound: float, threshold: float, mode: str) -> dict:
            endpoint_node = curve_node(bound, mode)
            key = mode + "_rise"
            crossed = endpoint_node[key] > threshold
            if crossed:
                if curve_node(center, mode)[key] > threshold:
                    raise ValueError(
                        "retained point lies outside declared objective-rise threshold"
                    )
                value = brentq(
                    lambda point: curve_node(point, mode, "threshold_crossing")[key]
                    - threshold,
                    min(bound, center),
                    max(bound, center),
                    xtol=policy.root_relative_tolerance
                    * (upper[target_index] - lower[target_index]),
                    rtol=4 * np.finfo(float).eps,
                    maxiter=100,
                )
                endpoint_node = curve_node(value, mode, "threshold_crossing")
            active = (
                endpoint_node["nuisance_active_bounds"]
                if mode == "profiled"
                else _active(
                    vector[nuisance],
                    lower[nuisance],
                    upper[nuisance],
                    nuisance_names,
                    policy,
                )
            )
            classification = (
                "gain_domain_clipped"
                if not crossed
                else "nuisance_domain_assisted"
                if active and mode == "profiled"
                else "interior_data_limited"
                if mode == "profiled"
                else "frozen_nuisance_slice"
            )
            return {
                "value": endpoint_node["value"],
                "status": "threshold_crossed" if crossed else "domain_limited",
                "objective_rise": endpoint_node[key],
                "nuisance_active_bounds": active,
                "classification": classification,
            }

        for threshold in policy.thresholds:
            support = {"threshold": threshold}
            for mode in ("profiled", "frozen"):
                left, right = [
                    support_end(bound, threshold, mode) for bound in result["domain"]
                ]
                clipped = any(
                    end["status"] == "domain_limited" for end in (left, right)
                )
                support[mode] = {
                    "lower": left["value"],
                    "upper": right["value"],
                    "width": right["value"] - left["value"],
                    "domain_clipped": clipped,
                    "classification": (
                        "gain_domain_clipped"
                        if clipped
                        else "nuisance_domain_assisted"
                        if any(
                            end["classification"] == "nuisance_domain_assisted"
                            for end in (left, right)
                        )
                        else "interior_data_limited"
                        if mode == "profiled"
                        else "frozen_nuisance_slice"
                    ),
                    "left": left,
                    "right": right,
                }
            result["supports"].append(support)
            persist({"phase": "support", "status": "completed", "support": support})
        result["status"] = "complete"
    except (ValueError, RuntimeError, TimeoutError, np.linalg.LinAlgError) as error:
        result["status"] = (
            "partial" if qualified or isinstance(error, TimeoutError) else "unqualified"
        )
        result["reason"] = str(error)
    result["elapsed_seconds"] = monotonic() - started
    result["accounting_complete"] = result["calls"] == result["completed_calls"]
    try:
        persist(
            {
                "phase": "diagnostic",
                "status": result["status"],
                "reason": result["reason"],
            }
        )
    except _SaveFailure as error:
        result["status"] = "partial" if qualified else "unqualified"
        result["reason"] = str(error)
    if timing is not None:
        if monotonic() >= deadline and result["status"] == "complete":
            result["status"] = "partial"
            result["reason"] = "practical profile wall allowance exhausted during save"
        result["timing"] = timing.snapshot()
        result["elapsed_seconds"] = result["timing"]["total_wall_seconds"]
    return result

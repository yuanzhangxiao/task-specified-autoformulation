"""M41 bounded nonlinear nuisance profiles for one declared training control.

Both searches use the exact exponential solution and bounded affine projection.
Their search algorithms differ; agreement is numerical evidence, not a global
certificate. An original-equation rollout independently checks that shared math.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from itertools import pairwise
from time import monotonic, process_time
from typing import Literal

import numpy as np
from pydantic import Field
from scipy.optimize import least_squares, minimize_scalar

from autoformalism.fitting import practical_profile as affine
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import robust_measurement
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.measurement_fitting import analysis_layout
from autoformalism.schemas.sensor_drift import DriftProblem

PROTOCOL = "bounded-nonlinear-gaussian-practical-profile-1"
NAMES = ("g", "k", "x0", "sensor_drift")
DOMAINS = {
    "g": (0.05, 3.0),
    "k": (0.02, 1.5),
    "x0": (0.01, 2.0),
    "sensor_drift": (-2.0, 2.0),
}


class Policy(StrictSchema):
    """Frozen numerical conventions and total, per-operation work allowances."""

    protocol: Literal["nonlinear-practical-profile-policy-1"] = (
        "nonlinear-practical-profile-policy-1"
    )
    thresholds: tuple[Literal[0.5], Literal[2.0]] = (0.5, 2.0)
    grid_points: int = Field(default=31, ge=3, le=301, strict=True)
    local_sampling: bool = True
    reference_points: int = Field(default=101, ge=9, le=1001, strict=True)
    nuisance_starts: int = Field(default=5, ge=2, le=15, strict=True)
    maximum_calls: int = Field(default=15000, ge=1, le=100000, strict=True)
    operation_calls: int = Field(default=1000, ge=1, le=5000, strict=True)
    max_nfev: int = Field(default=80, ge=1, le=500, strict=True)
    wall_seconds: FiniteFloat = Field(default=60, gt=0, le=600)
    operation_seconds: FiniteFloat = Field(default=10, gt=0, le=120)
    callback_seconds: FiniteFloat = Field(default=1, gt=0, le=30)
    objective_absolute_tolerance: FiniteFloat = Field(default=1e-5, gt=0, le=1e-2)
    objective_relative_tolerance: FiniteFloat = Field(default=1e-7, gt=0, le=1e-3)
    rollout_tolerance: FiniteFloat = Field(default=1e-6, gt=0, le=1e-3)
    rank_relative_tolerance: FiniteFloat = Field(default=1e-9, gt=0, le=1e-4)


POLICY = Policy().model_dump(mode="json")


def _seal(state: dict) -> dict:
    """Hash the complete checkpoint body, including completed work and its cost."""
    state.pop("state_sha256", None)
    state["state_sha256"] = content_sha256(state)
    return state


def _public_arrays(problem: DriftProblem) -> tuple[np.ndarray, ...]:
    """Fail closed outside the isolated known-structure acquisition contract."""
    candidate = problem.request.base_candidate
    domains = {p.name: (p.bounds.lower, p.bounds.upper) for p in candidate.parameters}
    equations = [(e.state, e.rhs.replace(" ", "")) for e in candidate.state_equations]
    outputs = [(m.channel, m.expression) for m in candidate.observation_mappings]
    if (
        domains != {key: DOMAINS[key] for key in ("g", "k", "x0")}
        or equations not in ([("x", "-k*x+g*u")], [("x", "g*u-k*x")])
        or outputs != [("y", "x")]
        or candidate.processes
        or tuple(problem.request.context.targets) != ("y",)
        or tuple(problem.request.context.external_inputs) != ("u",)
        or len(problem.initial_rules) != 1
        or (
            problem.initial_rules[0].state,
            problem.initial_rules[0].mode,
            problem.initial_rules[0].shared_coordinate,
        )
        != ("x", "shared_value", "x0")
        or (problem.drift.name, problem.drift.channel, problem.drift.time_origin)
        != ("sensor_drift", "y", 0.0)
        or (problem.drift.lower, problem.drift.upper) != DOMAINS["sensor_drift"]
    ):
        raise ValueError("unsupported nonlinear control structure or domains")
    layout = analysis_layout(problem.base)
    if (
        len(problem.training.rows) != 3
        or len(layout.values) != 51
        or not all(layout.included)
        or set(layout.channels) != {"y"}
    ):
        raise ValueError("nonlinear control requires three complete 17-reading series")
    channel = next(c for c in problem.manifest.channels if c.id == "u")
    forcings = {}
    for row in problem.training.rows:
        times = [
            t
            for t, name in zip(
                layout.analysis_times, layout.trajectory_ids, strict=True
            )
            if name == row.trajectory_id
        ]
        values = [
            channel.gain * r.raw_value + channel.offset
            for r in row.readings
            if r.channel == "u" and r.condition == "present"
        ]
        if not np.allclose(times, np.linspace(0, 4, 17), rtol=0, atol=1e-12) or (
            len(values) != 17 or not np.all(np.asarray(values) == values[0])
        ):
            raise ValueError("nonlinear control requires constant inputs at times 0..4")
        forcings[row.trajectory_id] = values[0]
    arrays = (
        np.asarray(layout.analysis_times, float),
        np.asarray([forcings[name] for name in layout.trajectory_ids], float),
        np.asarray(layout.values, float),
        np.asarray(layout.sigma, float),
    )
    if not all(np.isfinite(a).all() for a in arrays) or np.any(arrays[3] <= 0):
        raise ValueError("finite public readings and positive known sigma required")
    return arrays


def _basis(arrays: tuple[np.ndarray, ...], k: float) -> np.ndarray:
    """Exact columns for gain, shared initial and observation drift at fixed k."""
    time, forcing, _, sigma = arrays
    decay = np.exp(-k * time)
    return (
        np.column_stack((forcing * -np.expm1(-k * time) / k, decay, time))
        / (sigma[:, None])
    )


def _residual(arrays: tuple[np.ndarray, ...], parameters: dict) -> np.ndarray:
    """Whiten the exact complete prediction, preserving acquisition order."""
    coefficients = np.asarray([parameters[n] for n in ("g", "x0", "sensor_drift")])
    return _basis(arrays, parameters["k"]) @ coefficients - arrays[2] / arrays[3]


def _project(arrays, k: float, gain: float | None) -> tuple[dict, np.ndarray, dict]:
    """Bounded affine nuisance projection with a checked convex KKT condition."""
    names = ("g", "x0", "sensor_drift") if gain is None else ("x0", "sensor_drift")
    design, offset = _basis(arrays, k), -arrays[2] / arrays[3]
    if gain is not None:
        offset, design = offset + gain * design[:, 0], design[:, 1:]
    bounds = np.asarray([DOMAINS[name] for name in names])
    values, audit = affine._solve(
        design, offset, bounds[:, 0], bounds[:, 1], affine.Policy()
    )
    parameters = dict(zip(names, map(float, values), strict=True)) | {"k": float(k)}
    if gain is not None:
        parameters["g"] = float(gain)
    return parameters, design @ values + offset, audit


def _tolerance(policy: Policy, *objectives: float) -> float:
    """Compare objectives in the identical fixed Gaussian sum convention."""
    return policy.objective_absolute_tolerance + policy.objective_relative_tolerance * (
        max(1.0, *(abs(value) for value in objectives))
    )


class _Budget(RuntimeError):
    """Numerical work reached a frozen total or operation allowance."""


class _Persistence(RuntimeError):
    """No additional numerical work is safe after a failed checkpoint callback."""


class _Execution:
    """Charge work before execution and time numerical work and callbacks apart."""

    def __init__(
        self,
        state: dict,
        policy: Policy,
        save: Callable | None,
        *,
        invocation: tuple[float, float] | None = None,
    ):
        self.state, self.policy, self.save = state, policy, save
        self.wall, self.cpu = invocation or (monotonic(), process_time())
        self.old_wall = state["timing"]["total_wall_seconds"]
        self.old_cpu = state["timing"]["total_cpu_seconds"]
        self.old_charge = state["budget_charge_seconds"]
        self.deadline = self.wall + max(0.0, policy.wall_seconds - self.old_charge)
        self.operation_deadline = self.deadline
        self.operation_calls = 0

    def update(self) -> None:
        """Include orchestration and final callback work in cumulative totals."""
        timing = self.state["timing"]
        elapsed = monotonic() - self.wall
        timing["total_wall_seconds"] = self.old_wall + elapsed
        timing["total_cpu_seconds"] = self.old_cpu + process_time() - self.cpu
        self.state["budget_charge_seconds"] = self.old_charge + elapsed

    def check(self) -> None:
        """Apply total and operation limits without renewing either allowance."""
        if monotonic() >= min(self.deadline, self.operation_deadline):
            raise _Budget("nonlinear profile wall allowance exhausted")
        if self.state["calls"] >= self.policy.maximum_calls or (
            self.state["pending"] is not None
            and self.operation_calls >= self.state["pending"]["call_allowance"]
        ):
            raise _Budget("nonlinear profile call allowance exhausted")

    def persist(self, event: str) -> None:
        """Publish a sealed full checkpoint; an unfinished callback stays visible."""
        self.update()
        self.state["event"] = event
        if self.save is None:
            return
        timing = self.state["timing"]
        self.state["callback_pending"] = True
        timing["save_callback_calls"] += 1
        snapshot = deepcopy(_seal(self.state))
        wall, cpu = monotonic(), process_time()
        try:
            self.save(snapshot)
        except Exception as error:
            timing["save_callback_failures"] += 1
            self.state["journal_persistence_complete"] = False
            raise _Persistence(
                f"nonlinear profile save callback failed: {error}"
            ) from error
        finally:
            timing["save_callback_wall_seconds"] += monotonic() - wall
            timing["save_callback_cpu_seconds"] += process_time() - cpu
            self.state["callback_pending"] = False
            self.update()

    def call(self, function: Callable):
        """Count every basis/affine evaluation and rollout, including failures."""
        self.check()
        self.state["calls"] += 1
        self.operation_calls += 1
        wall, cpu = monotonic(), process_time()
        try:
            value = function()
        except Exception:
            self.state["failed_calls"] += 1
            raise
        finally:
            self.state["completed_calls"] += 1
            timing = self.state["timing"]
            timing["numerical_wall_seconds"] += monotonic() - wall
            timing["numerical_cpu_seconds"] += process_time() - cpu
        self.check_time()
        return value

    def check_time(self) -> None:
        """Accept a last allowed call, but never a result beyond its wall limit."""
        if monotonic() >= min(self.deadline, self.operation_deadline):
            raise _Budget("nonlinear profile operation exceeded wall allowance")

    def observe(self, parameters: dict, objective: float, source: str) -> None:
        """Retain feasible objective evidence even if a later search step fails."""
        previous = self.state["best_observed"]
        if previous is None or objective < previous["objective"]:
            self.state["best_observed"] = {
                "parameters": parameters,
                "objective": objective,
                "source": source,
            }

    def operation(self, key: str, function: Callable) -> dict:
        """Checkpoint start and completion; interrupted starts are not retried."""
        known = next((op for op in self.state["operations"] if op["key"] == key), None)
        if known is not None:
            return known
        self.check()
        self.operation_calls = 0
        self.operation_deadline = min(
            self.deadline, monotonic() + self.policy.operation_seconds
        )
        self.state["pending"] = {
            "key": key,
            "calls_before": self.state["calls"],
            "call_allowance": min(
                self.policy.operation_calls,
                self.policy.maximum_calls - self.state["calls"],
            ),
            "wall_allowance": max(0.0, self.operation_deadline - monotonic()),
        }
        self.persist("operation_started")
        try:
            self.check_time()
            value = function()
            self.check_time()
            record = {"key": key, "status": "complete", "value": value}
        except (
            ValueError,
            RuntimeError,
            ArithmeticError,
            np.linalg.LinAlgError,
        ) as error:
            record = {"key": key, "status": "unresolved", "reason": str(error)}
        self.state["operations"].append(record)
        self.state["pending"] = None
        self.operation_deadline = self.deadline
        self.persist("operation_completed")
        return record


def _dense_reference(arrays, gain: float | None, policy: Policy, work: _Execution):
    """Search all sampled local k basins and both endpoints, without global proof."""
    cache = {}

    def evaluate(k):
        k = float(k)
        if k not in cache:
            parameters, residual, audit = work.call(lambda: _project(arrays, k, gain))
            cache[k] = {
                "parameters": parameters,
                "objective": float(residual @ residual / 2),
                "audit": audit,
            }
            work.observe(parameters, cache[k]["objective"], "dense_reference")
        return cache[k]["objective"]

    grid = np.linspace(*DOMAINS["k"], policy.reference_points)
    values = [evaluate(k) for k in grid]
    basins = [
        i
        for i in range(1, len(grid) - 1)
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]
    ]
    refinements = []
    for index in basins:
        solved = minimize_scalar(
            evaluate,
            bounds=(grid[index - 1], grid[index + 1]),
            method="bounded",
            options={"xatol": 1e-11, "maxiter": policy.max_nfev},
        )
        refinements.append(
            {
                "sample_index": index,
                "success": bool(solved.success),
                "evaluations": int(solved.nfev),
            }
        )
        if not solved.success or not np.isfinite(solved.fun):
            raise ValueError("dense reference local basin refinement failed")
        evaluate(solved.x)
    best = min(cache.values(), key=lambda item: item["objective"])
    return best | {
        "grid_points": len(grid),
        "sampled_local_basins": len(basins),
        "refinements": refinements,
        "endpoints_checked": True,
        "evaluations": len(cache),
        "global_optimum_proved": False,
        "method": "dense k grid; bounded refinement of every sampled basin",
    }


def _multistart(
    arrays, gain: float, parameters: dict, policy: Policy, work: _Execution
):
    """Optimize the nonlinear rate from frozen starts, profiling x0 and drift."""
    candidates, starts = [], []
    lower, upper = DOMAINS["k"]

    def residual(vector):
        point, values, audit = work.call(
            lambda: _project(arrays, float(vector[0]), gain)
        )
        candidates.append(
            {
                "parameters": point,
                "objective": float(values @ values / 2),
                "audit": audit,
            }
        )
        work.observe(point, candidates[-1]["objective"], "multistart_profile")
        return values

    points = np.unique(
        np.r_[np.linspace(lower, upper, policy.nuisance_starts), parameters["k"]]
    )
    successful = []
    for start in points:
        solved = least_squares(
            residual,
            [start],
            bounds=([lower], [upper]),
            max_nfev=policy.max_nfev,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-9,
        )
        starts.append(
            {
                "k": float(start),
                "success": bool(solved.success),
                "nfev": int(solved.nfev),
                "status": int(solved.status),
            }
        )
        if solved.success and np.isfinite(solved.fun).all():
            values = residual(solved.x)
            successful.append(
                candidates[-1] | {"objective": float(values @ values / 2)}
            )
    if not successful:
        raise ValueError("all bounded nonlinear nuisance starts failed to converge")
    return min(successful, key=lambda item: item["objective"]) | {
        "starts": starts,
        "successful_starts": len(successful),
        "evaluations": len(candidates),
        "global_optimum_proved": False,
        "method": "bounded multistart TRF in k; BVLS in x0 and sensor_drift",
    }


def _node(arrays, gain, parameters, policy, work):
    """Compare a nuisance optimum, dense reference and feasible frozen slice."""
    primary = _multistart(arrays, gain, parameters, policy, work)
    reference = _dense_reference(arrays, gain, policy, work)
    frozen = work.call(lambda: _residual(arrays, parameters | {"g": gain}))
    frozen_objective = float(frozen @ frozen / 2)
    gap = primary["objective"] - reference["objective"]
    tolerance = _tolerance(policy, primary["objective"], reference["objective"])
    qualified = abs(gap) <= tolerance and (
        primary["objective"] <= frozen_objective + tolerance
    )
    values = np.asarray([primary["parameters"][name] for name in NAMES[1:]])
    bounds = np.asarray([DOMAINS[name] for name in NAMES[1:]])
    return {
        "value": gain,
        "status": "complete" if qualified else "unresolved",
        "parameters": primary["parameters"],
        "profiled_objective": primary["objective"],
        "reference_objective": reference["objective"],
        "frozen_objective": frozen_objective,
        "objective_gap": gap,
        "objective_tolerance": tolerance,
        "primary": primary,
        "reference": reference,
        "nuisance_active_bounds": affine._active(
            values, bounds[:, 0], bounds[:, 1], NAMES[1:], affine.Policy()
        ),
        "reason": None if qualified else "nuisance/reference numerical disagreement",
    }


def _curvature(arrays, parameters, policy):
    """Local unconstrained nuisance-aware information, including nonlinear k."""
    time, forcing, _, sigma = arrays
    gain, rate, initial = (parameters[name] for name in ("g", "k", "x0"))
    decay = np.exp(-rate * time)
    derivative = (
        -initial * time * decay
        + gain * forcing * (rate * time * decay + np.expm1(-rate * time)) / rate**2
    )
    basis = _basis(arrays, rate)
    jacobian = np.column_stack((basis[:, 0], derivative / sigma, basis[:, 1:]))
    result = affine._curvature(
        jacobian,
        0,
        affine.Policy(rank_relative_tolerance=policy.rank_relative_tolerance),
    )
    return result | {
        "parameter_order": list(NAMES),
        "scope": "local unconstrained data sensitivity; not global identifiability",
        "nonlinear_nuisance_included": "k",
        "curvature_kind": (
            "local Gauss-Newton sensitivity, not observed profile Hessian"
        ),
    }


def _supports(state: dict, policy: Policy) -> list[dict]:
    """Summarize sampled support only; do not infer crossings between samples."""
    qualified = state["endpoint_qualification"]
    if (
        state["status"] != "complete"
        or qualified is None
        or not qualified["numerically_qualified"]
        or (
            len(state["nodes"]) != len(state["grid"])
            or any(node["status"] != "complete" for node in state["nodes"])
        )
    ):
        return []
    nodes = sorted(state["nodes"], key=lambda node: node["value"])
    base = qualified["retained_objective"]
    supports = []
    for threshold in policy.thresholds:
        summary = {"threshold": threshold}
        for mode in ("profiled", "frozen"):
            accepted = [
                i
                for i, node in enumerate(nodes)
                if node[mode + "_objective"] - base <= threshold
            ]
            brackets = [
                [nodes[i]["value"], nodes[i + 1]["value"]]
                for i in range(len(nodes) - 1)
                if (i in accepted) != (i + 1 in accepted)
            ]
            values = [nodes[i]["value"] for i in accepted]
            components = []
            for index in accepted:
                if not components or index != components[-1][-1] + 1:
                    components.append([])
                components[-1].append(index)
            clipped = bool(accepted and (0 in accepted or len(nodes) - 1 in accepted))
            active = any(nodes[i]["nuisance_active_bounds"] for i in accepted)
            summary[mode] = {
                "status": "sampled_support_only",
                "sampled_values": values,
                "lower": min(values) if values else None,
                "upper": max(values) if values else None,
                "sampled_span": max(values) - min(values) if values else None,
                "width": None,
                "sampled_components": [
                    [nodes[i]["value"] for i in component] for component in components
                ],
                "boundary_brackets": brackets,
                "domain_clipped": clipped,
                "nuisance_bounds_active": active,
                "unresolved_between_nodes": True,
                "classification": "gain_domain_clipped"
                if clipped
                else (
                    "nuisance_domain_assisted"
                    if active and mode == "profiled"
                    else "sampled_objective_support"
                ),
                "resolution": max(b["value"] - a["value"] for a, b in pairwise(nodes)),
            }
        supports.append(summary)
    return supports


def _initial_state(identity: dict, grid: list[float]) -> dict:
    """Start a bounded diagnostic whose entire durable state is serializable."""
    return {
        "protocol": PROTOCOL,
        "identity": identity,
        "status": "running",
        "reason": None,
        "grid": grid,
        "operations": [],
        "pending": None,
        "nodes": [],
        "supports": [],
        "endpoint_qualification": None,
        "curvature": None,
        "independent_rollout": None,
        "best_observed": None,
        "calls": 0,
        "completed_calls": 0,
        "failed_calls": 0,
        "interrupted_calls_charged": 0,
        "interrupted_operations": 0,
        "budget_charge_seconds": 0.0,
        "accounting_complete": True,
        "callback_pending": False,
        "journal_persistence_complete": True,
        "timing": dict.fromkeys(
            (
                "total_wall_seconds",
                "total_cpu_seconds",
                "numerical_wall_seconds",
                "numerical_cpu_seconds",
                "save_callback_wall_seconds",
                "save_callback_cpu_seconds",
            ),
            0.0,
        )
        | {"save_callback_calls": 0, "save_callback_failures": 0},
        "confidence_probability": None,
        "test_data_opened": False,
        "reference_values_used": False,
        "validation_used_for_selection": False,
        "global_optimum_proved": False,
        "interpretation": (
            "sampled bounded Gaussian objective-rise support; no coverage probability"
        ),
        "reference_independence": (
            "Multistart TRF and dense scalar search share exact exponential "
            "basis and affine BVLS/KKT math. Original-equation DOP853 is an "
            "independent endpoint prediction check; agreement is numerical, "
            "not a global nonlinear optimum proof."
        ),
        "timing_scope": (
            "Cumulative measured process CPU and monotonic wall include callbacks. "
            "Interrupted work is charged its reserved calls/wall and has unknown CPU. "
            "Checkpoint snapshots exclude their own callback duration; recovery "
            "charges its frozen reservation; this is not an upper bound on "
            "unknown elapsed callback time."
        ),
    }


def _restore(state: dict, identity: dict) -> dict:
    """Reject detached or altered states before reusing any numerical evidence."""
    result = deepcopy(state)
    digest = result.pop("state_sha256", None)
    if digest != content_sha256(result) or result.get("identity") != identity:
        raise ValueError("nonlinear profile checkpoint identity or content differs")
    if result.get("protocol") != PROTOCOL:
        raise ValueError("nonlinear profile checkpoint protocol differs")
    return result


def run(
    problem: DriftProblem,
    parameters: dict,
    *,
    policy: Policy,
    state: dict | None = None,
    save: Callable[[dict], None] | None = None,
) -> dict:
    """Profile a retained backend endpoint using only public training acquisition.

    Full snapshots are accepted unchanged on resume. An interrupted operation
    consumes its frozen reservation and remains unresolved; it is never retried.
    Endpoint inadequacy preserves the endpoint and reports the better numerical
    reference instead of quietly replacing the fit or presenting narrow support.
    """
    invocation = monotonic(), process_time()
    if not isinstance(problem, DriftProblem) or not isinstance(policy, Policy):
        raise TypeError(
            "typed public drift problem and nonlinear profile policy required"
        )
    problem = DriftProblem.model_validate(problem.model_dump(mode="json"))
    policy = Policy.model_validate(policy.model_dump(mode="json"))
    arrays = _public_arrays(problem)
    if (
        not isinstance(parameters, dict)
        or set(parameters) != set(NAMES)
        or any(
            isinstance(parameters[name], bool)
            or not np.isfinite(parameters[name])
            or not DOMAINS[name][0] <= parameters[name] <= DOMAINS[name][1]
            for name in NAMES
        )
    ):
        raise ValueError("complete finite retained parameters within domains required")
    parameters = {name: float(parameters[name]) for name in NAMES}
    curvature = _curvature(arrays, parameters, policy)
    local = []
    information = curvature["conditional_information"]
    if policy.local_sampling and information > 0:
        local = np.clip(
            parameters["g"]
            + np.asarray([-4, -2, -1, -0.5, 0.5, 1, 2, 4]) / np.sqrt(information),
            *DOMAINS["g"],
        )
    identity = {
        "problem_sha256": content_sha256(problem.model_dump(mode="json")),
        "parameters": parameters,
        "policy": policy.model_dump(mode="json"),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
    }
    grid = list(
        map(
            float,
            np.unique(
                np.r_[
                    np.linspace(*DOMAINS["g"], policy.grid_points),
                    parameters["g"],
                    local,
                ]
            ),
        )
    )
    result = (
        _initial_state(identity, grid) if state is None else _restore(state, identity)
    )
    if (
        result["status"] in {"complete", "unqualified"}
        and not result["callback_pending"]
    ):
        return _seal(result)
    if result["callback_pending"]:
        result["budget_charge_seconds"] += policy.callback_seconds
        result["callback_pending"] = False
        result["accounting_complete"] = False
    if result["pending"] is not None:
        pending = result["pending"]
        charge = pending["call_allowance"]
        result["calls"] = pending["calls_before"] + charge
        result["interrupted_calls_charged"] += charge
        result["interrupted_operations"] += 1
        result["budget_charge_seconds"] += pending["wall_allowance"]
        result["operations"].append(
            {
                "key": pending["key"],
                "status": "unresolved",
                "reason": "interrupted work charged; no silent retry",
            }
        )
        result["pending"] = None
        result["accounting_complete"] = False
    result["status"], result["reason"] = "running", None
    work = _Execution(result, policy, save, invocation=invocation)
    result["sampling"] = (
        "Uniform domain grid plus retained gain; optional local offsets "
        "(+/-0.5,1,2,4)/sqrt(projected information) are sampling heuristics. "
        "No quadratic profile or between-node crossing is assumed."
    )

    def endpoint():
        oracle = robust_measurement.build(
            problem, robust_measurement.Loss(kind="gaussian")
        )[0].base
        actual, jacobian = work.call(
            lambda: oracle.evaluate(
                oracle.vector(parameters), work.operation_deadline, method="DOP853"
            )
        )
        exact = work.call(lambda: _residual(arrays, parameters))
        difference = float(np.max(np.abs(actual - exact)))
        if difference > policy.rollout_tolerance:
            raise ValueError(
                "exact exponential prediction disagrees with original rollout"
            )
        if jacobian is None or not np.isfinite(jacobian).all():
            raise ValueError("independent rollout sensitivity unavailable")
        return {
            "retained_objective": float(exact @ exact / 2),
            "backend_objective": float(actual @ actual / 2),
            "maximum_standardized_difference": difference,
            "passed": True,
            "method": "exact exponential versus original DOP853",
        }

    try:
        work.persist("diagnostic_started")
        checked = work.operation("endpoint", endpoint)
        if checked["status"] != "complete":
            result["status"] = "unqualified"
            result["reason"] = checked["reason"]
        else:
            result["independent_rollout"] = checked["value"]
            reference = work.operation(
                "endpoint_reference",
                lambda: _dense_reference(arrays, None, policy, work),
            )
            result["curvature"] = curvature
            ordered = [parameters["g"], *(v for v in grid if v != parameters["g"])]
            for value in ordered:
                work.operation(
                    "node:" + float(value).hex(),
                    lambda value=value: _node(arrays, value, parameters, policy, work),
                )
            result["status"] = "complete"
            if reference["status"] != "complete":
                result["status"], result["reason"] = "partial", reference["reason"]
    except (_Budget, _Persistence) as error:
        result["status"], result["reason"] = "partial", str(error)
    _summarize(result, parameters, policy)
    try:
        if result["journal_persistence_complete"]:
            work.persist("diagnostic_finished")
    except _Persistence as error:
        result["status"], result["reason"] = "partial", str(error)
        result["supports"] = []
    work.update()
    if (
        result["budget_charge_seconds"] >= policy.wall_seconds
        and result["status"] == "complete"
    ):
        result["status"], result["reason"] = (
            "partial",
            "wall allowance exhausted during save",
        )
        result["supports"] = []
    return _seal(result)


def _summarize(result: dict, parameters: dict, policy: Policy) -> None:
    """Derive conclusions from completed operations, retaining unresolved evidence."""
    result["nodes"] = []
    for operation in result["operations"]:
        if operation["key"].startswith("node:"):
            result["nodes"].append(
                operation["value"]
                if operation["status"] == "complete"
                else {
                    "value": float.fromhex(operation["key"][5:]),
                    "status": "unresolved",
                    "reason": operation["reason"],
                }
            )
    reference = next(
        (op for op in result["operations"] if op["key"] == "endpoint_reference"), None
    )
    if (
        result["independent_rollout"] is not None
        and result["best_observed"] is not None
    ):
        best = result["best_observed"]
        reference_complete = reference is not None and reference["status"] == "complete"
        if reference_complete and reference["value"]["objective"] < best["objective"]:
            best = reference["value"]
        for node in result["nodes"]:
            if "primary" in node and "reference" in node:
                for candidate in (node["primary"], node["reference"]):
                    if candidate["objective"] < best["objective"]:
                        best = candidate
        retained = result["independent_rollout"]["retained_objective"]
        gap = retained - best["objective"]
        tolerance = _tolerance(policy, retained, best["objective"])
        for node in result["nodes"]:
            for mode in ("profiled", "reference", "frozen"):
                if mode + "_objective" in node:
                    node[mode + "_rise"] = node[mode + "_objective"] - retained
        result["endpoint_qualification"] = {
            "retained_parameters": parameters,
            "retained_objective": retained,
            "reference_objective": best["objective"],
            "objective_gap": gap,
            "reference_parameters": best["parameters"],
            "objective_tolerance": tolerance,
            "retained_fit_inadequate": gap > tolerance,
            "numerically_qualified": reference_complete and abs(gap) <= tolerance,
            "global_optimum_proved": False,
            "reference_search": reference["value"] if reference_complete else None,
        }
        if gap > tolerance or (reference_complete and gap < -tolerance):
            result["status"] = "unqualified"
            result["reason"] = (
                "retained fit inadequate: better numerical objective found"
                if gap > 0
                else "numerical reference misses retained fit"
            )
    unresolved = any(node["status"] != "complete" for node in result["nodes"])
    if result["status"] == "complete" and (
        unresolved or len(result["nodes"]) != len(result["grid"])
    ):
        result["status"], result["reason"] = (
            "partial",
            "nonlinear nuisance nodes unresolved",
        )
    result["nodes_completed"] = sum(
        node["status"] == "complete" for node in result["nodes"]
    )
    result["unresolved_nodes"] = sum(
        node["status"] != "complete" for node in result["nodes"]
    )
    result["supports"] = _supports(result, policy)

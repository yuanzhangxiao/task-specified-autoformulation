"""Finite local derivative evidence for unchanged, possibly piecewise laws.

This module schedules no solves and performs no fitting. Its inputs are saved
normalized residuals from independent integrations and the joint physical AD
Jacobian. Agreement at a finite set of points is never a differentiability or
global stationarity certificate.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Literal, Protocol

import numpy as np
from pydantic import Field, model_validator

from autoformalism.schemas.base import StrictSchema


class AssessmentOracle(Protocol):
    """Frozen physical coordinates used by the existing rollout adapter."""

    names: Sequence[str]
    units: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


class PiecewiseAssessmentPolicy(StrictSchema):
    """Predeclared optimizer-unit steps and empirical numerical tolerances."""

    steps: tuple[float, ...] = (1e-3, 3e-4)
    relative_tolerance: float = Field(default=0.02, gt=0, le=0.1)
    absolute_tolerance: float = Field(default=1e-8, gt=0, le=1e-4)
    residual_noise_floor: float = Field(default=1e-12, gt=0, le=1e-6)
    noise_multiplier: float = Field(default=4.0, ge=1)
    minimum_signal_to_noise: float = Field(default=8.0, ge=2)
    branch_margin_tolerance: float = Field(default=1e-8, gt=0, le=1e-3)
    include_weak_direction: bool = False

    @model_validator(mode="after")
    def ordered_steps(self) -> PiecewiseAssessmentPolicy:
        if (
            len(self.steps) < 2
            or len(self.steps) > 8
            or any(not np.isfinite(h) or h <= 0 or h >= 1 for h in self.steps)
            or tuple(sorted(set(self.steps), reverse=True)) != self.steps
        ):
            raise ValueError(
                "two to eight distinct decreasing steps in (0, 1) required"
            )
        return self


class Perturbation(StrictSchema):
    """One unmodified physical point; infeasible points must never be clipped."""

    key: str
    direction: str
    parameter: str | None
    initial_parameter: bool
    direction_vector: tuple[float, ...]
    step: float
    sign: Literal[-1, 1]
    vector: tuple[float, ...]
    in_domain: bool
    reason: str | None = None


class PerturbationEvaluation(StrictSchema):
    """Two independent integrations at the exact planned vector, or a failure."""

    status: Literal["complete", "unavailable"] = "complete"
    residual: tuple[float, ...] | None = None
    alternate_residual: tuple[float, ...] | None = None
    error: str | None = None


AssessmentStatus = Literal[
    "supported",
    "unsupported_kink",
    "numerical_noise",
    "infeasible_direction",
    "derivative_mismatch",
    "unavailable",
]


class StepAssessment(StrictSchema):
    """RMS residual-space derivative comparisons in fixed optimizer units."""

    step: float
    central_error: float
    forward_error: float
    backward_error: float
    one_sided_disagreement: float
    agreement_tolerance: float
    derivative_noise_floor: float
    minimum_signal_to_noise: float
    noise_resolved: bool
    central_agrees: bool
    one_sided_agrees: bool
    sides_agree: bool


class DirectionAssessment(StrictSchema):
    """Finite evidence for one axis or the optionally checked weak direction."""

    direction: str
    parameter: str | None
    initial_parameter: bool
    direction_vector: tuple[float, ...]
    status: AssessmentStatus
    reason: str
    ad_derivative_rms: float
    steps: tuple[StepAssessment, ...] = ()
    converged_across_steps: bool = False


class BranchActivity(StrictSchema):
    """Activity of one primitive or one binary fold at saved output samples."""

    key: str
    location: str
    expression: str
    primitive: Literal["abs", "min", "max"]
    fold: int
    samples: int
    minimum_absolute_margin: float
    minimum_scaled_margin: float
    near_tie_samples: int
    left_or_positive_samples: int
    right_or_negative_samples: int
    sampled_activity_changes: int


class BranchSummary(StrictSchema):
    """Margins at observations and causal initial maps, never between them."""

    status: Literal["complete", "unavailable"] = "complete"
    sample_scope: Literal["stored_output_states_and_causal_initial_maps"] = (
        "stored_output_states_and_causal_initial_maps"
    )
    branches: tuple[BranchActivity, ...] = ()
    trajectory_ids: tuple[str, ...] = ()
    near_tie_samples: int = 0
    sampled_activity_changes: int = 0
    between_sample_behavior_certified: Literal[False] = False
    error: str | None = None


class PiecewiseAssessmentResult(StrictSchema):
    """Local numerical qualification only; not calibrated confidence."""

    status: AssessmentStatus
    reason: str
    active_parameters: tuple[str, ...]
    initial_parameters_checked: tuple[str, ...]
    checked_direction_scope: Literal[
        "coordinate_axes", "coordinate_axes_and_weak_joint"
    ]
    perturbations_required: int
    perturbations_available: int
    directions: tuple[DirectionAssessment, ...]
    branch_summary: BranchSummary | None = None
    supported_classical: Literal[False] = False
    supported_global: Literal[False] = False
    calibrated_confidence_claimed: Literal[False] = False
    solver_discrepancy_is_rigorous_error_bound: Literal[False] = False
    limitation: str = (
        "Finite local numerical evidence at the declared optimizer-unit steps and "
        "checked directions only. Independent-solver discrepancies give an "
        "empirical noise floor, not a rigorous integration error bound. Stored "
        "samples do not certify branch behavior between samples."
    )


def _coordinates(
    oracle: AssessmentOracle, vector: np.ndarray, active: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    names, active = tuple(oracle.names), tuple(active)
    if len(set(names)) != len(names) or len(set(active)) != len(active):
        raise ValueError("parameter and active names must be unique")
    if not set(active).issubset(names):
        raise ValueError("active parameters must belong to the full joint vector")
    x, units = np.asarray(vector, dtype=float), np.asarray(oracle.units, dtype=float)
    lower = np.asarray(oracle.lower, dtype=float)
    upper = np.asarray(oracle.upper, dtype=float)
    if any(a.shape != (len(names),) for a in (x, units, lower, upper)):
        raise ValueError("coordinate arrays must match the full joint parameter order")
    if (
        not np.isfinite(x).all()
        or not np.isfinite(units).all()
        or np.any(units <= 0)
        or np.isnan(lower).any()
        or np.isnan(upper).any()
        or np.any(lower > upper)
        or np.any(x < lower)
        or np.any(x > upper)
    ):
        raise ValueError("invalid units, bounds, or saved parameter vector")
    return x, units, lower, upper, tuple(n for n in names if n in active)


def _initial_names(oracle: AssessmentOracle) -> set[str]:
    return set(getattr(getattr(oracle, "system", None), "initial_parameter_names", ()))


def perturbation_plan(
    oracle: AssessmentOracle,
    vector: np.ndarray,
    active: Sequence[str],
    policy: PiecewiseAssessmentPolicy,
    *,
    jacobian: np.ndarray | None = None,
) -> tuple[Perturbation, ...]:
    """Enumerate every active axis, including initializers, without running solves.

    ``active`` must be the caller's complete structurally active parameter set;
    numerical weakness must not be used to remove columns. The optional weakest
    right-singular direction is formed from the same saved physical Jacobian.
    """
    x, units, lower, upper, active = _coordinates(oracle, vector, active)
    names, initial_names = tuple(oracle.names), _initial_names(oracle)
    directions: list[tuple[str, str | None, np.ndarray]] = []
    indices = [names.index(n) for n in active]
    for name, index in zip(active, indices, strict=True):
        d = np.zeros(len(names))
        d[index] = 1.0
        directions.append((f"coordinate:{name}", name, d))
    if policy.include_weak_direction and active:
        if jacobian is None:
            raise ValueError("saved joint Jacobian required for weak direction")
        j = np.asarray(jacobian, dtype=float)
        if j.ndim != 2 or not j.shape[0] or j.shape[1] != len(names):
            raise ValueError("Jacobian must include every joint parameter column")
        if not np.isfinite(j).all():
            raise ValueError("nonfinite joint Jacobian")
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = j[:, indices] * units[indices]
        if not np.isfinite(scaled).all():
            raise ValueError("nonfinite scaled joint Jacobian")
        # Full V is needed only when there are fewer residuals than columns.
        # A tall residual Jacobian must not allocate an unused quadratic-size U.
        _, _, vh = np.linalg.svd(
            scaled, full_matrices=scaled.shape[0] < scaled.shape[1]
        )
        weak = vh[-1].copy()
        if weak[int(np.argmax(np.abs(weak)))] < 0:
            weak *= -1
        d = np.zeros(len(names))
        d[indices] = weak
        directions.append(("weak_joint", None, d))
    points = []
    for direction_number, (label, parameter, d) in enumerate(directions):
        for step_number, step in enumerate(policy.steps):
            for sign in (-1, 1):
                delta = sign * step * units * d
                with np.errstate(over="ignore", invalid="ignore"):
                    point = x + delta
                reason = None
                if not np.isfinite(point).all():
                    # Preserve a serializable vector; this point cannot be executed.
                    point = x.copy()
                    reason = "nonfinite_perturbation"
                elif np.any(point < lower) or np.any(point > upper):
                    reason = "outside_declared_bounds"
                # Judge rounding in the prescribed optimizer coordinates. A
                # negligible component of a joint direction may round away;
                # losing a material fraction of the whole step is unavailable.
                elif np.linalg.norm((point - x) / units - sign * step * d) > (
                    1e-8 * step
                ):
                    reason = "unrepresentable_step"
                points.append(
                    Perturbation(
                        key=f"direction_{direction_number:03d}_step_{step_number:02d}_"
                        + ("minus" if sign < 0 else "plus"),
                        direction=label,
                        parameter=parameter,
                        initial_parameter=bool(
                            initial_names.intersection(
                                n for n, value in zip(names, d, strict=True) if value
                            )
                        ),
                        direction_vector=tuple(d),
                        step=step,
                        sign=sign,
                        vector=tuple(point),
                        in_domain=reason is None,
                        reason=reason,
                    )
                )
    return tuple(points)


def _rms(values: np.ndarray) -> float:
    # Scaling avoids overflow in a sum of squares of large finite residuals.
    largest = float(np.max(np.abs(values)))
    return (
        float(largest * np.sqrt(np.mean((values / largest) ** 2))) if largest else 0.0
    )


def _residual(
    values: Sequence[float] | np.ndarray, size: int | None = None
) -> np.ndarray:
    value = np.asarray(values, dtype=float)
    if value.ndim != 1 or not len(value) or (size is not None and len(value) != size):
        raise ValueError("saved residuals must have the same nonempty vector shape")
    if not np.isfinite(value).all():
        raise ValueError("nonfinite saved residuals")
    return value


def _step_evidence(
    base: np.ndarray,
    alternate: np.ndarray,
    plus: PerturbationEvaluation,
    minus: PerturbationEvaluation,
    ad: np.ndarray,
    step: float,
    policy: PiecewiseAssessmentPolicy,
) -> tuple[StepAssessment, np.ndarray]:
    rp = _residual(plus.residual, len(base))
    rm = _residual(minus.residual, len(base))
    ap = _residual(plus.alternate_residual, len(base))
    am = _residual(minus.alternate_residual, len(base))
    # Both solver realizations must agree. Norms are over the complete normalized
    # residual, with the original ordering and no row or weak-column removal.
    derivatives = np.array(
        [
            [(rp - rm) / (2 * step), (rp - base) / step, (base - rm) / step],
            [(ap - am) / (2 * step), (ap - alternate) / step, (alternate - am) / step],
        ]
    )
    if not np.isfinite(derivatives).all():
        raise ValueError("nonfinite finite-difference derivatives")
    errors = [max(_rms(row[i] - ad) for row in derivatives) for i in range(3)]
    disagreement = max(_rms(row[1] - row[2]) for row in derivatives)
    magnitude = max(_rms(ad), *(_rms(d) for row in derivatives for d in row))
    tolerance = policy.absolute_tolerance + policy.relative_tolerance * magnitude
    roundoff = (
        8
        * np.finfo(float).eps
        * max(1.0, *(_rms(r) for r in (base, alternate, rp, rm, ap, am)))
    )
    noise = policy.noise_multiplier * max(
        policy.residual_noise_floor,
        roundoff,
        _rms(base - alternate),
        _rms(rp - ap),
        _rms(rm - am),
    )
    derivative_noise = 2 * noise / step
    signal = min(
        _rms(delta) for delta in (rp - base, base - rm, ap - alternate, alternate - am)
    )
    ratio = signal / (2 * noise)
    resolved = ratio >= policy.minimum_signal_to_noise and derivative_noise <= tolerance
    return (
        StepAssessment(
            step=step,
            central_error=errors[0],
            forward_error=errors[1],
            backward_error=errors[2],
            one_sided_disagreement=disagreement,
            agreement_tolerance=tolerance,
            derivative_noise_floor=derivative_noise,
            minimum_signal_to_noise=ratio,
            noise_resolved=resolved,
            central_agrees=errors[0] + derivative_noise <= tolerance,
            one_sided_agrees=max(errors[1:]) + derivative_noise <= tolerance,
            sides_agree=disagreement + derivative_noise <= tolerance,
        ),
        derivatives,
    )


def assess_derivatives(
    oracle: AssessmentOracle,
    vector: np.ndarray,
    active: Sequence[str],
    policy: PiecewiseAssessmentPolicy,
    base_residual: Sequence[float] | np.ndarray,
    jacobian: np.ndarray,
    evaluations: Mapping[str, PerturbationEvaluation],
    *,
    base_alternate_residual: Sequence[float] | np.ndarray | None = None,
    branch_summary: BranchSummary | None = None,
) -> PiecewiseAssessmentResult:
    """Assess complete saved perturbations; missing evidence never earns support."""
    _, units, _, _, active = _coordinates(oracle, vector, active)
    plan = perturbation_plan(oracle, vector, active, policy, jacobian=jacobian)
    base = _residual(base_residual)
    j = np.asarray(jacobian, dtype=float)
    if j.shape != (len(base), len(oracle.names)) or not np.isfinite(j).all():
        raise ValueError("finite full joint AD Jacobian must match saved residuals")
    alternate = (
        _residual(base_alternate_residual, len(base))
        if base_alternate_residual is not None
        else None
    )
    reports = []
    labels = dict.fromkeys(p.direction for p in plan)
    piecewise = getattr(getattr(oracle, "system", None), "has_piecewise", True)
    for label in labels:
        points = [p for p in plan if p.direction == label]
        first = points[0]
        with np.errstate(over="ignore", invalid="ignore"):
            ad = (j * units) @ np.asarray(first.direction_vector)
        if not np.isfinite(ad).all():
            raise ValueError("nonfinite scaled directional derivative")
        details, slopes = [], []
        status: AssessmentStatus = "supported"
        reason = "both signs, independent solvers and all declared steps agree with AD"
        if any(not p.in_domain for p in points):
            status, reason = (
                "infeasible_direction",
                "a required point is infeasible; no clipping or one-sided substitution",
            )
        elif alternate is None:
            status, reason = (
                "unavailable",
                "independent-solver base residual is missing",
            )
        elif any(
            p.key not in evaluations
            or evaluations[p.key].status != "complete"
            or evaluations[p.key].residual is None
            or evaluations[p.key].alternate_residual is None
            for p in points
        ):
            status, reason = (
                "unavailable",
                "a required independent perturbation integration is missing or failed",
            )
        else:
            for step in policy.steps:
                sides = {p.sign: evaluations[p.key] for p in points if p.step == step}
                detail, values = _step_evidence(
                    base, alternate, sides[1], sides[-1], ad, step, policy
                )
                details.append(detail)
                slopes.append(values)
        converged = bool(slopes) and all(
            _rms(right[solver, kind] - left[solver, kind])
            <= policy.absolute_tolerance
            + policy.relative_tolerance
            * max(_rms(left[solver, kind]), _rms(right[solver, kind]), _rms(ad))
            for left, right in pairwise(slopes)
            for solver in range(2)
            for kind in range(3)
        )
        if status == "supported":
            if not all(d.noise_resolved for d in details):
                status, reason = (
                    "numerical_noise",
                    "a required derivative signal is unresolved above the "
                    "empirical solver/roundoff floor",
                )
            elif not all(d.sides_agree for d in details):
                status = "unsupported_kink" if piecewise else "derivative_mismatch"
                reason = (
                    "positive and negative directional slopes disagree "
                    "at the declared scales"
                )
            elif not converged or not all(
                d.central_agrees and d.one_sided_agrees for d in details
            ):
                status, reason = (
                    "derivative_mismatch",
                    "AD, finite differences, or successive step estimates disagree",
                )
        reports.append(
            DirectionAssessment(
                direction=label,
                parameter=first.parameter,
                initial_parameter=first.initial_parameter,
                direction_vector=first.direction_vector,
                status=status,
                reason=reason,
                ad_derivative_rms=_rms(ad),
                steps=tuple(details),
                converged_across_steps=converged,
            )
        )
    statuses = {r.status for r in reports}
    priority: tuple[AssessmentStatus, ...] = (
        "infeasible_direction",
        "unavailable",
        "unsupported_kink",
        "numerical_noise",
        "derivative_mismatch",
        "supported",
    )
    overall = next((s for s in priority if s in statuses), "unavailable")
    reason = (
        "all declared directions have supported finite local numerical evidence"
        if overall == "supported"
        else "no active directions were supplied"
        if not reports
        else "at least one required direction remains unqualified; inspect directions"
    )
    if branch_summary is not None and branch_summary.status == "unavailable":
        overall, reason = "unavailable", "sampled branch evidence is unavailable"
    return PiecewiseAssessmentResult(
        status=overall,
        reason=reason,
        active_parameters=active,
        initial_parameters_checked=tuple(
            n for n in active if n in _initial_names(oracle)
        ),
        checked_direction_scope=(
            "coordinate_axes_and_weak_joint"
            if policy.include_weak_direction
            else "coordinate_axes"
        ),
        perturbations_required=len(plan),
        perturbations_available=sum(
            p.in_domain
            and p.key in evaluations
            and evaluations[p.key].status == "complete"
            and evaluations[p.key].residual is not None
            and evaluations[p.key].alternate_residual is not None
            for p in plan
        ),
        directions=tuple(reports),
        branch_summary=branch_summary,
    )


def sampled_branch_summary(
    oracle: AssessmentOracle,
    vector: np.ndarray,
    states: Mapping[str, np.ndarray],
    policy: PiecewiseAssessmentPolicy,
) -> BranchSummary:
    """Probe expanded restricted ASTs at saved state outputs and initial maps.

    Each multiargument min/max is inspected in the same left-fold order used by
    CasADi. Sampling neither locates every event nor certifies an interval free
    of kinks. A caller must save this summary alongside the underlying states.
    """
    import casadi as ca

    from autoformalism.fitting.casadi_initializer import _walk
    from autoformalism.fitting.sensitivity_contract import certify_expressions
    from autoformalism.fitting.simulation import (
        _known_initial_values,
        trajectory_forcing,
    )

    model, system, train = oracle.model, oracle.system, oracle.train
    if not train.trajectories or any(
        row.trajectory_id not in states for row in train.trajectories
    ):
        raise ValueError("stored branch states are required for every trajectory")
    x, _, _, _, _ = _coordinates(oracle, vector, oracle.names)
    parameters = dict(zip(oracle.names, x, strict=True))
    equations, observations, smoothness = certify_expressions(
        model, allow_piecewise=True
    )
    expressions = [
        (f"{kind}:{name}", expression)
        for kind, values in (
            ("equation", equations),
            ("observation", observations),
            ("initial_condition", smoothness.initial_expressions),
        )
        for name, expression in values.items()
    ]
    rows = []
    for location, expression in expressions:
        function_names = {
            id(node.func)
            for node in ast.walk(expression.tree)
            if isinstance(node, ast.Call)
        }
        symbols = tuple(
            sorted(
                {
                    node.id
                    for node in ast.walk(expression.tree)
                    if isinstance(node, ast.Name) and id(node) not in function_names
                }
            )
        )
        argument = ca.SX.sym("branch_arguments", len(symbols))
        env = dict(zip(symbols, ca.vertsplit(argument), strict=True))
        primitives = [
            node
            for node in ast.walk(expression.tree)
            if isinstance(node, ast.Call) and node.func.id in {"abs", "min", "max"}
        ]
        for ordinal, node in enumerate(primitives):
            primitive = node.func.id
            left = _walk(ca, node.args[0], env)
            arguments = [ast.Constant(0)] if primitive == "abs" else node.args[1:]
            for fold, other in enumerate(arguments):
                right = _walk(ca, other, env)
                function = ca.Function(
                    "branch_margin",
                    [argument],
                    [
                        ca.vertcat(
                            left - right,
                            ca.fmax(1, ca.fmax(ca.fabs(left), ca.fabs(right))),
                        )
                    ],
                )
                margins, scaled, activities, changes = [], [], [], 0
                for trajectory in train.trajectories:
                    initial = location.startswith("initial_condition:")
                    if initial:
                        environments = [
                            parameters | _known_initial_values(model, trajectory)
                        ]
                    else:
                        values = np.asarray(
                            states[trajectory.trajectory_id], dtype=float
                        )
                        if (
                            values.shape != (len(trajectory.time), system.state_count)
                            or not np.isfinite(values).all()
                        ):
                            raise ValueError(
                                "stored branch states must match every "
                                "trajectory output"
                            )
                        forcing = trajectory_forcing(model, trajectory)
                        environments = [
                            parameters
                            | dict(zip(model.state_names, row, strict=True))
                            | {
                                name: forcing.value(name, float(t))
                                for name in system.inputs
                            }
                            | {model.validated.context.time_symbol: float(t)}
                            for t, row in zip(trajectory.time, values, strict=True)
                        ]
                    previous = None
                    for environment in environments:
                        margin, scale = np.asarray(
                            function([environment[n] for n in symbols])
                        ).ravel()
                        if not np.isfinite([margin, scale]).all():
                            raise ValueError("nonfinite sampled branch margin")
                        normalized = abs(float(margin)) / float(scale)
                        activity = (
                            0
                            if normalized <= policy.branch_margin_tolerance
                            else (1 if margin > 0 else -1)
                        )
                        if primitive == "min":
                            activity *= -1
                        changes += int(previous is not None and activity != previous)
                        previous = activity
                        margins.append(abs(float(margin)))
                        scaled.append(normalized)
                        activities.append(activity)
                if not margins:
                    raise ValueError("branch sampling requires training trajectories")
                rows.append(
                    BranchActivity(
                        key=f"{location}:{ordinal}:{fold}",
                        location=location,
                        expression=ast.unparse(node)[:512],
                        primitive=primitive,
                        fold=fold,
                        samples=len(margins),
                        minimum_absolute_margin=min(margins),
                        minimum_scaled_margin=min(scaled),
                        near_tie_samples=activities.count(0),
                        left_or_positive_samples=activities.count(1),
                        right_or_negative_samples=activities.count(-1),
                        sampled_activity_changes=changes,
                    )
                )
                if primitive != "abs":
                    left = (
                        ca.fmin(left, right)
                        if primitive == "min"
                        else ca.fmax(left, right)
                    )
    return BranchSummary(
        branches=tuple(rows),
        trajectory_ids=tuple(t.trajectory_id for t in train.trajectories),
        near_tie_samples=sum(row.near_tie_samples for row in rows),
        sampled_activity_changes=sum(row.sampled_activity_changes for row in rows),
    )

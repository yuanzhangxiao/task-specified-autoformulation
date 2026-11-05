"""Bounded, curvature-aware diagnostics of a conditionally profiled loss.

Two finite-difference scales distinguish curvature from a one-sided secant's
step dependence. Mixed derivatives use the actual objective, not only the
positive-semidefinite Gauss--Newton approximation. Every trial resolves the
same inner problem; finite improvements still require independent ODE replay.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from itertools import product

import numpy as np

from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting import prefix_execution as execution
from autoformalism.fitting import profile_endpoint_probe as endpoint_probe
from autoformalism.fitting.prefix_meal_joint_profile import JointMealProfile
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial

PROTOCOL = "profile-curvature-audit-1"
STEP = 1e-3
CANDIDATE_RADIUS = 0.02


class _CallLimit(RuntimeError):
    """A nonrenewable profile-call allowance has been consumed."""


def _active_set(endpoint: dict) -> tuple | None:
    """Compare inner bounds without mistaking missing diagnostics for stability."""
    if "joint_solve" in endpoint:
        audit = endpoint["joint_solve"]
        if "active_mask" not in audit:
            return None
        return tuple(audit["active_mask"])
    audits = endpoint.get("root_solve")
    if not audits or any("active_mask" not in row for row in audits):
        return None
    return tuple(tuple(row["active_mask"]) for row in audits)


def _stencil(step: float, side: int) -> tuple[dict[float, float], dict[float, float]]:
    """First/second derivative weights in one frozen optimizer coordinate."""
    if side == 0:
        return (
            {-step: -0.5 / step, step: 0.5 / step},
            {-step: 1 / step**2, 0.0: -2 / step**2, step: 1 / step**2},
        )
    return (
        {
            0.0: -1.5 * side / step,
            side * step: 2 * side / step,
            side * 2 * step: -0.5 * side / step,
        },
        {0.0: 1 / step**2, side * step: -2 / step**2, side * 2 * step: 1 / step**2},
    )


def _weighted(samples, weights, key: str = "loss_difference") -> float:
    """Accumulate baseline-relative differences without subtracting large losses."""
    return float(
        np.sum(
            np.asarray([s[key] for s in samples], dtype=np.longdouble)
            * np.asarray(weights, dtype=np.longdouble),
            dtype=np.longdouble,
        )
    )


def _local_chart(oracle, parameters, outer, fixed, chart):
    coordinates, units, _ = endpoint_probe._chart(
        oracle, parameters, outer, fixed, chart
    )
    initial = oracle.vector(parameters) if chart is None else chart.encode(parameters)

    def point(displacement):
        trial = initial.copy()
        for delta, (index, _, logarithmic, lo, hi) in zip(
            displacement, coordinates, strict=True
        ):
            if not np.isfinite(delta) or delta < lo or delta > hi:
                raise NumericalInvalidTrial("curvature probe outside frozen bounds")
            if delta == 0:
                continue
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                trial[index] = (
                    np.exp(np.log(initial[index]) + delta)
                    if logarithmic
                    else initial[index] + delta * units[index]
                )
            if trial[index] == initial[index] or (logarithmic and trial[index] <= 0):
                raise NumericalInvalidTrial("nonrepresentable curvature probe")
        if not np.isfinite(trial).all():
            raise NumericalInvalidTrial("nonfinite curvature coordinates")
        vector = trial if chart is None else chart.decode(trial)
        values = oracle.parameters(vector)
        oracle.vector(values)
        if any(values[name] != value for name, value in fixed.items()):
            raise NumericalInvalidTrial("curvature probe changed a fixed coordinate")
        return values

    return coordinates, units, point


def _differences(sample, baseline, residual, reference):
    """Use (r-r0)(r+r0), retaining a separate conservative reporting floor."""
    loss = float(sample["training_nmse"])
    if residual is None or reference is None:
        return loss - baseline["training_nmse"], "loss_subtraction", None
    if residual.shape != reference.shape:
        raise NumericalInvalidTrial("profiled residual shape changed")
    product = (residual.astype(np.longdouble) - reference) * (
        residual.astype(np.longdouble) + reference
    )
    difference = float(np.mean(product, dtype=np.longdouble))
    arithmetic_floor = float(
        64 * np.finfo(float).eps * np.mean(np.abs(product), dtype=np.longdouble)
    )
    if not np.isfinite([difference, arithmetic_floor]).all():
        raise NumericalInvalidTrial("nonfinite residual loss difference")
    return difference, "paired_residual_difference", arithmetic_floor


def curvature_audit(
    oracle,
    parameters: Mapping[str, float],
    seconds: float,
    checkpoint: Callable[[dict], None],
    *,
    profiler_factory: Callable = JointMealProfile,
    coordinate_chart=None,
    fixed: Mapping[str, float] | None = None,
    maximum_calls: int = 220,
) -> dict:
    """Audit a small profiled outer problem under nonrenewable time/call caps.

    Numerical work has ``seconds``; persistence shares an overall ``seconds+240``
    allowance. Returned candidates are observations, not automatic adoptions.
    Callers own durable operation journals and independent original-model replay.
    No loss, gradient, rank, or absence-of-descent result certifies a global fit.
    """
    budget = execution._budget(seconds, min(seconds + 240, 3600))
    result = {
        "protocol": PROTOCOL,
        "parameters_sha256": content_sha256(parameters),
        "status": "not_started",
        "complete": False,
        "calls_attempted": 0,
        "calls_started": 0,
        "calls_completed": 0,
        "calls_failed": 0,
        "maximum_calls": maximum_calls,
        "baseline": None,
        "best": None,
        "best_candidate": None,
        "material_improvement": False,
        "coordinates": [],
        "mixed_curvature": {
            "scope": "Hessian in the frozen optimizer coordinate chart",
            "negative_curvature_implies_constrained_saddle": False,
            "negative_curvature_implies_feasible_descent": False,
            "stationarity_and_feasibility_needed_for_saddle_interpretation": True,
            "actual_feasible_objective_decrease_required": True,
        },
        "residual_jacobian": {},
        "coupled_probes": [],
        "stencil_coverage": {
            "axial_planned": None,
            "axial_attempted": 0,
            "axial_completed": 0,
            "mixed_planned": None,
            "mixed_attempted": 0,
            "mixed_completed": 0,
            "scope": "two-scale derivative stencils; excludes finite candidate probes",
        },
        "samples": [],
        "stationarity_certified": False,
        "parameter_confidence_certified": False,
        "global_minimum_certified": False,
        "independent_original_verification_required": True,
        "scope": "training objective; fixed and profiled coordinates excluded",
        "rule": {
            "derivative_steps": [STEP, STEP / 2],
            "loss_resolution": "max(1e-12, 1e-8*abs(baseline_loss))",
            "loss_resolution_is_rigorous_error_bound": False,
            "material_drop": "greater than 10 times loss resolution",
            "candidate_maximum_scaled_radius": CANDIDATE_RADIUS,
            "candidate_steps_are_derivative_checks": False,
            "actual_objective_hessian_required": True,
            "gauss_newton_alone_excludes_saddles": False,
            "active_set_changes_defer_derivative_conclusions": True,
            "unresolved_directions_are_not_parameter_confidence": True,
        },
    }
    interrupted = None
    cache, residuals = {}, {}
    reference = None
    baseline_signature = None

    def save():
        budget.callback(lambda: checkpoint(deepcopy(result)))

    def evaluate(displacement, stage, *, keep_residual=False):
        nonlocal reference, baseline_signature
        budget.check()
        key = tuple(float(v) for v in displacement)
        if key in cache:
            return cache[key]
        if result["calls_attempted"] >= maximum_calls:
            raise _CallLimit("curvature profile call limit reached")
        values = point(displacement)
        result["calls_attempted"] += 1

        def call(deadline):
            result["calls_started"] += 1
            try:
                endpoint = profiler.profile(values, deadline)
                loss = float(endpoint["training_nmse"])
                if not np.isfinite(loss) or loss < 0:
                    raise NumericalInvalidTrial("invalid profiled objective")
                residual = getattr(profiler, "last_residual", None)
                if residual is not None:
                    residual = np.asarray(residual, dtype=float).ravel().copy()
                    if not residual.size or not np.isfinite(residual).all():
                        raise NumericalInvalidTrial("invalid paired profile residual")
                    if abs(float(np.mean(residual**2)) - loss) > max(
                        1e-12, 1e-10 * loss
                    ):
                        raise NumericalInvalidTrial(
                            "profile residual does not match loss"
                        )
            except BaseException:
                result["calls_failed"] += 1
                raise
            result["calls_completed"] += 1
            return endpoint, residual

        try:
            endpoint, residual = budget.numerical(call)
        except (NumericalInvalidTrial, ValueError, FloatingPointError) as error:
            sample = {
                "displacement": list(key),
                "stage": stage,
                "status": "numerically_unresolved",
                "error": str(error)[:500],
            }
            cache[key] = sample
            result["samples"].append(sample)
            save()
            return sample
        if result["baseline"] is None:
            result["baseline"] = deepcopy(endpoint)
            reference = residual
            baseline_signature = _active_set(endpoint)
            result["loss_resolution"] = max(
                1e-12, 1e-8 * abs(endpoint["training_nmse"])
            )
        difference, method, arithmetic_floor = _differences(
            endpoint, result["baseline"], residual, reference
        )
        signature = _active_set(endpoint)
        stable = (
            None
            if signature is None or baseline_signature is None
            else signature == baseline_signature
        )
        sample = {
            "displacement": list(key),
            "stage": stage,
            "status": "evaluated",
            "training_nmse": float(endpoint["training_nmse"]),
            "loss_difference": difference,
            "loss_difference_method": method,
            "arithmetic_difference_floor": arithmetic_floor,
            "active_set_stable": stable,
        }
        cache[key] = sample
        result["samples"].append(sample)
        if keep_residual and residual is not None:
            residuals[key] = residual
        # Preserve every strictly improving finite endpoint, including partial
        # audits and actual probes whose derivative model was inconclusive.
        improved = result["best"] is None or difference < result.get(
            "best_loss_difference", 0
        )
        if improved:
            result["best"] = deepcopy(endpoint)
            result["best_loss_difference"] = difference
            result["best_origin"] = stage
            result["best_displacement"] = list(key)
            material = difference < -10 * result["loss_resolution"]
            result["material_improvement"] = material
            result["best_candidate"] = deepcopy(endpoint) if material else None
            save()
        elif result["calls_completed"] % 8 == 0:
            save()
        budget.check()
        return sample

    def linear_combination(weights, stage, *, keep=False):
        samples = [evaluate(q, stage, keep_residual=keep) for q in weights]
        if any(s["status"] != "evaluated" for s in samples):
            return None, None, False
        value = _weighted(samples, list(weights.values()))
        error = sum(abs(w) for w in weights.values()) * result["loss_resolution"]
        stable = all(s["active_set_stable"] is True for s in samples)
        return value, error, stable

    try:
        save()
        if (
            not isinstance(maximum_calls, int)
            or isinstance(maximum_calls, bool)
            or maximum_calls < 1
        ):
            raise ValueError("positive integer profile allowance required")
        profiler = budget.numerical(lambda _: profiler_factory(oracle))
        coordinates, units, point = _local_chart(
            oracle, parameters, profiler.outer, dict(fixed or {}), coordinate_chart
        )
        n = len(coordinates)
        if n > 8:
            raise ValueError(
                "curvature audit is restricted to at most eight outer coordinates"
            )
        result["profile_certificate_sha256"] = profiler.audit["sha256"]
        result["fixed_coordinates"] = dict(fixed or {})
        result["inner_coordinates"] = [oracle.names[i] for i in profiler.inner]
        result["status"] = "running"
        zero = np.zeros(n)
        evaluate(zero, "baseline", keep_residual=True)
        if result["baseline"] is None:
            raise NumericalInvalidTrial("baseline profile unavailable")
        axes = []
        for index, name, log, lo, hi in coordinates:
            if lo < 0 < hi:
                side, step = 0, min(STEP, -lo, hi)
            elif hi > 0:
                side, step = 1, min(STEP, hi / 2)
            else:
                side, step = -1, min(STEP, -lo / 2)
            if not np.isfinite(step) or step <= 0:
                raise NumericalInvalidTrial("no representable derivative stencil")
            axes.append((side, step))
            result["coordinates"].append(
                {
                    "name": name,
                    "coordinate": "log_ratio" if log else "scaled_physical_difference",
                    "unit": float(units[index]),
                    "side": side,
                    "step": step,
                    "scales": [],
                }
            )
        gradients, hessians, uncertainties, stable_scales, jacobians = (
            [],
            [],
            [],
            [],
            [],
        )
        coverage = result["stencil_coverage"]
        coverage.update(axial_planned=2 * n, mixed_planned=n * (n - 1))
        for scale in (1.0, 0.5):
            gradient, hessian = np.full(n, np.nan), np.full((n, n), np.nan)
            errors, gradient_errors = np.full((n, n), np.nan), np.full(n, np.nan)
            stable_matrix = np.zeros((n, n), dtype=bool)
            first_weights, jacobian_columns = [], []
            for i, (side, step) in enumerate(axes):
                coverage["axial_attempted"] += 1
                first, second = _stencil(step * scale, side)

                def embed(weights, index=i):
                    points = {}
                    for delta, weight in weights.items():
                        q = zero.copy()
                        q[index] = delta
                        points[tuple(q)] = weight
                    return points

                first = embed(first)
                first_weights.append(first)
                g, ge, gs = linear_combination(first, "axial", keep=True)
                c, ce, cs = linear_combination(embed(second), "axial", keep=True)
                row = {
                    "step": step * scale,
                    "gradient": g,
                    "curvature": c,
                    "gradient_resolution": ge,
                    "curvature_resolution": ce,
                    "active_set_stable": gs and cs,
                }
                result["coordinates"][i]["scales"].append(row)
                if g is not None and c is not None:
                    coverage["axial_completed"] += 1
                    gradient[i], gradient_errors[i], hessian[i, i], errors[i, i] = (
                        g,
                        ge,
                        c,
                        ce,
                    )
                    stable_matrix[i, i] = gs and cs
                if reference is not None and all(q in residuals for q in first):
                    derivative = np.zeros_like(reference)
                    for q, weight in first.items():
                        derivative += weight * (residuals[q] - reference)
                    jacobian_columns.append(derivative)
                else:
                    jacobian_columns.append(None)
            # Tensor-product first-derivative stencils measure actual mixed
            # curvature, including r * d2r omitted by Gauss--Newton.
            for i in range(n):
                for j in range(i):
                    coverage["mixed_attempted"] += 1
                    weights = {}
                    for (qi, wi), (qj, wj) in product(
                        first_weights[i].items(), first_weights[j].items()
                    ):
                        q = tuple(np.asarray(qi) + np.asarray(qj))
                        weights[q] = weights.get(q, 0.0) + wi * wj
                    value, error, stable = linear_combination(
                        weights, "mixed_curvature"
                    )
                    if value is not None:
                        coverage["mixed_completed"] += 1
                        hessian[i, j] = hessian[j, i] = value
                        errors[i, j] = errors[j, i] = error
                        stable_matrix[i, j] = stable_matrix[j, i] = stable
            gradients.append((gradient, gradient_errors))
            hessians.append(hessian)
            uncertainties.append(errors)
            stable_scales.append(stable_matrix)
            jacobians.append(
                np.column_stack(jacobian_columns)
                if n and all(c is not None for c in jacobian_columns)
                else None
            )
            result["mixed_curvature"]["completed_scales"] = len(hessians)
            save()
        all_finite = all(np.isfinite(h).all() for h in hessians)
        all_stable = all(m.all() for m in stable_scales)
        candidate_directions = []
        if all_finite and n:
            h0, h1 = hessians
            g0, g1 = gradients[0][0], gradients[1][0]
            error = np.maximum(np.abs(h1 - h0), uncertainties[1])
            spectral_resolution = float(np.linalg.norm(error, ord=2))
            gradient_error = np.maximum(np.abs(g1 - g0), gradients[1][1])
            consistent = bool(
                np.all(
                    np.abs(h1 - h0)
                    <= np.maximum(
                        0.2 * np.maximum(np.abs(h0), np.abs(h1)),
                        uncertainties[0] + uncertainties[1],
                    )
                )
            )
            gradient_consistent = bool(
                np.all(
                    np.abs(g1 - g0)
                    <= np.maximum(
                        0.2 * np.maximum(np.abs(g0), np.abs(g1)),
                        gradients[0][1] + gradients[1][1],
                    )
                )
            )
            eigenvalues, eigenvectors = np.linalg.eigh(h1)
            qualified = all_stable and consistent and gradient_consistent
            result["mixed_curvature"].update(
                {
                    "status": "two_scale_local_model"
                    if qualified
                    else "derivative_conclusion_deferred",
                    "coordinate_names": [c[1] for c in coordinates],
                    "gradients": [g0.tolist(), g1.tolist()],
                    "gradient_resolution": gradient_error.tolist(),
                    "hessians": [h0.tolist(), h1.tolist()],
                    "entrywise_resolution": error.tolist(),
                    "eigenvalues": eigenvalues.tolist(),
                    "spectral_resolution": spectral_resolution,
                    "two_scale_consistent": consistent,
                    "gradient_two_scale_consistent": gradient_consistent,
                    "active_set_stable": all_stable,
                    "qualified_local_model": qualified,
                    "resolved_negative_curvature": bool(
                        qualified and eigenvalues[0] < -spectral_resolution
                    ),
                    "positive_curvature_is_global_certificate": False,
                }
            )
            # Check all local eigendirections by actual loss. When the derivative
            # model is deferred these remain exploratory finite probes only.
            for i in range(n):
                candidate_directions.append(
                    (f"actual_hessian_eigenvector_{i}", eigenvectors[:, i])
                )
            floor = max(spectral_resolution, np.max(np.abs(eigenvalues)) * 1e-10, 1e-12)
            correction = -eigenvectors @ (
                (eigenvectors.T @ g1) / np.maximum(eigenvalues, floor)
            )
            if (
                np.isfinite(correction).all()
                and np.max(np.abs(correction), initial=0) > 0
            ):
                correction *= min(1.0, CANDIDATE_RADIUS / np.max(np.abs(correction)))
                predicted = float(g1 @ correction + 0.5 * correction @ h1 @ correction)
                if predicted < -result["loss_resolution"]:
                    candidate_directions.append(
                        ("regularized_local_correction", correction)
                    )
            result["residual_jacobian"] = _jacobian_audit(
                jacobians, h1, all_stable, reference
            )
        elif n == 0:
            result["mixed_curvature"]["status"] = "no_free_outer_coordinates"
            result["residual_jacobian"]["status"] = "no_free_outer_coordinates"
        else:
            result["mixed_curvature"]["status"] = "numerically_unresolved"
            result["residual_jacobian"]["status"] = "numerically_unresolved"
        for row in result["coordinates"]:
            large, small = row["scales"]
            if large["gradient"] is None or small["gradient"] is None:
                row["status"] = "numerically_unresolved"
            elif not large["active_set_stable"] or not small["active_set_stable"]:
                row["status"] = "active_set_or_diagnostic_unresolved"
            else:
                difference = abs(large["gradient"] - small["gradient"])
                tolerance = max(
                    large["gradient_resolution"] + small["gradient_resolution"],
                    0.2 * max(abs(large["gradient"]), abs(small["gradient"])),
                )
                row["status"] = (
                    "two_scale_consistent"
                    if difference <= tolerance
                    else "step_sensitive"
                )
        # Predetermined mixed probes also cover cases with missing or changing
        # active-set derivatives, without pretending to certify local curvature.
        if n > 1:
            candidate_directions += [
                ("balanced_mixed", np.ones(n)),
                ("alternating_mixed", (-1.0) ** np.arange(n)),
            ]
        for label, direction in candidate_directions:
            norm = float(np.max(np.abs(direction), initial=0))
            if not np.isfinite(norm) or norm == 0:
                continue
            direction = np.asarray(direction) / norm
            for sign in (-1, 1):
                # A proposed Newton-like correction has its own predicted
                # magnitude. Do not silently expand a tiny correction to the
                # exploratory eigendirection radius.
                feasible = (
                    min(CANDIDATE_RADIUS, norm)
                    if label == "regularized_local_correction"
                    else CANDIDATE_RADIUS
                )
                for component, (_, _, _, lo, hi) in zip(
                    sign * direction, coordinates, strict=True
                ):
                    if component > 0:
                        feasible = min(feasible, hi / component)
                    elif component < 0:
                        feasible = min(feasible, lo / component)
                if feasible <= 0:
                    result["coupled_probes"].append(
                        {"direction": label, "sign": sign, "status": "bound_blocked"}
                    )
                    continue
                for radius in (feasible, feasible / 2):
                    sample = evaluate(sign * radius * direction, "coupled_candidate")
                    result["coupled_probes"].append(
                        {
                            "direction": label,
                            "sign": sign,
                            "radius": radius,
                            "displacement": sample["displacement"],
                            "status": sample["status"],
                            "loss_difference": sample.get("loss_difference"),
                            "active_set_stable": sample.get("active_set_stable"),
                        }
                    )
        result["complete"] = True
        result["status"] = (
            "material_descent_detected"
            if result["material_improvement"]
            else "numerically_unresolved"
            if result["calls_failed"]
            else "no_resolved_descent_in_audit"
        )
    except Exception as error:
        result.update(
            status="incomplete", error=f"{type(error).__name__}: {error}"[:500]
        )
    except BaseException as error:
        interrupted = error
        result.update(status="interrupted", error=type(error).__name__)
    output = continuation._finish(
        result, budget, checkpoint, "curvature_audit", interrupted
    )
    if output["status"] in {"incomplete", "interrupted"}:
        output["complete"] = False
    return output


def _jacobian_audit(jacobians, hessian, stable, reference):
    """Report scaled profiled residual sensitivity separately from true curvature."""
    if reference is None or any(j is None for j in jacobians):
        return {
            "status": "paired_residual_unavailable",
            "parameter_confidence_certified": False,
        }
    a, b = jacobians
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {
            "status": "numerically_unresolved",
            "parameter_confidence_certified": False,
        }
    delta = float(np.linalg.norm(a - b))
    norm = float(np.linalg.norm(b))
    column_differences = np.linalg.norm(a - b, axis=0)
    column_norms = np.linalg.norm(b, axis=0)
    column_consistent = column_differences <= np.maximum(1e-8, 0.2 * column_norms)
    consistent = bool(column_consistent.all())
    weighted = b / np.sqrt(len(reference))
    singular = np.linalg.svd(weighted, compute_uv=False)
    coarse_singular = np.linalg.svd(a / np.sqrt(len(reference)), compute_uv=False)
    threshold = max(
        float(singular[0]) * max(weighted.shape) * np.finfo(float).eps, 1e-12
    )
    gram = 2 * weighted.T @ weighted
    return {
        "status": "two_scale_sensitivity"
        if stable and consistent
        else "derivative_conclusion_deferred",
        "active_set_stable": stable,
        "two_scale_consistent": consistent,
        "difference_norm": delta,
        "jacobian_norm": norm,
        "column_difference_norms": column_differences.tolist(),
        "column_jacobian_norms": column_norms.tolist(),
        "column_two_scale_consistent": column_consistent.tolist(),
        "scaled_singular_values": singular.tolist(),
        "coarse_scaled_singular_values": coarse_singular.tolist(),
        "rank": int(np.count_nonzero(singular > threshold)),
        "rank_threshold": threshold,
        "profile_gradient": (
            2 * weighted.T @ (reference / np.sqrt(len(reference)))
        ).tolist(),
        "gauss_newton_hessian": gram.tolist(),
        "actual_hessian_minus_gauss_newton": (hessian - gram).tolist(),
        "residual_arrays_persisted": False,
        "parameter_confidence_certified": False,
        "scope": "profiled residual sensitivity in frozen optimizer coordinates",
    }

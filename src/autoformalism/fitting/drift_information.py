"""Bounded local physical/initial/sensor information, without statistical claims.

Diagnostics see only the candidate, training measurement contract and fitted
endpoint. Finite differences qualify the raw standardized sensitivity; Huber's
transformed residual is deliberately not used as a Gaussian information matrix.
"""

from __future__ import annotations

from time import monotonic

import numpy as np

from autoformalism.fitting import measurement_rollout
from autoformalism.fitting.robust_measurement import Loss

PROTOCOL = "sensor-drift-local-information-1"
POLICY = {
    "scaled_steps": [1e-4, 5e-5],
    "maximum_relative_column_error": 1e-4,
    "rank_relative_tolerance": 1e-8,
    "rank_absolute_tolerance": 1e-10,
    "conditional_rank_reference": "unprojected physical spectral norm",
    "weak_singular_ratio": 1e-6,
    "active_bound_scaled_distance": 1e-7,
}


def _spectrum(
    matrix: np.ndarray, names: list[str], *, reference_scale: float = 0.0
) -> dict:
    """Apply an explicit numerical rank convention in frozen coordinate units."""
    if matrix.ndim != 2 or matrix.shape[1] != len(names) or matrix.shape[0] == 0:
        raise ValueError("information matrix/name dimensions differ")
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite sensitivity")
    if not names:
        return {"names": [], "rank": 0, "singular_values": [], "null_directions": []}
    _, singular, right = np.linalg.svd(matrix, full_matrices=True)
    tolerance = max(
        POLICY["rank_absolute_tolerance"],
        POLICY["rank_relative_tolerance"] * max(reference_scale, float(singular[0])),
    )
    rank = int(np.sum(singular > tolerance))
    ratio = float(singular[-1] / singular[0]) if rank == len(names) else 0.0
    return {
        "names": names,
        "rank": rank,
        "rank_tolerance": tolerance,
        "rank_reference_scale": max(reference_scale, float(singular[0])),
        "singular_values": singular.tolist(),
        "smallest_largest_ratio": ratio,
        "weak_under_declared_convention": bool(
            rank < len(names) or ratio < POLICY["weak_singular_ratio"]
        ),
        "null_directions": [
            dict(zip(names, map(float, direction), strict=True))
            for direction in right[rank:]
        ],
        "direction_units": (
            "frozen scaled coordinates; physical step = scale * direction"
        ),
    }


def decompose(jacobian, units, names, nuisance_names) -> dict:
    """Project physical sensitivity off all declared initial/sensor directions.

    This is an unconstrained local linearization. Boundary constraints do not
    add measurement information, and this function never converts ranks to a
    probability or a claim of global identifiability.
    """
    names = list(names)
    nuisance_names = list(nuisance_names)
    units = np.asarray(units, float)
    if (
        len(set(names)) != len(names)
        or len(set(nuisance_names)) != len(nuisance_names)
        or set(nuisance_names) - set(names)
        or units.shape != (len(names),)
        or not np.isfinite(units).all()
        or np.any(units <= 0)
    ):
        raise ValueError("invalid joint sensitivity coordinates")
    raw = np.asarray(jacobian, float)
    if raw.ndim != 2 or raw.shape[1] != len(names) or raw.shape[0] == 0:
        raise ValueError("invalid joint sensitivity dimensions")
    if not np.isfinite(raw).all():
        raise ValueError("nonfinite sensitivity")
    scaled = raw * units
    nuisance = [names.index(name) for name in nuisance_names]
    physical = [i for i, name in enumerate(names) if name not in nuisance_names]
    physical_names = [names[i] for i in physical]
    jp, jn = scaled[:, physical], scaled[:, nuisance]
    if nuisance:
        left, singular, _ = np.linalg.svd(jn, full_matrices=False)
        tolerance = max(
            POLICY["rank_absolute_tolerance"],
            POLICY["rank_relative_tolerance"] * float(singular[0]),
        )
        q = left[:, singular > tolerance]
        residual = jp - q @ (q.T @ jp)
    else:
        residual = jp.copy()
    physical_spectrum = _spectrum(jp, physical_names)
    # Cancellation after nuisance projection must not be reinterpreted as
    # strong information by normalizing roundoff against its own tiny size.
    reference = max(physical_spectrum["singular_values"], default=0.0)
    return {
        "joint": _spectrum(scaled, names),
        "nuisance": _spectrum(jn, nuisance_names),
        "physical_before_nuisance": physical_spectrum,
        "physical_after_nuisance": _spectrum(
            residual, physical_names, reference_scale=reference
        ),
        "physical_conditional_gram": (residual.T @ residual).tolist(),
        "coordinate_scales": dict(zip(names, map(float, units), strict=True)),
        "projection": "J_physical - Q_nuisance @ (Q_nuisance.T @ J_physical)",
    }


def evaluate(problem, parameters, *, loss: Loss, deadline: float, save=None) -> dict:
    """Qualify joint sensitivities at one sealed endpoint within a separate budget.

    The campaign owns durable start/result journals. The progress callback keeps
    actually started/completed calls available if this diagnostic is interrupted.
    No optimization, evaluator data, future observations or profile search occurs.
    """
    from autoformalism.fitting import sensor_drift
    from autoformalism.schemas.sensor_drift import DriftProblem

    save = save or (lambda _: None)
    counts = {"calls": 0, "completed_calls": 0}

    def persist():
        save(dict(counts))

    persist()
    if monotonic() >= deadline:
        raise TimeoutError("sensor information allowance exhausted")
    if isinstance(problem, DriftProblem):
        oracle = sensor_drift.build(problem, loss)[0].base
        base = problem.base
        sensor_names = [problem.drift.name]
    else:
        oracle = measurement_rollout.MeasurementRollouts(problem)
        base, sensor_names = problem, []
    vector = oracle.vector(parameters)
    names, units = list(oracle.names), np.asarray(oracle.units)

    def call(point, *, jacobian=False):
        if monotonic() >= deadline:
            raise TimeoutError("sensor information allowance exhausted")
        counts["calls"] += 1
        persist()
        result = oracle.evaluate(point, deadline, method="DOP853", jacobian=jacobian)
        counts["completed_calls"] += 1
        persist()
        return result

    residual, jacobian = call(vector, jacobian=True)
    scaled = jacobian * units
    columns = []
    for i, name in enumerate(names):
        errors, schemes = [], []
        for step in POLICY["scaled_steps"]:
            distance = step * units[i]
            plus, minus = vector.copy(), vector.copy()
            plus[i] += distance
            minus[i] -= distance
            if minus[i] >= oracle.lower[i] and plus[i] <= oracle.upper[i]:
                derivative = (call(plus)[0] - call(minus)[0]) / (2 * step)
                schemes.append("central")
            else:
                room_up = oracle.upper[i] - vector[i]
                room_down = vector[i] - oracle.lower[i]
                direction = 1 if room_up >= room_down else -1
                room = max(room_up, room_down)
                actual = min(distance, room / 2)
                if actual <= np.finfo(float).eps * max(1, abs(vector[i])):
                    errors.append(None)
                    schemes.append("unavailable_domain_width")
                    continue
                p1, p2 = vector.copy(), vector.copy()
                p1[i] += direction * actual
                p2[i] += direction * 2 * actual
                derivative = (
                    direction
                    * (-3 * residual + 4 * call(p1)[0] - call(p2)[0])
                    / (2 * actual / units[i])
                )
                schemes.append(
                    "forward_second_order" if direction > 0 else "backward_second_order"
                )
            errors.append(
                float(
                    np.linalg.norm(derivative - scaled[:, i])
                    / max(1.0, np.linalg.norm(scaled[:, i]))
                )
            )
        columns.append({"name": name, "schemes": schemes, "relative_errors": errors})
    passed = bool(
        all(
            e is not None
            and np.isfinite(e)
            and e <= POLICY["maximum_relative_column_error"]
            for c in columns
            for e in c["relative_errors"]
        )
    )
    initial_names = list(
        dict.fromkeys(
            rule.shared_coordinate
            for rule in base.initial_rules
            if rule.mode == "shared_value"
        )
    )
    nuisance_names = list(dict.fromkeys(initial_names + sensor_names))
    active = [
        name
        for i, name in enumerate(names)
        if min(vector[i] - oracle.lower[i], oracle.upper[i] - vector[i]) / units[i]
        <= POLICY["active_bound_scaled_distance"]
    ]
    information = decompose(jacobian, units, names, nuisance_names) if passed else None
    if not passed:
        status = "derivatives_unqualified"
    elif active:
        status = "boundary_limited"
    elif information["joint"]["rank"] < len(names):
        status = "local_rank_deficient"
    elif information["joint"]["weak_under_declared_convention"]:
        status = "locally_weak_under_declared_coordinates"
    else:
        status = "locally_full_rank"
    return {
        "protocol": PROTOCOL,
        "policy": POLICY,
        **counts,
        "derivatives_passed": passed,
        "derivative_columns": columns,
        "information_status": status,
        "information": information,
        "initial_nuisance_names": initial_names,
        "sensor_nuisance_names": sensor_names,
        "active_bounds": active,
        "parameters": dict(parameters),
        "loss": loss.model_dump(mode="json"),
        "information_interpretation": (
            "Local expected Gaussian information conditional on the declared model "
            "and known sigma; "
            "not a full nonlinear observed Hessian or calibrated confidence."
            if loss.kind == "gaussian"
            else "Descriptive raw standardized sensitivity; not Huber likelihood "
            "information or robust confidence."
        ),
        "boundary_interpretation": (
            "Unconstrained linear projection only; "
            "active bounds withhold interior conclusions."
        ),
        "confidence_probability": None,
        "global_identifiability_certified": False,
        "data_fault_attribution": None,
    }

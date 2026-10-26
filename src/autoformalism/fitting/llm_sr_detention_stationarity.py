"""Opt-in interpretation of authenticated detention endpoint diagnostics.

This module performs no fitting or differentiation.  The caller must validate
the original diagnostic against its immutable coordinate receipts first.  The
additional assessment distinguishes a measured feasible descent direction from
insufficient evidence of stationarity without changing the historical result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import isfinite
from numbers import Real

from autoformalism.fitting.public_fitting import content_sha256

PROTOCOL = "llm-sr-detention-stationarity-1"


def _number(value: object, label: str, *, nonnegative: bool = False) -> float:
    """Reject nonfinite, Boolean or incorrectly signed numerical evidence."""
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{label} must be a finite real number")
    if nonnegative and value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return float(value)


def _domain(parameters: Mapping, bounds: Mapping) -> tuple[dict, dict]:
    """Validate the exact retained point against the supplied original bounds."""
    if not parameters or set(parameters) != set(bounds):
        raise ValueError("parameter and bound rosters differ")
    point, domain = {}, {}
    for name, value in parameters.items():
        if not isinstance(name, str) or not name:
            raise ValueError("parameter names must be nonempty strings")
        pair = bounds[name]
        if not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError("each parameter needs two bounds")
        low, high = (_number(v, "parameter bound") for v in pair)
        value = _number(value, "parameter")
        if not low <= value <= high:
            raise ValueError("retained parameter lies outside its bounds")
        point[name], domain[name] = value, [low, high]
    return point, domain


def _rank_assessment(diagnosis: Mapping, parameters: Mapping) -> dict:
    """Intersect numerical evidence with an independently certified rank cap."""
    interval = diagnosis.get("parameter_rank_interval")
    if interval is not None and (
        not isinstance(interval, (list, tuple))
        or len(interval) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in interval)
        or not 0 <= interval[0] <= interval[1] <= len(parameters)
    ):
        raise ValueError("invalid numerical rank interval")
    certificate = diagnosis.get("exact_reduction")
    ceiling = None
    if certificate:
        ceiling = certificate.get("rank_ceiling")
        if (
            certificate.get("parameters") != parameters
            or certificate.get("exact_combinations_preserved") is not True
            or certificate.get("original_bounds_satisfied") is not True
            or isinstance(ceiling, bool)
            or not isinstance(ceiling, int)
            or not 0 <= ceiling <= len(parameters)
        ):
            raise ValueError("exact reduction certificate does not match the point")
    conflict = bool(
        interval is not None and ceiling is not None and interval[0] > ceiling
    )
    combined = None
    if interval is not None and not conflict:
        combined = (
            [interval[0], min(interval[1], ceiling)]
            if ceiling is not None
            else list(interval)
        )
    return {
        "numerical_rank_interval": deepcopy(interval),
        "exact_rank_ceiling": ceiling,
        "rank_interval_after_exact_ceiling": combined,
        "numerical_evidence_conflicts_with_exact_ceiling": conflict,
        "full_original_parameter_rank_excluded": ceiling is not None
        and ceiling < len(parameters),
        "scope": (
            "Numerical rank evidence plus an exact algebraic cap; "
            "not a uniqueness certificate."
        ),
    }


def _coordinate_evidence(row: Mapping, value: float, unit: float, bounds, tolerance):
    """Bound the possible feasible descent on one scaled coordinate axis."""
    gradient = _number(row.get("gradient"), "gradient")
    error = row.get("gradient_error_bound")
    if error is None:
        return None
    error = _number(error, "gradient error", nonnegative=True)
    low, high = bounds
    margin = tolerance * max(unit, abs(value))
    decrease = value - low > margin
    increase = high - value > margin
    choices = []
    if increase:
        choices.append((1, -gradient - error, max(0.0, -gradient + error)))
    if decrease:
        choices.append((-1, gradient - error, max(0.0, gradient + error)))
    sign, lower, _ = (
        max(choices, key=lambda item: item[1]) if choices else (0, 0.0, 0.0)
    )
    return {
        "coordinate": row["name"],
        "gradient": gradient,
        "empirical_gradient_error": error,
        "gradient_interval": [gradient - error, gradient + error],
        "unit": unit,
        "parameter_value": value,
        "bounds": list(bounds),
        "decrease_feasible": decrease,
        "increase_feasible": increase,
        "descent_direction_sign": sign,
        "lower_empirical_descent_magnitude": max(0.0, lower),
        "upper_empirical_projected_gradient": max((v[2] for v in choices), default=0.0),
        "local_derivative_status": row.get("status"),
    }


def enrich_diagnosis(
    diagnosis: Mapping,
    *,
    parameters: Mapping[str, float],
    bounds: Mapping[str, Sequence[float]],
    gradient_tolerance: float = 1e-8,
    bound_tolerance: float = 1e-10,
) -> dict:
    """Add three-way stationarity and exact-rank interpretation without mutation.

    ``parameters`` is the independently verified retained original-model vector.
    Coordinate gradients use the original diagnostic's dimensionless units.
    A passed, feasible coordinate with ``abs(gradient)-error > tolerance`` can
    establish empirical descent even when other coordinates remain uncertain.
    Near-zero qualification still requires complete strict derivative evidence.
    This is numerical evidence, not a formal interval or statistical guarantee.
    """
    if not isinstance(diagnosis, Mapping):
        raise ValueError("diagnosis must be a mapping")
    if "stationarity_assessment" in diagnosis or "rank_assessment" in diagnosis:
        raise ValueError("enrich the authenticated raw diagnostic only once")
    point, domain = _domain(parameters, bounds)
    gradient_tolerance = _number(
        gradient_tolerance, "gradient tolerance", nonnegative=True
    )
    bound_tolerance = _number(bound_tolerance, "bound tolerance", nonnegative=True)
    point_hash = content_sha256(point)
    result = deepcopy(dict(diagnosis))
    assessment = {
        "protocol": PROTOCOL,
        "status": "numerically_unresolved",
        "source_diagnosis_sha256": content_sha256(diagnosis),
        "parameters_sha256": point_hash,
        "bounds_sha256": content_sha256(domain),
        "gradient_tolerance": gradient_tolerance,
        "bound_tolerance": bound_tolerance,
        "supporting_coordinate": None,
        "coordinate_evidence": [],
        "qualified_full_model_stationarity": False,
        "scope": (
            "Empirical local feasible first-order evidence; no minimum, global "
            "optimum, coefficient accuracy, or confidence probability is certified."
        ),
    }
    result["stationarity_assessment"] = assessment
    result["rank_assessment"] = _rank_assessment(diagnosis, point)
    if diagnosis.get("parameters") != point:
        raise ValueError("diagnosis is bound to another retained point")
    if diagnosis.get("status") != "complete":
        assessment["reason"] = "endpoint diagnosis is unavailable or incomplete"
        return result
    if (
        "diagnosed_parameters_sha256" in diagnosis
        and diagnosis["diagnosed_parameters_sha256"] != point_hash
    ):
        raise ValueError("diagnostic parameter digest differs")
    gate = diagnosis.get("qualification", {})
    digest = gate.get("qualification_sha256")
    if digest != content_sha256(
        {k: v for k, v in gate.items() if k != "qualification_sha256"}
    ):
        raise ValueError("qualification digest differs")
    if (
        gate.get("parameters") != point
        or gate.get("identity", {}).get("parameters_sha256") != point_hash
        or gate.get("purpose") != "diagnosis"
        or gate.get("precision") != "tight"
    ):
        raise ValueError("qualification is not an exact retained-point diagnosis")
    names, units, rows = (
        gate.get("coordinate_names", []),
        gate.get("coordinate_units", []),
        gate.get("rows", []),
    )
    if (
        len(names) != len(point)
        or len(set(names)) != len(names)
        or set(names) != set(point)
        or len(units) != len(names)
    ):
        raise ValueError("diagnostic coordinate roster differs")
    units = [_number(unit, "coordinate unit") for unit in units]
    if any(unit <= 0 for unit in units):
        raise ValueError("coordinate units must be positive")
    indexed = {}
    for row in rows:
        index = row.get("index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(names)
            or index in indexed
            or row.get("name") != names[index]
        ):
            raise ValueError("diagnostic row roster differs")
        indexed[index] = row
    evidence = []
    for index, row in sorted(indexed.items()):
        item = _coordinate_evidence(
            row,
            point[names[index]],
            units[index],
            domain[names[index]],
            bound_tolerance,
        )
        if item is not None:
            evidence.append(item)
    assessment["qualification_sha256"] = digest
    assessment["coordinate_evidence"] = evidence
    candidates = [
        item
        for item in evidence
        if item["local_derivative_status"] == "passed"
        and item["lower_empirical_descent_magnitude"] > gradient_tolerance
    ]
    if candidates:
        assessment.update(
            status="reliable_descent_remains",
            supporting_coordinate=max(
                candidates, key=lambda item: item["lower_empirical_descent_magnitude"]
            ),
            reason=(
                "at least one checked feasible coordinate has a nonzero gradient "
                "beyond its empirical uncertainty"
            ),
        )
    elif (
        gate.get("complete") is True
        and gate.get("strict_gradient_qualified") is True
        and not gate.get("failed_indices")
        and len(evidence) == len(names)
        and all(item["local_derivative_status"] != "failed" for item in evidence)
        and max(item["upper_empirical_projected_gradient"] for item in evidence)
        <= gradient_tolerance
    ):
        assessment.update(
            status="qualified",
            qualified_full_model_stationarity=True,
            reason=(
                "all feasible projected coordinate gradients satisfy "
                "the strict numerical tolerance"
            ),
        )
    else:
        assessment["reason"] = (
            "available derivative evidence establishes neither full stationarity "
            "nor a reliable feasible descent coordinate"
        )
    return result

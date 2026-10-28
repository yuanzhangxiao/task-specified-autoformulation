"""Bounded endpoint polishing and conservative, exact-point interpretation.

This opt-in layer never opens held-out data or changes the model. Cached initial
samples support search only; independently verified retained endpoints require
their own coordinate evidence. Empirical derivative envelopes are not rigorous
interval arithmetic, statistical confidence, or parameter-identification proof.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import isfinite
from numbers import Real

import numpy as np

from autoformalism.fitting import llm_sr_detention_continuation as chart
from autoformalism.fitting import llm_sr_detention_lazy_controls as controls
from autoformalism.fitting import llm_sr_detention_refit as original
from autoformalism.fitting import llm_sr_detention_stationarity as stationarity
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_numerics as numerics
from autoformalism.fitting.llm_sr_detention_reduction import reduction_certificate

PROTOCOL = "llm-sr-detention-polishing-numerics-1"
Mode = controls.Mode
Policy = chart.Policy


def fit(
    item,
    context,
    training,
    parameters: Mapping[str, float],
    *,
    qualification: dict,
    mode: Mode = "original",
    seconds: float = 600,
    policy: Policy | None = None,
    checkpoint: Callable[[dict], None] | None = None,
    active_names: Sequence[str] | None = None,
) -> dict:
    """Run one at-most-600-second lazy segment from authenticated saved evidence.

    The campaign owns the nonrenewable reservation and subsequent independent
    verification. By default all search-qualified coordinates are selected;
    explicit subsets retain exact held values and NaN derivative sentinels.
    """
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, Real)
        or not isfinite(seconds)
        or not 0 < seconds <= 600
    ):
        raise ValueError("polishing allowance must be positive and at most 600 seconds")
    eligible = controls.select_active(
        item,
        context,
        training,
        parameters,
        mode=mode,
        qualification=qualification,
        policy=policy,
    )
    binding = {
        "polishing_protocol": PROTOCOL,
        "maximum_segments": 1,
        "allowance_seconds": float(seconds),
        "historical_allowances_renewed": False,
    }

    def save(value: dict) -> None:
        if checkpoint:
            checkpoint(value | binding)

    return (
        controls.fit_control(
            item,
            context,
            training,
            parameters,
            mode=mode,
            lazy_jacobian=True,
            active_names=eligible if active_names is None else active_names,
            qualification=qualification,
            seconds=seconds,
            policy=policy,
            checkpoint=save,
            reuse_initial_qualification=True,
        )
        | binding
    )


def _coordinate(row, record, coordinates, policy) -> dict | None:
    """Project a scaled-gradient interval onto locally feasible coordinate signs."""
    index, name = row["index"], row["name"]
    value = coordinates.anchor_vector[index]
    evidence = stationarity._coordinate_evidence(
        row,
        value,
        coordinates.units[index],
        (coordinates.lower[index], coordinates.upper[index]),
        policy.bound_tolerance,
    )
    if evidence is None:
        return None
    low, high = evidence["gradient_interval"]
    if not evidence["decrease_feasible"]:
        low, high = min(0.0, low), min(0.0, high)
    if not evidence["increase_feasible"]:
        low, high = max(0.0, low), max(0.0, high)
    check = record["derivative"]
    projected_error = row.get("projected_gradient_error_bound")
    if projected_error is not None:
        projected_error = stationarity._number(
            projected_error, "projected gradient error", nonnegative=True
        )
    strict = (
        record.get("status") == "complete"
        and check.get("status") == "passed"
        and not check.get("error")
        and not check.get("failed_indices")
        and not check.get("omitted_indices")
        and not check.get("inconclusive_indices")
        and check.get("gradient_precision_qualified") is True
        and row["status"] == "passed"
        and projected_error is not None
        and projected_error
        <= policy.gradient_tolerance * policy.precision_gradient_fraction
        and "Search-purpose stopping" not in check.get("scope", "")
        and record.get("probe_recovery", {}).get("search_error_limits") is None
    )
    return evidence | {
        "projected_gradient_interval": [low, high],
        "projected_gradient_error_bound": projected_error,
        "strict_gradient_qualified": strict,
        "evidence_sha256": record["evidence_sha256"],
        "coordinate": name,
    }


def assess_endpoint(
    item,
    context,
    training,
    parameters: Mapping[str, float],
    *,
    qualification: dict,
    verification: Mapping,
    mode: Mode = "original",
    policy: Policy | None = None,
) -> dict:
    """Interpret authenticated matching evidence without rollout or differentiation.

    A passed coordinate can demonstrate remaining feasible descent despite
    missing evidence elsewhere. Full-chart stationarity requires strict complete
    evidence for every coordinate; zero/inconclusive columns cannot supply it.
    The caller authenticates the independent verification's campaign receipt.
    """
    if mode not in ("original", "reduced_model15_consistent"):
        raise ValueError("polishing assessment requires original or consistent chart")
    policy = chart.effective_policy(policy, mode=mode)
    chart.validate_qualification(
        item,
        context,
        training,
        parameters,
        qualification.get("directions", []),
        mode=mode,
        qualification=qualification,
        precision="tight",
        policy=policy,
    )
    domain = original.domain_policy(item, context, training)
    coordinates = chart.CoordinateChart(domain, parameters, mode)
    result = {
        "protocol": PROTOCOL,
        "status": "unresolved",
        "parameters": dict(parameters),
        "parameters_sha256": public.content_sha256(parameters),
        "qualification_sha256": qualification["qualification_sha256"],
        "verification_sha256": public.content_sha256(verification),
        "mode": mode,
        "chart": coordinates.domain,
        "gradient_tolerance": policy.numerical.derivative.gradient_tolerance,
        "bound_tolerance": policy.numerical.derivative.bound_tolerance,
        "coordinate_evidence": [],
        "supporting_coordinate": None,
        "projected_gradient_upper_bound": None,
        "qualified_full_chart_stationarity": False,
        "full_model_convergence_claimed": False,
        "parameter_information_qualified": False,
        "parameter_confidence_claimed": False,
        "coefficient_accuracy_claimed": False,
        "global_optimum_certified": False,
        "validation_or_test_access": False,
        "scope": (
            "Empirical feasible local first-order evidence in the stated chart; "
            "not rigorous intervals, a minimum, global optimality, coefficient "
            "accuracy, parameter confidence, or scientific validity."
        ),
    }
    if item.task == 15:
        result["exact_reduction"] = reduction_certificate(parameters, domain["bounds"])
        result["original_parameter_uniqueness"] = "excluded_by_exact_ambiguity"
    if verification.get("parameters") != parameters:
        raise ValueError("verification belongs to another retained endpoint")
    if verification.get("status") != "complete" or not verification.get(
        "usable_for_retention"
    ):
        return result | {"reason": "matching independent verification unavailable"}
    for key in (
        "training_nmse",
        "alternate_training_nmse",
        "solver_loss_discrepancy",
        "solver_prediction_discrepancy",
    ):
        stationarity._number(verification.get(key), key, nonnegative=True)
    if not numerics.usable(verification):
        return result | {"reason": "independent original-equation solvers disagree"}
    records = {record["index"]: record for record in qualification["directions"]}
    evidence = []
    for row in qualification["rows"]:
        record = records[row["index"]]
        residual = np.asarray(record["base_residual"], dtype=float)
        if not np.isclose(
            np.mean(residual**2), verification["training_nmse"], rtol=1e-10, atol=1e-14
        ):
            raise ValueError(
                "qualification residual differs from retained verification"
            )
        coordinate = _coordinate(row, record, coordinates, policy.numerical.derivative)
        if coordinate is not None:
            evidence.append(coordinate)
    result["coordinate_evidence"] = evidence
    tolerance = policy.numerical.derivative.gradient_tolerance
    candidates = [
        row
        for row in evidence
        if row["local_derivative_status"] == "passed"
        and row["lower_empirical_descent_magnitude"] > tolerance
    ]
    covered = qualification["complete"] and len(evidence) == len(coordinates.names)
    if covered:
        result["projected_gradient_upper_bound"] = max(
            row["upper_empirical_projected_gradient"] for row in evidence
        )
    if candidates:
        result.update(
            status="reliable_descent_remains",
            supporting_coordinate=max(
                candidates, key=lambda row: row["lower_empirical_descent_magnitude"]
            ),
            reason=(
                "a checked feasible coordinate has descent beyond empirical uncertainty"
            ),
        )
    elif (
        covered
        and all(row["strict_gradient_qualified"] for row in evidence)
        and result["projected_gradient_upper_bound"] <= tolerance
    ):
        result.update(
            status="qualified_local_stationarity",
            qualified_full_chart_stationarity=True,
            reason="all feasible projected coordinate gradients meet strict precision",
        )
    else:
        result["reason"] = (
            "neither reliable feasible descent nor complete strict stationarity "
            "established"
        )
    return result

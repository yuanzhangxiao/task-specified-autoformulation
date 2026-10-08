"""Actionable training-only fitting evidence, without a correctness probability."""

from typing import Literal

from pydantic import Field

from autoformalism.fitting import recovery_numerics as numerical
from autoformalism.fitting.affine_propagation import numerically_verified
from autoformalism.schemas.base import StrictSchema


class FittingAssessment(StrictSchema):
    """Separate independent numerical checks, search status, and parameter evidence."""

    protocol: Literal["fitting-assessment-1"] = "fitting-assessment-1"
    fit_quality: Literal[
        "unverified", "above_numerical_target", "numerical_target_reached"
    ]
    numerical_status: Literal[
        "unavailable", "fit_usable_uncertainty_unresolved", "strictly_verified"
    ]
    search_status: Literal[
        "target_reached", "further_search_needed", "verification_needed"
    ]
    parameter_information: Literal[
        "unavailable",
        "weak_sensitivity",
        "alternative_vectors_found",
        "local_sensitivity_resolved_profiles_not_tested",
    ]
    recommended_action: Literal[
        "verify_numerics",
        "diversify_or_extend_search",
        "assess_parameter_uncertainty",
        "retain_with_local_evidence",
    ]
    training_nmse: float | None = None
    target_nmse: float = Field(gt=0)
    sensitivity: dict | None = None
    alternative_witnesses: list[dict] = Field(default_factory=list)
    search_stop_reasons: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    probability: None = None
    global_identifiability_certified: Literal[False] = False
    coefficient_accuracy_available_to_fitter: Literal[False] = False
    profiles_minimized: Literal[False] = False


def assess(
    selected: dict | None,
    sensitivity: dict | None,
    alternatives: list[dict],
    stops: list[str],
    target: float,
) -> dict:
    """Do not conflate a weak direction, a stalled search, and numerical failure."""
    available = numerical.usable(selected)
    strict = available and numerically_verified(selected, 1e-12)
    value = (
        max(selected["training_nmse"], selected["alternate_training_nmse"])
        if available
        else None
    )
    accurate = available and value <= target
    witnesses = []
    if strict and sensitivity:
        for point in alternatives:
            if not numerically_verified(point, 1e-12):
                continue
            delta = (
                max(point["training_nmse"], point["alternate_training_nmse"]) - value
            )
            if delta > 1e-12:
                continue
            for name, unit in zip(
                sensitivity["parameters"], sensitivity["units"], strict=True
            ):
                distance = (
                    abs(point["parameters"][name] - selected["parameters"][name]) / unit
                )
                if distance >= 0.01:
                    witnesses.append(
                        {
                            "parameter": name,
                            "value": point["parameters"][name],
                            "scaled_distance": distance,
                            "loss_increase": delta,
                            "origin": point.get("origin"),
                        }
                    )
    weak = bool(
        sensitivity
        and (
            sensitivity["rank"] < sensitivity["parameter_count"]
            or sensitivity["condition_number"] is None
            or sensitivity["condition_number"] > 1e6
        )
    )
    information = (
        "unavailable"
        if not strict or not sensitivity
        else "alternative_vectors_found"
        if witnesses
        else "weak_sensitivity"
        if weak
        else "local_sensitivity_resolved_profiles_not_tested"
    )
    action = (
        "verify_numerics"
        if not available or not strict
        else "diversify_or_extend_search"
        if not accurate
        else "assess_parameter_uncertainty"
        if information != "local_sensitivity_resolved_profiles_not_tested"
        else "retain_with_local_evidence"
    )
    reasons = [
        "No reference values, validation or test observations enter this assessment.",
        "Finite local searches and sensitivity do not certify coefficient accuracy "
        "or global uniqueness.",
    ]
    if available and not strict:
        reasons.append(
            "Retain the usable fit; uncertainty at 1e-12 is numerically unresolved."
        )
    if weak:
        reasons.append(
            "Scaled joint sensitivity includes hidden initials; weak directions "
            "are not proof of structural non-identifiability."
        )
    if not accurate and available:
        reasons.append(
            "The numerical fitting target remains unmet; ambiguity witnesses "
            "concern this retained solution, not a certified global minimum."
        )
    if witnesses:
        reasons.append(
            "Verified search alternatives are witnesses; no full profile "
            "minimization was performed."
        )
    return FittingAssessment(
        fit_quality="unverified"
        if not available
        else "numerical_target_reached"
        if accurate
        else "above_numerical_target",
        numerical_status="unavailable"
        if not available
        else "strictly_verified"
        if strict
        else "fit_usable_uncertainty_unresolved",
        search_status="verification_needed"
        if not available
        else "target_reached"
        if accurate
        else "further_search_needed",
        parameter_information=information,
        recommended_action=action,
        training_nmse=value,
        target_nmse=target,
        sensitivity=sensitivity,
        alternative_witnesses=witnesses,
        search_stop_reasons=stops,
        reasons=reasons,
    ).model_dump(mode="json")

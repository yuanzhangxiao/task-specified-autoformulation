"""Endpoint interpretation separates reliable descent from unresolved precision."""

from copy import deepcopy

import pytest

from autoformalism.fitting.llm_sr_detention_stationarity import enrich_diagnosis
from autoformalism.fitting.public_fitting import content_sha256


def diagnostic(*, gradients=(1e-3, 0.0), errors=(1e-7, 1e-10)):
    """Minimal authenticated-shaped fixture; no numerical integrations needed."""
    parameters = {"a": 0.5, "b": 0.5}
    names = list(parameters)
    gate = {
        "identity": {"parameters_sha256": content_sha256(parameters)},
        "parameters": parameters,
        "purpose": "diagnosis",
        "precision": "tight",
        "coordinate_names": names,
        "coordinate_units": [1.0, 1.0],
        "complete": True,
        "strict_gradient_qualified": False,
        "failed_indices": [],
        "rows": [
            {
                "index": i,
                "name": name,
                "status": "passed",
                "gradient": gradients[i],
                "gradient_error_bound": errors[i],
                "projected_gradient_error_bound": errors[i],
            }
            for i, name in enumerate(names)
        ],
    }
    reseal(gate)
    result = {
        "parameters": parameters,
        "status": "complete",
        "diagnosed_parameters_sha256": content_sha256(parameters),
        "qualification": gate,
        "parameter_rank_interval": [1, 2],
        "local_first_order_stationarity": "unresolved",
        "parameter_information_qualified": False,
        "confidence_probability": None,
    }
    return result, parameters, {"a": [0.0, 1.0], "b": [0.0, 1.0]}


def reseal(gate):
    gate["qualification_sha256"] = content_sha256(
        {k: v for k, v in gate.items() if k != "qualification_sha256"}
    )


def test_passed_coordinate_can_show_descent_while_another_is_uncertain():
    raw, parameters, bounds = diagnostic()
    raw["qualification"]["rows"][1]["status"] = "failed"
    raw["qualification"]["failed_indices"] = [1]
    reseal(raw["qualification"])
    unchanged = deepcopy(raw)
    result = enrich_diagnosis(raw, parameters=parameters, bounds=bounds)
    assert raw == unchanged
    assert {
        k: v
        for k, v in result.items()
        if k not in ("stationarity_assessment", "rank_assessment")
    } == raw
    assessment = result["stationarity_assessment"]
    assert assessment["status"] == "reliable_descent_remains"
    support = assessment["supporting_coordinate"]
    assert support["coordinate"] == "a"
    assert support["descent_direction_sign"] == -1
    assert support["lower_empirical_descent_magnitude"] == pytest.approx(0.0009999)
    assert not assessment["qualified_full_model_stationarity"]
    assert result["local_first_order_stationarity"] == "unresolved"
    assert result["confidence_probability"] is None


def test_qualified_requires_complete_strict_near_zero_evidence():
    raw, parameters, bounds = diagnostic(
        gradients=(1e-11, -1e-11), errors=(1e-12, 1e-12)
    )
    raw["qualification"]["strict_gradient_qualified"] = True
    reseal(raw["qualification"])
    result = enrich_diagnosis(raw, parameters=parameters, bounds=bounds)
    assert result["stationarity_assessment"]["status"] == "qualified"
    raw["qualification"]["complete"] = False
    reseal(raw["qualification"])
    result = enrich_diagnosis(raw, parameters=parameters, bounds=bounds)
    assert result["stationarity_assessment"]["status"] == "numerically_unresolved"


@pytest.mark.parametrize("status", ["failed", "inconclusive"])
def test_unpassed_coordinate_cannot_establish_descent(status):
    raw, parameters, bounds = diagnostic()
    raw["qualification"]["rows"][0]["status"] = status
    reseal(raw["qualification"])
    result = enrich_diagnosis(raw, parameters=parameters, bounds=bounds)
    assert result["stationarity_assessment"]["status"] == "numerically_unresolved"


@pytest.mark.parametrize(
    ("gradient", "pair", "expected", "sign"),
    [
        (1.0, [0.5, 1.0], "numerically_unresolved", None),
        (-1.0, [0.5, 1.0], "reliable_descent_remains", 1),
        (-1.0, [0.0, 0.5], "numerically_unresolved", None),
        (1.0, [0.0, 0.5], "reliable_descent_remains", -1),
        (1.0, [0.5, 0.5], "numerically_unresolved", None),
    ],
)
def test_descent_must_be_feasible_at_lower_upper_and_fixed_bounds(
    gradient, pair, expected, sign
):
    raw, parameters, bounds = diagnostic(gradients=(gradient, 0.0))
    bounds["a"] = pair
    result = enrich_diagnosis(raw, parameters=parameters, bounds=bounds)
    assessment = result["stationarity_assessment"]
    assert assessment["status"] == expected
    support = assessment["supporting_coordinate"]
    assert (
        support is None if sign is None else support["descent_direction_sign"] == sign
    )


def test_uncertainty_containing_zero_does_not_prove_descent():
    raw, parameters, bounds = diagnostic(gradients=(1e-3, 0.0), errors=(2e-3, 1e-10))
    result = enrich_diagnosis(raw, parameters=parameters, bounds=bounds)
    assert result["stationarity_assessment"]["status"] == "numerically_unresolved"


def test_exact_rank_ceiling_restricts_display_not_saved_numerics_or_confidence():
    raw, parameters, bounds = diagnostic()
    raw["exact_reduction"] = {
        "parameters": dict(parameters),
        "rank_ceiling": 1,
        "exact_combinations_preserved": True,
        "original_bounds_satisfied": True,
    }
    result = enrich_diagnosis(raw, parameters=parameters, bounds=bounds)
    rank = result["rank_assessment"]
    assert result["parameter_rank_interval"] == [1, 2]
    assert rank["rank_interval_after_exact_ceiling"] == [1, 1]
    assert rank["full_original_parameter_rank_excluded"]
    assert not result["parameter_information_qualified"]
    raw["parameter_rank_interval"] = [2, 2]
    rank = enrich_diagnosis(raw, parameters=parameters, bounds=bounds)[
        "rank_assessment"
    ]
    assert rank["rank_interval_after_exact_ceiling"] is None
    assert rank["numerical_evidence_conflicts_with_exact_ceiling"]


def test_unavailable_diagnosis_stays_unresolved():
    raw, parameters, bounds = diagnostic()
    raw["status"] = "unavailable"
    raw.pop("qualification")
    result = enrich_diagnosis(raw, parameters=parameters, bounds=bounds)
    assert result["stationarity_assessment"]["status"] == "numerically_unresolved"


def test_raw_diagnose_output_is_bound_through_qualification_before_campaign_seal():
    raw, parameters, bounds = diagnostic()
    raw.pop("diagnosed_parameters_sha256")
    result = enrich_diagnosis(raw, parameters=parameters, bounds=bounds)
    assert "diagnosed_parameters_sha256" not in result
    assert result["stationarity_assessment"]["parameters_sha256"] == content_sha256(
        parameters
    )
    assert result["stationarity_assessment"]["status"] == "reliable_descent_remains"


@pytest.mark.parametrize(
    "mutation",
    ["point", "digest", "gate", "index", "purpose", "unit", "nan", "negative_error"],
)
def test_reject_mismatched_or_invalid_evidence(mutation):
    raw, parameters, bounds = diagnostic()
    parameters = dict(parameters)
    gate = raw["qualification"]
    if mutation == "point":
        parameters["a"] += 0.1
    elif mutation == "digest":
        raw["diagnosed_parameters_sha256"] = "0" * 64
    elif mutation == "gate":
        gate["rows"][0]["gradient"] *= 2
    elif mutation == "index":
        gate["rows"][1]["index"] = 0
        reseal(gate)
    elif mutation == "purpose":
        gate["purpose"] = "search"
        reseal(gate)
    elif mutation == "unit":
        gate["coordinate_units"][0] = 0.0
        reseal(gate)
    elif mutation == "nan":
        parameters["a"] = float("nan")
    elif mutation == "negative_error":
        gate["rows"][0]["gradient_error_bound"] = -1.0
        reseal(gate)
    with pytest.raises(ValueError):
        enrich_diagnosis(raw, parameters=parameters, bounds=bounds)


def test_reject_reenrichment_and_inconsistent_exact_certificate():
    raw, parameters, bounds = diagnostic()
    result = enrich_diagnosis(raw, parameters=parameters, bounds=bounds)
    with pytest.raises(ValueError, match="only once"):
        enrich_diagnosis(result, parameters=parameters, bounds=bounds)
    raw["exact_reduction"] = {"parameters": parameters | {"a": 0.8}, "rank_ceiling": 1}
    with pytest.raises(ValueError, match="certificate"):
        enrich_diagnosis(raw, parameters=parameters, bounds=bounds)

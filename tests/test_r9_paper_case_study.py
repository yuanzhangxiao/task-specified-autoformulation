"""Paper exports must distinguish baseline bias, response and model selection."""

import copy

import pytest

from scripts.build_r9_paper_case_study import score, selected_sol, validate_records


def record(observed, predicted, nmse):
    return {
        "success": True,
        "time": [0, 1],
        "observed": observed,
        "predicted": predicted,
        "nmse": nmse,
    }


def test_effect_subtracts_own_control_and_retains_absolute_bias():
    control = record([10, 10], [20, 20], 100)
    intervention = record([10, 12], [20, 22], 100)
    result = score(intervention, 1, control)
    assert result == {"nmse": 100, "response_rse": 0}
    insensitive = record([10, 12], [20, 20], 82)
    assert score(insensitive, 1, control)["response_rse"] == 1


def test_selection_uses_original_validation_not_training_or_probe():
    rows = [
        {
            "model": "sol0",
            "validation_nmse": 0.01,
            "train_nmse": 0.5,
            "worst_response_rse": 100,
        },
        {
            "model": "sol1",
            "validation_nmse": 0.02,
            "train_nmse": 0.01,
            "worst_response_rse": 0,
        },
        {"model": "r9", "validation_nmse": 0, "train_nmse": 0, "worst_response_rse": 0},
    ]
    assert selected_sol(rows) == "sol0"


@pytest.mark.parametrize("change", ["grid", "reference", "nan", "length", "failure"])
def test_invalid_curve_cannot_enter_paper_export(change):
    a = record([1, 2], [1, 2], 0)
    b = copy.deepcopy(a)
    if change == "grid":
        b["time"] = [0, 2]
    elif change == "reference":
        b["observed"] = [1, 3]
    elif change == "nan":
        b["predicted"] = [1, float("nan")]
    elif change == "length":
        b["predicted"] = [1]
    else:
        b["success"] = False
    with pytest.raises(ValueError):
        validate_records({"a": a, "b": b})


def test_mismatched_metric_and_negligible_contrast_rejected():
    a = record([1, 2], [1, 2], 0.1)
    with pytest.raises(ValueError, match="Saved absolute"):
        score(a, 1)
    a["nmse"] = 0
    with pytest.raises(ValueError, match="Negligible"):
        score(a, 1, a)

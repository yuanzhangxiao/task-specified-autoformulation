"""Prevent misleading pairing of frozen intervention curves."""

import pytest

from scripts.plot_t1_four_model_interventions import MODELS, aligned, differences


def test_differences_remove_each_models_own_baseline():
    control = [
        {
            "case_id": "control",
            "time_min": 0,
            "reference": 10,
            **{key: 100 * (i + 1) for i, key in enumerate(MODELS)},
        }
    ]
    changed = [
        {
            "case_id": "changed",
            "time_min": 0,
            "reference": 13,
            **{key: 100 * (i + 1) + 2 for i, key in enumerate(MODELS)},
        }
    ]
    result = differences(changed, control)[0]
    assert result["reference"] == 3
    assert all(result[key] == 2 for key in MODELS)
    assert result["control_case_id"] == "control"


def test_control_time_grid_must_match():
    with pytest.raises(ValueError, match="time grids"):
        differences([{"time_min": 0}], [{"time_min": 1}])


@pytest.mark.parametrize(
    "change,message",
    [
        ({"success": False}, "Failed"),
        ({"time": [0, 2]}, "time grid"),
        ({"observed": [1, 3]}, "physical reference"),
        ({"predicted": [1]}, "Incomplete prediction"),
        ({"predicted": [1, float("nan")]}, "Nonfinite"),
    ],
)
def test_reject_invalid_or_mismatched_rollouts(change, message):
    row = {"success": True, "time": [0, 1], "observed": [1, 2], "predicted": [1, 2]}
    aligned({"a": row, "b": row.copy()})
    with pytest.raises(ValueError, match=message):
        aligned({"a": row, "b": {**row, **change}})

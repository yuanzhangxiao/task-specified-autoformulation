"""Fixed checkpoint selection, missing-model denominators, and response pooling."""

from copy import deepcopy

import pytest

from scripts import iclr_anytime_response as study


def example_history():
    model = {
        "candidate": {"states": [{"name": "y", "kind": "observed"}]},
        "context": {"targets": ["y"]},
        "parameters": {"a": 1},
        "validation": {"normalized_mse": 1},
    }
    return {
        "artifact_sha256": "history",
        "models": {"a": model, "b": {**model, "parameters": {"a": 2}}},
        "runs": [
            {
                "task": {
                    "cell": "test",
                    "treatment": "full",
                    "task_id": "seed0",
                    "seed": 0,
                },
                "visits": [
                    {"round": i, "selected_model": "a" if i < 9 else "b"}
                    for i in range(15)
                ],
                "final_model": "a",
            }
        ],
    }


def test_prescribed_incumbents_and_pruning_are_kept_separate():
    source = example_history()
    conditions = {"test": {"family": "t2", "source": "physical"}}
    rows = study.checkpoint_rows(source, conditions, "cohort")
    assert [r["checkpoint"] for r in rows] == ["1", "5", "10", "15", "pruned"]
    assert [r["model_id"] for r in rows] == ["a", "a", "b", "b", "a"]
    source["runs"][0]["visits"][0]["selected_model"] = None
    assert (
        study.checkpoint_rows(source, conditions, "cohort")[0]["status"]
        == "model_unavailable"
    )
    incomplete = deepcopy(source)
    incomplete["runs"][0]["visits"].pop()
    with pytest.raises(ValueError, match="complete fifteen"):
        study.checkpoint_rows(incomplete, conditions, "cohort")


def test_pooling_is_median_rse_mean_direction_shape_and_preserves_failures():
    cases = [
        {
            "quantity": "T",
            "status": "complete",
            "test_nmse": v,
            "response_rse": v,
            "directional_accuracy": v / 10,
            "shape_error": v / 10,
        }
        for v in (1, 2, 9)
    ]
    result = study.pooled(cases, {"T"}, 3)
    assert result["response_rse"] == 2
    assert result["test_nmse"] == 2
    assert result["directional_accuracy"] == pytest.approx(0.4)
    assert result["shape_error"] == pytest.approx(0.4)
    assert study.pooled(cases[:-1], {"T"}, 3)["status"] == "incomplete_panel"


def test_continuous_response_scores_include_missing_mechanism_as_zero():
    result = study.continuous_score(
        [0, 1, 2], [0, 1, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0], 1
    )
    assert result["response_rse"] == 1
    assert result["directional_accuracy"] == 0
    assert result["shape_error"] == 1
    signed = study.continuous_score(
        [0, 1, 2], [0, 1, 2], [0, 0, 0], [0, -1, 2], [0, 0, 0]
    )
    assert signed["directional_accuracy"] == pytest.approx(2 / 3)

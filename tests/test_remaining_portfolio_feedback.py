"""Training-only residual descriptions preserve sample order and qualifications."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.fitting import remaining_portfolio_feedback as feedback
from autoformalism.schemas.public_fitting import PublicSplit


def training():
    return PublicSplit.model_validate(
        {
            "name": "train",
            "fingerprint": "fixture",
            "rows": [
                {
                    "trajectory_id": "meal",
                    "time": [0, 1, 2, 3, 4],
                    "targets": {"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]},
                    "external_inputs": {feedback.MEAL_INPUT: [0, 1, 0, 1, 0]},
                },
                {
                    "trajectory_id": "no-meal",
                    "time": [0, 1],
                    "targets": {"a": [1, 1], "b": [2, 2]},
                    "external_inputs": {feedback.MEAL_INPUT: [0, 0]},
                },
            ],
        }
    )


def test_multioutput_unequal_row_lengths_and_window_partition():
    residual = np.arange(14, dtype=float) / 10
    result = feedback.summarize_residuals(
        training(), ("b", "a"), {"a": 2, "b": 3}, residual
    )
    assert result["training_nmse"] == pytest.approx(np.mean(residual**2))
    assert result["rows"][0]["targets"]["b"]["prediction"] == pytest.approx(
        [5, 4.6, 4.2, 3.8, 3.4]
    )
    assert result["rows"][1]["targets"]["a"]["prediction"] == pytest.approx([3.2, 3.6])
    windows = result["rows"][0]["targets"]["a"]["windows"]
    assert {name: value["samples"] for name, value in windows.items()} == {
        "pre_meal": 1,
        "ingestion": 2,
        "post_ingestion": 2,
    }
    assert set(result["rows"][1]["targets"]["a"]["windows"]) == {"no_meal_schedule"}
    assert not result["scientific_cause_identified"]
    assert not result["global_minimum_certified"]


def test_plateau_is_not_reported_as_unique_precise_peak():
    result = feedback.summarize_residuals(
        training(), ("a", "b"), {"a": 1, "b": 1}, np.zeros(14)
    )
    peak = result["rows"][1]["targets"]["a"]["sampled_peak"]
    assert peak["observation_plateau_samples"] == 2
    assert peak["observation_at_boundary"]


@pytest.mark.parametrize("bad", [np.zeros(13), np.zeros((7, 2)), np.full(14, np.nan)])
def test_shape_and_nonfinite_residuals_fail(bad):
    with pytest.raises(ValueError, match="residual"):
        feedback.summarize_residuals(training(), ("a", "b"), {"a": 1, "b": 1}, bad)


def test_validation_data_or_wrong_output_contract_rejected():
    with pytest.raises(ValueError, match="training only"):
        feedback.summarize_residuals(
            training().model_copy(update={"name": "val"}),
            ("a", "b"),
            {"a": 1, "b": 1},
            np.zeros(14),
        )
    with pytest.raises(ValueError, match="target contract"):
        feedback.summarize_residuals(training(), ("a",), {"a": 1}, np.zeros(7))


def test_missing_input_is_unclassified_and_negative_rate_rejected():
    payload = training().model_dump(mode="json")
    payload["rows"][0]["external_inputs"] = {}
    result = feedback.summarize_residuals(
        PublicSplit.model_validate(payload), ("a", "b"), {"a": 1, "b": 1}, np.zeros(14)
    )
    assert set(result["rows"][0]["targets"]["a"]["windows"]) == {"unclassified"}
    payload["rows"][0]["external_inputs"] = {feedback.MEAL_INPUT: [-1, 0, 0, 0, 0]}
    with pytest.raises(ValueError, match="nonnegative"):
        feedback.summarize_residuals(
            PublicSplit.model_validate(payload),
            ("a", "b"),
            {"a": 1, "b": 1},
            np.zeros(14),
        )


def test_endpoint_replay_is_frozen_and_no_derivatives_requested():
    calls, checkpoints = [], []

    def evaluate(point, deadline, **kwargs):
        calls.append((point.tolist(), kwargs))
        return np.ones(14), None

    oracle = SimpleNamespace(
        training=training(),
        system=SimpleNamespace(channels=("a", "b")),
        scales={"a": 1, "b": 1},
        vector=lambda p: np.array([p["p"]]),
        evaluate=evaluate,
    )
    endpoint = {
        "parameters": {"p": 2},
        "training_nmse": 1,
        "original_equations_verified": True,
    }
    before = deepcopy(endpoint)
    result = feedback.describe_endpoint(oracle, endpoint, 10, checkpoints.append)
    assert result["passed"] and result["status"] == "complete"
    assert endpoint == before
    assert calls == [([2], {"jacobian": False, "method": "Radau"})]
    assert checkpoints[-1]["passed"]
    endpoint["training_nmse"] = 2
    result = feedback.describe_endpoint(oracle, endpoint, 10, checkpoints.append)
    assert not result["passed"]
    assert "rows" not in result
    assert "disagrees" in str(result["error"])
    endpoint["original_equations_verified"] = False
    with pytest.raises(ValueError, match="verified"):
        feedback.describe_endpoint(oracle, endpoint, 10, checkpoints.append)

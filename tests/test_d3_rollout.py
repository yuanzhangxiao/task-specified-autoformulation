"""Native increments, frozen values, and Phase-B information permissions."""

import numpy as np
import pytest

from autoformalism.baselines.d3_rollout import (
    NativeMap,
    evaluate_validation,
    finite_mean,
    predict,
)
from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.schemas import CandidateModel


def candidate(rhs="k * x", auxiliary=False):
    names = ["x", "a"] if auxiliary else ["x"]
    return CandidateModel.model_validate(
        {
            "candidate_id": "native",
            "parent_candidate_id": None,
            "states": [{"name": name, "kind": "observed"} for name in names],
            "state_equations": [{"state": "x", "rhs": rhs}]
            + ([{"state": "a", "rhs": "100 * a"}] if auxiliary else []),
            "observation_mappings": [{"channel": "x", "expression": "x"}],
            "parameters": [
                {
                    "name": "k",
                    "scope": "global",
                    "bounds": {"lower": 0, "upper": 1},
                    "initialization_range": {"lower": 0.1, "upper": 0.5},
                }
            ],
            "initial_conditions": [
                {"state": name, "scope": "global", "expression": name} for name in names
            ],
        }
    )


def trajectory(values=(1, 2, 3), aux=None, dt=0.1):
    return Trajectory(
        trajectory_id="t",
        time=np.arange(len(values)) * dt,
        targets={"x": np.array(values, dtype=float)},
        auxiliaries={} if aux is None else {"a": np.array(aux, dtype=float)},
        external_inputs={},
        fixed_covariates={},
        derivatives={},
    )


def splits(values=(1, 2, 3), aux=None):
    return (
        DatasetSplit(SplitName.TRAIN, (trajectory(),), "train"),
        DatasetSplit(SplitName.VALIDATION, (trajectory(values, aux),), "val"),
    )


def test_increment_not_derivative_and_no_future_target_reset():
    model = NativeMap.build(candidate(), {"k": 1.0}, ("x",), ())
    data = trajectory((1, 200, 300))
    np.testing.assert_array_equal(
        predict(model, data, teacher_forced=False)[:, 0], [1, 2, 4]
    )
    np.testing.assert_array_equal(
        predict(model, data, teacher_forced=True)[:, 0], [1, 2, 400]
    )


def test_supplied_auxiliary_path_used_each_step():
    model = NativeMap.build(candidate("k * a", True), {"k": 1}, ("x", "a"), ())
    actual = predict(model, trajectory((1, 999, 999), (2, 7, 99)), teacher_forced=False)
    np.testing.assert_array_equal(actual[:, 0], [1, 3, 10])


def test_exact_native_coefficients_preserved_and_bounds_audited():
    train, val = splits((1, 0.5, 0.25))
    result = evaluate_validation(
        candidate(), {"k": -0.5}, train, val, saved_one_step_nmse=0
    )
    assert result["phase_b_rollout"]["normalized_mse"] == 0
    assert result["saved_one_step_matches"] is True
    assert result["bounds_violations"][0]["value"] == -0.5
    assert result["saved_parameter_values_preserved"] is True


@pytest.mark.parametrize("wrong", [SplitName.TRAIN, SplitName.TEST])
def test_test_split_rejected(wrong):
    train, val = splits()
    with pytest.raises(ValueError, match="never TEST"):
        evaluate_validation(
            candidate(), {"k": 0}, train, DatasetSplit(wrong, val.trajectories, "wrong")
        )


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"k": float("nan")},
        {"k": 1, "z": 2},
        {"k": True},
    ],
)
def test_invalid_saved_vector_rejected(parameters):
    with pytest.raises(ValueError):
        NativeMap.build(candidate(), parameters, ("x",), ())


def test_zero_division_is_not_silently_guarded():
    train, val = splits()
    result = evaluate_validation(candidate("1 / (k - k)"), {"k": 0}, train, val)
    assert result["phase_b_rollout"]["status"] == "rollout_failed"
    assert result["phase_b_rollout"]["normalized_mse"] is None


def test_train_normalization_and_sample_policy():
    train, val = splits()
    result = evaluate_validation(candidate(), {"k": 0}, train, val)
    assert result["phase_b_rollout"]["normalized_mse"] == pytest.approx(2.5)
    assert result["native_one_step"]["normalized_mse"] == pytest.approx(1.5)
    assert result["phase_b_rollout"]["sample_policy"] == "include_initial"
    assert result["native_one_step"]["sample_policy"] == "exclude_initial"


def test_native_metric_matches_torch_when_available():
    pytest.importorskip("torch")
    from autoformalism.baselines.d3_native import _target_scales, evaluate_native_d3

    train, val = splits()
    model = candidate("k * tanh(x) + exp(-x) + softplus(x) / (1 + x)")
    expected, _ = evaluate_native_d3(
        model, val, {"k": -0.5}, ("x",), _target_scales(train, ("x",))
    )
    result = evaluate_validation(model, {"k": -0.5}, train, val)
    assert result["native_one_step"]["normalized_mse"] == pytest.approx(expected)


def test_large_finite_errors_do_not_overflow_when_averaged():
    with np.errstate(over="raise", invalid="raise"):
        assert finite_mean([1e308, 1e308, 1e308]) == 1e308


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
def test_invalid_deadline_refused(seconds):
    train, val = splits()
    with pytest.raises(ValueError, match="trajectory time limit"):
        evaluate_validation(candidate(), {"k": 0}, train, val, seconds=seconds)

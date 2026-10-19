"""Separate common-horizon scoring and prefix-only conditional sensitivity."""

from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.fitting import prefix_initialization as p
from autoformalism.fitting import prefix_window_diagnostics as d
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.public_fitting import PublicTrajectory
from tests.test_prefix_initialization import problem, row, training


@pytest.fixture(scope="module")
def fitted():
    model = problem()
    policy = p.NumericalPolicy()
    fit = p.fit_training(model, training(), policy)
    assert fit["fit_success"]
    return model, fit, policy


def prediction(cutoff=1.0):
    heldout = PublicTrajectory(
        trajectory_id="common", time=(0, 1, 2, 3), targets={"y": (1, 2, 3, 4)}
    )
    forecast = {
        "trajectory_id": "common",
        "time": [0, 1, 2, 3],
        "cutoff": cutoff,
        "forecast_success": True,
        "predictions": {"y": [1000, 2000, 3000, 6]},
        "inference_sha256": "a" * 64,
        "schedule_sha256": "b" * 64,
    }
    return forecast, heldout


def test_common_horizon_excludes_boundary_and_preserves_original_receipt():
    forecast, heldout = prediction()
    before = deepcopy(forecast)
    result = d.common_horizon_score(forecast, heldout, 2.0, {"y": 2.0})
    assert result["available"]
    assert result["normalized_mse"] == 1.0
    assert result["per_target_normalized_mse"] == {"y": 1.0}
    assert result["sample_count"] == result["horizon_sample_count"] == 1
    assert result["forecast_cutoff"] == 1.0 and result["score_start"] == 2.0
    assert result["forecast_sha256"] == content_sha256(before)
    assert result["inference_sha256"] == before["inference_sha256"]
    assert result["schedule_sha256"] == before["schedule_sha256"]
    assert forecast == before
    later, _ = prediction(2.0)
    same_horizon = d.common_horizon_score(later, heldout, 2.0, {"y": 2.0})
    assert same_horizon["normalized_mse"] == result["normalized_mse"]
    assert same_horizon["forecast_cutoff"] == 2.0


@pytest.mark.parametrize("start", [0.5, 3.0, 4.0, float("nan"), float("inf")])
def test_common_horizon_rejects_early_empty_and_nonfinite_windows(start):
    forecast, heldout = prediction()
    with pytest.raises(ValueError, match="scoring start"):
        d.common_horizon_score(forecast, heldout, start, {"y": 1.0})


@pytest.mark.parametrize(
    "kind", ["row", "time", "channels", "length", "nonfinite", "scale"]
)
def test_common_horizon_rejects_incompatible_scoring_inputs(kind):
    forecast, heldout = prediction()
    scales = {"y": 1.0}
    if kind == "row":
        forecast["trajectory_id"] = "other"
    elif kind == "time":
        forecast["time"][1] = 1.1
    elif kind == "channels":
        forecast["predictions"] = {"hidden": [0] * 4}
    elif kind == "length":
        forecast["predictions"]["y"].pop()
    elif kind == "nonfinite":
        forecast["predictions"]["y"][-1] = float("nan")
    else:
        scales["y"] = 0.0
    with pytest.raises(ValueError):
        d.common_horizon_score(forecast, heldout, 2.0, scales)


def test_failed_forecast_is_unavailable_on_the_same_intended_horizon():
    forecast, heldout = prediction()
    forecast.update(forecast_success=False, predictions={})
    result = d.common_horizon_score(forecast, heldout, 2.0, {"y": 1.0})
    assert not result["available"] and result["normalized_mse"] is None
    assert result["per_target_normalized_mse"] == {}
    assert result["sample_count"] == 0 and result["horizon_sample_count"] == 1


def infer(fitted, duration, *, start=None):
    model, fit, policy = fitted
    prefix = p.make_prefix(row("information", 1.7), p.PrefixPolicy(duration=duration))
    inference = p.infer_prefix(model, fit, prefix, policy, root_guesses=start)
    assert inference["fit_success"]
    return prefix, inference


def test_longer_prefix_has_more_information_without_changing_native_units(fitted):
    model, fit, numerical = fitted
    short, short_fit = infer(fitted, 0.1, start={"init_r_value": 5.0})
    long, long_fit = infer(fitted, 1.0, start={"init_r_value": 5.0})
    brief = d.prefix_information(model, fit, short, short_fit, numerical, 0.05)
    extended = d.prefix_information(model, fit, long, long_fit, numerical, 0.05)
    assert brief["available"] and extended["available"]
    assert brief["rank"] == extended["rank"] == 1
    assert extended["singular_values"][0] > brief["singular_values"][0]
    assert extended["one_noise_unit_root_scale"] < brief["one_noise_unit_root_scale"]
    assert extended["coordinate_system"] == "native_physical_roots"
    assert extended["conditional_on_frozen_shared_coefficients"]
    assert not extended["confidence_interval"]
    assert (
        extended["diagnostic_calls_started"]
        == extended["diagnostic_calls_completed"]
        == 1
    )
    theta, phi = (
        fit["shared_parameters"]["theta"],
        fit["shared_parameters"]["init_z_phi"],
    )
    time = np.asarray(long.time)
    derivative = phi * (np.exp(-theta * time) - np.exp(-time)) / (1 - theta)
    derivative += (np.exp(-theta * time) - np.exp(-2 * time)) / (2 - theta)
    norm = np.linalg.norm(derivative)
    assert extended["physical_output_singular_values"] == pytest.approx(
        [norm], rel=1e-7
    )
    assert extended["singular_values"] == pytest.approx(
        [norm / fit["training_scales"]["y"]], rel=1e-7
    )
    assert extended["one_noise_unit_root_scale"] == pytest.approx(0.05 / norm, rel=1e-7)


def test_zero_root_information_and_noiseless_label_are_explicit(fitted):
    model, fit, numerical = fitted
    zero, zero_fit = infer(fitted, 0)
    empty = d.prefix_information(model, fit, zero, zero_fit, numerical, 0.05)
    assert empty["available"] and empty["rank"] == 0
    assert empty["one_noise_unit_root_scale"] is None
    assert empty["scale_status"] == "uninformative"
    assert empty["residual_count"] == 1 and empty["noise_scale_residual_count"] == 0
    assert (
        empty["noise_scale_observation_scope"]
        == "prefix_samples_strictly_after_initial_time"
    )
    positive, positive_fit = infer(fitted, 0.5)
    noiseless = d.prefix_information(model, fit, positive, positive_fit, numerical, 0)
    assert noiseless["rank"] == 1
    assert noiseless["one_noise_unit_root_scale"] is None
    assert noiseless["scale_status"] == "noiseless"


def test_diagnostic_only_reads_prefix_at_retained_roots(fitted, monkeypatch):
    model, fit, numerical = fitted
    prefix, inference = infer(fitted, 0.4)
    clean = row("information", 1.7)
    raw = clean.model_dump(mode="json")
    raw["targets"]["y"][5:] = [10000] * 36
    poisoned = PublicTrajectory.model_validate(raw)
    same_prefix = p.make_prefix(poisoned, prefix.policy)
    assert same_prefix == prefix
    original = p.PrefixOracle.evaluate
    calls = []

    def evaluate(oracle, vector, deadline, **kwargs):
        calls.append(1)
        assert oracle.rows[0].time.tolist() == list(prefix.time)
        assert oracle.rows[0].targets["y"].tolist() == list(prefix.targets["y"])
        assert oracle.parameters(vector) == inference["roots"]
        assert oracle.frozen == fit["shared_parameters"]
        return original(oracle, vector, deadline, **kwargs)

    monkeypatch.setattr(p.PrefixOracle, "evaluate", evaluate)
    before = content_sha256(fit), content_sha256(inference)
    clean_info = d.prefix_information(model, fit, prefix, inference, numerical, 0.1)
    poisoned_info = d.prefix_information(
        model, fit, same_prefix, inference, numerical, 0.1
    )
    assert len(calls) == 2
    assert clean_info["singular_values"] == poisoned_info["singular_values"]
    assert (
        clean_info["one_noise_unit_root_scale"]
        == poisoned_info["one_noise_unit_root_scale"]
    )
    assert before == (content_sha256(fit), content_sha256(inference))


@pytest.mark.parametrize(
    "field",
    [
        "prefix_sha256",
        "fit_sha256",
        "problem_sha256",
        "shared_parameters",
        "training_scales",
    ],
)
def test_diagnostic_rejects_mismatched_inference_without_work(
    fitted, monkeypatch, field
):
    model, fit, numerical = fitted
    prefix, inference = infer(fitted, 0.4)
    invalid = deepcopy(inference)
    invalid[field] = (
        {} if field in {"shared_parameters", "training_scales"} else "0" * 64
    )
    monkeypatch.setattr(
        p.PrefixOracle, "evaluate", lambda *a, **k: pytest.fail("unexpected rollout")
    )
    with pytest.raises(ValueError, match=r"differs|differ"):
        d.prefix_information(model, fit, prefix, invalid, numerical, 0.1)


@pytest.mark.parametrize("sigma", [-0.1, float("nan"), float("inf"), True, [0.1]])
def test_noise_input_is_predeclared_finite_nonnegative_scalar(fitted, sigma):
    model, fit, numerical = fitted
    prefix, inference = infer(fitted, 0.1)
    with pytest.raises(ValueError, match="noise sigma"):
        d.prefix_information(model, fit, prefix, inference, numerical, sigma)


def test_diagnostic_timeout_charges_started_call_and_never_retries(fitted, monkeypatch):
    model, fit, numerical = fitted
    prefix, inference = infer(fitted, 0.2)
    progress = []

    def unavailable(*args, **kwargs):
        assert progress[-1]["diagnostic_calls_started"] == 1
        assert progress[-1]["diagnostic_calls_completed"] == 0
        raise TimeoutError("test diagnostic exhausted")

    monkeypatch.setattr(p.PrefixOracle, "evaluate", unavailable)
    result = d.prefix_information(
        model, fit, prefix, inference, numerical, 0.1, progress.append
    )
    assert not result["available"] and result["status"] == "budget_exhausted"
    assert (
        result["diagnostic_calls_started"] == 1
        and result["diagnostic_calls_completed"] == 0
    )
    assert result["one_noise_unit_root_scale"] is None
    assert progress[-1] == result


def test_diagnostic_interruption_preserves_started_reservation(fitted, monkeypatch):
    model, fit, numerical = fitted
    prefix, inference = infer(fitted, 0.2)
    progress = []

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(p.PrefixOracle, "evaluate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        d.prefix_information(
            model, fit, prefix, inference, numerical, 0.1, progress.append
        )
    assert progress[-1]["diagnostic_calls_started"] == 1
    assert progress[-1]["diagnostic_calls_completed"] == 0


def test_failed_inference_has_no_diagnostic_call(fitted, monkeypatch):
    model, fit, numerical = fitted
    prefix, inference = infer(fitted, 0.2)
    inference.update(fit_success=False, roots=None)
    monkeypatch.setattr(
        p.PrefixOracle, "evaluate", lambda *a, **k: pytest.fail("unexpected rollout")
    )
    result = d.prefix_information(model, fit, prefix, inference, numerical, 0.1)
    assert not result["available"] and result["status"] == "inference_unavailable"
    assert (
        result["diagnostic_calls_started"] == result["diagnostic_calls_completed"] == 0
    )

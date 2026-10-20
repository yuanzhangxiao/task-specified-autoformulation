"""Independent roots, prefix-only convex checks, and conditional SVD evidence."""

import json
from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_multiroot_controls as controls
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.public_fitting import PublicSplit, PublicTrajectory


@pytest.fixture(scope="module", params=controls.SCENARIOS)
def fitted(request):
    record = controls.fixture(request.param)
    model = kernel.PrefixFitProblem.model_validate(record["problem"])
    numerical = kernel.NumericalPolicy()
    result = kernel.fit_training(
        model, PublicSplit.model_validate(record["training"]), numerical
    )
    assert result["fit_success"], result
    return record, model, result, numerical


def infer(fitted, duration=1.0, start=(0.3, 1.3)):
    record, model, fit, numerical = fitted
    row = PublicSplit.model_validate(record["validation"]).rows[0]
    prefix = kernel.make_prefix(row, kernel.PrefixPolicy(duration=duration))
    result = kernel.infer_prefix(
        model,
        fit,
        prefix,
        numerical,
        root_guesses=dict(zip(controls.ROOT_NAMES, start, strict=True)),
    )
    assert result["fit_success"], result
    return prefix, result


@pytest.mark.parametrize("scenario", controls.SCENARIOS)
def test_fixture_split_identity_noise_and_truth_separation(scenario):
    record = controls.fixture(scenario, noise_sigma=0.01, repeat=2)
    another_start = controls.fixture(scenario, seed=1, noise_sigma=0.01, repeat=2)
    another_repeat = controls.fixture(scenario, noise_sigma=0.01, repeat=3)
    assert record == controls.fixture(scenario, noise_sigma=0.01, repeat=2)
    assert not record["test_data_opened"]
    assert record["followup_scope"] == "new_synthetic_qualification_trajectories"
    ids = []
    for key, count in (("training", 3), ("validation", 4), ("followup", 4)):
        split = PublicSplit.model_validate(record[key])
        assert split.name == ("train" if key == "training" else "val")
        assert len(split.rows) == count
        assert controls.fingerprint(record[key]) == record[key]
        assert record[key] == another_start[key]
        assert record[key] != another_repeat[key]
        for row in split.rows:
            assert row.time == tuple(np.linspace(0, 4, 41))
            assert row.targets["y"][0] == 0.4
            assert set(row.targets) == {"y"}
            assert not row.auxiliaries and not row.external_inputs
            assert not row.fixed_covariates
            ids.append(row.trajectory_id)
        assert record[key] != record["evaluation"]["clean"][key]
    assert len(ids) == len(set(ids)) == 11
    assert set(ids) == set(record["evaluation"]["roots"])
    assert record["problem"] != another_start["problem"]
    record["evaluation"]["clean"]["training"]["rows"][0]["targets"]["y"][1] = 1e9
    assert record["training"] == another_start["training"]
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize(
    "arguments",
    [
        {"scenario": "missing"},
        {"scenario": "separated", "seed": True},
        {"scenario": "separated", "seed": 2},
        {"scenario": "separated", "repeat": -1},
        {"scenario": "separated", "repeat": False},
        {"scenario": "separated", "noise_sigma": -1},
        {"scenario": "separated", "noise_sigma": True},
        {"scenario": "separated", "noise_sigma": float("nan")},
    ],
)
def test_fixture_rejects_invalid_declarations(arguments):
    with pytest.raises(ValueError):
        controls.fixture(**arguments)


def test_declared_roots_are_independent_and_shared_theta_is_fitted(fitted):
    record, model, fit, numerical = fitted
    assert set(fit["shared_parameters"]) == {"theta"}
    assert fit["shared_parameters"]["theta"] == pytest.approx(0.35, abs=1e-6)
    assert fit["normalized_mse"] < 1e-14
    for state in ("z", "r"):
        rule = model.graph.rules[state].initial
        assert rule.mode == "value"
        assert rule.scope == "trajectory"
        assert rule.role == "nonnegative_coefficient"
    oracle = kernel.TrainingOracle(
        model, PublicSplit.model_validate(record["training"]), numerical
    )
    assert oracle.shared == ("theta",) and oracle.roots == controls.ROOT_NAMES
    assert len(oracle.names) == 7
    for row_id, roots in fit["training_roots"].items():
        truth = record["evaluation"]["roots"][row_id]
        if record["scenario"] == "exact_confounded":
            assert sum(roots.values()) == pytest.approx(sum(truth.values()), abs=1e-6)
        else:
            assert roots == pytest.approx(truth, abs=1e-6)


@pytest.mark.parametrize("duration", [0.0, 0.1, 1.0])
def test_analytic_design_matches_compiled_root_sensitivities(fitted, duration):
    _, model, fit, numerical = fitted
    prefix, inference = infer(fitted, duration)
    oracle, design, target, _ = controls._design(
        model, fit, prefix, inference, numerical
    )
    vector = oracle.vector(inference["roots"])
    residual, jacobian = oracle.evaluate(vector, monotonic() + 20)
    np.testing.assert_allclose(design, jacobian, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(design @ vector - target, residual, atol=2e-8)


def test_two_root_svd_reports_native_weak_direction_and_conditional_scales(fitted):
    record, model, fit, numerical = fitted
    prefix, inference = infer(fitted)
    information = controls.prefix_information(
        model, fit, prefix, inference, numerical, 0.01
    )
    assert information["available"], information
    assert information["root_names"] == list(controls.ROOT_NAMES)
    assert information["diagnostic_calls_started"] == 1
    assert information["diagnostic_calls_completed"] == 1
    assert information["integration_calls"] == 0
    assert information["conditional_on_frozen_shared_coefficients"]
    assert not information["confidence_interval"]
    assert not information["domain_bounds_in_noise_scales"]
    _, design, _, physical = controls._design(
        model, fit, prefix, inference, numerical
    )
    singular = np.linalg.svd(physical, compute_uv=False)
    np.testing.assert_allclose(information["physical_output_singular_values"], singular)
    weak = np.asarray(list(information["weak_direction"].values()))
    assert np.linalg.norm(weak) == pytest.approx(1.0)
    assert np.linalg.norm(design @ weak) == pytest.approx(
        information["singular_values"][-1], abs=1e-14
    )
    assert information["noise_scale_residual_count"] == len(prefix.time) - 1
    if record["scenario"] == "exact_confounded":
        assert information["rank"] == 1
        assert information["condition_number"] is None
        assert information["weakest_one_noise_unit_root_scale"] is None
        assert information["scale_status"] == "partially_uninformative"
        np.testing.assert_allclose(weak, np.asarray([1, -1]) / np.sqrt(2), atol=1e-12)
    else:
        assert information["rank"] == 2
        assert information["scale_status"] == "available"
        np.testing.assert_allclose(
            information["one_noise_unit_root_scales"], 0.01 / singular
        )
    json.dumps(information, allow_nan=False)


def test_zero_prefix_is_rank_zero_and_noiseless_scales_are_unavailable(fitted):
    _, model, fit, numerical = fitted
    prefix, inference = infer(fitted, 0)
    zero = controls.prefix_information(model, fit, prefix, inference, numerical, 0.01)
    assert zero["rank"] == 0
    assert zero["singular_values"] == [0.0, 0.0]
    assert zero["one_noise_unit_root_scales"] == [None, None]
    assert zero["scale_status"] == "uninformative"
    assert not zero["weak_direction_unique"]
    prefix, inference = infer(fitted)
    clean = controls.prefix_information(model, fit, prefix, inference, numerical, 0)
    assert clean["scale_status"] == "noiseless"
    assert clean["one_noise_unit_root_scales"] == [None, None]


def test_conditional_reference_uses_only_prefix_and_preserves_retained_fit(fitted):
    record, model, fit, numerical = fitted
    prefix, inference = infer(fitted)
    before = content_sha256(fit), content_sha256(inference)
    reference = controls.conditional_linear_optimum(
        model, fit, prefix, inference, numerical
    )
    assert reference["available"] and reference["objective_optimal"], reference
    assert reference["reference_normalized_mse"] < 1e-14
    assert abs(reference["objective_gap"]) < 1e-14
    assert not reference["roots_replaced"] and not reference["truth_used"]
    assert all(
        bound == {"lower": 0.0, "upper": None}
        for bound in reference["root_bounds"].values()
    )
    altered = deepcopy(record["validation"]["rows"][0])
    altered["targets"]["y"][11:] = [1e9] * 30
    poisoned = kernel.make_prefix(
        PublicTrajectory.model_validate(altered), prefix.policy
    )
    assert poisoned == prefix
    repeated = controls.conditional_linear_optimum(
        model, fit, poisoned, inference, numerical
    )
    assert repeated["reference_roots"] == reference["reference_roots"]
    assert repeated["reference_normalized_mse"] == reference["reference_normalized_mse"]
    assert before == (content_sha256(fit), content_sha256(inference))
    json.dumps(reference, allow_nan=False)


def test_conditional_reference_exposes_bad_retained_objective(fitted):
    _, model, fit, numerical = fitted
    prefix, inference = infer(fitted)
    altered = deepcopy(inference)
    altered["roots"] = dict.fromkeys(controls.ROOT_NAMES, 3.0)
    reference = controls.conditional_linear_optimum(
        model, fit, prefix, altered, numerical
    )
    assert reference["available"]
    assert not reference["objective_optimal"]
    assert reference["objective_gap"] > 0.1


@pytest.mark.parametrize(
    ("target", "expected"),
    [([-2, 0.5], [0, 0.5]), ([4, 4], [1, 1]), ([0.3, 0.7], [0.3, 0.7])],
)
def test_convex_reference_checks_interior_faces_and_corners(target, expected):
    vector, _ = controls._box_optimum(
        np.eye(2), np.asarray(target), np.zeros(2), np.ones(2)
    )
    np.testing.assert_allclose(vector, expected)


def test_equal_rates_allow_different_roots_with_matching_predictions():
    time = np.linspace(0, 4, 41)
    one = controls.analytic(time, 0.4, 0.35, 0.2, 1.8, 1, 1)
    other = controls.analytic(time, 0.4, 0.35, 1.4, 0.6, 1, 1)
    np.testing.assert_allclose(one, other, atol=1e-15)
    equal_poles = controls.analytic(time, 0.4, 1, 0.2, 1.8, 1, 1)
    np.testing.assert_allclose(equal_poles, (0.4 + 2 * time) * np.exp(-time))


def test_failed_inference_and_failed_diagnostic_keep_unavailability(
    fitted, monkeypatch
):
    _, model, fit, numerical = fitted
    prefix, inference = infer(fitted)
    failed = deepcopy(inference)
    failed.update(fit_success=False, roots=None)
    for function, args in (
        (controls.prefix_information, (0.01,)),
        (controls.conditional_linear_optimum, ()),
    ):
        saved = []
        result = function(
            model, fit, prefix, failed, numerical, *args, checkpoint=saved.append
        )
        assert not result["available"]
        assert result["diagnostic_calls_started"] == 0
        assert len(saved) == 1

    def broken(*args):
        raise np.linalg.LinAlgError("deliberate failure")

    monkeypatch.setattr(controls, "_spectrum", broken)
    result = controls.prefix_information(model, fit, prefix, inference, numerical, 0.01)
    assert not result["available"] and result["status"] == "numerical_failure"
    assert result["diagnostic_calls_started"] == 1
    assert result["diagnostic_calls_completed"] == 0


def test_diagnostics_reject_mismatched_identity_before_work(fitted):
    _, model, fit, numerical = fitted
    prefix, inference = infer(fitted)
    altered = deepcopy(inference)
    altered["prefix_sha256"] = "0" * 64
    for function, args in (
        (controls.prefix_information, (0.01,)),
        (controls.conditional_linear_optimum, ()),
    ):
        with pytest.raises(ValueError, match="identity"):
            function(model, fit, prefix, altered, numerical, *args)


def test_analytic_reference_rejects_unrecognized_equations():
    payload = controls.problem("separated").model_dump(mode="json")
    payload["base_candidate"]["state_equations"][0]["rhs"] = "-theta*x+2*z+r"
    with pytest.raises(ValueError, match="declared multiple-root control"):
        controls._control_rates(kernel.PrefixFitProblem.model_validate(payload))


def test_near_confounded_control_has_a_weaker_resolved_direction():
    time = np.linspace(0, 1, 11)
    singular = {}
    for scenario, rates in controls.RATES.items():
        design = np.column_stack(
            [controls.response(time, rate, 0.35) for rate in rates]
        )
        values, _, rank, _ = controls._spectrum(design)
        singular[scenario] = values
        assert rank == (1 if scenario == "exact_confounded" else 2)
    assert singular["near_confounded"][-1] < singular["separated"][-1] / 10


def test_noisy_prefix_reference_respects_nonnegative_boundary(fitted):
    record, model, fit, numerical = fitted
    noisy = controls.fixture(record["scenario"], noise_sigma=0.1, repeat=5)
    row = PublicSplit.model_validate(noisy["validation"]).rows[0]
    prefix = kernel.make_prefix(row, kernel.PrefixPolicy(duration=0.2))
    inference = kernel.infer_prefix(model, fit, prefix, numerical)
    reference = controls.conditional_linear_optimum(
        model, fit, prefix, inference, numerical
    )
    assert reference["available"] and reference["objective_optimal"], reference
    assert min(reference["reference_roots"].values()) >= 0
    assert min(inference["roots"].values()) >= 0


def evaluated_windows(fitted):
    """Build real retained forecasts for a complete post-selection evaluation."""
    record, model, fit, numerical = fitted
    entries = []
    for row in PublicSplit.model_validate(record["validation"]).rows:
        policy = kernel.PrefixPolicy(duration=1.0)
        prefix = kernel.make_prefix(row, policy)
        inference = kernel.infer_prefix(model, fit, prefix, numerical)
        forecast = kernel.forecast(
            model,
            fit,
            inference,
            kernel.make_forecast_schedule(row, policy),
            numerical,
        )
        assert inference["fit_success"] and forecast["forecast_success"]
        entries.append(
            {
                "trajectory_id": row.trajectory_id,
                "attempts": [{"inference": inference, "forecast": forecast}],
                "chosen_start": 0,
            }
        )
    return [{"duration": 1.0, "trajectories": entries}]


def test_postselection_evaluator_preserves_choices_and_reports_exact_ambiguity(fitted):
    record, _, fit, _ = fitted
    windows = evaluated_windows(fitted)
    selection = {
        "status": "selected",
        "duration": 1.0,
        "common_start": 2.0,
        "immediate_duration": 0.5,
    }
    before = content_sha256(windows), content_sha256(selection), content_sha256(fit)
    result = controls.evaluate(record, fit, windows, [], selection)
    assert result["available"] and not result["truth_used_for_selection"]
    assert result["followup_status"] == "missing"
    assert result["followup"] == []
    assert abs(result["shared_theta_error"]) < 1e-6
    window = result["validation"][0]
    assert window["complete"] and window["rows_available"] == 4
    assert window["maximum_absolute_sum_root_error"] < 1e-6
    assert window["selected_duration"]
    assert max(window["mean_clean_scores"].values()) < 1e-14
    for row in window["rows"]:
        assert row["clean_scores"]["common"]["sample_count"] == 20
        assert row["clean_scores"]["immediate"]["sample_count"] == 5
        assert row["clean_scores"]["own_suffix"]["sample_count"] == 30
        assert row["analytic_max_absolute_difference"] < 1e-8
    if record["scenario"] == "exact_confounded":
        assert window["maximum_root_l2_error"] > 0.1
    else:
        assert window["maximum_root_l2_error"] < 1e-5
    assert before == (
        content_sha256(windows),
        content_sha256(selection),
        content_sha256(fit),
    )
    assert controls.evaluate(record, fit, windows, [], selection) == result
    json.dumps(result, allow_nan=False)

    missing = deepcopy(windows)
    missing[0]["trajectories"][0]["chosen_start"] = None
    failed = controls.evaluate(record, fit, missing, [], selection)["validation"][0]
    assert not failed["complete"] and failed["rows_available"] == 3
    assert failed["mean_clean_scores"] is None
    assert failed["maximum_root_l2_error"] is None


def test_distinct_inferred_roots_can_have_equal_exact_confounded_forecasts():
    record = controls.fixture("exact_confounded")
    model = kernel.PrefixFitProblem.model_validate(record["problem"])
    numerical = kernel.NumericalPolicy()
    fit = kernel.fit_training(
        model, PublicSplit.model_validate(record["training"]), numerical
    )
    first = infer((record, model, fit, numerical), start=(0.3, 1.3))[1]
    second = infer((record, model, fit, numerical), start=(1.3, 0.3))[1]
    assert abs(first["roots"]["init_z_value"] - second["roots"]["init_z_value"]) > 0.1
    assert sum(first["roots"].values()) == pytest.approx(
        sum(second["roots"].values()), abs=1e-7
    )
    time = np.linspace(0, 4, 41)
    outputs = [
        controls.analytic(
            time, 0.4, fit["shared_parameters"]["theta"], *inf["roots"].values(), 1, 1
        )
        for inf in (first, second)
    ]
    np.testing.assert_allclose(*outputs, atol=1e-7)

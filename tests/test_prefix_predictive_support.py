"""Prefix-only noise support, conservative forecast bounds, and failure receipts."""

import json
from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_multiroot_controls as controls
from autoformalism.fitting import prefix_predictive_support as support
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.public_fitting import PublicSplit, PublicTrajectory


@pytest.fixture(scope="module")
def fitted():
    results = {}
    numerical = kernel.NumericalPolicy()
    for scenario in controls.SCENARIOS:
        record = controls.fixture(scenario)
        model = kernel.PrefixFitProblem.model_validate(record["problem"])
        fit = kernel.fit_training(
            model, PublicSplit.model_validate(record["training"]), numerical
        )
        assert fit["fit_success"], fit
        results[scenario] = record, model, fit, numerical
    return results


def endpoint(fitted, scenario="separated", duration=1.0, noisy=False):
    """Fit a retained root endpoint using permitted prefix observations only."""
    record, model, fit, numerical = fitted[scenario]
    if noisy:
        record = controls.fixture(scenario, noise_sigma=0.01, repeat=5)
    row = PublicSplit.model_validate(record["validation"]).rows[0]
    prefix = kernel.make_prefix(row, kernel.PrefixPolicy(duration=duration))
    inference = kernel.infer_prefix(model, fit, prefix, numerical)
    assert inference["fit_success"], inference
    return model, fit, prefix, inference, numerical


def future(prefix):
    """Declare two target-free grids relative to the prefix cutoff."""
    return {
        "immediate": (prefix.cutoff + np.linspace(0.1, 0.5, 5)).tolist(),
        "common": np.linspace(2.1, 4, 20).tolist(),
    }


@pytest.mark.parametrize("scenario", controls.SCENARIOS)
def test_support_contains_sampled_nonnegative_roots_and_preserves_inputs(
    fitted, scenario
):
    args = endpoint(fitted, scenario, duration=0.5, noisy=True)
    model, fit, prefix, inference, _ = args
    grids = future(prefix)
    before = [content_sha256(value) for value in (*args, grids)]
    result = support.predictive_support(*args, 0.01, grids)
    assert result["available"] and result["all_horizons_finite"], result
    assert result["objective_rise_units"] == 4
    assert result["diagnostic_calls_started"] == 1
    assert result["diagnostic_calls_completed"] == 1
    assert result["conditional_reference_calls_completed"] == 1
    assert result["horizon_evaluations_started"] == 2
    assert result["horizon_evaluations_completed"] == 2
    assert result["integration_calls"] == 0
    assert not result["confidence_interval"] and not result["calibrated_probability"]
    assert result["conditional_on_frozen_shared_coefficients"]
    assert result["conditional_on_exact_initial_observation"]
    assert not result["shared_parameter_uncertainty_included"]
    assert not result["initial_observation_uncertainty_included"]
    assert not result["roots_replaced"] and not result["truth_used"]
    assert not result["future_observations_used"]
    assert result["normalized_sse_rise"] == pytest.approx(
        4 * (0.01 / fit["training_scales"]["y"]) ** 2
    )
    oracle, design, target, _ = controls._design(*args)
    singular, right, rank, _ = controls._spectrum(design)
    center = np.asarray(list(result["unconstrained_center_roots"].values()))
    retained = oracle.vector(inference["roots"])
    rng = np.random.default_rng(42)
    directions = rng.normal(size=(rank, 4000))
    directions /= np.linalg.norm(directions, axis=0)
    directions *= rng.uniform(0, 1, size=4000)
    roots = (
        center[:, None]
        + result["ellipsoid_radius"] * (right[:rank].T / singular[:rank]) @ directions
    )
    if rank < 2:
        roots += right[rank:].T @ rng.uniform(-2, 2, size=(2 - rank, 4000))
    supported = np.all(roots >= 0, axis=0) & (
        np.sum((design @ roots - target[:, None]) ** 2, axis=0)
        <= result["support_normalized_sse_limit"] + 1e-14
    )
    assert supported.sum() > 10
    for name, times in grids.items():
        future_design = (
            np.column_stack(
                [
                    controls.response(np.asarray(times), rate, oracle.frozen["theta"])
                    for rate in controls._control_rates(model)
                ]
            )
            / oracle.scales["y"]
        )
        differences = future_design @ (roots[:, supported] - retained[:, None])
        actual = np.sqrt(np.mean(differences**2, axis=0))
        horizon = result["horizons"][name]
        assert actual.max() <= horizon["normalized_rms_upper_bound"] + 1e-11
        assert horizon["normalized_rms_upper_bound"] == pytest.approx(
            horizon["retained_shift_contribution"]
            + horizon["noise_support_contribution"]
        )
    assert before == [content_sha256(value) for value in (*args, grids)]
    json.dumps(result, allow_nan=False)


def test_exact_confounding_keeps_prediction_finite_despite_root_ambiguity(fitted):
    args = endpoint(fitted, "exact_confounded", duration=0.2)
    result = support.predictive_support(*args, 0.01, future(args[2]))
    assert result["available"] and result["root_rank"] == 1
    assert result["root_ambiguous"] and not result["root_uniqueness_claimed"]
    for horizon in result["horizons"].values():
        assert horizon["status"] == "finite"
        assert horizon["normalized_rms_upper_bound"] > 0
        assert horizon["nullspace_dimension"] == 1
        assert horizon["nullspace_cancellation_certified"]
        assert horizon["nullspace_cancellation_basis"] == (
            "declared_equal_rates_and_identical_columns"
        )
        assert horizon["nullspace_leak_below_tolerance"]


@pytest.mark.parametrize(
    ("scenario", "duration", "rank"),
    [
        ("separated", 0, 0),
        ("separated", 0.1, 1),
        ("near_confounded", 0.1, 1),
        ("exact_confounded", 0, 0),
    ],
)
def test_future_prefix_null_directions_have_no_finite_relaxed_bound(
    fitted, scenario, duration, rank
):
    args = endpoint(fitted, scenario, duration=duration)
    result = support.predictive_support(*args, 0.01, future(args[2]))
    assert result["available"] and result["status"] == "complete", result
    assert result["root_rank"] == rank and result["root_ambiguous"]
    assert not result["all_horizons_finite"]
    for horizon in result["horizons"].values():
        assert horizon["status"] == "unbounded_relaxation"
        assert horizon["normalized_rms_upper_bound"] is None
        assert horizon["unbounded_domain_relaxation_only"]
        assert not horizon["nullspace_cancellation_certified"]
    json.dumps(result, allow_nan=False)


def test_small_numeric_null_leak_does_not_alone_certify_cancellation():
    rates = (1.0, 1.0 + 1e-14)
    design = np.array([[0.0, 0.0], [0.1, 0.1]])
    singular, right, rank, _ = controls._spectrum(design)
    horizon = support._horizon_bound(
        [0.2, 0.3],
        0,
        0.35,
        rates,
        1,
        right,
        singular,
        rank,
        np.ones(2),
        np.ones(2),
        0.02,
        design,
    )
    assert horizon["nullspace_leak_below_tolerance"]
    assert horizon["status"] == "unbounded_relaxation"


def test_noisy_short_near_confounded_prefix_has_weaker_future_support(fitted):
    results = {}
    for scenario in ("separated", "near_confounded"):
        args = endpoint(fitted, scenario, duration=0.2, noisy=True)
        results[scenario] = support.predictive_support(*args, 0.01, future(args[2]))
        assert results[scenario]["available"]
    near, separated = results["near_confounded"], results["separated"]
    assert near["singular_values"][-1] < separated["singular_values"][-1] / 10
    assert near["horizons"]["common"]["normalized_rms_upper_bound"] > (
        1.5 * separated["horizons"]["common"]["normalized_rms_upper_bound"]
    )


def test_zero_noise_and_scaling_with_declared_noise_units(fitted):
    args = endpoint(fitted)
    grids = future(args[2])
    zero = support.predictive_support(*args, 0, grids)
    one = support.predictive_support(*args, 0.01, grids, objective_rise_units=1)
    four = support.predictive_support(*args, 0.01, grids)
    assert zero["available"] and zero["normalized_sse_rise"] == 0
    for name in grids:
        assert zero["horizons"][name]["normalized_rms_upper_bound"] < 1e-5
        assert four["horizons"][name]["noise_support_contribution"] == pytest.approx(
            2 * one["horizons"][name]["noise_support_contribution"], rel=1e-8
        )


def test_nonnegative_boundary_gap_is_retained_in_relaxed_ellipsoid(fitted):
    args = endpoint(fitted, duration=0.2, noisy=True)
    result = support.predictive_support(*args, 0.01, future(args[2]))
    assert result["available"]
    assert min(result["unconstrained_center_roots"].values()) < 0
    assert min(result["conditional_reference"]["reference_roots"].values()) == 0
    assert result["domain_relaxation_normalized_sse_gap"] > 0
    assert result["ellipsoid_radius"] ** 2 == pytest.approx(
        result["domain_relaxation_normalized_sse_gap"] + result["normalized_sse_rise"]
    )


def test_frozen_shared_values_and_scales_cannot_differ(fitted):
    args = list(endpoint(fitted))
    original = args[3]
    for field in ("shared_parameters", "training_scales"):
        args[3] = deepcopy(original)
        args[3][field] = dict.fromkeys(args[3][field], 0.2)
        with pytest.raises(ValueError, match="frozen"):
            support.predictive_support(*args, 0.01, future(args[2]))


def test_unavailable_reference_cannot_qualify_support(fitted, monkeypatch):
    args = endpoint(fitted)
    unavailable = {
        "available": False,
        "status": "budget_exhausted",
        "diagnostic_calls_completed": 0,
    }
    monkeypatch.setattr(
        controls, "conditional_linear_optimum", lambda *args: unavailable
    )
    result = support.predictive_support(*args, 0.01, future(args[2]))
    assert not result["available"] and result["status"] == "budget_exhausted"
    assert result["conditional_reference"] == unavailable
    assert result["conditional_reference_calls_started"] == 1
    assert result["conditional_reference_calls_completed"] == 0
    assert result["horizon_evaluations_started"] == 0


def test_future_observations_and_evaluator_truth_cannot_change_support(fitted):
    args = endpoint(fitted)
    record, model, fit, numerical = fitted["separated"]
    altered = deepcopy(record)
    altered["evaluation"] = {"inaccessible": "not accepted by diagnostic API"}
    altered["followup"] = None
    row = altered["validation"]["rows"][0]
    row["targets"]["y"][11:] = [1e90] * 30
    poisoned_prefix = kernel.make_prefix(
        PublicTrajectory.model_validate(row), args[2].policy
    )
    assert poisoned_prefix == args[2]
    normal = support.predictive_support(*args, 0.01, future(args[2]))
    changed = support.predictive_support(
        model, fit, poisoned_prefix, args[3], numerical, 0.01, future(args[2])
    )
    assert normal["horizons"] == changed["horizons"]
    assert normal["normalized_sse_rise"] == changed["normalized_sse_rise"]
    assert normal["unconstrained_center_roots"] == changed["unconstrained_center_roots"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"known_noise_sigma": True},
        {"known_noise_sigma": -1},
        {"known_noise_sigma": np.inf},
        {"known_noise_sigma": np.nan},
        {"objective_rise_units": 0},
        {"objective_rise_units": -1},
        {"objective_rise_units": True},
        {"objective_rise_units": np.nan},
        {"maximum_seconds": 0},
        {"maximum_seconds": False},
        {"maximum_seconds": np.inf},
        {"horizon_times": {}},
        {"horizon_times": {"common": []}},
        {"horizon_times": {"common": [1.0]}},
        {"horizon_times": {"common": [2.0, 2.0]}},
        {"horizon_times": {"common": [3.0, 2.0]}},
        {"horizon_times": {"common": [np.inf]}},
        {"horizon_times": {"common": [True]}},
        {"horizon_times": {"common": [np.nan]}},
        {"horizon_times": {"common": [2, {"target": 1}]}},
        {"horizon_times": {"": [2]}},
        {"horizon_times": {"common": (2, 3)}},
    ],
)
def test_invalid_inputs_are_rejected_before_numerical_work(fitted, kwargs):
    args = endpoint(fitted)
    options = {"known_noise_sigma": 0.01, "horizon_times": future(args[2])} | kwargs
    with pytest.raises(ValueError):
        support.predictive_support(*args, **options)


def test_untyped_model_prefix_and_numerical_policy_rejected(fitted):
    args = list(endpoint(fitted))
    for index in (0, 2, 4):
        untyped = list(args)
        untyped[index] = untyped[index].model_dump(mode="json")
        with pytest.raises(TypeError, match="typed"):
            support.predictive_support(*untyped, 0.01, future(args[2]))


@pytest.mark.parametrize("field", ["problem_sha256", "fit_sha256", "prefix_sha256"])
def test_identity_mismatch_fails_before_any_reference_call(fitted, monkeypatch, field):
    args = list(endpoint(fitted))
    args[3] = deepcopy(args[3])
    args[3][field] = "0" * 64

    def forbidden(*args, **kwargs):
        raise AssertionError("identity must be checked before numerical work")

    monkeypatch.setattr(controls, "conditional_linear_optimum", forbidden)
    with pytest.raises(ValueError, match="identity"):
        support.predictive_support(*args, 0.01, future(args[2]))


def test_failed_inference_and_bad_retained_objective_stay_unavailable(fitted):
    args = list(endpoint(fitted))
    args[3] = deepcopy(args[3])
    args[3].update(fit_success=False, roots=None)
    failed = support.predictive_support(*args, 0.01, future(args[2]))
    assert not failed["available"] and failed["status"] == "inference_unavailable"
    assert failed["diagnostic_calls_started"] == 0
    assert failed["diagnostic_wall_seconds"] >= 0
    args[3].update(fit_success=True, roots=dict.fromkeys(controls.ROOT_NAMES, 4.0))
    poor = support.predictive_support(*args, 0.01, future(args[2]))
    assert not poor["available"]
    assert poor["status"] == "retained_objective_not_optimal"
    assert poor["conditional_reference_calls_completed"] == 1
    assert poor["horizon_evaluations_started"] == 0


def test_numerical_failure_retains_reference_receipt(fitted, monkeypatch):
    args = endpoint(fitted)

    def broken(*args, **kwargs):
        raise np.linalg.LinAlgError("deliberate horizon failure")

    monkeypatch.setattr(support, "_horizon_bound", broken)
    result = support.predictive_support(*args, 0.01, future(args[2]))
    assert not result["available"] and result["status"] == "numerical_failure"
    assert result["diagnostic_calls_started"] == 1
    assert result["diagnostic_calls_completed"] == 0
    assert result["conditional_reference_calls_completed"] == 1
    assert result["horizon_evaluations_started"] == 1
    assert result["horizon_evaluations_completed"] == 0
    json.dumps(result, allow_nan=False)


def test_nonfinite_intermediate_cannot_qualify(fitted):
    args = endpoint(fitted)
    result = support.predictive_support(*args, 1e308, future(args[2]))
    assert not result["available"] and result["status"] == "numerical_failure"
    json.dumps(result, allow_nan=False)


def test_expiry_before_work_does_not_claim_a_started_call(fitted, monkeypatch):
    args = endpoint(fitted)
    ticks = iter((10, 12, 13))
    monkeypatch.setattr(support, "monotonic", lambda: next(ticks))
    result = support.predictive_support(*args, 0.01, future(args[2]), maximum_seconds=1)
    assert not result["available"] and result["status"] == "budget_exhausted"
    assert result["diagnostic_calls_started"] == 0
    assert result["horizon_evaluations_started"] == 0
    assert result["diagnostic_wall_seconds"] == 3


def test_expiry_during_second_horizon_preserves_first(fitted, monkeypatch):
    args = endpoint(fitted)
    clock = [10.0]
    real_bound = support._horizon_bound

    def slow_second(times, *args, **kwargs):
        result = real_bound(times, *args, **kwargs)
        if len(times) == 20:
            clock[0] += 2
        return result

    monkeypatch.setattr(support, "monotonic", lambda: clock[0])
    monkeypatch.setattr(support, "_horizon_bound", slow_second)
    result = support.predictive_support(*args, 0.01, future(args[2]), maximum_seconds=1)
    assert not result["available"] and result["status"] == "budget_exhausted"
    assert result["horizon_evaluations_started"] == 2
    assert result["horizon_evaluations_completed"] == 1
    assert result["horizons"]["immediate"]["status"] == "finite"
    assert result["horizons"]["common"]["status"] == "unavailable"
    assert result["diagnostic_calls_completed"] == 0
    assert result["diagnostic_wall_seconds"] == 2

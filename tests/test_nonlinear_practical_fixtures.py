"""Nonlinear public acquisition provenance and isolated exact-response evaluation."""

from copy import deepcopy

import numpy as np
import pytest

from autoformalism.fitting import nonlinear_practical_fixtures as fixtures
from autoformalism.fitting import sensor_drift
from autoformalism.fitting.robust_measurement import Loss


def test_frozen_factorial_and_generic_unknown_decay_shared_initial_profile():
    first = fixtures.make_inputs()
    assert first == fixtures.make_inputs()
    assert len(first["problems"]) == 18
    fixtures.verify_raw_artifacts(first)
    problem, evaluator = fixtures.make()
    assert problem.start == fixtures.START | {"sensor_drift": 0.0}
    assert problem.base.initial_rules[0].shared_coordinate == "x0"
    assert len(evaluator["observations"]) == 51
    assert "truth" not in problem.model_dump()
    oracle, profile, route = sensor_drift.build(problem, Loss(kind="gaussian"))
    assert profile is not None
    assert tuple(oracle.names[index] for index in profile.outer) == ("k",)
    assert set(route["certificate"]["inner_names"]) == {"g", "x0", "sensor_drift"}
    assert route["certificate"]["profiled_initial_parameters"] == ["x0"]
    assert set(oracle.names) == set(fixtures.PARAMETERS)


@pytest.mark.parametrize("seed", fixtures.SEEDS)
def test_noise_is_paired_across_separations_and_scales(seed):
    baseline = None
    for separation in fixtures.SEPARATIONS:
        for sigma in fixtures.SIGMAS:
            problem, evaluator = fixtures.make(separation, sigma, seed)
            noise = np.array(
                [
                    (row["value"] - row["clean_measured"]) / sigma
                    for row in evaluator["observations"]
                ]
            )
            if baseline is None:
                baseline = noise
            np.testing.assert_allclose(noise, baseline, atol=2e-12, rtol=2e-12)
            assert set(problem.manifest.noise.sigma) == {sigma}


def test_zero_separation_is_not_assumed_to_be_a_gain_drift_ridge():
    _, evaluator = fixtures.make(0, 0.01, 0)
    truth = evaluator["truth"]
    actual = fixtures.evaluate(evaluator, truth)
    shifted = fixtures.evaluate(evaluator, truth | {"g": 2.3, "sensor_drift": -0.94})
    assert actual["clean_measured_prediction_rmse_evaluator_only"] < 1e-12
    assert shifted["clean_measured_prediction_rmse_evaluator_only"] > 1
    assert not actual["physical_recovery_identified_by_objective"]


@pytest.fixture
def one_acquisition():
    problem, evaluator, artifacts = fixtures._build(0.02, 0.2, 0)
    key = fixtures.case_key(0.02, 0.2, 0)
    return {
        "problems": {key: problem.model_dump(mode="json")},
        "evaluation": {key: evaluator},
        "raw_artifacts": artifacts,
    }


@pytest.mark.parametrize("field", ["value", "sigma", "forcing", "time"])
def test_public_readings_and_evaluator_observation_tampering_fail_closed(
    one_acquisition, field
):
    next(iter(one_acquisition["evaluation"].values()))["observations"][0][field] += 0.01
    with pytest.raises(ValueError, match="evaluator observation differs"):
        fixtures.verify_raw_artifacts(one_acquisition)


@pytest.mark.parametrize("field", ["raw", "k_bounds", "k_start", "k_scale", "initial"])
def test_imported_nonlinear_design_must_match_the_frozen_contract(
    one_acquisition, field
):
    base = next(iter(one_acquisition["problems"].values()))["base"]
    if field == "raw":
        base["training"]["rows"][0]["readings"][0]["raw_value"] += 1
    elif field == "k_bounds":
        base["request"]["base_candidate"]["parameters"][1]["bounds"]["lower"] = 0.1
    elif field == "k_start":
        base["start"]["k"] = 0.65
    elif field == "k_scale":
        base["coordinates"]["parameters"]["k"]["scale"] = 2
    else:
        base["initial_rules"][0]["shared_coordinate"] = "g"
    with pytest.raises(ValueError):
        fixtures.verify_raw_artifacts(one_acquisition)


def test_consistently_rehashed_calibration_change_rejects_wrong_sigma(one_acquisition):
    base = next(iter(one_acquisition["problems"].values()))["base"]
    source = next(
        row for row in base["manifest"]["sources"] if row["id"] == "calibration"
    )
    changed = deepcopy(one_acquisition["raw_artifacts"][source["artifact_sha256"]])
    changed["sigma"][0] *= 2
    source["artifact_sha256"] = fixtures.public.content_sha256(changed)
    one_acquisition["raw_artifacts"][source["artifact_sha256"]] = changed
    with pytest.raises(ValueError, match="calibration source differs"):
        fixtures.verify_raw_artifacts(one_acquisition)


def test_truth_is_not_used_by_public_verification(one_acquisition):
    evaluator = next(iter(one_acquisition["evaluation"].values()))
    evaluator["truth"] = "forbidden"
    for row in evaluator["observations"]:
        row["clean_physical"] = row["clean_measured"] = "forbidden"
    fixtures.verify_raw_artifacts(one_acquisition)


def test_unknown_acquisitions_and_invalid_parameters_fail_closed():
    with pytest.raises(ValueError, match="unknown"):
        fixtures.make(0.1, 0.01, 0)
    _, evaluator = fixtures.make()
    for invalid in ({"g": 1}, evaluator["truth"] | {"k": float("nan")}):
        with pytest.raises(ValueError, match="names/values"):
            fixtures.evaluate(evaluator, invalid)
    assert fixtures.evaluate(evaluator, None)["status"] == "no_parameters"

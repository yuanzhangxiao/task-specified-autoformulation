"""M39 fixed acquisitions, matched noise and exact Gaussian evaluator arithmetic."""

from copy import deepcopy

import numpy as np
import pytest

from autoformalism.fitting import practical_uncertainty_fixtures as fixtures


def test_full_roster_is_frozen_and_raw_provenance_is_exact():
    first, second = fixtures.make_inputs(), fixtures.make_inputs()
    assert first == second
    assert len(first["problems"]) == 75
    fixtures.verify_raw_artifacts(first)
    for key, payload in first["problems"].items():
        assert set(payload) == {"protocol", "base", "drift"}
        assert "truth" not in payload and "evaluation" not in payload
        assert payload["drift"]["lower"] == -2 and payload["drift"]["upper"] == 2
        assert payload["base"]["initial_rules"][0]["mode"] == "shared_value"
        assert payload["base"]["initial_rules"][0]["shared_coordinate"] == "x0"
        evaluator = first["evaluation"][key]
        assert len(evaluator["observations"]) == 51
        for row in payload["base"]["training"]["rows"]:
            assert len(row["clock_values"]) == 17
            assert row["clock_values"][-1] == 340
        assert evaluator["acquisition_sha256"] in first["raw_artifacts"]


@pytest.mark.parametrize("seed", fixtures.SEEDS)
def test_standardized_noise_draws_match_across_scales_and_separations(seed):
    baseline = None
    for separation in fixtures.SEPARATIONS:
        for sigma in fixtures.SIGMAS:
            problem, evaluator = fixtures.make(separation, sigma, seed)
            noise = np.array(
                [
                    (row["value"] - row["clean_measured"]) / row["sigma"]
                    for row in evaluator["observations"]
                ]
            )
            if baseline is None:
                baseline = noise
            np.testing.assert_allclose(noise, baseline, atol=2e-12, rtol=2e-12)
            assert set(problem.manifest.noise.sigma) == {sigma}
            assert problem.base.start == {"g": 0.45, "x0": 0.35}
            assert problem.drift.start == problem.drift.incumbent == 0
            assert problem.manifest.channels[0].gain == 0.01
            assert problem.manifest.time_bases[0].factor == 1 / 60


@pytest.mark.parametrize("separation", fixtures.SEPARATIONS)
@pytest.mark.parametrize("sigma", fixtures.SIGMAS)
@pytest.mark.parametrize("seed", fixtures.SEEDS)
def test_independent_bounded_reference_checks_objective_without_parameter_uniqueness(
    separation, sigma, seed
):
    _, evaluator = fixtures.make(separation, sigma, seed)
    reference = fixtures.analytic_optimum(evaluator)
    result = fixtures.evaluate(evaluator, reference["analytic_parameters"])
    assert result["analytic_objective_passed"]
    assert reference["analytic_projected_gradient_maximum"] <= 1e-5
    assert reference["analytic_design_rank"] == (2 if separation == 0 else 3)
    assert reference["rank_deficient_optimum"] == (separation == 0)
    assert not reference["unique_parameters_assessed"]
    assert not result["physical_recovery_identified_by_objective"]
    assert result["confidence_probability"] is None
    assert result["data_fault_attribution"] is None


@pytest.mark.parametrize("separation", fixtures.SEPARATIONS)
@pytest.mark.parametrize("sigma", fixtures.SIGMAS)
def test_exact_conditional_information_has_declared_scale_dependence(separation, sigma):
    _, evaluator = fixtures.make(separation, sigma, 0)
    matrix, _ = fixtures._design(evaluator)
    nuisance = matrix[:, 1:] / sigma
    gain = matrix[:, 0] / sigma
    remaining = gain - nuisance @ np.linalg.lstsq(nuisance, gain, rcond=None)[0]
    assert remaining @ remaining == pytest.approx(
        187 * separation**2 / sigma**2, abs=1e-12, rel=1e-12
    )


def test_rank_deficient_gain_drift_ridge_can_change_physical_truth_error():
    _, evaluator = fixtures.make(0, 0.05, 0)
    a = {"g": 1.3, "x0": 0.4, "sensor_drift": 0.06}
    b = {"g": 2.3, "x0": 0.4, "sensor_drift": -0.94}
    first, second = fixtures.evaluate(evaluator, a), fixtures.evaluate(evaluator, b)
    assert first["independent_analytic_endpoint_objective"] == pytest.approx(
        second["independent_analytic_endpoint_objective"]
    )
    assert second["clean_measured_prediction_rmse_evaluator_only"] < 1e-12
    assert second["clean_physical_prediction_rmse_evaluator_only"] > 2
    assert not second["physical_recovery_identified_by_objective"]


def test_reference_does_not_consult_truth_or_clean_companions():
    _, evaluator = fixtures.make(0.02, 0.2, 3)
    expected = fixtures.analytic_optimum(evaluator)
    changed = deepcopy(evaluator)
    changed["truth"] = "forbidden"
    for row in changed["observations"]:
        row["clean_physical"] = row["clean_measured"] = "forbidden"
    assert fixtures.analytic_optimum(changed) == expected


def test_malformed_inputs_and_raw_tampering_fail_closed():
    data = fixtures.make_inputs()
    first = next(iter(data["problems"].values()))
    first["base"]["training"]["rows"][0]["readings"][0]["raw_value"] += 1
    with pytest.raises(ValueError, match="reading differs"):
        fixtures.verify_raw_artifacts(data)
    with pytest.raises(ValueError, match="unknown"):
        fixtures.make(0.33, 0.01, 0)
    _, evaluator = fixtures.make()
    with pytest.raises(ValueError, match="names/values"):
        fixtures.evaluate(evaluator, {"g": 1, "x0": 0.4})
    with pytest.raises(ValueError, match="names/values"):
        fixtures.evaluate(evaluator, {"g": float("nan"), "x0": 0.4, "sensor_drift": 0})


@pytest.fixture
def one_acquisition():
    problem, evaluator, artifacts = fixtures._build(0.1, 0.05, 0)
    key = fixtures.case_key(0.1, 0.05, 0)
    return {
        "problems": {key: problem.model_dump(mode="json")},
        "evaluation": {key: evaluator},
        "raw_artifacts": artifacts,
    }


@pytest.mark.parametrize(
    "field", ["value", "time", "forcing", "sigma", "reading_id", "trajectory"]
)
def test_evaluator_observations_must_match_actual_public_readings(
    one_acquisition, field
):
    evaluator = next(iter(one_acquisition["evaluation"].values()))
    row = evaluator["observations"][0]
    row[field] = "wrong" if isinstance(row[field], str) else row[field] + 0.01
    with pytest.raises(ValueError, match="evaluator observation differs"):
        fixtures.verify_raw_artifacts(one_acquisition)


@pytest.mark.parametrize(
    "mutation",
    [
        "sigma",
        "bounds",
        "drift_bounds",
        "time",
        "calibration",
        "initial",
        "coordinates",
    ],
)
def test_imported_public_design_cannot_disagree_with_factorial_metadata(
    one_acquisition, mutation
):
    problem = next(iter(one_acquisition["problems"].values()))
    base = problem["base"]
    if mutation == "sigma":
        base["manifest"]["noise"]["sigma"][0] *= 2
    elif mutation == "bounds":
        base["request"]["base_candidate"]["parameters"][0]["bounds"]["upper"] = 4
    elif mutation == "drift_bounds":
        problem["drift"]["upper"] = 3
    elif mutation == "time":
        base["manifest"]["time_bases"][0]["factor"] = 1 / 30
    elif mutation == "calibration":
        base["manifest"]["channels"][0]["gain"] = 0.02
    elif mutation == "initial":
        base["initial_rules"][0]["shared_coordinate"] = "g"
    else:
        base["coordinates"]["parameters"]["g"]["scale"] = 2
    with pytest.raises(ValueError, match="practical uncertainty"):
        fixtures.verify_raw_artifacts(one_acquisition)


def test_consistently_rehashed_forcing_change_still_rejects_wrong_design(
    one_acquisition,
):
    payload = next(iter(one_acquisition["problems"].values()))["base"]
    evaluator = next(iter(one_acquisition["evaluation"].values()))
    old = evaluator["acquisition_sha256"]
    raw = deepcopy(one_acquisition["raw_artifacts"][old])
    for record in raw["records"]:
        if record["channel"] == "u":
            record["raw_value"] += 0.01
    digest = fixtures.public.content_sha256(raw)
    one_acquisition["raw_artifacts"][digest] = raw
    evaluator["acquisition_sha256"] = digest
    payload["manifest"]["artifacts"][0]["sha256"] = digest
    next(
        source
        for source in payload["manifest"]["sources"]
        if source["id"] == "acquisition"
    )["artifact_sha256"] = digest
    for row in payload["training"]["rows"]:
        for reading in row["readings"]:
            reading["artifact_sha256"] = digest
            if reading["channel"] == "u":
                reading["raw_value"] += 0.01
    with pytest.raises(ValueError, match="forcing design differs"):
        fixtures.verify_raw_artifacts(one_acquisition)


def test_hash_valid_calibration_artifact_still_must_match_public_sigma(one_acquisition):
    payload = next(iter(one_acquisition["problems"].values()))["base"]
    source = next(
        source
        for source in payload["manifest"]["sources"]
        if source["id"] == "calibration"
    )
    calibration = deepcopy(one_acquisition["raw_artifacts"][source["artifact_sha256"]])
    calibration["sigma"][0] *= 2
    digest = fixtures.public.content_sha256(calibration)
    one_acquisition["raw_artifacts"][digest] = calibration
    source["artifact_sha256"] = digest
    with pytest.raises(ValueError, match="calibration source differs"):
        fixtures.verify_raw_artifacts(one_acquisition)


def test_generating_truth_remains_separate_from_public_design_checks(one_acquisition):
    evaluator = next(iter(one_acquisition["evaluation"].values()))
    evaluator["truth"] = "not consulted"
    for row in evaluator["observations"]:
        row["clean_physical"] = row["clean_measured"] = "not consulted"
    fixtures.verify_raw_artifacts(one_acquisition)

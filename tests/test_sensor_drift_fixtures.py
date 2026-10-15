"""M36 raw provenance, matched hypotheses, affine references and ambiguity."""

from copy import deepcopy

import numpy as np
import pytest

from autoformalism.fitting import sensor_drift_fixtures as fixtures


def test_acquisitions_are_deterministic_and_have_retained_original_bytes():
    first, second = fixtures.make_inputs(), fixtures.make_inputs()
    assert first == second
    assert len(first["problems"]) == 18
    fixtures.verify_raw_artifacts(first)
    for key, evaluator in first["evaluation"].items():
        assert evaluator["original_acquisition_sha256"] in first["raw_artifacts"]
        assert evaluator["fitting_acquisition_sha256"] in first["raw_artifacts"]
        payload = first["problems"][key]
        assert set(payload) == {"protocol", "base", "drift"}
        assert (
            not {"truth", "evaluation", "ambiguity_witness_evaluator_only"}
            & payload.keys()
        )
        assert (
            payload["base"]["request"]["base_candidate"]["candidate_id"]
            == "m36_fixed_physical_contract"
        )


@pytest.mark.parametrize("seed", fixtures.SEEDS)
def test_clean_and_drift_exact_share_physical_noise_and_public_hypothesis(seed):
    clean, first = fixtures.make("clean_exact", seed)
    drifted, second = fixtures.make("drift_exact", seed)
    assert clean.drift == drifted.drift
    assert clean.base.request == drifted.base.request
    assert clean.base.start == drifted.base.start
    assert clean.base.manifest.noise == drifted.base.manifest.noise
    assert clean.base.initial_rules == drifted.base.initial_rules
    for a, b in zip(first["observations"], second["observations"], strict=True):
        assert a["clean_physical"] == b["clean_physical"]
        assert b["value"] - a["value"] == pytest.approx(0.06 * a["time"])
        assert b["value"] - b["clean_measured"] == pytest.approx(
            a["value"] - a["clean_measured"]
        )


@pytest.mark.parametrize("case", fixtures.CASES)
@pytest.mark.parametrize("seed", fixtures.SEEDS)
@pytest.mark.parametrize("hypothesis", ("no_drift", "shared_drift"))
@pytest.mark.parametrize("kind", ("gaussian", "huber"))
def test_own_loss_affine_optimum_and_exact_endpoint_replay(
    case, seed, hypothesis, kind
):
    problem, evaluator = fixtures.make(case, seed)
    reference = fixtures.analytic_optimum(evaluator, hypothesis, kind)
    start = problem.start if hypothesis == "shared_drift" else problem.base.start
    endpoint = reference["analytic_parameters"] or start
    result = fixtures.evaluate(evaluator, endpoint, hypothesis, kind)
    if case == "drift_nonlinear_decay":
        assert result["analytic_objective_passed"] is None
    else:
        assert result["analytic_objective_passed"] is True
        assert reference["analytic_projected_gradient_maximum"] <= 1e-5
        assert reference["unique_parameters_assessed"] is False
        assert reference["rank_deficient_optimum"] == (
            case == "confounded_constant_input" and hypothesis == "shared_drift"
        )
    assert result["observation_hypothesis_selected"] is None
    assert result["data_fault_attribution"] is None
    assert result["analysis_residual_unit"] == "m"
    assert np.isfinite(result["analysis_observed_nmse_descriptive_only"])


def test_equal_measured_predictions_do_not_recover_physical_gain():
    _, evaluator = fixtures.make("confounded_constant_input", 0)
    first, second = evaluator["ambiguity_witness_evaluator_only"]
    physical_a, measured_a = fixtures.predictions(evaluator, first, "shared_drift")
    physical_b, measured_b = fixtures.predictions(evaluator, second, "shared_drift")
    assert np.max(abs(measured_a - measured_b)) < 1e-12
    assert np.max(abs(physical_a - physical_b)) > 0.5
    a = fixtures.evaluate(evaluator, first, "shared_drift", "huber")
    b = fixtures.evaluate(evaluator, second, "shared_drift", "huber")
    assert a["independent_analytic_endpoint_objective"] == pytest.approx(
        b["independent_analytic_endpoint_objective"]
    )
    assert a["clean_measured_prediction_rmse_evaluator_only"] < 1e-12
    assert b["clean_physical_prediction_rmse_evaluator_only"] > 0.3
    assert not b["physical_recovery_identified_by_objective"]


def test_varied_input_separates_the_same_physical_and_sensor_directions():
    _, evaluator = fixtures.make("resolved_varied_input", 0)
    first = fixtures.predictions(
        evaluator, {"g": 1.3, "sensor_drift": 0.06}, "shared_drift"
    )[1]
    second = fixtures.predictions(
        evaluator, {"g": 1.5, "sensor_drift": -0.08}, "shared_drift"
    )[1]
    assert np.max(abs(first - second)) > 0.4
    reference = fixtures.analytic_optimum(evaluator, "shared_drift", "gaussian")
    assert reference["analytic_design_rank"] == 2


def test_affine_reference_never_uses_clean_truth_or_ambiguity_witnesses():
    _, evaluator = fixtures.make("drift_shared_initial", 0)
    expected = fixtures.analytic_optimum(evaluator, "shared_drift", "huber")
    changed = deepcopy(evaluator)
    changed["truth"] = "forbidden"
    changed["ambiguity_witness_evaluator_only"] = "forbidden"
    for row in changed["observations"]:
        row["clean_physical"] = row["clean_measured"] = "forbidden"
    assert fixtures.analytic_optimum(changed, "shared_drift", "huber") == expected


def test_raw_tampering_and_malformed_requests_fail_before_fitting():
    data = fixtures.make_inputs()
    first = next(iter(data["problems"].values()))
    first["base"]["training"]["rows"][0]["readings"][0]["raw_value"] += 1
    with pytest.raises(ValueError, match="reading differs"):
        fixtures.verify_raw_artifacts(data)
    with pytest.raises(ValueError, match="unknown"):
        fixtures.make("wrong", 0)
    _, evaluator = fixtures.make()
    with pytest.raises(ValueError, match="parameter names/values"):
        fixtures.evaluate(evaluator, {"g": float("nan")}, "no_drift", "gaussian")
    with pytest.raises(ValueError, match="parameter names/values"):
        fixtures.evaluate(evaluator, {"sensor_drift": 0.0}, "shared_drift", "gaussian")

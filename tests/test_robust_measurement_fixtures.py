"""M35 exact acquisition pairing, evaluator isolation and loss-specific references."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from autoformalism.fitting import robust_measurement_fixtures as fixtures


def test_frozen_inputs_have_original_raw_provenance_and_deterministic_hashes():
    first, second = fixtures.make_inputs(), fixtures.make_inputs()
    assert first == second
    assert len(first["problems"]) == 24
    fixtures.verify_raw_artifacts(first)
    for name, evaluator in first["evaluation"].items():
        problem = first["problems"][name]
        assert evaluator["original_acquisition_sha256"] in first["raw_artifacts"]
        assert evaluator["fitting_acquisition_sha256"] in first["raw_artifacts"]
        assert (
            not {"truth", "clean", "corruption_evaluator_only", "evaluation"}
            & problem.keys()
        )
        assert (
            problem["request"]["base_candidate"]["candidate_id"] == "m35_fixed_contract"
        )


@pytest.mark.parametrize("clean,contaminated", fixtures.PAIRS)
@pytest.mark.parametrize("seed", fixtures.SEEDS)
def test_corruption_changes_only_four_target_readings_and_retains_originals(
    clean, contaminated, seed
):
    before, clean_evaluator = fixtures.make(clean, seed)
    after, evaluator = fixtures.make(contaminated, seed)
    assert before.manifest.noise == after.manifest.noise
    assert before.initial_rules == after.initial_rules
    assert before.start == after.start
    assert before.incumbent == after.incumbent
    assert before.request == after.request
    assert (
        clean_evaluator["original_acquisition_sha256"]
        == evaluator["original_acquisition_sha256"]
    )
    assert len(evaluator["corruption_evaluator_only"]) == 4
    assert clean_evaluator["corruption_evaluator_only"] == []
    changes = []
    for old_row, new_row in zip(before.training.rows, after.training.rows, strict=True):
        for old, new in zip(old_row.readings, new_row.readings, strict=True):
            if old.raw_value != new.raw_value:
                assert old.channel == "y"
                changes.append(old.id)
    assert set(changes) == {
        r["reading_id"] for r in evaluator["corruption_evaluator_only"]
    }
    for original, modified in zip(
        clean_evaluator["observations"], evaluator["observations"], strict=True
    ):
        assert original["clean"] == modified["clean"]
        assert original["sigma"] == modified["sigma"]


@pytest.mark.parametrize("case", fixtures.CASES)
@pytest.mark.parametrize("seed", fixtures.SEEDS)
@pytest.mark.parametrize("kind", ("gaussian", "huber"))
def test_independent_references_check_declared_candidate_and_loss(case, seed, kind):
    problem, evaluator = fixtures.make(case, seed)
    reference = fixtures.analytic_optimum(evaluator, kind)
    endpoint = reference["analytic_parameters"] or problem.start
    value = fixtures.evaluate(evaluator, endpoint, kind)
    if case.startswith("nonlinear"):
        assert reference["analytic_objective"] is None
        assert value["analytic_objective_passed"] is None
    else:
        assert value["analytic_objective_passed"] is True
        assert reference["analytic_projected_gradient_maximum"] <= 1e-5
    assert value["data_fault_attribution"] is None
    assert value["confidence_probability"] is None
    assert np.isfinite(value["analysis_observed_nmse_descriptive_only"])
    if case == "wrong_gain_bound":
        assert reference["analytic_active_gain_bound"]
        assert reference["analytic_parameters"]["g"] == pytest.approx(0.8)
        assert value["clean_prediction_rmse_evaluator_only"] > 0.4
    if case == "wrong_dynamics":
        assert value["clean_prediction_rmse_evaluator_only"] > 0.05


@pytest.mark.parametrize("clean,contaminated", fixtures.PAIRS[:2])
def test_huber_limits_frozen_sparse_contamination_bias(clean, contaminated):
    _, evaluator = fixtures.make(contaminated, 0)
    outcomes = {
        kind: fixtures.evaluate(
            evaluator,
            fixtures.analytic_optimum(evaluator, kind)["analytic_parameters"],
            kind,
        )
        for kind in ("gaussian", "huber")
    }
    assert (
        outcomes["huber"]["clean_prediction_rmse_evaluator_only"]
        < outcomes["gaussian"]["clean_prediction_rmse_evaluator_only"]
    )
    assert max(outcomes["huber"]["robust_weights"]) == 1
    assert min(outcomes["huber"]["robust_weights"]) < 0.1
    assert min(outcomes["gaussian"]["robust_weights"]) == 1


def test_wrong_dynamics_can_be_downweighted_without_fault_attribution():
    _, evaluator = fixtures.make("wrong_dynamics", 0)
    fit = fixtures.analytic_optimum(evaluator, "huber")
    value = fixtures.evaluate(evaluator, fit["analytic_parameters"], "huber")
    assert evaluator["corruption_evaluator_only"] == []
    assert value["downweighted_reading_ids"]
    assert value["data_fault_attribution"] is None
    assert value["analysis_residual_rmse"] > 0.05


def test_reference_does_not_use_clean_truth_or_corruption_labels():
    _, evaluator = fixtures.make("shared_outliers", 0)
    expected = fixtures.analytic_optimum(evaluator, "huber")
    changed = deepcopy(evaluator)
    changed["truth"] = "forbidden"
    changed["corruption_evaluator_only"] = "forbidden"
    for row in changed["observations"]:
        row["clean"] = "forbidden"
    assert fixtures.analytic_optimum(changed, "huber") == expected


def test_invalid_fixture_loss_and_parameters_fail_closed():
    with pytest.raises(ValueError, match="unknown"):
        fixtures.make("unknown", 0)
    with pytest.raises(ValueError, match="invalid evaluator loss"):
        fixtures.contributions(np.array([1.0]), "wrong", 1.345)
    _, evaluator = fixtures.make()
    with pytest.raises(ValueError, match="parameter names"):
        fixtures.evaluate(evaluator, {"k": 1.0}, "huber")
    with pytest.raises(ValueError, match="finite"):
        fixtures.evaluate(evaluator, {"g": float("nan")}, "huber")

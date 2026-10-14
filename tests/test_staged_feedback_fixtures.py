"""Real tiny fits and independent numerical witnesses for staged qualification."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import fitting_feedback as feedback
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import public_rollout as bridge
from autoformalism.fitting import staged_feedback_fixtures as fixtures
from autoformalism.fitting.rollout_adapter import FitResult


def _problem(value: dict) -> checks.TrainingProblem:
    """Preserve the explicit piecewise opt-in when loading each bridge problem."""
    schema = (
        checks.PiecewiseTrainingProblem
        if value["problem"].get("allow_piecewise")
        else checks.TrainingProblem
    )
    return schema.model_validate(value["problem"])


def _oracle(value: dict) -> tuple[checks.Rollouts, np.ndarray]:
    """Use the full production residual, including initialization derivatives."""
    oracle = checks.Rollouts(_problem(value), checks.ConfidencePolicy())
    return oracle, oracle.vector(value["fit"]["parameters"])


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """Fit each control once and confirm ordinary checkpoint resume is exact."""
    folder = tmp_path_factory.mktemp("staged_feedback_fixtures")
    results = {}
    source = public._source_identity()
    with pytest.MonkeyPatch.context() as patch:
        # Other collaborators may edit unrelated source during this test session.
        patch.setattr(public, "_source_identity", lambda: source)
        for item in fixtures.roster():
            case = item["case_id"]
            destination = folder / case
            results[case] = fixtures.fit_fixture(case, destination)
            assert fixtures.fit_fixture(case, destination) == results[case]
    return results


def test_roster_training_only_boundary_and_mutation_isolation():
    assert len(fixtures.roster()) == 7
    for item in fixtures.roster():
        inputs, policy, evaluator = fixtures.make(item["case_id"])
        allowed = inputs.model_dump(mode="json")
        assert set(allowed) == {"request", "training", "incumbent"}
        assert inputs.training.name == "train" and inputs.incumbent is None
        assert len(inputs.training.rows) == 1
        assert inputs.training.rows[0].auxiliaries == {}
        search = policy.adapter.search
        assert search.maximum_blocks * search.block_calls <= 80
        assert policy.allow_piecewise == item["case_id"].startswith("piecewise_")
        assert not evaluator["benchmark_claim"]
        assert not evaluator["reference_values_given_to_fitter"]
        assert not evaluator["validation_or_test_data_opened"]
        assert "expected" not in str(allowed)
        assert "analytic_argument" not in str(allowed)
        with pytest.raises(ValueError):
            bridge.Inputs.model_validate(allowed | evaluator)
        allowed["training"]["rows"][0]["targets"]["y"][0] = 999.0
        assert fixtures.make(item["case_id"])[0].training.rows[0].targets["y"][0] != 999
    with pytest.raises(ValueError, match="unknown staged feedback fixture"):
        fixtures.make("unknown")


def test_all_fits_have_unmodified_bridge_integrity_and_expected_accuracy(fitted):
    for case, value in fitted.items():
        assert set(value) == {"problem", "fit", "backend"}
        fit = FitResult.model_validate(value["fit"])
        feedback.validate_evidence(_problem(value), fit, value["backend"])
        evaluator = fixtures.make(case)[2]
        assert fit.status == "complete"
        assert fit.verification.target_reached == evaluator["expected_accurate"]
        assert fit.cost.accounting_complete
        assert fit.cost.search_calls_started <= 80
        assert not fit.reference_values_used
        assert not fit.validation_used_for_selection and not fit.test_data_opened
        assert (
            fit.assessment.output_activity["active_parameters"]
            == evaluator["expected_active_parameters"]
        )


@pytest.mark.parametrize(
    "case", ["piecewise_inactive_guard", "piecewise_clean_crossing"]
)
def test_smooth_piecewise_controls_have_real_matching_perturbed_rollouts(case, fitted):
    value = fitted[case]
    assert value["fit"]["cost"]["search_calls_started"] > 0
    assert value["fit"]["parameters"]["a"] == pytest.approx(1.0, abs=1e-6)
    oracle, vector = _oracle(value)
    end = monotonic() + 10
    residual, jacobian = oracle.evaluate(vector, end)
    assert np.mean(residual**2) < 1e-12
    for step in (1e-3, 3e-4):
        plus = oracle.evaluate(vector + step, end, jacobian=False)[0]
        minus = oracle.evaluate(vector - step, end, jacobian=False)[0]
        central = (plus - minus) / (2 * step)
        np.testing.assert_allclose(central, jacobian[:, 0], atol=2e-4, rtol=2e-4)
    rows = fixtures.make(case)[0].training.rows
    if case == "piecewise_clean_crossing":
        assert min(rows[0].targets["y"]) < 10 < max(rows[0].targets["y"])
    else:
        assert max(rows[0].targets["y"]) < 10


def test_kink_control_exposes_false_confidence_from_central_difference_alone(fitted):
    value = fitted["piecewise_kink"]
    assert value["fit"]["parameters"] == {"a": 1.0}
    assert value["fit"]["cost"]["search_calls_started"] == 0
    assert "declared kink" in fixtures.make("piecewise_kink")[2]["starting_point_scope"]
    oracle, vector = _oracle(value)
    end = monotonic() + 5
    residual, jacobian = oracle.evaluate(vector, end)
    plus = oracle.evaluate(vector + 1e-3, end, jacobian=False)[0]
    minus = oracle.evaluate(vector - 1e-3, end, jacobian=False)[0]
    np.testing.assert_allclose((plus - minus) / 2e-3, jacobian[:, 0], atol=1e-10)
    slope = 1 / oracle.scales["y"]
    np.testing.assert_allclose((plus - residual) / 1e-3, slope, atol=1e-10)
    np.testing.assert_allclose((residual - minus) / 1e-3, -slope, atol=1e-10)


def test_bound_control_has_no_legal_symmetric_perturbation(fitted):
    value = fitted["piecewise_bound_limited"]
    assert value["fit"]["parameters"] == {"a": 0.0}
    assert value["fit"]["cost"]["search_calls_started"] == 0
    oracle, vector = _oracle(value)
    assert vector[0] == oracle.lower[0]
    with pytest.raises(ValueError, match="outside declared domain"):
        oracle.evaluate(vector - 1e-3, monotonic() + 5, jacobian=False)


def test_stationarity_is_genuine_at_the_retained_poor_point(fitted, tmp_path):
    value = fitted["stationary_poor_retained"]
    before = deepcopy(value)
    fit = FitResult.model_validate(value["fit"])
    assert fit.parameters == {"a": 0.0}
    oracle, vector = _oracle(value)
    residual, jacobian = oracle.evaluate(vector, monotonic() + 5)
    assert np.mean(residual**2) > 1e-3
    np.testing.assert_array_equal(jacobian, np.zeros_like(jacobian))
    np.testing.assert_array_equal(jacobian.T @ residual, [0.0])
    witness = feedback.search_witness(value["backend"], fit.parameters)
    assert witness.kind == "local_stationarity"
    assert witness.parameters_sha256 == public.content_sha256(fit.parameters)
    operation = next(
        op
        for op in value["backend"]["operations"]
        if op["operation"] == witness.operation
    )
    assert operation["value"]["stop_reason"] == "stationary_poor_fit"
    assert operation["value"]["calls"] > 0
    assert not operation["value"]["stationarity"]["global_minimum_certified"]
    packet = feedback.run(
        _problem(value),
        fit,
        value["backend"],
        tmp_path,
        feedback.Policy(check_seconds=5),
    )
    assert packet.diagnosis == "stationary_poor_fit"
    assert packet.specification_verdict == "not_assessed"
    assert value == before


@pytest.mark.parametrize("case", ["initializer_path", "piecewise_initializer_path"])
def test_initializer_is_an_actual_fitted_output_sensitivity(case, fitted):
    value = fitted[case]
    assert value["fit"]["parameters"]["init_z_value"] == pytest.approx(0.7, abs=1e-6)
    assert value["fit"]["cost"]["search_calls_started"] > 0
    oracle, vector = _oracle(value)
    assert oracle.names == ("init_z_value",)
    residual, jacobian = oracle.evaluate(vector, monotonic() + 5)
    assert np.mean(residual**2) < 1e-12
    time = np.array(value["problem"]["training"]["rows"][0]["time"])
    expected = time / oracle.scales["y"]
    np.testing.assert_allclose(jacobian[:, 0], expected, atol=1e-10)
    paths = feedback.dependencies(oracle)
    assert paths[0]["initial_parameters"] == ["init_z_value"]

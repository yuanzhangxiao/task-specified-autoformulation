"""Real small basin fits, independent witness checks and evaluator isolation."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

from autoformalism.fitting import feedback_fixtures as fixtures
from autoformalism.fitting import fitting_feedback, nonlinear_rollout
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import public_rollout as bridge
from autoformalism.fitting.confidence_checks import TrainingProblem
from autoformalism.fitting.rollout_adapter import FitResult


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """Exercise genuine fitting once per control, including checkpoint resume."""
    folder = tmp_path_factory.mktemp("basin_feedback")
    values = {}
    # Freeze the loaded-source identity while other collaborators edit files;
    # all numerical fits and independent checks remain real.
    source = public._source_identity()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(public, "_source_identity", lambda: source)
        for item in fixtures.roster():
            case = item["case_id"]
            destination = folder / case
            values[case] = fixtures.fit_fixture(case, destination)
            assert fixtures.fit_fixture(case, destination) == values[case]
    return values


def test_roster_and_inputs_separate_reviewer_information():
    assert len(fixtures.roster()) == 5
    for item in fixtures.roster():
        inputs, policy, evaluator = fixtures.make(item["case_id"])
        allowed = inputs.model_dump(mode="json")
        assert set(allowed) == {"request", "training", "incumbent"}
        assert inputs.training.name == "train"
        assert inputs.incumbent is None
        search = policy.adapter.search
        assert search.maximum_blocks * search.block_calls <= 40
        assert evaluator["reference_values_given_to_fitter"] is False
        assert evaluator["benchmark_claim"] is False
        assert evaluator["validation_or_test_data_opened"] is False
        assert set(inputs.request.parameter_guesses.values()) == {1.5}
        assert set(evaluator).isdisjoint(allowed)
        assert "expected" not in str(allowed)
        assert "witness" not in str(allowed)
        assert "certificate" not in str(allowed)
        for row in inputs.training.rows:
            assert set(row.targets) == {"h"}
            assert set(row.external_inputs) == {"u"}
            assert not row.auxiliaries
        # Returned mutable dictionaries cannot contaminate later fixture calls.
        allowed["training"]["rows"][0]["targets"]["h"][0] = 999
        assert fixtures.make(item["case_id"])[0].training.rows[0].targets["h"][0] != 999
    with pytest.raises(ValueError, match="unknown feedback fixture"):
        fixtures.make("unknown")


def test_actual_fit_classifications_and_exact_optimum(fitted):
    for case, value in fitted.items():
        assert set(value) == {"problem", "fit", "backend"}
        _, _, evaluator = fixtures.make(case)
        fit = value["fit"]
        assessment = fit["assessment"]
        assert fit["status"] == "complete"
        assert assessment["search_status"] == evaluator["expected_search_status"]
        assert fit["verification"]["target_reached"] == evaluator["expected_accurate"]
        assert assessment["sensitivity"]["rank"] == evaluator["expected_rank"]
        assert fit["cost"]["accounting_complete"]
        assert not fit["reference_values_used"]
        assert not fit["validation_used_for_selection"]
        assert not fit["test_data_opened"]
        if "analytic_candidate_optimum" in evaluator:
            assert fit["parameters"]["g"] == pytest.approx(
                evaluator["analytic_candidate_optimum"]["g"], abs=1e-10
            )
            assert fit["cost"]["search_calls_started"] == 1
            assert assessment["recommended_action"] == "review_equations"
    ambiguity = fitted["ambiguous_transfer"]["fit"]["assessment"]
    assert ambiguity["parameter_information"] == "weak_sensitivity"
    assert ambiguity["sensitivity"]["parameter_count"] == 2
    assert fitted["budget_limited_drainage"]["fit"]["cost"]["search_calls_started"] == 8


@pytest.mark.parametrize(
    "case", ["adequate_drainage", "budget_limited_drainage", "ambiguous_transfer"]
)
def test_evaluator_witness_replays_original_equations_accurately(case, fitted):
    _, _, evaluator = fixtures.make(case)
    problem = TrainingProblem.model_validate(fitted[case]["problem"])
    oracle, _, _ = nonlinear_rollout.build(problem, reduce_inactive=True)
    witness = evaluator["adequate_witness"]
    check = nonlinear_rollout.verify(
        oracle, witness, monotonic() + 5, lambda _: None
    )
    assert check["training_nmse"] < 1e-12
    assert check["alternate_training_nmse"] < 1e-12
    if case == "budget_limited_drainage":
        fit = fitted[case]["fit"]
        assert fit["assessment"]["training_nmse"] > 1e-8
        assert fit["assessment"]["recommended_action"] == "diversify_or_extend_search"
        assert fit["parameters"] != witness
    if case == "ambiguous_transfer":
        other = nonlinear_rollout.verify(
            oracle, evaluator["ambiguity_witness"], monotonic() + 5, lambda _: None
        )
        assert other["training_nmse"] < 1e-12
        assert other["alternate_training_nmse"] < 1e-12
        assert evaluator["ambiguity_witness"] != witness


def test_candidate_failure_and_false_claim_are_separate_evaluator_cases(fitted):
    omitted, _, ordinary = fixtures.make("omitted_drainage")
    false, _, contradicted = fixtures.make("false_no_drainage_claim")
    adequate, _, _ = fixtures.make("adequate_drainage")
    assert omitted == false
    assert omitted.training == adequate.training
    assert "incompatibility_certificate" not in ordinary
    assert ordinary["expected_claim_compatibility"] == "compatible_witness_exists"
    assert contradicted["expected_claim_compatibility"] == "contradicted"
    cert = contradicted["incompatibility_certificate"]
    assert cert["evaluated_training_sha256"] == public.content_sha256(
        false.training.model_dump(mode="json")
    )
    assert cert["contradicted"] and cert["violations"]
    assert not cert["uses_fit_failure"]
    assert all(v["decrease"] > 0 for v in cert["violations"])
    assert fitted["adequate_drainage"]["fit"]["verification"]["target_reached"]


def test_real_feedback_diagnoses_do_not_receive_evaluator_claims(fitted, tmp_path):
    for case, value in fitted.items():
        _, _, evaluator = fixtures.make(case)
        packet = fitting_feedback.run(
            TrainingProblem.model_validate(value["problem"]),
            FitResult.model_validate(value["fit"]),
            value["backend"],
            tmp_path / case,
            fitting_feedback.Policy(check_seconds=5),
        )
        assert packet.diagnosis == evaluator["expected_diagnosis"]
        assert packet.diagnostics.status == "complete"
        serialized = packet.model_dump(mode="json")
        assert "adequate_witness" not in str(serialized)
        assert "incompatibility_certificate" not in str(serialized)
        assert "synthetic_explicit_claim" not in str(serialized)


@pytest.mark.parametrize("change", ["split", "negative", "nonfinite", "order"])
def test_monotonicity_certificate_rejects_invalid_assumptions(change):
    inputs, _, _ = fixtures.make("false_no_drainage_claim")
    training = inputs.training.model_dump(mode="json")
    row = training["rows"][0]
    if change == "split":
        training["name"] = "val"
    elif change == "negative":
        row["external_inputs"]["u"][0] = -1
    elif change == "nonfinite":
        row["targets"]["h"][0] = np.nan
    else:
        row["time"][1] = row["time"][0]
    with pytest.raises(ValueError, match=r"training|observations"):
        fixtures.no_drainage_certificate(training)


def test_monotonicity_certificate_does_not_invent_a_contradiction():
    inputs, _, _ = fixtures.make("false_no_drainage_claim")
    training = deepcopy(inputs.training.model_dump(mode="json"))
    for row in training["rows"]:
        row["targets"]["h"] = (2 + 0.3 * np.asarray(row["time"])).tolist()
    certificate = fixtures.no_drainage_certificate(training)
    assert not certificate["contradicted"]
    assert certificate["violations"] == []


def test_fitter_cannot_receive_reviewer_dictionary():
    inputs, _, evaluator = fixtures.make("adequate_drainage")
    with pytest.raises(ValueError):
        bridge.Inputs.model_validate(inputs.model_dump(mode="json") | evaluator)

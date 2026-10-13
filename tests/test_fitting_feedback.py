"""Candidate-bound numerical evidence, honest uncertainty and immutable replay."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import fitting_feedback as feedback
from autoformalism.fitting import public_rollout as bridge
from autoformalism.fitting.confidence_checks import TrainingProblem
from tests.test_output_activity import inputs, policy


@pytest.fixture
def fitted(tmp_path, monkeypatch):
    # Another agent may edit independent source files during this test session.
    monkeypatch.setattr(feedback.public, "_source_identity", lambda: "test-source")
    data = inputs()
    result = bridge.run(data, policy(), tmp_path / "fit")
    problem = TrainingProblem.model_validate(result.preparation["value"]["problem"])
    backend = read_seal(tmp_path / "fit/fit/backend.json")
    return problem, result.fit, backend


def test_solved_bad_equation_is_not_false_specification(fitted, tmp_path):
    problem, fit, backend = fitted
    before = deepcopy((problem.model_dump(), fit.model_dump(), backend))
    packet = feedback.run(problem, fit, backend, tmp_path / "feedback")
    assert packet.diagnosis == "solved_restricted_problem_poor_fit"
    assert packet.recommended_action == "review_equations"
    assert packet.specification_verdict == "not_assessed"
    assert not packet.automatic_specification_revision_allowed
    assert packet.search_witness.scope == "fixed_candidate_output"
    assert packet.search_witness.parameters_sha256 == packet.identity.parameters_sha256
    assert packet.diagnostics.active_sensitivity.parameters == ["a"]
    assert packet.diagnostics.inactive_parameters == ["b", "c", "d", "init_z_value"]
    assert packet.diagnostics.disconnected_states == ["z"]
    assert packet.cost.calls == 3 and packet.cost.optimization_calls == 0
    assert packet.training_nmse == pytest.approx(fit.assessment.training_nmse)
    assert (problem.model_dump(), fit.model_dump(), backend) == before


def test_resume_does_not_run_check_or_recharge_and_detects_tampering(
    fitted, tmp_path, monkeypatch
):
    p, fit, backend = fitted
    folder = tmp_path / "feedback"
    first = feedback.run(p, fit, backend, folder)
    monkeypatch.setattr(feedback, "diagnose", lambda *a: pytest.fail("repeated check"))
    assert feedback.run(p, fit, backend, folder) == first
    assert feedback.completed(folder) == first
    path = folder / "diagnostics/result.json"
    changed = read_seal(path)
    changed["value"]["inactive_parameters"] = []
    path.unlink()
    seal(path, changed)
    with pytest.raises(ValueError, match="diagnostics differ"):
        feedback.completed(folder)


def test_interrupt_preserves_unknown_cost_and_does_not_claim_new_check(
    fitted, tmp_path, monkeypatch
):
    p, fit, backend = fitted
    original = tmp_path / "source"
    feedback.run(p, fit, backend, original)
    folder = tmp_path / "interrupted"
    seal(
        folder / "diagnostics/started.json",
        read_seal(original / "diagnostics/started.json"),
    )
    monkeypatch.setattr(feedback, "diagnose", lambda *a: pytest.fail("new allowance"))
    packet = feedback.run(p, fit, backend, folder)
    assert packet.diagnostics.status == "interrupted"
    assert packet.cost.calls is None and packet.cost.wall_seconds is None
    assert packet.cost.budget_charge_seconds == 180
    assert not packet.cost.accounting_complete
    assert packet.parameter_information == "unavailable"
    # Historical solved-output evidence remains labeled and bound, not erased.
    assert packet.diagnosis == "solved_restricted_problem_poor_fit"
    assert feedback.completed(folder) == packet


def test_cross_candidate_or_parameter_or_policy_substitution_is_rejected(
    fitted, tmp_path
):
    p, fit, backend = fitted
    bad = p.model_copy(update={"start": p.start | {"a": 0.4}})
    with pytest.raises(ValueError, match="identity differs"):
        feedback.run(bad, fit, backend, tmp_path / "different")
    with pytest.raises(ValueError, match="identity differs"):
        feedback.run(
            p,
            fit.model_copy(update={"parameters": p.start}),
            backend,
            tmp_path / "changed",
        )
    changed = deepcopy(backend)
    changed["identity"]["policy"]["target_nmse"] = 1e-6
    with pytest.raises(ValueError, match="identity differs"):
        feedback.run(p, fit, changed, tmp_path / "policy")


def test_stationarity_on_other_vector_never_follows_the_incumbent(fitted):
    _, fit, backend = fitted
    changed = deepcopy(backend)
    changed["routing"] = {}
    op = next(o for o in changed["operations"] if o["operation"].endswith("/rollout"))
    op["value"]["stop_reason"] = "stationary_poor_fit"
    op["value"]["stationarity"] = {
        "parameters_sha256": feedback.public.content_sha256(fit.parameters)
    }
    assert feedback.search_witness(changed, fit.parameters).kind == "local_stationarity"
    op["value"]["stationarity"]["parameters_sha256"] = "different-vector"
    assert feedback.search_witness(changed, fit.parameters) is None
    op["value"]["stationarity"]["parameters_sha256"] = feedback.public.content_sha256(
        fit.parameters
    )
    op["value"]["best"]["parameters"]["a"] += 0.1
    assert feedback.search_witness(changed, fit.parameters) is None


def test_replayed_accuracy_and_worst_trajectory_override_old_success_flag(
    fitted, tmp_path
):
    p, fit, backend = fitted
    packet = feedback.run(p, fit, backend, tmp_path / "feedback")
    flagged = fit.model_copy(
        update={
            "verification": fit.verification.model_copy(update={"target_reached": True})
        }
    )
    # Reproduction tolerance alone is intentionally insufficient for accuracy.
    diagnostic = packet.diagnostics.model_copy(
        update={
            "replay_training_nmse": 5e-11,
            "replay_alternate_training_nmse": 5e-11,
            "replay_worst_trajectory_nmse": 5e-11,
        }
    )
    assert not feedback.classify(flagged, backend, diagnostic)[0].startswith("accurate")
    diagnostic = diagnostic.model_copy(
        update={
            "replay_training_nmse": 1e-15,
            "replay_alternate_training_nmse": 1e-15,
            "replay_solver_loss_discrepancy": 0,
            "replay_solver_prediction_discrepancy": 0,
        }
    )
    assert not feedback.classify(flagged, backend, diagnostic)[0].startswith("accurate")
    diagnostic = diagnostic.model_copy(update={"replay_worst_trajectory_nmse": 1e-15})
    assert (
        feedback.classify(flagged, backend, diagnostic)[0] == "accurate_local_evidence"
    )
    nonsmooth = diagnostic.model_copy(
        update={"derivative_scope": "piecewise_branch_convention"}
    )
    assert feedback.classify(flagged, backend, nonsmooth)[0] == "accurate_unresolved"


def test_derivative_failure_preserves_completed_fixed_parameter_rollouts(
    fitted, tmp_path, monkeypatch
):
    evaluate = feedback.checks.Rollouts.evaluate

    def fail_derivative(self, vector, deadline, **kwargs):
        if kwargs.get("jacobian", True):
            raise TimeoutError("derivative allowance")
        return evaluate(self, vector, deadline, **kwargs)

    monkeypatch.setattr(feedback.checks.Rollouts, "evaluate", fail_derivative)
    packet = feedback.run(*fitted, tmp_path / "feedback")
    assert packet.diagnostics.status == "complete"
    assert packet.diagnostics.solver_agreement
    assert (
        packet.diagnostics.residuals and packet.diagnostics.active_sensitivity is None
    )
    assert packet.diagnostics.sensitivity_status == "unavailable"
    assert packet.parameter_information == "unavailable"
    assert packet.cost.calls == 3


def test_multiple_outputs_and_initial_map_dependencies_are_joint():
    raw = inputs(rhs="a*z").model_dump(mode="json")
    raw["request"]["context"]["targets"].append("other")
    raw["request"]["base_candidate"]["observation_mappings"].append(
        {"channel": "other", "expression": "c*z+d"}
    )
    raw["training"]["rows"][0]["targets"]["other"] = [1.0] * 9
    value = bridge.prepare(bridge.Inputs.model_validate(raw), policy())
    problem = TrainingProblem.model_validate(value["problem"])
    oracle = feedback.checks.Rollouts(problem, feedback.checks.ConfidencePolicy())
    paths = {d["target"]: d for d in feedback.dependencies(oracle)}
    assert paths["y"]["initial_parameters"] == ["init_z_value"]
    assert paths["other"]["parameters"] == ["b", "c", "d", "init_z_value"]
    assert set(feedback.certificate(oracle.system)["active_parameters"]) == set(
        oracle.names
    )
    # Different lengths and opposing target residuals must not cancel in NMSE.
    row = oracle.train.trajectories[0]
    other = replace(row, trajectory_id="short", time=np.array([0.0, 1.0]))
    oracle.train = replace(oracle.train, trajectories=(row, other))
    residual = np.tile([1.0, -3.0], len(row.time) + 2)
    summaries = feedback.residual_summary(oracle, residual, residual)
    assert [(r["target"], r["primary_nmse"]) for r in summaries] == [
        ("y", 1),
        ("other", 9),
    ]
    assert all(r["sample_count"] == 11 for r in summaries)


@pytest.mark.parametrize("field", ["validation", "test", "reference_parameters"])
def test_training_problem_rejects_evaluator_channels(fitted, field):
    p, _, _ = fitted
    with pytest.raises(ValueError):
        TrainingProblem.model_validate(
            p.model_dump(mode="json") | {field: {"hidden": 1}}
        )


def test_no_retained_vector_produces_no_diagnostics_or_fabricated_scores(
    fitted, tmp_path, monkeypatch
):
    p, fit, backend = fitted
    changed = deepcopy(backend)
    changed.update(
        selected=None,
        retained_parameters=None,
        accuracy_verified=False,
        status="retained_unverified",
    )
    changed["assessment"] = feedback.assess(None, None, [], [], 1e-12)
    lost = feedback.adapter.summarize(changed, fit.identity)
    monkeypatch.setattr(feedback, "diagnose", lambda *a: pytest.fail("no vector"))
    packet = feedback.run(p, lost, changed, tmp_path / "feedback")
    assert packet.diagnosis == "rollout_unverified"
    assert packet.training_nmse is None and packet.identity.parameters_sha256 is None
    assert packet.diagnostics.status == "skipped" and packet.cost.calls == 0
    assert feedback.completed(tmp_path / "feedback") == packet

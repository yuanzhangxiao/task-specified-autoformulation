"""Durable independent stages, conservative piecewise claims, and fixed fits."""

from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import staged_feedback as feedback
from autoformalism.fitting import staged_feedback_fixtures as fixtures
from tests.test_fitting_feedback import fitted as fitted_fixture

fitted = fitted_fixture


@pytest.fixture(autouse=True)
def source(monkeypatch):
    monkeypatch.setattr(feedback.public, "_source_identity", lambda: "test-source")


def test_smooth_retained_point_and_resume(fitted, tmp_path, monkeypatch):
    before = deepcopy(
        [x.model_dump() if hasattr(x, "model_dump") else x for x in fitted]
    )
    folder = tmp_path / "feedback"
    packet = feedback.run(*fitted, folder)
    assert packet.diagnosis == "solved_restricted_problem_poor_fit"
    assert packet.cost.calls == packet.cost.calls_completed == 3
    assert packet.cost.stages_completed == 3
    assert packet.diagnostics.output_status == "verified"
    assert packet.cost.optimization_calls == 0
    assert not packet.parameters_changed and not packet.validation_used
    monkeypatch.setattr(
        feedback, "evaluate_stage", lambda *a, **kw: pytest.fail("rerun")
    )
    assert feedback.run(*fitted, folder) == packet
    assert feedback.completed(folder) == packet
    assert [x.model_dump() if hasattr(x, "model_dump") else x for x in fitted] == before


def test_alternate_timeout_preserves_primary_and_reason(fitted, tmp_path, monkeypatch):
    evaluate = feedback.evaluate_stage

    def fail(oracle, vector, deadline, checkpoint, method, sensitivities, **kw):
        if method == "Radau":
            checkpoint({"calls": 1, "calls_completed": 0, "failure_kind": "timeout"})
            raise TimeoutError("independent rollout timeout")
        return evaluate(
            oracle, vector, deadline, checkpoint, method, sensitivities, **kw
        )

    monkeypatch.setattr(feedback, "evaluate_stage", fail)
    packet = feedback.run(*fitted, tmp_path / "feedback")
    assert packet.diagnostics.status == "unavailable"
    assert packet.diagnostics.replay_training_nmse is not None
    assert packet.diagnostics.replay_alternate_training_nmse is None
    assert "ALTERNATE_TIMEOUT" in packet.diagnostics.reason_codes
    assert packet.parameter_information == "unavailable"
    assert packet.cost.calls == 2 and packet.cost.calls_completed == 1
    assert feedback.completed(tmp_path / "feedback") == packet


def test_sensitivity_failure_preserves_two_outputs(fitted, tmp_path, monkeypatch):
    evaluate = feedback.evaluate_stage

    def fail(oracle, vector, deadline, checkpoint, method, sensitivities, **kw):
        if sensitivities:
            checkpoint({"calls": 1, "calls_completed": 0, "failure_kind": "timeout"})
            raise TimeoutError("derivative timeout")
        return evaluate(
            oracle, vector, deadline, checkpoint, method, sensitivities, **kw
        )

    monkeypatch.setattr(feedback, "evaluate_stage", fail)
    packet = feedback.run(*fitted, tmp_path / "feedback")
    assert packet.diagnostics.status == "complete"
    assert packet.diagnostics.output_status == "verified"
    assert (
        packet.diagnostics.residuals and packet.diagnostics.active_sensitivity is None
    )
    assert "SENSITIVITY_TIMEOUT" in packet.diagnostics.reason_codes
    assert packet.cost.calls == 3 and packet.cost.calls_completed == 2


def test_interrupted_stage_not_retried_other_stages_survive(
    fitted, tmp_path, monkeypatch
):
    folder = tmp_path / "feedback"
    packet = feedback.run(*fitted, folder)
    for path in (
        folder / "result.json",
        folder / "stages.json",
        folder / "stages/alternate/result.json",
        folder / "stages/alternate/record.json",
    ):
        path.unlink()
    monkeypatch.setattr(
        feedback, "evaluate_stage", lambda *a, **kw: pytest.fail("new budget")
    )
    resumed = feedback.run(*fitted, folder)
    assert (
        resumed.diagnostics.replay_training_nmse
        == packet.diagnostics.replay_training_nmse
    )
    assert resumed.cost.calls is None and not resumed.cost.accounting_complete
    assert resumed.cost.budget_charge_seconds >= feedback.Policy().alternate_seconds
    assert "ALTERNATE_INTERRUPTED" in resumed.diagnostics.reason_codes
    assert feedback.completed(folder) == resumed


def test_resume_after_process_interrupt_runs_only_unstarted_stage(
    fitted, tmp_path, monkeypatch
):
    folder = tmp_path / "feedback"
    evaluate = feedback.evaluate_stage

    def stop(oracle, vector, deadline, checkpoint, method, sensitivities, **kw):
        if method == "Radau":
            raise KeyboardInterrupt
        return evaluate(
            oracle, vector, deadline, checkpoint, method, sensitivities, **kw
        )

    monkeypatch.setattr(feedback, "evaluate_stage", stop)
    with pytest.raises(KeyboardInterrupt):
        feedback.run(*fitted, folder)
    assert (folder / "stages/primary/result.json").exists()
    monkeypatch.setattr(
        feedback, "evaluate_stage", lambda *a, **kw: pytest.fail("retry")
    )
    packet = feedback.run(*fitted, folder)
    assert packet.diagnostics.replay_training_nmse is not None
    assert packet.cost.calls is None


def test_resealed_stage_tampering_rejected(fitted, tmp_path):
    folder = tmp_path / "feedback"
    feedback.run(*fitted, folder)
    path = folder / "stages/primary/result.json"
    raw = read_seal(path)
    raw["value"]["residual"][0] += 1
    path.unlink()
    seal(path, raw)
    with pytest.raises(ValueError, match="envelope differs"):
        feedback.completed(folder)


@pytest.mark.parametrize(
    ("case", "status"),
    [
        ("piecewise_inactive_guard", "supported"),
        ("piecewise_clean_crossing", "supported"),
        ("piecewise_kink", "unsupported_kink"),
        ("piecewise_bound_limited", "infeasible_direction"),
        ("piecewise_initializer_path", "supported"),
    ],
)
def test_real_piecewise_assessment(case, status, tmp_path):
    saved = fixtures.fit_fixture(case, tmp_path / "fit")
    problem = feedback.training_problem(saved["problem"])
    fit = feedback.adapter.FitResult.model_validate(saved["fit"])
    packet = feedback.run(problem, fit, saved["backend"], tmp_path / "feedback")
    result = packet.diagnostics.piecewise_assessment
    assert result is not None and result.status == status
    assert result.supported_classical is False and result.supported_global is False
    assert packet.diagnostics.derivative_scope == (
        "piecewise_numerically_supported"
        if status == "supported"
        else "piecewise_branch_convention"
    )
    assert feedback.completed(tmp_path / "feedback") == packet
    if case == "piecewise_initializer_path":
        assert result.initial_parameters_checked == ("init_z_value",)
    if status != "supported":
        assert packet.parameter_information == "unavailable"


def test_stationarity_is_bound_to_retained_real_fit(tmp_path):
    saved = fixtures.fit_fixture("stationary_poor_retained", tmp_path / "fit")
    packet = feedback.run(
        feedback.training_problem(saved["problem"]),
        feedback.adapter.FitResult.model_validate(saved["fit"]),
        saved["backend"],
        tmp_path / "feedback",
    )
    assert packet.diagnosis == "stationary_poor_fit"
    assert packet.search_witness.parameters_sha256 == packet.identity.parameters_sha256
    assert packet.specification_verdict == "not_assessed"

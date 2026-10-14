"""Contradictory partial replay and independently bound stage identity checks."""

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_rollout as bridge
from autoformalism.fitting import staged_feedback as feedback
from autoformalism.fitting import staged_feedback_fixtures as fixtures


@pytest.fixture(autouse=True)
def source(monkeypatch):
    """Keep identity deterministic while independent collaborators edit source."""
    monkeypatch.setattr(feedback.public, "_source_identity", lambda: "test-source")


@pytest.fixture
def fitted(tmp_path):
    """Create a real accurate backend with a fitted initializer parameter."""
    saved = fixtures.fit_fixture("initializer_path", tmp_path / "fit")
    return (
        feedback.training_problem(saved["problem"]),
        feedback.adapter.FitResult.model_validate(saved["fit"]),
        saved["backend"],
    )


def _replace(path, value):
    """Reseal altered evidence to test semantic binding beyond a file checksum."""
    path.unlink()
    seal(path, value)


def test_partial_replay_contradiction_cannot_inherit_historical_accuracy(
    tmp_path, monkeypatch
):
    saved = fixtures.fit_fixture("initializer_path", tmp_path / "fit")
    problem = feedback.training_problem(saved["problem"])
    fit = feedback.adapter.FitResult.model_validate(saved["fit"])
    assert fit.verification.target_reached
    evaluate = feedback.evaluate_stage

    def contradict(oracle, vector, deadline, checkpoint, method, sensitivities, **kw):
        if method == "Radau":
            checkpoint({"calls": 1, "calls_completed": 0, "failure_kind": "timeout"})
            raise TimeoutError("independent replay unavailable")
        value = evaluate(
            oracle, vector, deadline, checkpoint, method, sensitivities, **kw
        )
        value["residual"] = [1.0] * len(value["residual"])
        return value

    monkeypatch.setattr(feedback, "evaluate_stage", contradict)
    folder = tmp_path / "feedback"
    packet = feedback.run(problem, fit, saved["backend"], folder)
    assert packet.diagnostics.replay_training_nmse == 1.0
    assert packet.diagnostics.replay_alternate_training_nmse is None
    assert packet.diagnosis == "rollout_unverified"
    assert packet.fit_quality == "unverified"
    assert packet.training_nmse == 1.0
    assert packet.parameter_information == "unavailable"
    assert feedback.completed(folder) == packet


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("point_sha256", "another-parameter-vector"),
        ("solver", "Radau"),
        ("sensitivities", True),
        ("stage", "alternate"),
        ("allowance_seconds", 17.0),
    ],
)
def test_resealed_stage_identity_must_match_the_expected_plan(
    fitted, tmp_path, field, replacement
):
    folder = tmp_path / "feedback"
    feedback.run(*fitted, folder)
    manifest = read_seal(folder / "stages.json")
    operations = {
        key: read_seal(folder / "stages" / key / "record.json")
        for key, _ in manifest
    }
    primary = operations["primary"]
    primary["identity"][field] = replacement
    path = folder / "stages/primary"
    _replace(path / "record.json", primary)
    _replace(
        path / "result.json", {k: v for k, v in primary.items() if k != "progress"}
    )
    _replace(
        folder / "stages.json",
        [[key, feedback.public.content_sha256(op)] for key, op in operations.items()],
    )
    # Recompute the outer summaries too. A checksum-consistent artifact still
    # must not claim to assess a different point, solver, or stage allowance.
    packet = read_seal(folder / "result.json")
    packet["diagnostics"]["stages"] = [
        feedback._record(key, op).model_dump(mode="json")
        for key, op in operations.items()
    ]
    packet["diagnostics"]["artifact_sha256"] = feedback.public.content_sha256(
        operations
    )
    _replace(folder / "result.json", packet)
    with pytest.raises(ValueError, match=r"stage|identity|plan|point"):
        feedback.completed(folder)


def test_saved_context_cannot_add_a_reviewer_channel(fitted, tmp_path):
    folder = tmp_path / "feedback"
    feedback.run(*fitted, folder)
    context = read_seal(folder / "context.json")
    context["problem"]["reference_parameters"] = {"a": 0.7}
    _replace(folder / "context.json", context)
    with pytest.raises(ValueError):
        feedback.completed(folder)


def test_interrupted_perturbation_consumes_remaining_budget_without_new_solves(
    tmp_path, monkeypatch
):
    saved = fixtures.fit_fixture("piecewise_inactive_guard", tmp_path / "fit")
    problem = feedback.training_problem(saved["problem"])
    fit = feedback.adapter.FitResult.model_validate(saved["fit"])
    policy = feedback.Policy(
        perturbation_seconds=1,
        perturbation_total_seconds=1,
    )
    evaluate = feedback.evaluate_stage

    def stop(oracle, vector, deadline, checkpoint, method, sensitivities, **kw):
        if method == "Radau" and kw.get("metadata_policy") is None:
            checkpoint({"calls": 1, "calls_completed": 0})
            raise KeyboardInterrupt
        return evaluate(
            oracle, vector, deadline, checkpoint, method, sensitivities, **kw
        )

    folder = tmp_path / "feedback"
    monkeypatch.setattr(feedback, "evaluate_stage", stop)
    with pytest.raises(KeyboardInterrupt):
        feedback.run(problem, fit, saved["backend"], folder, policy)
    perturbations = folder / "stages/perturb"
    complete = list(perturbations.glob("*/primary/record.json"))
    assert len(complete) == 1
    previous = read_seal(complete[0])
    monkeypatch.setattr(
        feedback, "evaluate_stage", lambda *a, **kw: pytest.fail("renewed allowance")
    )
    packet = feedback.run(problem, fit, saved["backend"], folder, policy)
    stages = [
        row for row in packet.diagnostics.stages if row.key.startswith("perturb/")
    ]
    assert [row.status for row in stages] == ["complete", "interrupted"]
    assert read_seal(complete[0]) == previous
    assert sum(row.budget_charge_seconds for row in stages) == pytest.approx(1)
    assert packet.cost.calls is None and not packet.cost.accounting_complete
    assert packet.cost.stages_completed == 4
    assert "PERTURBATION_BUDGET_EXHAUSTED" in packet.diagnostics.reason_codes
    assert packet.parameter_information == "unavailable"
    assert feedback.completed(folder) == packet


def test_real_piecewise_stationarity_requires_current_derivative_qualification(
    tmp_path,
):
    inputs, policy, _ = fixtures.make("stationary_poor_retained")
    raw = inputs.model_dump(mode="json")
    raw["request"]["base_candidate"]["state_equations"][0]["rhs"] = "abs(a)"
    result = bridge.run(
        bridge.Inputs.model_validate(raw),
        policy.model_copy(update={"allow_piecewise": True}),
        tmp_path / "fit",
    )
    fit = result.fit
    assert fit.parameters == {"a": 0.0} and not fit.verification.target_reached
    backend = read_seal(tmp_path / "fit/fit/backend.json")
    original = feedback.legacy.search_witness(backend, fit.parameters)
    assert original.kind == "local_stationarity"
    problem = feedback.training_problem(result.preparation["value"]["problem"])
    folder = tmp_path / "feedback"
    packet = feedback.run(problem, fit, backend, folder)
    assert packet.diagnostics.piecewise_assessment.status == "unsupported_kink"
    assert packet.diagnostics.derivative_scope == "piecewise_branch_convention"
    assert packet.diagnosis == "search_incomplete"
    assert packet.recommended_action == "diversify_or_extend_search"
    assert packet.search_witness is None
    assert packet.historical_search_witness == original
    assert "HISTORICAL_STATIONARITY_UNQUALIFIED" in packet.diagnostics.reason_codes
    assert packet.parameter_information == "unavailable"
    assert packet.cost.optimization_calls == 0
    assert feedback.completed(folder) == packet


def test_partial_worst_trajectory_contradiction_overrides_passing_aggregate(
    tmp_path, monkeypatch
):
    inputs, policy, _ = fixtures.make("initializer_path")
    raw = inputs.model_dump(mode="json")
    row = raw["training"]["rows"][0]
    time = np.linspace(0, 3, 401)
    row["time"], row["targets"]["y"] = time.tolist(), (0.7 * time).tolist()
    raw["training"]["rows"].append(
        {
            "trajectory_id": "train_short",
            "time": [0.0, 3.0],
            "targets": {"y": [0.0, 2.1]},
            "external_inputs": {},
            "auxiliaries": {},
        }
    )
    raw["training"]["fingerprint"] = feedback.public.content_sha256(
        raw["training"]["rows"]
    )
    result = bridge.run(bridge.Inputs.model_validate(raw), policy, tmp_path / "fit")
    fit = result.fit
    assert fit.verification.target_reached
    backend = read_seal(tmp_path / "fit/fit/backend.json")
    problem = feedback.training_problem(result.preparation["value"]["problem"])
    evaluate = feedback.evaluate_stage

    def contradict(oracle, vector, deadline, checkpoint, method, sensitivities, **kw):
        if method == "Radau":
            checkpoint({"calls": 1, "calls_completed": 0, "failure_kind": "timeout"})
            raise TimeoutError("independent replay unavailable")
        value = evaluate(
            oracle, vector, deadline, checkpoint, method, sensitivities, **kw
        )
        value["residual"] = [0.0] * len(value["residual"])
        value["residual"][-1] = float(np.sqrt(4e-11))
        return value

    monkeypatch.setattr(feedback, "evaluate_stage", contradict)
    folder = tmp_path / "feedback"
    packet = feedback.run(problem, fit, backend, folder)
    assert packet.training_nmse < packet.target_nmse
    assert packet.diagnostics.replay_worst_trajectory_nmse == pytest.approx(
        2e-11, abs=1e-20
    )
    assert (
        packet.diagnostics.replay_worst_trajectory_nmse
        > packet.worst_trajectory_target_nmse
    )
    assert "PARTIAL_REPLAY_METRICS_DIFFER" not in packet.diagnostics.reason_codes
    assert "PARTIAL_REPLAY_ACCURACY_CONTRADICTION" in packet.diagnostics.reason_codes
    assert packet.diagnosis == "rollout_unverified"
    assert packet.fit_quality == "unverified"
    assert feedback.completed(folder) == packet

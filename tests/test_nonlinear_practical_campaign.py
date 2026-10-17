"""M41 paired public fits, sealed stage boundaries and deterministic checkpoints."""

import json
from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import nonlinear_practical_campaign as campaign


@pytest.fixture(autouse=True)
def stable_source(monkeypatch):
    monkeypatch.setattr(campaign.public, "_source_identity", lambda: "a" * 64)


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "campaign"
    assert campaign.prepare(root)["tasks"] == 36
    return root


def fake_fit(problem, policy, folder, *, strategy, loss):
    """Keep the public start as an intentionally inadequate retained endpoint."""
    return {
        "status": "complete",
        "identity": {
            "protocol": "sensor-drift-fitting-execution-1",
            "source_sha256": campaign.public._source_identity(),
            "runtime": campaign.public._runtime(),
            "problem_sha256": campaign.public.content_sha256(
                problem.model_dump(mode="json")
            ),
            "base_request_sha256": campaign.public.content_sha256(
                problem.request.model_dump(mode="json")
            ),
            "policy": policy.model_dump(mode="json"),
            "strategy": strategy,
            "objective_definition": loss.model_dump(mode="json"),
        },
        "parameters": dict(problem.start),
        "objective_value": 12.0,
        "verification": {"passed": True},
        "cost": {"wall_seconds": 0.1, "cpu_seconds": 0.05, "accounting_complete": True},
    }


def diagnostic(status="complete"):
    """Match the real nested timing contract without numerical work."""
    return {
        "status": status,
        "calls": 14,
        "completed_calls": 14,
        "failed_calls": 0,
        "budget_charge_seconds": 0.2,
        "accounting_complete": True,
        "timing": {
            "total_wall_seconds": 0.2,
            "total_cpu_seconds": 0.1,
            "numerical_wall_seconds": 0.15,
            "numerical_cpu_seconds": 0.08,
            "save_callback_wall_seconds": 0.03,
            "save_callback_cpu_seconds": 0.01,
        },
        "endpoint_qualification": {
            "numerically_qualified": False,
            "retained_fit_inadequate": True,
        },
        "supports": [],
        "nodes": [{"value": 0.45, "status": "complete"}],
    }


def fake_profile(problem, parameters, *, policy, state=None, save=None):
    assert "truth" not in problem.model_dump()
    assert parameters == dict(problem.start)
    value = diagnostic()
    save(value)
    return value


@pytest.fixture
def fake_backend(monkeypatch):
    monkeypatch.setattr(campaign.measurement_fit, "run", fake_fit)
    monkeypatch.setattr(campaign.profiles, "run", fake_profile)


def test_roster_pairing_public_boundary_and_missing_denominators(prepared, monkeypatch):
    plan, data = campaign.verify(prepared)
    first, second = plan["tasks"][:2]
    assert campaign.training_problem(data, first) == campaign.training_problem(
        data, second
    )
    a, b = (
        campaign.fit_identity(plan, data, first),
        campaign.fit_identity(plan, data, second),
    )
    assert a["problem_sha256"] == b["problem_sha256"] and a["policy"] == b["policy"]
    assert a["strategy"] != b["strategy"]
    data["evaluation"] = dict.fromkeys(data["evaluation"], "forbidden")
    assert campaign.training_problem(data, first).start["k"] == 1.1
    monkeypatch.setattr(
        campaign.fixtures, "make_inputs", lambda: pytest.fail("redrawn")
    )
    campaign.prepare(prepared)
    summary = campaign.report(prepared)
    assert summary["status_counts"] == {"missing": 36}
    assert summary["profile_status_counts"] == {"missing": 36}
    assert len(summary["groups"]) == 12 and len(summary["paired_objectives"]) == 18
    assert all(group["expected"] == 3 for group in summary["groups"])
    assert summary["cost"]["profile"]["calls_total"] is None
    assert not summary["cost_complete"]


def test_fit_and_profile_sealed_before_evaluator_and_terminal_resume(
    prepared, fake_backend, monkeypatch
):
    original = campaign.evaluate
    events = []

    def evaluate(data, task, backend):
        folder = prepared / "results" / task["task_id"]
        assert read_seal(folder / "backend.json") == backend
        assert read_seal(folder / "profile.json")["status"] == "complete"
        events.append("evaluated")
        return original(data, task, backend)

    monkeypatch.setattr(campaign, "evaluate", evaluate)
    first = campaign.run_task(prepared, 0)
    assert first["evaluation"]["analytic_rollout_objective_agreement_passed"] is False
    assert first["profile"]["wall_seconds"] == 0.2
    monkeypatch.setattr(
        campaign.measurement_fit, "run", lambda *a, **k: pytest.fail("refit")
    )
    monkeypatch.setattr(
        campaign.profiles, "run", lambda *a, **k: pytest.fail("reprofile")
    )
    monkeypatch.setattr(campaign, "evaluate", lambda *a, **k: pytest.fail("reevaluate"))
    assert campaign.run_task(prepared, 0) == first
    assert events == ["evaluated"]
    summary = campaign.report(prepared)
    assert summary["recorded"] == 1
    assert summary["cost"]["profile"]["calls_observed"] == 14
    assert summary["cost"]["profile"]["cpu_seconds_observed"] == 0.1
    assert summary["cost"]["profile"]["numerical_wall_seconds_observed"] == 0.15
    assert summary["cost"]["profile"]["save_callback_cpu_seconds_observed"] == 0.01
    assert "nodes" not in summary["rows"][0]["profile"]
    assert "backend" not in summary["rows"][0]
    assert summary["groups"][0]["retained_fit_inadequate"] == 1


def test_profile_interruption_resumes_exact_saved_state_without_refit(
    prepared, fake_backend, monkeypatch
):
    partial = diagnostic("running") | {
        "calls": 21,
        "completed_calls": 20,
        "endpoint_qualification": None,
    }

    def interrupted(problem, parameters, *, policy, state=None, save=None):
        save(partial)
        raise KeyboardInterrupt

    monkeypatch.setattr(campaign.profiles, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        campaign.run_task(prepared, 0)
    report = campaign.report(prepared)
    assert report["recorded"] == 0
    assert report["profile_status_counts"]["unfinished"] == 1
    assert report["cost"]["profile"]["calls_observed"] == 21
    assert report["cost"]["profile"]["calls_total"] is None
    monkeypatch.setattr(
        campaign.measurement_fit, "run", lambda *a, **k: pytest.fail("refit")
    )

    def resume(problem, parameters, *, policy, state=None, save=None):
        assert state == partial
        value = diagnostic() | {
            "calls": 35,
            "completed_calls": 34,
            "accounting_complete": False,
        }
        save(value)
        return value

    monkeypatch.setattr(campaign.profiles, "run", resume)
    completed = campaign.run_task(prepared, 0)
    assert completed["profile"]["calls"] == 35
    assert completed["evaluation"]["status"] == "evaluated"
    assert not campaign.report(prepared)["cost_complete"]


@pytest.mark.parametrize("outcome", ["failed", "partial", "unqualified"])
def test_incomplete_diagnostics_do_not_block_postfit_truth(
    prepared, fake_backend, monkeypatch, outcome
):
    def incomplete(problem, parameters, *, policy, state=None, save=None):
        save(diagnostic("running"))
        if outcome == "failed":
            raise RuntimeError("deliberate profile failure")
        return diagnostic(outcome) | {"endpoint_qualification": None}

    monkeypatch.setattr(campaign.profiles, "run", incomplete)
    result = campaign.run_task(prepared, 0)
    assert result["profile"]["status"] == outcome
    assert result["evaluation"]["status"] == "evaluated"
    assert campaign.report(prepared)["groups"][0]["numerically_qualified"] == 0


def test_fit_failure_and_invalid_task_indices(prepared, monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError("fit failed")

    monkeypatch.setattr(campaign.measurement_fit, "run", failed)
    monkeypatch.setattr(
        campaign.profiles, "run", lambda *a, **k: pytest.fail("profile")
    )
    result = campaign.run_task(prepared, 0)
    assert result["status"] == "failed"
    assert result["profile"]["status"] == "no_parameters"
    assert result["evaluation"]["status"] == "no_parameters"
    for index in (-1, 36, True):
        with pytest.raises(ValueError, match="task index"):
            campaign.run_task(prepared, index)


@pytest.mark.parametrize("field", ["strategy", "problem_sha256", "policy"])
def test_changed_backend_identity_rejected(prepared, fake_backend, field):
    result = campaign.run_task(prepared, 0)
    folder = prepared / "results" / result["task_id"]
    backend = read_seal(folder / "backend.json")
    backend["identity"][field] = "wrong"
    (folder / "backend.json").unlink()
    seal(folder / "backend.json", backend)
    with pytest.raises(ValueError, match="backend identity differs"):
        campaign.run_task(prepared, 0)
    with pytest.raises(ValueError, match="backend identity differs"):
        campaign.report(prepared)


def test_copied_postfit_stage_from_another_endpoint_is_rejected(prepared, fake_backend):
    first = campaign.run_task(prepared, 0)
    second = campaign.run_task(prepared, 1)
    folder = prepared / "results" / second["task_id"]
    (folder / "profile.json").unlink()
    seal(folder / "profile.json", first["profile"])
    with pytest.raises(ValueError, match="profile identity differs"):
        campaign.run_task(prepared, 1)


def test_raw_sigma_policy_and_source_changes_reject_resume(prepared, monkeypatch):
    with pytest.raises(ValueError, match="sealed artifact differs"):
        campaign.prepare(prepared, policy=campaign.Policy(maximum_calls=101))
    plan, data = campaign.verify(prepared)
    changed = deepcopy(data)
    changed["problems"][plan["tasks"][0]["common"]]["base"]["manifest"]["noise"][
        "sigma"
    ][0] *= 2
    other = prepared.parent / "other.json"
    seal(other, changed)
    with pytest.raises(ValueError, match="known sigma/mask differs"):
        campaign.prepare(prepared, other)
    monkeypatch.setattr(campaign.public, "_source_identity", lambda: "b" * 64)
    with pytest.raises(ValueError, match="source/runtime differs"):
        campaign.verify(prepared)


@pytest.mark.parametrize("torn", [False, True])
def test_checkpoint_survives_interruption_before_journal_append(
    tmp_path, monkeypatch, torn
):
    problem, _ = campaign.fixtures.make()
    identity = {"test": "identity"}
    partial = diagnostic("running")

    def interrupted(problem, parameters, *, policy, state=None, save=None):
        save(partial)

    append = campaign._append_event

    def stop_append(path, event):
        if torn:
            path.write_bytes(b'{"torn":')
        raise KeyboardInterrupt

    monkeypatch.setattr(campaign.profiles, "run", interrupted)
    monkeypatch.setattr(campaign, "_append_event", stop_append)
    with pytest.raises(KeyboardInterrupt):
        campaign._profile_once(
            tmp_path, identity, problem, dict(problem.start), campaign.profiles.Policy()
        )
    checkpoint, records, truncated = campaign._checkpoint(tmp_path, identity)
    assert checkpoint["state"] == partial and records == [] and truncated is torn
    monkeypatch.setattr(campaign, "_append_event", append)

    def resume(problem, parameters, *, policy, state=None, save=None):
        assert state == partial
        save(diagnostic())
        return diagnostic()

    monkeypatch.setattr(campaign.profiles, "run", resume)
    assert (
        campaign._profile_once(
            tmp_path, identity, problem, dict(problem.start), campaign.profiles.Policy()
        )["status"]
        == "complete"
    )
    checkpoint, records, truncated = campaign._checkpoint(tmp_path, identity)
    assert len(records) == 2 and not truncated
    assert len(list(tmp_path.iterdir())) == 4
    journal = tmp_path / "profile-progress.jsonl"
    records[0]["calls"] += 1
    journal.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="journal differs"):
        campaign._checkpoint(tmp_path, identity)


def test_checkpoint_and_journal_corruption_is_not_silently_restarted(
    tmp_path, monkeypatch
):
    problem, _ = campaign.fixtures.make()
    identity = {"test": "identity"}

    def interrupted(problem, parameters, *, policy, state=None, save=None):
        save(diagnostic("running"))
        raise KeyboardInterrupt

    monkeypatch.setattr(campaign.profiles, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        campaign._profile_once(
            tmp_path, identity, problem, dict(problem.start), campaign.profiles.Policy()
        )
    path = tmp_path / "profile-progress.jsonl"
    record = json.loads(path.read_text())
    record["sequence"] = 99
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="journal differs"):
        campaign._profile_once(
            tmp_path, identity, problem, dict(problem.start), campaign.profiles.Policy()
        )


def test_unknown_diagnostic_timing_cannot_be_complete_or_zero_total():
    result = campaign._diagnostic_cost(
        [
            {
                "profile": {
                    "accounting_complete": True,
                    "calls": 7,
                    "completed_calls": 7,
                    "diagnostic": {"timing": {}},
                }
            }
        ]
    )
    assert result["calls_observed"] == 7 and result["calls_total"] is None
    assert result["wall_seconds_total"] is None and not result["accounting_complete"]


def test_interrupted_call_reservations_are_separate_from_actual_total():
    result = campaign._diagnostic_cost(
        [
            {
                "profile": {
                    "calls": 107,
                    "completed_calls": 7,
                    "wall_seconds": 1.0,
                    "cpu_seconds": 0.5,
                    "accounting_complete": False,
                    "diagnostic": {"interrupted_calls_charged": 100},
                }
            }
        ]
    )
    assert result["calls_observed"] == 107
    assert result["interrupted_calls_charged_observed"] == 100
    assert result["calls_total"] is None

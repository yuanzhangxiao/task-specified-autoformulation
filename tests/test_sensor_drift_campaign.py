"""M36 evaluator separation, bounded diagnostic journals and frozen identities."""

from __future__ import annotations

from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import sensor_drift_campaign as campaign
from autoformalism.schemas.sensor_drift import DriftProblem


@pytest.fixture(autouse=True)
def source_identity(monkeypatch):
    """Make unit identities independent of concurrent unrelated checkout edits."""
    monkeypatch.setattr(campaign.public, "_source_identity", lambda: "a" * 64)


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "campaign"
    assert campaign.prepare(root, None, campaign.Policy())["tasks"] == 144
    return root


def fake_fit(problem, policy, folder, *, strategy, loss):
    """Return public-only fit telemetry, with a deliberately nonoptimal endpoint."""
    assert (
        not {"truth", "evaluation", "ambiguity_witness_evaluator_only"}
        & problem.model_dump().keys()
    )
    return {
        "status": "complete",
        "identity": {
            "protocol": "sensor-drift-fitting-execution-1"
            if isinstance(problem, DriftProblem)
            else "robust-measurement-fitting-execution-1",
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
        "objective_definition": loss.model_dump(mode="json"),
        "verification": {"passed": True},
        "route": {"selected": strategy},
        "cost": {"wall_seconds": 0.1, "accounting_complete": True},
    }


def fake_diagnostic(problem, parameters, *, loss, deadline, save):
    """Exercise the bounded public-only diagnostic boundary and progress journal."""
    assert not {"truth", "evaluation"} & problem.model_dump().keys()
    assert set(parameters) == set(problem.start)
    assert deadline > campaign.monotonic()
    save({"calls": 3, "completed_calls": 3})
    return {
        "information_status": "qualified",
        "derivatives_passed": True,
        "information": {"physical_after_nuisance": {"rank": 1}},
        "calls": 3,
        "completed_calls": 3,
    }


@pytest.fixture
def fake_backend(monkeypatch):
    monkeypatch.setattr(campaign.measurement_fit, "run", fake_fit)
    monkeypatch.setattr(campaign.drift_information, "evaluate", fake_diagnostic)


def test_frozen_roster_pairs_physical_data_starts_and_budgets(prepared):
    plan, data = campaign.verify(prepared)
    tasks = plan["tasks"][:8]
    problems = [campaign.training_problem(data, task) for task in tasks]
    assert all(
        problem.model_dump() == problems[0].model_dump() for problem in problems[:4]
    )
    assert all(problem.base == problems[0] for problem in problems[4:])
    assert all(problem.drift.start == 0 for problem in problems[4:])
    assert all(
        dict(problem.base.start) == dict(problems[0].start) for problem in problems[4:]
    )
    identities = [campaign.fit_identity(plan, data, task) for task in tasks]
    assert len({value["base_request_sha256"] for value in identities}) == 1
    assert all(value["policy"] == plan["policy"] for value in identities)
    assert len({campaign.public.content_sha256(value) for value in identities}) == 8
    data["evaluation"] = dict.fromkeys(data["evaluation"], "forbidden")
    assert campaign.training_problem(data, tasks[4]) == problems[4]


def test_prepare_reuses_inputs_and_missing_arms_stay_in_denominators(
    prepared, monkeypatch
):
    monkeypatch.setattr(
        campaign.sensor_drift_fixtures,
        "make_inputs",
        lambda: pytest.fail("regenerated"),
    )
    campaign.prepare(prepared, None, campaign.Policy())
    summary = campaign.report(prepared)
    assert summary["expected"] == 144 and summary["recorded"] == 0
    assert summary["status_counts"] == {"missing": 144}
    assert summary["diagnostic_status_counts"] == {"missing": 144}
    assert len(summary["groups"]) == 48
    assert len(summary["paired_objectives"]) == 72
    assert len(summary["observation_hypothesis_comparisons"]) == 72
    assert len(summary["cost_by_hypothesis_loss_and_strategy"]) == 8
    assert all(group["expected"] == 3 for group in summary["groups"])
    assert summary["diagnostic_calls_total"] is None
    assert summary["diagnostic_calls_unknown_or_missing"] == 144
    assert summary["observation_hypothesis_selected"] is None
    assert not summary["cost_complete"]


def test_endpoint_and_diagnostic_are_sealed_before_evaluation_and_resume(
    prepared, fake_backend, monkeypatch
):
    events = []
    original = campaign.evaluate

    def score(data, task, backend):
        folder = prepared / "results" / task["task_id"]
        assert read_seal(folder / "backend.json") == backend
        assert read_seal(folder / "diagnostic.json")["status"] == "evaluated"
        events.append("evaluation")
        return original(data, task, backend)

    monkeypatch.setattr(campaign, "evaluate", score)
    row = campaign.run_task(prepared, 4)
    assert row["diagnostic"]["calls"] == 3
    assert row["evaluation"]["analytic_objective_passed"] is False
    monkeypatch.setattr(
        campaign.measurement_fit, "run", lambda *a, **kw: pytest.fail("refitted")
    )
    monkeypatch.setattr(
        campaign.drift_information,
        "evaluate",
        lambda *a, **kw: pytest.fail("rediagnosed"),
    )
    monkeypatch.setattr(
        campaign, "evaluate", lambda *a, **kw: pytest.fail("reevaluated")
    )
    assert campaign.run_task(prepared, 4) == row
    assert events == ["evaluation"]
    summary = campaign.report(prepared)
    assert summary["recorded"] == 1 and summary["diagnostic_calls_observed"] == 3
    assert summary["diagnostic_completed_calls_observed"] == 3


def test_interrupted_diagnostic_consumes_allowance_and_is_not_repeated(
    prepared, fake_backend, monkeypatch
):
    def interrupted(problem, parameters, *, loss, deadline, save):
        save({"calls": 2, "completed_calls": 1})
        raise KeyboardInterrupt("diagnostic interrupted")

    monkeypatch.setattr(campaign.drift_information, "evaluate", interrupted)
    with pytest.raises(KeyboardInterrupt, match="diagnostic interrupted"):
        campaign.run_task(prepared, 4)
    partial = campaign.report(prepared)
    assert partial["recorded"] == 0
    assert partial["diagnostic_status_counts"] == {"unfinished": 1, "missing": 143}
    assert partial["fit_wall_seconds_observed"] == 0.1
    assert partial["diagnostic_calls_observed"] == 2
    assert partial["diagnostic_completed_calls_observed"] == 1
    assert partial["diagnostic_calls_total"] is None
    assert partial["diagnostic_budget_charge_seconds"] == 15
    monkeypatch.setattr(
        campaign.measurement_fit, "run", lambda *a, **kw: pytest.fail("refitted")
    )
    monkeypatch.setattr(
        campaign.drift_information,
        "evaluate",
        lambda *a, **kw: pytest.fail("repeated interrupted diagnostic"),
    )
    resumed = campaign.run_task(prepared, 4)
    assert resumed["diagnostic"]["status"] == "interrupted"
    assert resumed["diagnostic"]["wall_seconds"] is None
    assert not resumed["diagnostic"]["accounting_complete"]
    assert resumed["evaluation"]["status"] == "evaluated"
    assert campaign.run_task(prepared, 4) == resumed


def test_failed_and_unqualified_diagnostics_do_not_block_fit_or_report(
    prepared, fake_backend, monkeypatch
):
    def failed(problem, parameters, *, loss, deadline, save):
        save({"calls": 2, "completed_calls": 1})
        raise ValueError("derivative numerical failure")

    monkeypatch.setattr(campaign.drift_information, "evaluate", failed)
    row = campaign.run_task(prepared, 4)
    assert row["status"] == "complete"
    assert row["diagnostic"]["status"] == "failed"
    assert row["diagnostic"]["observed_progress"]["calls"] == 2
    assert row["diagnostic"]["accounting_complete"]

    def unqualified(*args, **kwargs):
        return fake_diagnostic(*args, **kwargs) | {
            "information_status": "derivatives_unqualified",
            "derivatives_passed": False,
            "information": None,
        }

    monkeypatch.setattr(campaign.drift_information, "evaluate", unqualified)
    campaign.run_task(prepared, 5)
    summary = campaign.report(prepared)
    assert summary["recorded"] == 2
    assert summary["diagnostic_calls_observed"] == 5
    assert summary["diagnostic_calls_unknown_or_missing"] == 143
    assert (
        summary["information_by_hypothesis"][1]["physical_conditional_rank_counts"]
        == {}
    )


def test_fit_failure_counted_and_diagnostic_has_no_parameter_path(
    prepared, monkeypatch
):
    def broken(*args, **kwargs):
        raise RuntimeError("fit failed")

    monkeypatch.setattr(campaign.measurement_fit, "run", broken)
    monkeypatch.setattr(
        campaign.drift_information,
        "evaluate",
        lambda *a, **kw: pytest.fail("diagnosed missing endpoint"),
    )
    row = campaign.run_task(prepared, 4)
    assert row["status"] == "failed"
    assert row["diagnostic"]["status"] == "no_parameters"
    assert row["diagnostic"]["calls"] == 0
    assert row["evaluation"]["status"] == "no_parameters"
    assert campaign.report(prepared)["status_counts"] == {"failed": 1, "missing": 143}


def test_evaluator_interruption_preserves_completed_diagnostic(
    prepared, fake_backend, monkeypatch
):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("evaluation interrupted")

    monkeypatch.setattr(campaign, "evaluate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        campaign.run_task(prepared, 4)
    monkeypatch.setattr(
        campaign.drift_information,
        "evaluate",
        lambda *a, **kw: pytest.fail("repeated diagnostic"),
    )
    monkeypatch.setattr(
        campaign, "evaluate", lambda *a, **kw: pytest.fail("repeated evaluation")
    )
    row = campaign.run_task(prepared, 4)
    assert row["evaluation"]["status"] == "interrupted"
    assert row["diagnostic"]["status"] == "evaluated"
    assert campaign.run_task(prepared, 4) == row


@pytest.mark.parametrize(
    "field",
    ["strategy", "problem_sha256", "objective_definition", "policy", "source_sha256"],
)
def test_changed_backend_identity_rejected_before_diagnostics(
    prepared, fake_backend, monkeypatch, field
):
    row = campaign.run_task(prepared, 4)
    folder = prepared / "results" / row["task_id"]
    (folder / "result.json").unlink()
    backend = read_seal(folder / "backend.json")
    backend["identity"][field] = "wrong"
    (folder / "backend.json").unlink()
    seal(folder / "backend.json", backend)
    monkeypatch.setattr(
        campaign.drift_information,
        "evaluate",
        lambda *a, **kw: pytest.fail("stale backend diagnosed"),
    )
    with pytest.raises(ValueError, match="backend identity differs"):
        campaign.run_task(prepared, 4)
    with pytest.raises(ValueError, match="backend identity differs"):
        campaign.report(prepared)


@pytest.mark.parametrize("artifact", ["diagnostic.json", "evaluation.json"])
def test_copied_other_endpoint_diagnostic_or_evaluation_rejected(
    prepared, fake_backend, artifact
):
    first, second = campaign.run_task(prepared, 4), campaign.run_task(prepared, 5)
    folder = prepared / "results" / second["task_id"]
    (folder / "result.json").unlink()
    (folder / artifact).unlink()
    seal(folder / artifact, first[artifact.removesuffix(".json")])
    with pytest.raises(ValueError, match="identity differs"):
        campaign.run_task(prepared, 5)
    with pytest.raises(ValueError, match="identity differs"):
        campaign.report(prepared)


def test_diagnostic_progress_identity_and_terminal_content_are_checked(
    prepared, fake_backend, monkeypatch
):
    row = campaign.run_task(prepared, 4)
    folder = prepared / "results" / row["task_id"]
    diagnostic = read_seal(folder / "diagnostic.json")
    diagnostic["calls"] = 999
    (folder / "diagnostic.json").unlink()
    seal(folder / "diagnostic.json", diagnostic)
    with pytest.raises(ValueError, match="diagnostic differs"):
        campaign.run_task(prepared, 4)
    (folder / "result.json").unlink()
    (folder / "diagnostic.json").unlink()
    campaign.public._write(
        folder / "diagnostic-progress.json",
        {"identity": {}, "calls": 3, "completed_calls": 2},
    )
    monkeypatch.setattr(
        campaign.drift_information,
        "evaluate",
        lambda *a, **kw: pytest.fail("repeated diagnostic"),
    )
    with pytest.raises(ValueError, match="progress identity differs"):
        campaign.run_task(prepared, 4)


def test_policy_raw_inputs_and_diagnostic_policy_cannot_change_on_resume(
    prepared, monkeypatch
):
    with pytest.raises(ValueError, match="sealed artifact differs"):
        campaign.prepare(prepared, None, campaign.Policy(maximum_calls=101))
    plan, data = campaign.verify(prepared)
    changed = deepcopy(data)
    key = plan["tasks"][0]["common"]
    changed["problems"][key]["drift"]["upper"] = 0.3
    other = prepared.parent / "changed.json"
    seal(other, changed)
    with pytest.raises(ValueError, match="sealed artifact differs"):
        campaign.prepare(prepared, other, campaign.Policy())
    monkeypatch.setattr(campaign, "DIAGNOSTIC_SECONDS", 20.0)
    with pytest.raises(ValueError, match="campaign identity differs"):
        campaign.verify(prepared)


def test_partial_fit_result_cost_retained_before_diagnostic(prepared):
    plan, data = campaign.verify(prepared)
    task = plan["tasks"][4]
    backend = fake_fit(
        campaign.training_problem(data, task),
        campaign.Policy.model_validate(plan["policy"]),
        prepared,
        strategy=task["strategy"],
        loss=campaign.Loss.model_validate(task["loss"]),
    )
    seal(prepared / "results" / task["task_id"] / "fit/result.json", backend)
    summary = campaign.report(prepared)
    assert summary["recorded"] == 0
    assert summary["fit_wall_seconds_observed"] == 0.1
    assert summary["diagnostic_calls_observed"] == 0
    assert not summary["cost_complete"]


def test_unknown_task_index_fails_closed(prepared):
    with pytest.raises(ValueError, match="task index"):
        campaign.run_task(prepared, -1)
    with pytest.raises(ValueError, match="task index"):
        campaign.run_task(prepared, 144)

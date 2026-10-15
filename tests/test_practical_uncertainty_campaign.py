"""M39 frozen pairing, post-fit profile journals, support summaries and accounting."""

import json
from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import practical_uncertainty_campaign as campaign


@pytest.fixture(autouse=True)
def stable_source(monkeypatch):
    monkeypatch.setattr(campaign.public, "_source_identity", lambda: "a" * 64)


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "campaign"
    assert campaign.prepare(root, None, campaign.Policy())["tasks"] == 150
    return root


def fake_fit(problem, policy, folder, *, strategy, loss):
    """Use only the public problem, deliberately keeping the poor initial endpoint."""
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
        "objective_definition": loss.model_dump(mode="json"),
        "verification": {"passed": True},
        "cost": {"wall_seconds": 0.1, "accounting_complete": True},
    }


def support(*, active=False):
    """Supply an interior crossing with optional nuisance-domain assistance."""
    endpoint = {
        "status": "threshold_crossed",
        "nuisance_active_bounds": [{"name": "x0", "side": "lower", "value": 0.01}]
        if active
        else [],
    }
    return {
        "width": 1.0,
        "lower": 1.0,
        "upper": 2.0,
        "domain_clipped": False,
        "left": endpoint,
        "right": endpoint,
        "classification": "nuisance_domain_assisted"
        if active
        else "interior_data_limited",
    }


def fake_profile(problem, endpoint, *, target, policy, deadline, save):
    """Return public-only profile evidence and journal a completed numerical point."""
    assert not {"truth", "evaluation"} & problem.model_dump().keys()
    assert target == "g" and endpoint["parameters"] == dict(problem.start)
    assert deadline > campaign.monotonic()
    save(
        {
            "calls": 4,
            "completed_calls": 4,
            "solve_calls": 3,
            "rollout_calls": 1,
            "inner_iterations": 3,
            "nodes_completed": 1,
            "event": {"phase": "node_completed", "point": 1.3},
        }
    )
    sigma = problem.manifest.noise.sigma[0]
    levels = [
        next(reading.raw_value for reading in row.readings if reading.channel == "u")
        for row in problem.training.rows
    ]
    separation = (max(levels) - min(levels)) / 2
    return {
        "status": "complete",
        "calls": 4,
        "completed_calls": 4,
        "solve_calls": 3,
        "rollout_calls": 1,
        "inner_iterations": 3,
        "inner_iterations_complete": True,
        "nodes_completed": 1,
        "supports": [
            {"threshold": threshold, "profiled": support(), "frozen": support()}
            for threshold in policy.thresholds
        ],
        "curvature": {
            "conditional_information": 187 * separation**2 / sigma**2,
            "frozen_information": 93.5 * (3 + 2 * separation**2) / sigma**2,
            "joint_rank": 2 if separation == 0 else 3,
            "conditional_rank": int(separation > 0),
        },
        "nodes": [{"point": 1.3}],
        "confidence_probability": None,
    }


@pytest.fixture
def fake_backend(monkeypatch):
    monkeypatch.setattr(campaign.measurement_fit, "run", fake_fit)
    monkeypatch.setattr(campaign.practical_profile, "evaluate", fake_profile)


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
    assert campaign.training_problem(data, first).start["g"] == 0.45
    monkeypatch.setattr(
        campaign.fixtures, "make_inputs", lambda: pytest.fail("regenerated")
    )
    campaign.prepare(prepared, None, campaign.Policy())
    summary = campaign.report(prepared)
    assert summary["expected"] == 150 and summary["recorded"] == 0
    assert summary["status_counts"] == {"missing": 150}
    assert summary["profile_status_counts"] == {"missing": 150}
    assert len(summary["groups"]) == 30 and len(summary["paired_objectives"]) == 75
    assert all(group["expected"] == 5 for group in summary["groups"])
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
    assert first["evaluation"]["analytic_objective_passed"] is False
    monkeypatch.setattr(
        campaign.measurement_fit, "run", lambda *a, **kw: pytest.fail("refitted")
    )
    monkeypatch.setattr(
        campaign.practical_profile,
        "evaluate",
        lambda *a, **kw: pytest.fail("reprofiled"),
    )
    monkeypatch.setattr(
        campaign, "evaluate", lambda *a, **kw: pytest.fail("reevaluated")
    )
    assert campaign.run_task(prepared, 0) == first
    assert events == ["evaluated"]
    summary = campaign.report(prepared)
    assert summary["recorded"] == 1
    assert summary["cost"]["profile"]["calls_observed"] == 4
    group = summary["groups"][0]
    assert group["conditional_information_analytic_passed"] == 1
    assert group["frozen_information_analytic_passed"] == 1


def test_profile_interruption_retains_events_and_allowance_without_repeating(
    prepared, fake_backend, monkeypatch
):
    def interrupted(problem, endpoint, *, target, policy, deadline, save):
        save(
            {
                "calls": 8,
                "completed_calls": 7,
                "solve_calls": 7,
                "rollout_calls": 1,
                "inner_iterations": 9,
                "nodes_completed": 3,
                "event": {"phase": "node_completed", "point": 0.5},
            }
        )
        raise KeyboardInterrupt("profile interrupted")

    monkeypatch.setattr(campaign.practical_profile, "evaluate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        campaign.run_task(prepared, 0)
    partial = campaign.report(prepared)
    cost = partial["cost"]["profile"]
    assert partial["recorded"] == 0
    assert partial["cost"]["fit"]["observed"]["wall_seconds"] == 0.1
    assert cost["calls_observed"] == 8 and cost["completed_calls_observed"] == 7
    assert cost["solve_calls_observed"] == 7 and cost["nodes_completed_observed"] == 3
    assert cost["calls_total"] is None and cost["budget_charge_seconds"] == 30
    row = next(row for row in partial["rows"] if "profile" in row)
    assert len(row["profile"]["observed_progress"]["events"]) == 1
    monkeypatch.setattr(
        campaign.practical_profile,
        "evaluate",
        lambda *a, **kw: pytest.fail("repeated interrupted profile"),
    )
    resumed = campaign.run_task(prepared, 0)
    assert resumed["profile"]["status"] == "interrupted"
    assert resumed["profile"]["wall_seconds"] is None
    assert resumed["evaluation"]["status"] == "evaluated"
    assert campaign.run_task(prepared, 0) == resumed


def test_profile_event_journal_uses_bounded_file_count(tmp_path):
    identity = {"stage": "test"}

    def many_events(deadline, save):
        for index in range(500):
            save(
                {
                    "calls": index + 1,
                    "completed_calls": index + 1,
                    "event": {"point": index},
                }
            )
        return {"status": "complete", "calls": 500, "completed_calls": 500}

    value = campaign._stage_once(tmp_path, "profile", identity, many_events, 30)
    assert len(list(tmp_path.rglob("*"))) == 3
    assert len(value["event_history"]) == 500
    assert len(campaign._progress(tmp_path, "profile", identity)["events"]) == 500
    journal = tmp_path / "profile-progress.jsonl"
    assert len(journal.read_text().splitlines()) == 501
    assert journal.stat().st_size < 150_000


def test_torn_journal_tail_preserves_known_started_work_without_retry(tmp_path):
    identity = {"stage": "test"}

    def interrupted(deadline, save):
        save({"calls": 2, "completed_calls": 1, "event": {"status": "started"}})
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        campaign._stage_once(tmp_path, "profile", identity, interrupted, 30)
    journal = tmp_path / "profile-progress.jsonl"
    with journal.open("ab") as stream:
        stream.write(b'{"unfinished":')
    value = campaign._stage_once(
        tmp_path, "profile", identity, lambda *args: pytest.fail("retried"), 30
    )
    assert value["status"] == "interrupted"
    assert value["budget_charge_seconds"] == 30
    assert value["observed_progress"]["calls"] == 2
    assert value["observed_progress"]["completed_calls"] == 1
    assert value["observed_progress"]["truncated_final_record"]
    assert not value["accounting_complete"]


@pytest.mark.parametrize("field,value", [("identity_sha256", "wrong"), ("sequence", 9)])
def test_journal_complete_record_corruption_is_not_ignored(tmp_path, field, value):
    identity = {"stage": "test"}
    record = {
        "identity_sha256": campaign.public.content_sha256(identity),
        "sequence": 0,
        "value": {"calls": 0, "completed_calls": 0},
    }
    record[field] = value
    (tmp_path / "profile-progress.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="progress identity differs"):
        campaign._progress(tmp_path, "profile", identity)


def test_partial_inner_iterations_remain_observed_without_complete_total():
    value = campaign._stage_cost(
        [
            {
                "profile": {
                    "status": "partial",
                    "accounting_complete": True,
                    "calls": 8,
                    "completed_calls": 7,
                    "inner_iterations": 12,
                    "inner_iterations_complete": False,
                }
            }
        ],
        "profile",
    )
    assert value["inner_iterations_observed"] == 12
    assert value["inner_iterations_total"] is None
    assert not value["accounting_complete"]


@pytest.mark.parametrize("outcome", ["failed", "partial", "unqualified"])
def test_profile_failure_or_incomplete_evidence_does_not_block_evaluator(
    prepared, fake_backend, monkeypatch, outcome
):
    def incomplete(*args, **kwargs):
        value = fake_profile(*args, **kwargs)
        if outcome == "failed":
            raise ValueError("deliberate profile failure")
        return value | {"status": outcome}

    monkeypatch.setattr(campaign.practical_profile, "evaluate", incomplete)
    row = campaign.run_task(prepared, 0)
    assert row["status"] == "complete"
    assert row["profile"]["status"] == outcome
    assert row["evaluation"]["status"] == "evaluated"
    summary = campaign.report(prepared)
    assert summary["groups"][0]["supports"][0]["width"]["available"] == 0


def test_evaluator_interruption_keeps_completed_profile(
    prepared, fake_backend, monkeypatch
):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("evaluation interrupted")

    monkeypatch.setattr(campaign, "evaluate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        campaign.run_task(prepared, 0)
    monkeypatch.setattr(
        campaign.practical_profile,
        "evaluate",
        lambda *a, **kw: pytest.fail("repeated profile"),
    )
    row = campaign.run_task(prepared, 0)
    assert row["profile"]["status"] == "complete"
    assert row["evaluation"]["status"] == "interrupted"


def test_fit_failure_has_explicit_missing_parameter_profile(prepared, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("fit failed")

    monkeypatch.setattr(campaign.measurement_fit, "run", broken)
    monkeypatch.setattr(
        campaign.practical_profile,
        "evaluate",
        lambda *a, **kw: pytest.fail("profile without endpoint"),
    )
    row = campaign.run_task(prepared, 0)
    assert row["status"] == "failed"
    assert row["profile"]["status"] == "no_parameters"
    assert row["evaluation"]["status"] == "no_parameters"
    assert campaign.report(prepared)["status_counts"] == {"failed": 1, "missing": 149}


@pytest.mark.parametrize(
    "field", ["strategy", "problem_sha256", "policy", "objective_definition"]
)
def test_changed_backend_identity_rejected_before_postfit_work(
    prepared, fake_backend, field
):
    row = campaign.run_task(prepared, 0)
    folder = prepared / "results" / row["task_id"]
    (folder / "result.json").unlink()
    backend = read_seal(folder / "backend.json")
    backend["identity"][field] = "wrong"
    (folder / "backend.json").unlink()
    seal(folder / "backend.json", backend)
    with pytest.raises(ValueError, match="backend identity differs"):
        campaign.run_task(prepared, 0)
    with pytest.raises(ValueError, match="backend identity differs"):
        campaign.report(prepared)


@pytest.mark.parametrize("name", ["profile", "evaluation"])
def test_copied_stage_from_other_endpoint_cannot_resume(prepared, fake_backend, name):
    first, second = campaign.run_task(prepared, 0), campaign.run_task(prepared, 1)
    folder = prepared / "results" / second["task_id"]
    (folder / "result.json").unlink()
    (folder / f"{name}.json").unlink()
    seal(folder / f"{name}.json", first[name])
    with pytest.raises(ValueError, match="stage identity differs"):
        campaign.run_task(prepared, 1)
    with pytest.raises(ValueError, match="stage identity differs"):
        campaign.report(prepared)


def test_raw_sigma_policy_and_profile_allowance_changes_cannot_resume(
    prepared, monkeypatch
):
    with pytest.raises(ValueError, match="sealed artifact differs"):
        campaign.prepare(prepared, None, campaign.Policy(maximum_calls=101))
    plan, data = campaign.verify(prepared)
    changed = deepcopy(data)
    changed["problems"][plan["tasks"][0]["common"]]["base"]["manifest"]["noise"][
        "sigma"
    ][0] *= 2
    other = prepared.parent / "other.json"
    seal(other, changed)
    with pytest.raises(ValueError, match="known sigma/mask differs"):
        campaign.prepare(prepared, other, campaign.Policy())
    monkeypatch.setattr(campaign, "PROFILE_SECONDS", 31.0)
    with pytest.raises(ValueError, match="campaign identity differs"):
        campaign.verify(prepared)


def test_summary_separates_nuisance_bounds_partial_widths_and_frozen_slice():
    rows = [
        {
            "profile": {
                "status": "complete",
                "supports": [
                    {
                        "threshold": 0.5,
                        "profiled": support(active=True),
                        "frozen": support(),
                    }
                ],
            }
        },
        {
            "profile": {
                "status": "partial",
                "supports": [
                    {"threshold": 0.5, "profiled": support(), "frozen": support()}
                ],
            }
        },
    ]
    profiled = campaign._support_summary(rows, 0.5, "profiled")
    assert profiled["width"]["available"] == 1
    assert profiled["interior_data_limited_width"]["available"] == 0
    assert profiled["nuisance_domain_assisted"] == 1
    assert (
        campaign._support_summary(rows, 0.5, "frozen")["interior_data_limited_width"][
            "available"
        ]
        == 0
    )
    assert campaign._statistics([1.0, 3.0, 5.0, None]) == {
        "available": 3,
        "missing": 1,
        "median": 3.0,
        "mad": 2.0,
    }


def test_curvature_validation_uses_exact_public_design_and_rejects_wrong_value():
    row = {
        "separation": 0.02,
        "sigma": 0.05,
        "profile": {
            "curvature": {
                "conditional_information": 187 * 0.02**2 / 0.05**2,
                "frozen_information": 93.5 * (3 + 2 * 0.02**2) / 0.05**2,
            }
        },
    }
    value = campaign.curvature_validation(row)
    assert value["conditional_passed"] and value["frozen_passed"]
    row["profile"]["curvature"]["conditional_information"] *= 2
    assert campaign.curvature_validation(row)["conditional_passed"] is False
    assert (
        campaign.curvature_validation({"separation": 0, "sigma": 0.2})[
            "conditional_passed"
        ]
        is None
    )


def test_unknown_task_index_fails_closed(prepared):
    with pytest.raises(ValueError, match="task index"):
        campaign.run_task(prepared, -1)
    with pytest.raises(ValueError, match="task index"):
        campaign.run_task(prepared, 150)

"""M40 preserves M39 fits and costs while extending only incomplete diagnostics."""

import json
import shutil
from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import seal
from autoformalism.fitting import practical_uncertainty_supplement as campaign


@pytest.fixture(autouse=True)
def stable_source(monkeypatch):
    monkeypatch.setattr(campaign.public, "_source_identity", lambda: "a" * 64)


@pytest.fixture(scope="module")
def source_template(tmp_path_factory):
    """Retain 150 sealed endpoint stand-ins; numerical fitting has separate tests."""
    root = tmp_path_factory.mktemp("m39-source")
    previous = campaign.previous
    previous.prepare(root, None, previous.Policy())
    plan, data = previous.verify(root, runtime=False)
    for index, task in enumerate(plan["tasks"]):
        identity = {"plan_sha256": campaign.public.content_sha256(plan), "task": task}
        problem = previous.training_problem(data, task)
        backend = {
            "status": "complete",
            "identity": previous.fit_identity(plan, data, task),
            "campaign_identity": identity,
            "parameters": dict(problem.start),
            "objective_value": 10.0,
            "verification": {"passed": True},
            "cost": {
                "wall_seconds": 1.0,
                "cpu_seconds": 0.5,
                "accounting_complete": True,
            },
        }
        folder = root / "results" / task["task_id"]
        seal(folder / "backend.json", backend)
        stages = {}
        for name, definition in previous._stage_definitions(plan, data, task).items():
            stage_identity = previous.stage_identity(identity, backend, definition)
            status = "partial" if index < 9 else "complete"
            stage = {
                "identity": stage_identity,
                "status": status if name == "profile" else "evaluated",
                "wall_seconds": 30.0 if index < 9 else 10.0,
                "cpu_seconds": 2.0,
                "budget_charge_seconds": 30.0 if index < 9 else 10.0,
                "accounting_complete": True,
                "calls": 4,
                "completed_calls": 4,
                "solve_calls": 3,
                "rollout_calls": 1,
                "inner_iterations": 3,
                "inner_iterations_complete": True,
                "nodes_completed": 1,
                "failed_calls": 0,
            }
            seal(folder / f"{name}-started.json", stage_identity)
            seal(folder / f"{name}.json", stage)
            stages[name] = stage
        seal(
            folder / "result.json",
            task
            | {
                "identity": identity,
                "status": "complete",
                "backend": backend,
                "backend_sha256": campaign.public.content_sha256(backend),
            }
            | stages,
        )
    return root


@pytest.fixture
def source(source_template, tmp_path):
    destination = tmp_path / "source"
    shutil.copytree(source_template, destination)
    return destination


@pytest.fixture
def prepared(source, tmp_path):
    root = tmp_path / "supplement"
    assert campaign.prepare(root, source, campaign.Policy())["tasks"] == 9
    return root


def profile_result():
    return {
        "status": "complete",
        "calls": 3,
        "completed_calls": 3,
        "solve_calls": 1,
        "rollout_calls": 1,
        "frozen_calls": 1,
        "frozen_nodes_completed": 1,
        "inner_iterations": 1,
        "inner_iterations_complete": True,
        "nodes_completed": 1,
        "failed_calls": 0,
        "nodes": [{"point": 1.3}],
        "supports": [],
        "timing": {"numerical_wall_seconds": 0.01, "numerical_cpu_seconds": 0.005},
    }


def test_selection_preserves_every_source_endpoint_and_separates_allowances(
    source, prepared, monkeypatch
):
    before = campaign._files(source)
    monkeypatch.setattr(
        campaign.previous.measurement_fit, "run", lambda *a, **kw: pytest.fail("refit")
    )
    monkeypatch.setattr(
        campaign.previous, "evaluate", lambda *a, **kw: pytest.fail("truth evaluator")
    )
    plan, _ = campaign.verify(prepared)
    assert len(plan["tasks"]) == 9
    assert plan["diagnostic"]["maximum_seconds"] == 30
    assert plan["policy"]["maximum_seconds"] == 90
    summary = campaign.report(prepared)
    assert summary["source_verified_fits"] == 150
    assert summary["source_profile_status_counts"] == {"partial": 9, "complete": 141}
    assert (
        summary["combined_profile_status_counts"]
        == summary["source_profile_status_counts"]
    )
    assert summary["expected"] == 9 and summary["recorded"] == 0
    assert len(summary["rows"]) == 150
    assert summary["cost"]["source_fit_unchanged"]["observed"]["wall_seconds"] == 150
    assert summary["cost"]["extension_diagnostic"]["wall_seconds_total"] is None
    assert summary["cost"]["source_diagnostic"]["calls_total"] == 600
    assert before == campaign._files(source)


def test_public_endpoint_only_extension_is_immutable_and_portable(
    prepared, tmp_path, monkeypatch
):
    captured = []

    def evaluate(problem, endpoint, *, target, policy, execution, deadline, save):
        assert "truth" not in problem.model_dump()
        assert (
            target == "g"
            and policy.model_dump(mode="json") == campaign.practical_profile.POLICY
        )
        assert (
            execution.model_dump(mode="json")
            == campaign.practical_profile.EXECUTION_POLICY
        )
        assert deadline > campaign.monotonic()
        captured.append(endpoint)
        value = profile_result()
        save(
            {key: value[key] for key in campaign.COUNTERS}
            | {"event": {"status": "completed"}}
        )
        return value

    monkeypatch.setattr(campaign.practical_profile, "evaluate", evaluate)
    result = campaign.run_task(prepared, 0)
    monkeypatch.setattr(
        campaign.practical_profile,
        "evaluate",
        lambda *a, **kw: pytest.fail("reprofile"),
    )
    assert campaign.run_task(prepared, 0) == result
    assert len(captured) == 1
    assert result["profile"]["accounting_complete"]
    assert result["profile"]["journal_timing"]["records"] == 2
    assert result["profile"]["journal_timing"]["bytes"] > 0
    assert "excluded" in result["profile"]["timing_scope"]
    first = campaign.report(prepared)
    assert first["combined_profile_status_counts"] == {"complete": 142, "partial": 8}
    assert first["cost"]["combined_diagnostic"]["calls_observed"] == 603
    assert first["cost"]["extension_diagnostic"]["calls_total"] is None
    assert (
        first["cost"]["extension_diagnostic"]["timing"]["numerical_wall_seconds"][
            "observed"
        ]
        == 0.01
    )
    moved = tmp_path / "moved"
    shutil.move(str(prepared), moved)
    monkeypatch.setattr(campaign.public, "_source_identity", lambda: "other-platform")
    assert campaign.report(moved) == first
    with pytest.raises(ValueError, match="source/runtime"):
        campaign.run_task(moved, 0)


def test_started_extension_never_receives_a_second_budget(prepared, monkeypatch):
    def interrupted(*args, save, **kwargs):
        save(
            {
                "calls": 2,
                "completed_calls": 1,
                "solve_calls": 1,
                "event": {"phase": "solve", "status": "started"},
            }
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(campaign.practical_profile, "evaluate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        campaign.run_task(prepared, 0)
    pending = campaign.report(prepared)
    cost = pending["cost"]["extension_diagnostic"]
    assert cost["calls_observed"] == 2 and cost["completed_calls_observed"] == 1
    assert cost["calls_total"] is None and cost["budget_charge_seconds"] == 90
    monkeypatch.setattr(
        campaign.practical_profile, "evaluate", lambda *a, **kw: pytest.fail("retried")
    )
    result = campaign.run_task(prepared, 0)
    assert result["status"] == "interrupted"
    assert result["profile"]["wall_seconds"] is None
    assert result["profile"]["budget_charge_seconds"] == 90
    assert not result["profile"]["accounting_complete"]
    assert campaign.run_task(prepared, 0) == result


@pytest.mark.parametrize("kind", ["raised", "incomplete", "counters"])
def test_failed_or_unknown_numerical_usage_is_not_marked_complete(tmp_path, kind):
    def operation(deadline, save):
        save({"calls": 2, "completed_calls": 1})
        if kind == "raised":
            raise RuntimeError("failed before terminal callback")
        value = profile_result()
        if kind == "incomplete":
            value["accounting_complete"] = False
        else:
            value["calls"] = 1
        return value

    value = campaign._profile_once(tmp_path, {"task": "test"}, 90, operation)
    assert not value["accounting_complete"]
    assert value["wall_seconds"] is not None
    if kind == "raised":
        assert value["observed_progress"]["calls"] == 2
        assert campaign._cost([value])["calls_total"] is None


def test_journal_uses_constant_files_and_torn_tail_preserves_evidence(tmp_path):
    identity = {"task": "test"}

    def operation(deadline, save):
        for index in range(500):
            save(
                {
                    "calls": index + 1,
                    "completed_calls": index,
                    "event": {"status": "started"},
                }
            )
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        campaign._profile_once(tmp_path, identity, 90, operation)
    assert len(list(tmp_path.iterdir())) == 2
    journal = tmp_path / "profile-progress.jsonl"
    assert len(journal.read_text().splitlines()) == 501
    assert journal.stat().st_size < 150_000
    with journal.open("ab") as stream:
        stream.write(b'{"torn":')
    evidence = campaign._progress(tmp_path, identity)
    assert evidence["calls"] == 500 and evidence["completed_calls"] == 499
    assert evidence["truncated_final_record"]
    result = campaign._profile_once(
        tmp_path, identity, 90, lambda *a: pytest.fail("retry")
    )
    assert result["status"] == "interrupted"
    assert len(result["observed_progress"]["events"]) == 500


@pytest.mark.parametrize("field", ["sequence", "identity_sha256", "counter"])
def test_journal_corruption_is_rejected(tmp_path, field):
    identity = {"task": "test"}
    record = {
        "sequence": 0,
        "identity_sha256": campaign.public.content_sha256(identity),
        "value": {"calls": 1},
    }
    if field == "counter":
        record["value"]["calls"] = -1
    else:
        record[field] = "different"
    (tmp_path / "profile-progress.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="progress"):
        campaign._progress(tmp_path, identity)


def test_source_snapshot_and_policy_cannot_change(prepared, source):
    first = campaign.prepare(prepared, source, campaign.Policy())
    assert first["tasks"] == 9
    with pytest.raises(ValueError, match="source/policy"):
        campaign.prepare(prepared, source, campaign.Policy(maximum_seconds=100))
    source_path = prepared / "source_snapshot/inputs.json"
    source_path.write_bytes(source_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="snapshot/identity"):
        campaign.verify(prepared)


def test_source_without_verified_fit_cannot_prepare(source, tmp_path, monkeypatch):
    original = campaign.previous.report_row

    def unverified(*args):
        row = original(*args)
        row["backend"]["verification"]["passed"] = False
        return row

    monkeypatch.setattr(campaign.previous, "report_row", unverified)
    with pytest.raises(ValueError, match="every source fit to be verified"):
        campaign.prepare(tmp_path / "supplement", source, campaign.Policy())


def test_source_selection_ignores_retained_evaluator_values(source, monkeypatch):
    expected = campaign.audit(source)["selected_task_ids"]
    plan, data, rows = campaign._source_rows(source)
    for row in rows:
        row["evaluation"] = {"truth_error": 1e100, "analytic_objective_passed": False}
    monkeypatch.setattr(campaign, "_source_rows", lambda _: (plan, data, rows))
    assert campaign.audit(source)["selected_task_ids"] == expected


@pytest.mark.parametrize("failure", ["serialization", "write"])
def test_failed_journal_callbacks_preserve_timing_and_accounting(
    tmp_path, monkeypatch, failure
):
    def operation(deadline, save):
        value = profile_result()
        if failure == "write":
            original = campaign.os.fsync

            def fail_once(fd):
                monkeypatch.setattr(campaign.os, "fsync", original)
                raise OSError("journal disk failure")

            monkeypatch.setattr(campaign.os, "fsync", fail_once)
        try:
            save(
                {"calls": 1, "bad": float("nan")}
                if failure == "serialization"
                else {"calls": 1}
            )
        except (ValueError, OSError):
            value["journal_persistence_complete"] = False
            value["status"] = "partial"
        return value

    value = campaign._profile_once(tmp_path, {"task": "test"}, 90, operation)
    assert not value["journal_persistence_complete"]
    assert not value["accounting_complete"]
    key = (
        "serialization_wall_seconds"
        if failure == "serialization"
        else "write_fsync_wall_seconds"
    )
    assert value["journal_timing"][key] > 0


def test_unknown_index_and_overlapping_source_are_rejected(prepared, source):
    with pytest.raises(ValueError, match="unknown"):
        campaign.run_task(prepared, 9)
    with pytest.raises(ValueError, match="separate"):
        campaign.prepare(source / "extension", source, campaign.Policy())


def test_resealed_terminal_body_cannot_override_profile(prepared, monkeypatch):
    monkeypatch.setattr(
        campaign.practical_profile, "evaluate", lambda *a, **kw: profile_result()
    )
    result = campaign.run_task(prepared, 0)
    folder = prepared / "results" / result["task_id"]
    changed = deepcopy(result)
    changed["profile"]["calls"] += 1
    (folder / "result.json").unlink()
    seal(folder / "result.json", changed)
    with pytest.raises(ValueError, match="terminal identity"):
        campaign.run_task(prepared, 0)

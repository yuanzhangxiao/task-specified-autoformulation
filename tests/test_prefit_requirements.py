"""Saved-model campaign provenance, paired control, atomic rollback and resume."""

import copy
import json
from pathlib import Path

import pytest

from autoformalism.llm.staged_topology import DeferredCall, atomic_json
from autoformalism.rebuttal import prefit_requirements as campaign
from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts.smoke_prefit_requirements import repair_client, synthetic_plan


@pytest.fixture
def frozen(tmp_path):
    return tmp_path / "campaign", synthetic_plan(tmp_path)


def repair_task(plan):
    return next(
        t
        for t in plan["tasks"]
        if t["cohort"] == "repair" and t["arm"] == "requirement_feedback"
    )


def test_paired_runtime_baseline_and_preserved_controls(frozen):
    root, plan = frozen
    calls = []
    for task in plan["tasks"]:
        result = campaign.run_episode(
            root, plan, task, repair_client(root, plan, task, calls)
        )
        count = len(calls)
        assert (
            campaign.run_episode(
                root, plan, task, repair_client(root, plan, task, calls)
            )
            == result
        )
        assert len(calls) == count
    assert len(calls) == 1
    report = campaign.summarize(root)
    assert report["status"] == "complete" and report["source_models"] == 2
    assert report["distinct_source_gaps"] == 1
    assert report["paired_repairs"] == {"feedback_resolved_local_gap": 1}
    assert report["arms"]["local_only"]["repair"]["requirement_gap_remaining"] == 1
    for arm in campaign.ARMS:
        assert report["arms"][arm]["control"]["changed_models"] == 0
        assert report["arms"][arm]["control"]["physical_requests"] == 0
    assert campaign.verify(root) == plan


def test_freeze_opens_no_trajectory_or_fit_artifacts(frozen, monkeypatch):
    root, plan = frozen
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path.suffix != ".csv"
        assert not (path.suffix == ".json" and path.name.startswith("fit"))
        assert "private_reference" not in path.parts
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    assert (
        campaign.freeze(root.parent / "source", root.parent / "config.json", root)
        == plan
    )
    assert not list(root.rglob("*.csv"))


def test_resume_independent_of_original_source(frozen, monkeypatch):
    root, plan = frozen
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert not path.resolve().is_relative_to(root.parent / "source")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    task = repair_task(plan)
    result = campaign.run_episode(root, plan, task, repair_client(root, plan, task, []))
    assert not result["final"]["diagnosis"]["requirement_gap"]
    campaign.summarize(root)


def test_freeze_refuses_changed_source_model_or_stale_audit(frozen):
    root, _ = frozen
    source = root.parent / "source"
    audit = next(source.glob("results/*/audit.json"))
    payload = json.loads(audit.read_text())
    payload["handoff"]["candidate"]["change_summary"] = "changed"
    atomic_json(audit, payload)
    with pytest.raises(ValueError, match="digest"):
        campaign.freeze(source, root.parent / "config.json", root.parent / "other")


def test_wrong_expected_source_plan_is_rejected(frozen):
    root, _ = frozen
    path = root.parent / "config.json"
    config = json.loads(path.read_text())
    config["expected_source_plan_sha256"] = "1" * 64
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="pinned experiment"):
        campaign.freeze(root.parent / "source", path, root.parent / "new")


def test_no_op_retries_are_bounded_and_parent_remains_unchanged(frozen):
    root, plan = frozen
    task, calls = repair_task(plan), []
    before = copy.deepcopy(plan)
    result = campaign.run_episode(
        root, plan, task, repair_client(root, plan, task, calls, failing=True)
    )
    assert len(calls) == len(result["attempts"]) == 3
    assert result["stop_reason"] == "attempts_exhausted"
    assert result["final"]["diagnosis"]["requirement_gap"]
    assert all(
        e["feedback"]["transaction"] == "rolled_back" for e in result["attempts"]
    )
    assert (
        campaign.run_episode(root, plan, task, repair_client(root, plan, task, calls))
        == result
    )
    assert len(calls) == 3 and before == plan


def test_malformed_rhs_is_feedback_not_worker_failure(frozen):
    root, plan = frozen
    task, calls = repair_task(plan), []
    client = repair_client(root, plan, task, calls)
    valid = client.transport

    def transport(url, body, timeout):
        result = valid(url, body, timeout)
        if len(calls) == 1:
            raw = json.loads(result["choices"][0]["message"]["content"])
            raw["expression"] = "m = invalid"
            result["choices"][0]["message"]["content"] = json.dumps(raw)
        return result

    client.transport = transport
    result = campaign.run_episode(root, plan, task, client)
    assert len(calls) == 2 and result["stop_reason"] == "requirement_syntax_repaired"
    assert result["attempts"][0]["feedback"]["transaction"] == "rolled_back"
    retry = json.loads(calls[1]["messages"][1]["content"].split("\n", 1)[1])
    assert "SYNTAX_ERROR" in retry["retry_feedback"]["error"]


def test_deferred_work_remains_pending_and_resumes(frozen):
    root, plan = frozen
    task, calls = repair_task(plan), []
    with pytest.raises(DeferredCall):
        campaign.run_episode(
            root,
            plan,
            task,
            repair_client(root, plan, task, calls, can_start=lambda: False),
        )
    assert not calls and campaign.summarize(root)["status"] == "partial"
    assert (
        campaign.run_episode(root, plan, task, repair_client(root, plan, task, calls))[
            "status"
        ]
        == "complete"
    )


def test_uncertain_call_is_charged_and_not_resent(frozen):
    root, plan = frozen
    task, calls = repair_task(plan), []
    client = repair_client(root, plan, task, calls)
    client.transport = lambda *args: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        campaign.run_episode(root, plan, task, client)
    result = campaign.run_episode(
        root, plan, task, repair_client(root, plan, task, calls)
    )
    assert len(calls) == 1 and len(result["attempts"]) == 2
    row = next(
        r for r in campaign.summarize(root)["rows"] if r["task_id"] == task["task_id"]
    )
    assert row["physical_requests"] == 2 and row["unknown_usage_requests"] == 1


def test_cache_tampering_is_rejected_on_completed_resume(frozen):
    root, plan = frozen
    task = repair_task(plan)
    campaign.run_episode(root, plan, task, repair_client(root, plan, task, []))
    path = next((root / "results" / task["task_id"] / "calls").glob("*.json"))
    record = json.loads(path.read_text())
    record["budget_charge"] += 1
    atomic_json(path, record)
    with pytest.raises(ValueError, match="changed"):
        campaign.run_episode(root, plan, task, repair_client(root, plan, task, []))


def test_config_cannot_silently_expand_budget(frozen):
    root, plan = frozen
    config = copy.deepcopy(plan["config"])
    config["model_settings"]["maximum_requests"] = 4
    with pytest.raises(ValueError, match="shared physical"):
        campaign.RequirementConfig.model_validate(config)
    assert sealed_read(root / "plan.json") == plan

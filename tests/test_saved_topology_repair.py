"""Frozen starting models, explicit repair, finite budgets and exact cached resume."""

import json
from copy import deepcopy

import pytest

from autoformalism.llm.staged_topology import DeferredCall, StagedModelSettings
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import construction_training as training
from autoformalism.research import saved_topology_repair as campaign
from autoformalism.search import construction_evidence_repair as core
from autoformalism.search.public_graph_obligations import PublicGraphContract
from autoformalism.staged_topology import content_hash
from tests.test_construction_ledger import brief, context, equation, variable
from tests.test_evidence_feedback_schedule import complete_evidence
from tests.test_explicit_variable_bindings import response
from tests.test_phase_c_construction_baseline import tokenize


@pytest.fixture
def setup(monkeypatch):
    """Use real structural/data checks on a scalar reversal with constant input."""
    problem = object.__new__(campaign.Problem)
    public = brief().model_dump(mode="json")
    public["public_variables"].append({"name": "t", "data_role": "time"})
    problem.brief = type(brief()).model_validate(public)
    problem.context, problem.targets = context(), {}
    brief_hash = content_hash(problem.brief.model_dump(mode="json"))
    problem.graph = PublicGraphContract(
        source_brief_sha256=brief_hash, deferred_scientific_checks=()
    )
    problem.boundary = core.BoundaryContract(source_brief_sha256=brief_hash)
    problem.evidence = training.load_evidence(complete_evidence())
    problem.enriched = problem.brief.model_dump(mode="json")
    problem.highlights = training.initial_highlights(problem.evidence)
    draft = core.Draft.model_validate(
        {"variables": [variable("y")], "equations": [equation("y", "u")]}
    )
    monkeypatch.setattr(campaign, "Problem", lambda *args: problem)
    tasks = [
        {
            "task_id": arm,
            "source_task": "saved",
            "arm": arm,
            "seed": 2,
            "benchmark_id": "case",
        }
        for arm in campaign.ARMS
    ]
    plan = {
        "artifact_sha256": "fixture",
        "settings": StagedModelSettings(
            maximum_requests=campaign.MAX_CALLS,
            maximum_total_tokens=campaign.MAX_TOKENS,
        ).model_dump(mode="json"),
        "cells": {"case": {}},
        "starts": {
            "saved": {
                "task": {},
                "draft": draft.model_dump(mode="json"),
                "assessment": problem.assess(draft),
                "diagnostics": problem.diagnose(draft),
            }
        },
        "tasks": tasks,
    }
    return plan, tasks, problem


def execute(root, plan, task, send, **kwargs):
    return campaign.run_episode(
        root,
        plan,
        task,
        base_url="http://offline",
        transport=send,
        token_transport=tokenize,
        **kwargs,
    )


def test_noop_answers_do_not_complete_actionable_repair_and_resume_is_exact(
    tmp_path, setup
):
    plan, tasks, _ = setup
    packets, seeds = {}, {}
    for task in tasks:
        arm = task["arm"]
        packets[arm], seeds[arm] = [], []

        def send(url, body, timeout, arm=arm):
            packet = json.loads(body["messages"][1]["content"])
            packets[arm].append(packet)
            seeds[arm].append(body["seed"])
            edit = {**packet["response_template"], "stage_complete": True}
            edit["compatibility_decisions"] = [
                {
                    "question_id": q["question_id"],
                    "disposition": "retain_approximation"
                    if arm == "control"
                    else "revise",
                    "explanation": "Acknowledgment without a scientific edit.",
                    **({"supporting_evidence": []} if arm == "actionable" else {}),
                }
                for q in packet["compatibility_questions"]
            ]
            return response(edit)

        result = execute(tmp_path, plan, task, send)
        assert result["assessment"]["eligible"]
        assert result["scientific_state"]["repair_status"] == "repair_required"
        assert not result["scientific_accuracy_established"]
        assert (
            execute(
                tmp_path, plan, task, lambda *args: pytest.fail("cached generation")
            )
            == result
        )
    assert seeds["control"][0] == seeds["actionable"][0]
    assert len(packets["control"]) == 1
    assert len(packets["actionable"]) == 3
    assert "repeated_repair_feedback" not in packets["actionable"][0]["feedback"]
    assert packets["actionable"][1]["feedback"]["repeated_repair_feedback"]


def test_a_real_dependency_change_can_remove_the_supported_restriction(tmp_path, setup):
    plan, tasks, _ = setup
    task = tasks[1]

    def send(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"])
        edit = {**packet["response_template"], "stage_complete": True}
        edit["equations"] = [equation("y", "u", "t")]
        return response(edit)

    result = execute(tmp_path, plan, task, send)
    assert result["assessment"]["eligible"]
    assert result["event_count"] == 1
    assert result["scientific_state"]["repair_status"] == "checks_clear"
    assert result["stop_reason"] == "completed_requested_work"
    assert not result["scientific_accuracy_established"]


def test_unknown_latent_initialization_does_not_count_as_repaired(tmp_path, setup):
    plan, tasks, _ = setup

    def send(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"])
        edit = {**packet["response_template"], "stage_complete": True}
        edit["variables"] = [variable("z")]
        edit["equations"] = [equation("y", "z"), equation("z", "u")]
        return response(edit)

    result = execute(tmp_path, plan, tasks[1], send)
    assert result["assessment"]["eligible"]
    assert result["event_count"] == 1
    assert result["stop_reason"] == "scientific_review_deferred"
    assert result["scientific_state"]["repair_status"] == "review_deferred"


def test_interruption_and_resume_preserve_call_budget(tmp_path, setup):
    plan, tasks, _ = setup
    calls = []

    def send(url, body, timeout):
        calls.append(body)
        packet = json.loads(body["messages"][1]["content"])
        return response({**packet["response_template"], "stage_complete": True})

    with pytest.raises(DeferredCall):
        execute(tmp_path, plan, tasks[1], send, can_start=lambda: len(calls) < 1)
    assert len(calls) == 1
    result = execute(tmp_path, plan, tasks[1], send)
    assert len(calls) == result["cost"]["physical_requests"] == 3
    event = sealed_read(tmp_path / "results/actionable/events/000.json")
    assert event["before"] == plan["starts"]["saved"]["draft"]


def test_cached_response_tampering_is_rejected(tmp_path, setup):
    plan, tasks, _ = setup

    def send(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"])
        return response({**packet["response_template"], "stage_complete": True})

    execute(tmp_path, plan, tasks[1], send)
    path = next((tmp_path / "results/actionable/calls").glob("*.json"))
    record = json.loads(path.read_text())
    record["raw_response"]["usage"] = {"total_tokens": 1}
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="call missing or changed"):
        execute(tmp_path, plan, tasks[1], send)


def test_v2_completion_toggles_and_full_restatements_share_repetition_counter(
    tmp_path, setup
):
    plan, tasks, _ = setup
    plan["actionable_policy"] = campaign.actionable.POLICY_V2
    packets = []

    def send(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"])
        packets.append(packet)
        edit = {**packet["response_template"], "stage_complete": len(packets) % 2 == 1}
        if len(packets) % 2 == 0:
            edit["equations"] = [equation("y", "u")]
        return response(edit)

    result = execute(tmp_path, plan, tasks[1], send)
    assert result["event_count"] == 3
    assert result["stop_reason"] == "three_repeated_repairs"
    assert result["scientific_state"]["repair_status"] == "repair_required"
    assert (
        execute(tmp_path, plan, tasks[1], lambda *args: pytest.fail("new call"))
        == result
    )
    assert [
        sealed_read(tmp_path / f"results/actionable/events/{i:03d}.json")[
            "consecutive_unchanged"
        ]
        for i in range(3)
    ] == [1, 2, 3]


def test_v2_distinct_rejected_scientific_alternatives_can_continue(tmp_path, setup):
    plan, tasks, _ = setup
    plan["actionable_policy"] = campaign.actionable.POLICY_V2
    calls = []

    def send(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"])
        calls.append(packet)
        edit = {**packet["response_template"], "stage_complete": True}
        if len(calls) < 4:
            edit["mechanism_realizations"] = [
                {
                    "requirement_id": "mechanism",
                    "components": ["y"],
                    "memory_states": [f"alternative_{len(calls)}"],
                }
            ]
        else:
            edit["equations"] = [equation("y", "u", "t")]
        return response(edit)

    result = execute(tmp_path, plan, tasks[1], send)
    assert result["event_count"] == 4
    assert result["stop_reason"] == "completed_requested_work"
    assert result["scientific_state"]["repair_status"] == "checks_clear"


def test_v2_record_repetition_ignores_completion_and_explanation(tmp_path, setup):
    plan, tasks, _ = setup
    plan["actionable_policy"] = campaign.actionable.POLICY_V2
    calls = []

    def send(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"])
        calls.append(packet)
        edit = {**packet["response_template"], "stage_complete": len(calls) % 2 == 0}
        edit["mechanism_realizations"] = [
            {
                "requirement_id": "mechanism",
                "components": ["y"],
                "memory_states": ["z"],
            }
        ]
        return response(edit)

    result = execute(tmp_path, plan, tasks[1], send)
    assert result["event_count"] == 3
    assert result["stop_reason"] == "three_repeated_repairs"


def test_refreeze_keeps_starts_seeds_and_budget_and_preserves_original_plan(
    tmp_path, setup, monkeypatch
):
    from autoformalism.rebuttal.prefit_replay import sealed_write

    plan, _, _ = setup
    source, destination = tmp_path / "old", tmp_path / "new"
    parent = sealed_write(
        source / "plan.json",
        {
            **{k: v for k, v in plan.items() if k != "artifact_sha256"},
            "protocol": campaign.PROTOCOL,
            "test_data_opened": False,
        },
    )
    before = (source / "plan.json").read_bytes()
    monkeypatch.setattr(campaign, "source_identity", lambda: "test-source")
    frozen = campaign.refreeze(source, destination)
    assert frozen["starts"] == parent["starts"]
    assert frozen["settings"] == parent["settings"]
    assert frozen["tasks"] == [parent["tasks"][1]]
    assert frozen["historical_repair_plan_sha256"] == parent["artifact_sha256"]
    assert campaign.verify(destination) == frozen
    assert (source / "plan.json").read_bytes() == before
    summary = campaign.report(destination, frozen)
    assert summary["protocol"] == campaign.PROTOCOL_V2
    assert list(summary["groups"]) == ["actionable"]


def test_v2_memory_handoff_uses_qualified_eligibility():
    from autoformalism.expressions import ValidationContext
    from tests import test_required_memory_path_qualification as memory

    problem = object.__new__(campaign.Problem)
    problem.brief = memory.public()
    problem.context = ValidationContext(
        targets=("y", "I"), auxiliaries=("a", "b"), external_inputs=("u",)
    )
    problem.targets = {}
    brief_hash = content_hash(problem.brief.model_dump(mode="json"))
    problem.graph = PublicGraphContract(
        source_brief_sha256=brief_hash, deferred_scientific_checks=()
    )
    problem.boundary = core.BoundaryContract(source_brief_sha256=brief_hash)
    draft = core.Draft.model_validate(memory.draft().model_dump(mode="json"))
    old = problem.assess(draft)
    new = problem.assess(draft, qualified_memory=True)
    assert (
        old["eligible"]
        and old["mechanism_realization_handoff"]["structurally_eligible"]
    )
    assert not new["eligible"]
    handoff = new["mechanism_realization_handoff"]
    assert not handoff["structurally_eligible"]
    assert handoff["required_memory_path_qualification"]["status"] == "failed"


def test_report_separates_ended_control_from_scientific_resolution(tmp_path, setup):
    plan, tasks, _ = setup
    task = tasks[0]

    def send(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"])
        edit = {**packet["response_template"], "stage_complete": True}
        edit["compatibility_decisions"] = [
            {
                "question_id": q["question_id"],
                "disposition": "retain_approximation",
                "explanation": "Retained restriction.",
            }
            for q in packet["compatibility_questions"]
        ]
        return response(edit)

    execute(tmp_path, plan, task, send)
    result = campaign.report(tmp_path, plan)
    assert result["groups"]["control"]["status_counts"] == {"assessed": 1}
    assert result["groups"]["control"]["scientific_status_counts"] == {
        "repair_required": 1
    }
    assert result["groups"]["actionable"]["status_counts"] == {"pending": 1}
    assert result["physical_requests"] == 1


def test_report_writes_mixed_assessed_pending_and_partial_tasks_in_each_arm(
    tmp_path, setup
):
    plan, templates, _ = setup
    plan["tasks"] = [
        {**task, "task_id": f"{task['arm']}_{status}"}
        for task in templates
        for status in ("assessed", "pending", "partial")
    ]
    previous = campaign.report(tmp_path, plan)
    summary_path = tmp_path / "summary.json"
    assert summary_path.is_file()

    def finish(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"])
        edit = {**packet["response_template"], "stage_complete": True}
        edit["equations"] = [equation("y", "u", "t")]
        return response(edit)

    for task in plan["tasks"]:
        if task["task_id"].endswith("_assessed"):
            execute(tmp_path, plan, task, finish)
        elif task["task_id"].endswith("_partial"):
            calls = []

            def partial(url, body, timeout, calls=calls):
                calls.append(body)
                packet = json.loads(body["messages"][1]["content"])
                edit = {**packet["response_template"], "stage_complete": False}
                return response(edit)

            with pytest.raises(DeferredCall):
                execute(
                    tmp_path,
                    plan,
                    task,
                    partial,
                    can_start=lambda calls=calls: not calls,
                )
            assert len(calls) == 1

    result = campaign.report(tmp_path, plan)
    persisted = json.loads(summary_path.read_text())
    assert persisted == result
    assert persisted != previous
    assert not list(tmp_path.glob("summary.json.*.tmp"))
    assert result["physical_requests"] == 4
    assert result["observed_tokens"] == 400
    for arm in campaign.ARMS:
        assert result["groups"][arm]["status_counts"] == {
            "assessed": 1,
            "pending": 1,
            "partial": 1,
        }
        assert result["groups"][arm]["scientific_status_counts"] == {
            "checks_clear": 1,
            "not_assessed": 2,
        }
    for row in persisted["rows"]:
        if row["status"] == "assessed":
            assert row["scientific_status"] == "checks_clear"
        else:
            assert row["scientific_status"] is None
            assert row["structural_eligible"] is None
            assert row["cost"]["physical_requests"] == (row["status"] == "partial")
    assert campaign.report(tmp_path, plan) == result
    assert json.loads(summary_path.read_text()) == persisted


@pytest.mark.parametrize("arm", campaign.ARMS)
def test_distinct_rejected_alternatives_are_not_three_repeated_attempts(
    tmp_path, setup, arm
):
    plan, tasks, _ = setup
    task = next(task for task in tasks if task["arm"] == arm)
    packets = []

    def send(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"])
        packets.append(packet)
        edit = {**packet["response_template"], "stage_complete": True}
        edit["equations"] = (
            [{"name": "y", "terms": f"invalid-alternative-{len(packets)}"}]
            if len(packets) <= 4
            else [equation("y", "u", "t")]
        )
        return response(edit)

    result = execute(tmp_path, plan, task, send)
    assert len(packets) == result["event_count"] == 5
    assert result["cost"]["physical_requests"] == 5
    assert result["stop_reason"] == "completed_requested_work"
    assert result["scientific_state"]["repair_status"] == "checks_clear"
    events = [
        sealed_read(path)
        for path in sorted((tmp_path / "results" / arm / "events").glob("*.json"))
    ]
    assert [event["accepted"] for event in events] == [False] * 4 + [True]
    assert all(event["before"] == event["after"] for event in events[:4])


def test_progressive_control_answers_count_as_progress(tmp_path, setup):
    plan, tasks, problem = setup
    public = problem.brief.model_dump(mode="json")
    public["public_variables"].append({"name": "z", "data_role": "target"})
    problem.brief = type(problem.brief).model_validate(public)
    brief_hash = content_hash(public)
    problem.graph = PublicGraphContract(
        source_brief_sha256=brief_hash, deferred_scientific_checks=()
    )
    problem.boundary = core.BoundaryContract(source_brief_sha256=brief_hash)
    evidence = complete_evidence()
    evidence["context"]["targets"] = ["y", "z"]
    evidence["contract"]["initial_rules"]["z"] = {"mode": "observed_initial"}
    evidence["contract"]["absolute_error_bounds"]["z"] = 0
    evidence["trajectories"][0]["targets"]["z"] = [0.0, 2.0, 0.0]
    other = deepcopy(evidence["trajectories"][0])
    other.update(
        trajectory_id="train-1", targets={"y": [0.0, 3.0, 0.0], "z": [0.0, 4.0, 0.0]}
    )
    evidence["trajectories"].append(other)
    # Normalize defaults before sealing the self-contained synthetic evidence.
    evidence = type(problem.evidence).model_validate(evidence).model_dump(mode="json")
    evidence["sha256"] = content_hash(
        {key: value for key, value in evidence.items() if key != "sha256"}
    )
    problem.evidence = training.load_evidence(evidence)
    problem.context = problem.evidence.context
    problem.enriched = public
    problem.highlights = training.initial_highlights(problem.evidence)
    draft = core.Draft.model_validate(
        {
            "variables": [variable("y"), variable("z")],
            "equations": [equation("y", "u"), equation("z", "u")],
        }
    )
    plan["starts"]["saved"].update(
        draft=draft.model_dump(mode="json"),
        assessment=problem.assess(draft),
        diagnostics=problem.diagnose(draft),
    )
    assert len(plan["starts"]["saved"]["diagnostics"]["advisory_questions"]) == 4
    packets = []

    def send(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"])
        packets.append(packet)
        questions = packet["compatibility_questions"]
        edit = {**packet["response_template"], "stage_complete": len(questions) == 1}
        edit["compatibility_decisions"] = [
            {
                "question_id": questions[0]["question_id"],
                "disposition": "retain_approximation",
                "explanation": "Acknowledge this one restriction before the next.",
            }
        ]
        return response(edit)

    result = execute(tmp_path, plan, tasks[0], send)
    pending_counts = [len(packet["compatibility_questions"]) for packet in packets]
    assert pending_counts == [4, 3, 2, 1]
    assert result["event_count"] == 4
    assert result["stop_reason"] == "completed_requested_work"
    assert result["draft"] == plan["starts"]["saved"]["draft"]
    assert result["scientific_state"]["repair_status"] == "repair_required"


@pytest.mark.parametrize("field", ["identity", "task", "source_draft_sha256"])
def test_report_rejects_foreign_result_identity(tmp_path, setup, field):
    plan, tasks, _ = setup

    def send(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"])
        return response({**packet["response_template"], "stage_complete": True})

    execute(tmp_path, plan, tasks[1], send)
    path = tmp_path / "results/actionable/result.json"
    result = sealed_read(path)
    result.pop("artifact_sha256")
    result[field] = tasks[0] if field == "task" else "foreign-episode"
    # This is internally sealed, but belongs to a different logical episode.
    result["artifact_sha256"] = content_hash(result)
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match=r"identity|task|source|provenance"):
        campaign.report(tmp_path, plan)

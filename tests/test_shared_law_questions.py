"""Shared-law elicitation placement, actual assembly, failure isolation and resume."""

import json
from collections import Counter

import pytest

from autoformalism.llm.construction import ConstructionClient
from autoformalism.llm.staged_topology import DeferredCall, StagedModelSettings
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import construction_comparison as campaign
from autoformalism.search import construction_schedules as schedules
from tests.test_construction_comparison import source_fixture, transport
from tests.test_construction_ledger import (
    brief,
    context,
    equation,
    patch,
    process,
    variable,
)
from tests.test_explicit_variable_bindings import response
from tests.test_phase_c_construction_baseline import tokenize


def attempt(
    root,
    policy,
    placement,
    calls,
    *,
    broken=False,
    empty=False,
    limit=100,
    reject_first=False,
):
    """A toy transfer is declared only when explicitly asked; no live provider."""
    settings = StagedModelSettings(maximum_requests=128, maximum_total_tokens=524288)

    def send(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"])
        calls.append(payload)
        stage = payload["stage"]
        if stage in {"variables", "relationships"}:
            edits = patch(variables=[variable("x"), variable("y")], stage_complete=True)
        else:
            edits = patch(stage_complete=True)
        if payload["shared_law_question"]:
            if broken:
                return response({"not_the_schema": True})
            if not empty:
                edits = edits.model_copy(
                    update={
                        "processes": patch(processes=[process()]).processes,
                    }
                )
            if (
                reject_first
                and sum(p["shared_law_question"] is not None for p in calls) == 1
            ):
                # A real rejected transaction: law plus an out-of-stage equation.
                edits = edits.model_copy(
                    update={
                        "equations": patch(equations=[equation("y", "u")]).equations,
                    }
                )
        if stage == "equations":
            names = [payload["selected_lhs"]] if payload["selected_lhs"] else ["x", "y"]
            edits = patch(
                equations=[
                    equation(n, "u")
                    if n == "x" or empty or broken
                    else {"name": "y", "terms": []}
                    for n in names
                ],
                stage_complete=payload["selected_lhs"] is None,
            )
        return response(edits.model_dump(mode="json"))

    client = ConstructionClient(
        settings=settings,
        directory=root / "calls",
        namespace="placement-test",
        seed=0,
        base_url="http://offline",
        transport=send,
        token_transport=tokenize,
        can_start=lambda: len(calls) < limit,
    )
    result = schedules.run(
        brief(),
        context(),
        {},
        brief().model_dump(mode="json"),
        client,
        root / "construction",
        policy,
        process_question=placement,
    )
    assert client.settings == settings
    return result


@pytest.mark.parametrize("policy", schedules.POLICIES)
def test_placement_changes_call_location_not_single_law_assembly(tmp_path, policy):
    outcomes, counts, questions = [], [], []
    for placement in schedules.PROCESS_QUESTIONS:
        calls = []
        root = tmp_path / placement
        result = attempt(root, policy, placement, calls)
        assert result["status"] == "topology_complete"
        assert result["before_repair"]["assessment"]["eligible"]
        asked = [p for p in calls if p["shared_law_question"]]
        assert len(asked) == 1
        assert asked[0]["stage"] == (
            "relationships" if placement == "integrated" else "shared_laws"
        )
        questions.append(asked[0]["shared_law_question"])
        assert result["assessment"]["process_usage"][0]["scope"] == "shared"
        equations = {e["name"]: e for e in result["assessment"]["equations"]}
        assert sum(e["name"] == "p" for e in result["assessment"]["equations"]) == 1
        for name, sign in (("x", "negative"), ("y", "positive")):
            uses = [t for t in equations[name]["terms"] if t["sources"] == ["p"]]
            assert len(uses) == 1 and uses[0]["outer_weight_sign"] == sign
        outcomes.append(result["draft"])
        counts.append(len(calls))
        saved = (root / "construction/before_repair.json").read_bytes()
        assert attempt(root, policy, placement, calls) == result
        assert len(calls) == counts[-1]
        assert (root / "construction/before_repair.json").read_bytes() == saved
    assert questions[0] == questions[1]
    assert outcomes[0] == outcomes[1]
    assert counts[1] == counts[0] + 1  # Dedicated decision is charged, not hidden.


@pytest.mark.parametrize("broken", [False, True])
def test_empty_or_unavailable_dedicated_decision_does_not_stop_construction(
    tmp_path, broken
):
    calls = []
    result = attempt(
        tmp_path, "separate", "dedicated", calls, broken=broken, empty=True
    )
    assert result["status"] == "topology_complete"
    assert not result["draft"]["processes"]
    assert not any(p["stage"] == "repair" for p in calls)
    assert sum(p["stage"] == "shared_laws" for p in calls) == (3 if broken else 1)
    stage = next(
        s
        for s in result["before_repair"]["stage_outcomes"]
        if s["stage"] == "shared_laws"
    )
    assert stage["completed"] is not broken
    assert stage["continued_to_equations"] is broken


def test_resume_at_dedicated_boundary_reuses_relationship_reply(tmp_path):
    calls = []
    with pytest.raises(DeferredCall):
        attempt(tmp_path, "joint_adaptive", "dedicated", calls, limit=1)
    assert [p["stage"] for p in calls] == ["relationships"]
    result = attempt(tmp_path, "joint_adaptive", "dedicated", calls)
    assert result["status"] == "topology_complete"
    assert [p["stage"] for p in calls] == ["relationships", "shared_laws", "equations"]


def test_process_question_stage_cannot_emit_functions_or_ordinary_equations():
    for policy in schedules.POLICIES:
        with pytest.raises(ValueError, match="precedes equation"):
            schedules.validate_scope(
                policy, "shared_laws", None, patch(equations=[equation("y", "u")])
            )
        schedules.validate_scope(policy, "equations", "y", patch(processes=[process()]))


def test_paired_basin_plan_reports_question_placement_separately(tmp_path):
    source_fixture(tmp_path)
    root = tmp_path / "paired"
    plan = campaign.freeze(tmp_path / "new", root, study="shared_law_comparison")
    assert campaign.verify(root) == plan
    assert len(plan["tasks"]) == 12 and set(plan["cells"]) == set(campaign.BASIN_CASES)
    blocks = Counter((t["benchmark_id"], t["policy"]) for t in plan["tasks"])
    assert set(blocks.values()) == {2}
    assert {t["process_question"] for t in plan["tasks"]} == set(
        schedules.PROCESS_QUESTIONS
    )
    assert all(t["seed"] == 0 and t["arm"] == "full" for t in plan["tasks"])
    assert not plan["test_data_opened"] and not plan["automatic_followup"]
    calls = []
    base_transport = transport(calls)

    def send(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"])
        if payload["stage"] == "shared_laws":
            calls.append(payload)
            return response(patch(stage_complete=True).model_dump(mode="json"))
        reply = base_transport(url, body, timeout)
        message = reply["choices"][0]["message"]
        edits = json.loads(message["content"])
        for e in edits.get("equations", []):
            if e["name"] == "h_down":
                e["terms"][0]["sources"].append("h_down")
        message["content"] = json.dumps(edits)
        return reply

    for task in plan["tasks"]:
        result = campaign.propose(
            root, plan, task, "http://offline", transport=send, token_transport=tokenize
        )
        assert result["status"] == "topology_complete"
    summary = campaign.report(root, plan)
    assert summary["status_counts"] == {"topology_complete": 12}
    assert len(summary["groups"]) == 6
    for row in summary["rows"]:
        assert row["process_counts"]["replies_with_declarations"] == 0
        assert row["process_counts"]["retained_shared_laws"] == 0
        assert any(h["shared_law_question_displayed"] for h in row["process_history"])
    for group in summary["groups"].values():
        assert group["planned"] == group["finished"] == 2
    plan["tasks"][0]["process_question"] = "unplanned"
    plan.pop("artifact_sha256")
    (root / "plan.json").unlink()
    sealed_write(root / "plan.json", plan)
    with pytest.raises(ValueError, match="12-task"):
        campaign.verify(root)


def test_process_history_exposes_rejected_declaration_not_as_empty(tmp_path):
    calls = []
    result = attempt(tmp_path, "joint_adaptive", "dedicated", calls, reject_first=True)
    events = [
        sealed_read(p)
        for p in sorted((tmp_path / "construction/events").glob("*.json"))
    ]
    records = [json.loads(p.read_text()) for p in (tmp_path / "calls").glob("*.json")]
    history = campaign.process_history(records, events)
    entry = next(h for h in history if h["stage"] == "shared_laws")
    assert not entry["accepted"] and "precedes equation" in entry["error"]
    assert entry["proposed_processes"][0]["declared_shared"]
    assert entry["retained_names"] == []
    assert result["status"] == "topology_complete"
    assert history[-1]["retained_names"] == ["p"]
    count = len(calls)
    assert (
        attempt(tmp_path, "joint_adaptive", "dedicated", calls, reject_first=True)
        == result
    )
    assert len(calls) == count

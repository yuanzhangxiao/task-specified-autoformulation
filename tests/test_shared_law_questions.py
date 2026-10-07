"""Shared-law elicitation placement, actual assembly, failure isolation and resume."""

import json
from collections import Counter

import pytest

from autoformalism.llm.construction import ConstructionClient
from autoformalism.llm.staged_topology import DeferredCall, StagedModelSettings
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import construction_comparison as campaign
from autoformalism.search import construction_ledger as ledger
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


@pytest.mark.parametrize("policy", schedules.POLICIES)
def test_multi_lhs_batch_finishes_without_scope_retry_and_resumes(tmp_path, policy):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        if p["stage"] == "variables" or (
            p["stage"] == "relationships" and policy != "separate"
        ):
            result = patch(
                variables=[variable("x"), variable("y")], stage_complete=True
            )
        elif p["stage"] == "relationships":
            result = patch(stage_complete=True)
        else:
            result = patch(
                equations=[equation("x", "u"), equation("y", "x")],
                stage_complete=True,
            )
        return response(result.model_dump(mode="json"))

    def run():
        client = ConstructionClient(
            settings=StagedModelSettings(
                maximum_requests=128, maximum_total_tokens=524288
            ),
            directory=tmp_path / "calls",
            namespace="multi-lhs",
            seed=0,
            base_url="http://offline",
            transport=send,
            token_transport=tokenize,
        )
        return schedules.run(
            brief(),
            context(),
            {},
            brief().model_dump(mode="json"),
            client,
            tmp_path / "construction",
            policy,
        )

    result = run()
    assert result["status"] == "topology_complete"
    assert result["before_repair"]["assessment"]["eligible"]
    assert sum(p["stage"] == "equations" for p in calls) == 1
    assert not any(p["stage"] == "repair" for p in calls)
    count = len(calls)
    assert run() == result and len(calls) == count
    assert {e["name"] for e in result["draft"]["equations"]} == {"x", "y"}


def test_rejected_batch_receipt_shows_actual_retained_entries(tmp_path):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        if p["stage"] == "relationships":
            edits = patch(variables=[variable("x"), variable("y")], stage_complete=True)
        elif sum(c["stage"] == "equations" for c in calls) == 1:
            edits = patch(
                variables=[variable("z")],
                equations=[equation("x", "u"), equation("y", "x")],
                remove_equations=["x"],
                stage_complete=True,
            )
        else:
            receipt = p["last_edit_result"]
            assert not receipt["transaction_applied"]
            assert receipt == p["runtime_diagnostics"]
            assert {
                v["name"] for v in p["current_draft"]["declarations"]["variables"]
            } == {"x", "y"}
            z = next(x for x in receipt["uncommitted_edits"] if x["key"] == "z")
            assert z["retained_entry"] is None and not z["applied"]
            assert not p["current_draft"]["declarations"]["equations"]
            edits = patch(
                equations=[equation("x", "u"), equation("y", "x")], stage_complete=True
            )
        return response(edits.model_dump(mode="json"))

    client = ConstructionClient(
        settings=StagedModelSettings(maximum_requests=128, maximum_total_tokens=524288),
        directory=tmp_path / "calls",
        namespace="rejection",
        seed=0,
        base_url="http://offline",
        transport=send,
        token_transport=tokenize,
    )
    result = schedules.run(
        brief(),
        context(),
        {},
        brief().model_dump(mode="json"),
        client,
        tmp_path / "construction",
        "joint_guided",
    )
    assert result["status"] == "topology_complete"
    assert {v["name"] for v in result["draft"]["variables"]} == {"x", "y"}
    assert calls[0]["last_edit_result"] is None
    assert calls[1]["last_edit_result"]["transaction_applied"]


def test_multi_equation_removals_and_scientific_failures_remain_visible():
    d = ledger.Draft.model_validate(
        {
            "variables": [variable("x"), variable("y")],
            "equations": [equation("x", "u"), equation("y", "x")],
        }
    )
    edits = patch(equations=[equation("y", "missing")], remove_equations=["x"])
    for policy in schedules.POLICIES:
        schedules.validate_scope(policy, "equations", "x", edits, d)
    after = ledger.apply_patch(brief(), d, edits)
    check = ledger.assess(brief(), context(), {}, after)
    assert not check["eligible"]
    assert {e["code"] for e in check["errors"]} >= {
        "missing_declarations",
        "equations_to_define",
    }
    errors = [
        {"code": "public_path", "driver": "u"},
        {"code": "memory_driver_path", "driver": "u", "state": "x"},
        {"code": "memory_target_path", "target": "y"},
        {"code": "memory_type", "state": "x"},
    ]
    groups = schedules.repair_issue_groups(errors)
    assert groups["remaining_topology_failures"] == errors[:3]
    assert groups["binding_failures"] == errors[3:]
    raw = {"variables": [variable("x", "algebraic")], "remove_variables": ["y"]}
    receipt = schedules.uncommitted_edits(raw, d)
    assert receipt[0]["retained_entry"]["definition"] == "differential"
    assert receipt[1]["retained_entry"]["name"] == "y"
    assert schedules.uncommitted_edits(None, d) == []


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

"""Bounded inventory closure and factual feedback for current prompts only."""

import json

import pytest

from autoformalism.llm.construction import ConstructionClient
from autoformalism.llm.staged_topology import DeferredCall, StagedModelSettings
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import construction_comparison as campaign
from autoformalism.search import construction_bookkeeping as bookkeeping
from autoformalism.search import construction_feedback as feedback
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import construction_schedules as schedules
from tests.test_construction_comparison import source_fixture
from tests.test_construction_ledger import (
    apply,
    brief,
    context,
    equation,
    patch,
    process,
    variable,
)
from tests.test_explicit_variable_bindings import response
from tests.test_phase_c_construction_baseline import tokenize
from tests.test_public_graph_obligations import contract


def execute(root, send, *, policy=bookkeeping.FEEDBACK_POLICY, **kwargs):
    client = ConstructionClient(
        settings=StagedModelSettings(maximum_requests=128, maximum_total_tokens=524288),
        directory=root / "calls",
        namespace="specific-feedback",
        seed=0,
        base_url="http://offline",
        transport=send,
        token_transport=tokenize,
        **kwargs,
    )
    return schedules.run(
        brief(),
        context(),
        {},
        brief().model_dump(mode="json"),
        client,
        root / "construction",
        "separate",
        bookkeeping_policy=policy,
    )


def test_incomplete_inventory_remains_editable_with_cached_resume(tmp_path):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        if len(calls) == 1:
            edits = patch(variables=[variable("x")], stage_complete=True)
        elif p["stage"] == "variables":
            completion = p["runtime_diagnostics"]["stage_completion"]
            assert completion["missing_public_targets"] == ["y"]
            assert p["last_edit_result"]["transaction_applied"]
            assert p["current_draft"]["declarations"]["variables"][0]["name"] == "x"
            edits = patch(variables=[variable("y", "algebraic")], stage_complete=True)
        elif p["stage"] == "relationships":
            edits = patch(stage_complete=True)
        else:
            edits = patch(
                equations=[equation("y", "x"), equation("x", "u")],
                stage_complete=True,
            )
        return response(edits.model_dump(mode="json"))

    with pytest.raises(DeferredCall):
        execute(tmp_path, send, can_start=lambda: len(calls) < 1)
    event = sealed_read(tmp_path / "construction/events/000.json")
    assert event["accepted"] and not event["stage_complete"]
    assert event["stage_completion"]["status"] == "incomplete"
    result = execute(tmp_path, send)
    assert [p["stage"] for p in calls] == [
        "variables",
        "variables",
        "relationships",
        "equations",
    ]
    assert result["status"] == "topology_complete"
    assert result["before_repair"]["assessment"]["eligible"]
    assert result["cost"]["physical_requests"] == 4
    assert execute(tmp_path, send) == result and len(calls) == 4


def test_repeated_premature_completion_is_bounded_and_never_freezes_inventory(tmp_path):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        return response(
            patch(variables=[variable("x")], stage_complete=True).model_dump(
                mode="json"
            )
        )

    result = execute(tmp_path, send)
    assert [p["stage"] for p in calls] == ["variables"] * 3 + ["repair"] * 3
    assert result["status"] == "topology_incomplete"
    initial = result["before_repair"]
    assert initial["cost"]["physical_requests"] == 3
    assert not initial["stage_outcomes"][0]["completed"]
    assert "variables" in initial["stop_reason"]
    assert execute(tmp_path, send) == result and len(calls) == 6


def test_complete_inventory_needs_no_extra_confirmation_or_topology(tmp_path):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        edits = (
            patch(variables=[variable("y")], stage_complete=True)
            if p["stage"] == "variables"
            else patch(equations=[equation("y", "u")], stage_complete=True)
            if p["stage"] == "equations"
            else patch(stage_complete=True)
        )
        return response(edits.model_dump(mode="json"))

    assert execute(tmp_path, send)["status"] == "topology_complete"
    assert [p["stage"] for p in calls] == ["variables", "relationships", "equations"]
    d = apply(variables=[variable("y", "algebraic")])
    assert feedback.variable_completion(brief(), d)["status"] == "ready"
    assert not d.equations  # No topology or inferred science at this stage.


@pytest.mark.parametrize("policy", ["legacy", bookkeeping.POLICY])
def test_older_policies_keep_the_original_variable_transition(tmp_path, policy):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        assert feedback.VARIABLE_COMPLETION not in p["stage_instructions"]
        return response(patch(stage_complete=True).model_dump(mode="json"))

    execute(tmp_path, send, policy=policy)
    assert [p["stage"] for p in calls[:2]] == ["variables", "relationships"]
    event = sealed_read(tmp_path / "construction/events/000.json")
    assert "stage_completion" not in event and event["stage_complete"]


def assess(draft):
    old = ledger.assess(
        brief(), context(), {}, draft, graph_contract=contract("target_feedback")
    )
    new = bookkeeping.assessment_context(old, draft, policy=bookkeeping.FEEDBACK_POLICY)
    assert old["eligible"] == new["eligible"]
    assert "feedback_evidence" not in old["reviewed_public_graph_checks"][0]
    return new


def coordinate_draft(*, cycle=False, readout=True, binding=True):
    return apply(
        variables=[variable("y", "algebraic"), variable("x")],
        equations=[
            equation("y", "x" if readout else "u"),
            equation("x", "u", *(["x"] if cycle else [])),
        ],
        feedback_bindings=[{"target": "y", "states": ["x"]}] if binding else [],
    )


@pytest.mark.parametrize("cycle,readout", [(False, True), (True, False), (True, True)])
def test_feedback_separates_binding_cycle_and_algebraic_readout(cycle, readout):
    draft = coordinate_draft(cycle=cycle, readout=readout)
    original = draft.model_dump(mode="json")
    result = assess(draft)
    row = result["reviewed_public_graph_checks"][0]
    e = row["feedback_evidence"]
    assert e["binding_status"] == "present"
    state = e["coordinate_checks"][0]
    assert state["declared"] and state["differential"]
    assert state["feedback_cycle_exists"] == cycle
    assert state["algebraic_readout_path_exists"] == readout
    assert e["passed"] == row["passed"] == (cycle and readout)
    if not row["passed"]:
        assert {a["missing"] for a in row["repair_actions"]} == {
            "algebraic_readout_path" if cycle else "feedback_cycle"
        }
        error = next(
            e for e in result["errors"] if e["code"] == "reviewed_public_graph"
        )
        assert error["feedback_evidence"] == row["feedback_evidence"]
        assert "creates no graph edges" in error["annotation_limit"]
        assert "Coordinate binding is present." in error["reason"]
        assert f"feedback cycle: {'yes' if cycle else 'no'}" in error["reason"]
    assert draft.model_dump(mode="json") == original


def test_missing_binding_and_uncompiled_graph_are_not_false_cycle_diagnoses():
    row = assess(coordinate_draft(cycle=True, binding=False))[
        "reviewed_public_graph_checks"
    ][0]
    assert row["feedback_evidence"]["binding_status"] == "missing"
    assert row["repair_actions"][0]["missing"] == "coordinate_binding"
    broken = apply(coordinate_draft(), equations=[equation("x", "undeclared")])
    row = assess(broken)["reviewed_public_graph_checks"][0]
    assert row["feedback_evidence"]["graph_available"] is False
    assert row["feedback_evidence"]["passed"] is row["passed"] is None
    assert "repair_actions" not in row


def test_differential_target_and_multi_state_realizations_use_actual_cycles():
    d = apply(variables=[variable("y")], equations=[equation("y", "u", "y")])
    row = assess(d)["reviewed_public_graph_checks"][0]
    assert row["feedback_evidence"]["binding_status"] == "implicit_target_coordinate"
    assert row["passed"]
    d = apply(
        coordinate_draft(cycle=True),
        variables=[variable("z")],
        equations=[equation("y", "x", "z"), equation("z", "u")],
        feedback_bindings=[{"target": "y", "states": ["x", "z"]}],
    )
    row = assess(d)["reviewed_public_graph_checks"][0]
    assert not row["passed"]
    assert [a["state"] for a in row["repair_actions"]] == ["z"]


def test_shared_process_cycle_and_wrong_coordinate_type():
    d = apply(
        variables=[variable("x"), variable("y")],
        processes=[process()],
        equations=[equation("x", "u"), {"name": "y", "terms": []}],
    )
    # x -> p -> x is a valid cycle, y is an algebraic readout of x.
    d = apply(
        d,
        variables=[variable("y", "algebraic")],
        equations=[equation("y", "x")],
        remove_processes=["p"],
    )
    d = apply(
        d,
        processes=[
            {
                **process(),
                "uses": [{"target": "x", "sign": "negative", "conversion": None}],
                "kind": "influence",
            }
        ],
        feedback_bindings=[{"target": "y", "states": ["x"]}],
    )
    row = assess(d)["reviewed_public_graph_checks"][0]
    assert row["feedback_evidence"]["coordinate_checks"][0]["feedback_cycle"] == [
        "x",
        "p",
        "x",
    ]
    wrong = apply(
        coordinate_draft(),
        variables=[variable("x", "algebraic"), variable("z")],
        equations=[equation("z", "u")],
    )
    row = assess(wrong)["reviewed_public_graph_checks"][0]
    assert row["repair_actions"][0]["missing"] == "differential_coordinate"


def test_prompt_explains_missing_cycle_without_claiming_missing_binding(tmp_path):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        if p["stage"] == "relationships":
            edits = patch(
                variables=[variable("y", "algebraic"), variable("x")],
                feedback_bindings=[{"target": "y", "states": ["x"]}],
                stage_complete=True,
            )
        elif p["stage"] == "equations":
            edits = patch(
                equations=[equation("y", "x"), equation("x", "u")],
                stage_complete=True,
            )
        else:
            assert feedback.FEEDBACK_REPAIR in p["stage_instructions"]
            error = p["runtime_diagnostics"]["remaining_topology_failures"][0]
            assert error["feedback_evidence"]["binding_status"] == "present"
            assert error["repair_actions"][0]["missing"] == "feedback_cycle"
            # The test proposer chooses a coupled restoring path. Runtime does not.
            edits = patch(equations=[equation("x", "u", "y")], stage_complete=True)
        return response(edits.model_dump(mode="json"))

    def run():
        client = ConstructionClient(
            settings=StagedModelSettings(),
            directory=tmp_path / "calls",
            namespace="graph-feedback",
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
            "joint_adaptive",
            graph_contract=contract("target_feedback"),
            bookkeeping_policy=bookkeeping.FEEDBACK_POLICY,
        )

    result = run()
    assert result["status"] == "topology_complete"
    assert len(calls) == 3
    assert run() == result and len(calls) == 3


def test_new_policy_is_frozen_and_excluded_from_both_prompt_comparison_arms(tmp_path):
    source_fixture(tmp_path)
    source = tmp_path / "new"
    plan = campaign.freeze(
        source,
        tmp_path / "new-policy",
        study="shared_law_comparison",
        bookkeeping_policy=bookkeeping.FEEDBACK_POLICY,
    )
    assert campaign.verify(tmp_path / "new-policy") == plan
    assert plan["config"]["bookkeeping_policy"] == bookkeeping.FEEDBACK_POLICY
    with pytest.raises(ValueError, match="separate from the prompt comparison"):
        campaign.freeze(
            source,
            tmp_path / "disallowed",
            study="prompt_comparison",
            bookkeeping_policy=bookkeeping.FEEDBACK_POLICY,
        )
    for family in ("minimal", "current"):
        with pytest.raises(ValueError, match="separate from the prompt comparison"):
            bookkeeping.validate_policy(bookkeeping.FEEDBACK_POLICY, family)

"""Focused views, independent graph evidence and bounded stage continuation."""

import json
from copy import deepcopy

import pytest

from autoformalism.llm.construction import ConstructionClient
from autoformalism.llm.staged_topology import DeferredCall, StagedModelSettings
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import construction_comparison as campaign
from autoformalism.search import construction_focus as focus
from autoformalism.search import construction_schedules as schedules
from tests.test_construction_ledger import (
    apply,
    brief,
    context,
    equation,
    process,
    variable,
)
from tests.test_explicit_variable_bindings import response
from tests.test_phase_c_construction_baseline import tokenize
from tests.test_public_graph_obligations import contract


def packet(stage, draft, b=None, last=None, graph=None):
    b = b or brief()
    return focus.payload(
        stage, b, context(), {}, b.model_dump(mode="json"), draft, graph, last
    )


def test_model_view_keeps_joint_functions_conversions_signs_types_and_meanings():
    d = apply(
        variables=[variable("x"), variable("y")],
        processes=[process()],
        equations=[equation("x", "u", "x"), {"name": "y", "terms": []}],
    )
    view = focus.model_view(d)
    text = json.dumps(view)
    assert "phi_x_local_0(u, x)" in text
    assert "(1/area) * p" in text
    assert "- g_p_use_0" in text and "= g_p_use_1" in text
    assert "d(x)/dt" in text and "shared process" in text
    p = packet("equations", d)
    for key in (
        "current_draft",
        "last_edit_result",
        "runtime_diagnostics",
        "bookkeeping_policy",
        "interaction_review_handoff",
        "allowed_memory_bindings",
    ):
        assert key not in p
    assert p["public_brief"] == brief().model_dump(mode="json")
    assert "terms" not in text


def test_bindings_are_explicit_conditional_examples_not_runtime_assignments():
    b = brief(True)
    d = apply(public=b, variables=[variable("y"), variable("M"), variable("N")])
    before = d.model_dump()
    p = packet("variables", d, b)
    issue = next(x for x in p["feedback"]["issues"] if x["code"] == "memory_assignment")
    assert issue["type_eligible_states"] == ["M", "N"]
    assert issue["conditional_examples"][0]["edit"]["mechanism_bindings"] == [
        {"requirement_id": "memory", "memory_states": ["M"]}
    ]
    assert issue["conditional_examples"][1]["edit"]["mechanism_bindings"] == [
        {"requirement_id": "memory", "memory_states": ["N"]}
    ]
    assert d.model_dump() == before and not d.mechanism_bindings
    optional = b.model_copy(
        update={
            "requirements": tuple(
                r.model_copy(update={"requires_dynamic_memory": False})
                for r in b.requirements
            )
        }
    )
    p = packet("variables", d, optional)
    assert p["memory_binding_options"][0]["memory_assignment_required"] is False
    assert any(
        r["id"] == "mechanism_topology:memory" for r in p["requirements_checklist"]
    )


def test_receiver_feedback_preserves_driver_and_valid_use_without_applying_it():
    d = apply(
        variables=[variable("y")],
        processes=[
            process(
                kind="influence",
                depends_on=["y", "u"],
                uses=[
                    {"target": "y", "sign": "positive"},
                    {"target": "u", "sign": "negative"},
                ],
            )
        ],
    )
    old = deepcopy(d)
    p = packet("shared_laws", d)
    issue = next(x for x in p["feedback"]["issues"] if x["code"] == "supplied_receiver")
    example = issue["preserve_valid_effects"]["conditional_process_edit"]
    assert example["depends_on"] == ["y", "u"]
    assert example["uses"] == [{"target": "y", "sign": "positive", "conversion": None}]
    assert example["kind"] == "influence"
    assert p["receiver_choices"]["currently_generated"] == ["y"]
    assert p["receiver_choices"][
        "observations_needing_generated_declaration_first"
    ] == ["a"]
    assert d == old


def test_uncompiled_graph_shows_missing_memory_path_without_a_false_final_verdict():
    b = brief(True)
    d = apply(
        public=b,
        variables=[variable("x"), variable("y")],
        mechanism_bindings=[{"requirement_id": "memory", "memory_states": ["x"]}],
        equations=[equation("x", "u"), {"name": "y", "terms": []}],
    )
    p = packet("repair", d, b)
    assert any(x["code"] == "compiler" for x in p["feedback"]["issues"])
    paths = p["feedback"]["provisional_paths"]
    assert any(
        x["kind"] == "assigned_memory_to_target" and x["status"] == "missing_in_draft"
        for x in paths
    )
    assert any(
        x["kind"] == "driver_to_assigned_memory" and x["status"] == "present_in_draft"
        for x in paths
    )
    assert not any(
        r.get("status") == "passed"
        for r in p["requirements_checklist"]
        if r["id"].startswith("mechanism_topology")
    )
    complete = apply(d, public=b, equations=[equation("y", "x")])
    assert not packet("repair", complete, b)["feedback"]["provisional_paths"]


def test_provisional_forbidden_paths_and_feedback_include_known_edges_only():
    d = apply(variables=[variable("y")], equations=[equation("y", "y", "u")])
    rows = focus.provisional_paths(brief(), d, contract("target_feedback"))
    assert rows[-1]["status"] == "present_in_draft"
    rows = focus.provisional_paths(brief(), d, contract("forbidden_path"))
    assert (
        rows[-1]["kind"] == "forbidden_path"
        and rows[-1]["status"] == "present_in_draft"
    )


def execute(root, send, can_start=lambda: True):
    b = brief(True)
    c = ConstructionClient(
        settings=StagedModelSettings(maximum_requests=128, maximum_total_tokens=524288),
        directory=root / "calls",
        namespace="focused",
        seed=0,
        base_url="http://offline",
        transport=send,
        token_transport=tokenize,
        can_start=can_start,
    )
    return schedules.run(
        b,
        context(),
        {},
        b.model_dump(mode="json"),
        c,
        root / "construction",
        "joint_adaptive",
        prompt_family="minimal",
        process_question="dedicated",
        bookkeeping_policy=focus.POLICY,
    )


def test_local_binding_recovery_resumes_shared_stage_and_cached_replay(tmp_path):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        edit = {**p["response_template"], "stage_complete": True}
        if p["stage"] == "variables":
            edit["variables"] = [variable("y"), variable("M")]
            if len(calls) == 4:
                assert p["feedback"]["continuation"]["local_recovery"]
                edit["mechanism_bindings"] = [
                    {"requirement_id": "memory", "memory_states": ["M"]}
                ]
        elif p["stage"] == "equations":
            edit["equations"] = [equation("M", "u"), equation("y", "M")]
        elif p["stage"] != "shared_laws":
            pytest.fail(
                "Recovery must resume the stage sequence, not global construction"
            )
        return response(edit)

    with pytest.raises(DeferredCall):
        execute(tmp_path, send, lambda: len(calls) < 3)
    result = execute(tmp_path, send)
    assert result["status"] == "topology_complete"
    assert [p["stage"] for p in calls] == ["variables"] * 4 + [
        "shared_laws",
        "equations",
    ]
    assert execute(tmp_path, send) == result and len(calls) == 6


def test_exhausted_variable_recovery_stops_without_skipping_to_global_repair(tmp_path):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        return response(
            {
                **p["response_template"],
                "variables": [variable("y")],
                "stage_complete": True,
            }
        )

    result = execute(tmp_path, send)
    assert result["status"] == "topology_incomplete"
    assert len(calls) == 5 and {p["stage"] for p in calls} == {"variables"}
    assert execute(tmp_path, send) == result and len(calls) == 5


def test_receipt_keeps_rejection_distinct_from_current_model():
    d = apply(variables=[variable("y")])
    last = {
        "status": "rejected",
        "transaction_applied": False,
        "error": "invalid edit",
        "edit_effects": None,
    }
    p = packet("variables", d, last=last)
    assert p["previous_edit"]["applied"] is False
    assert p["model_view"]["variables"][0]["name"] == "y"
    assert p["feedback"]["issues"][-1]["code"] == "last_reply_rejected"


@pytest.mark.parametrize("progress", [True, False])
def test_global_repair_cap_and_no_progress_stop_preserve_cached_chain(
    tmp_path, progress
):
    calls, repairs = [], []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        edit = {**p["response_template"], "stage_complete": True}
        if p["stage"] == "variables":
            edit.update(
                variables=[variable("y"), variable("M")],
                mechanism_bindings=[
                    {"requirement_id": "memory", "memory_states": ["M"]}
                ],
            )
        elif p["stage"] == "equations":
            edit["equations"] = [equation("M", "u"), equation("y", "u")]
        elif p["stage"] == "repair":
            repairs.append(p)
            if progress:
                edit["variables"] = [
                    {**variable("M"), "scientific_role": f"revision {len(repairs)}"}
                ]
                if len(repairs) == 5:
                    edit["equations"] = [equation("y", "M")]
        return response(edit)

    result = execute(tmp_path, send)
    assert len(repairs) == (5 if progress else 2)
    assert result["status"] == (
        "topology_complete" if progress else "topology_incomplete"
    )
    if not progress:
        assert repairs[-1]["feedback"]["continuation"]["no_progress"]
        assert "no draft progress" in result["stop_reason"]
    assert result["bookkeeping_policy"] == focus.POLICY
    count = len(calls)
    assert execute(tmp_path, send) == result and len(calls) == count
    assert len(campaign.checked_records(tmp_path, "focused")) == count
    # The log retains full declarations; only the displayed projection changed.
    event = sealed_read(tmp_path / "construction/events/001.json")
    assert event["before"]["variables"]


def test_missing_public_target_is_repaired_locally_before_shared_stage(tmp_path):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        edit = {**p["response_template"], "stage_complete": True}
        if p["stage"] == "variables":
            edit["variables"] = [variable("M")]
            edit["mechanism_bindings"] = [
                {"requirement_id": "memory", "memory_states": ["M"]}
            ]
            if len(calls) == 2:
                assert any(
                    x["code"] == "public_target" for x in p["feedback"]["issues"]
                )
                edit["variables"].append(variable("y"))
        elif p["stage"] == "equations":
            edit["equations"] = [equation("M", "u"), equation("y", "M")]
        return response(edit)

    assert execute(tmp_path, send)["status"] == "topology_complete"
    assert [p["stage"] for p in calls] == [
        "variables",
        "variables",
        "shared_laws",
        "equations",
    ]

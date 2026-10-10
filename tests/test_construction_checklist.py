"""Early declaration feedback without scientific inference or topology leakage."""

import json

import pytest

from autoformalism.llm.construction import ConstructionClient
from autoformalism.llm.staged_topology import DeferredCall, StagedModelSettings
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import construction_comparison as campaign
from autoformalism.search import construction_bookkeeping as bookkeeping
from autoformalism.search import construction_checklist as checklist
from autoformalism.search import construction_schedules as schedules
from tests.test_construction_ledger import (
    apply,
    brief,
    context,
    equation,
    patch,
    variable,
)
from tests.test_explicit_variable_bindings import response
from tests.test_phase_c_construction_baseline import tokenize
from tests.test_public_graph_obligations import contract


def rows(draft, *, public=None, definitions=None, graph=None):
    return checklist.variable_checklist(
        public or brief(True), draft, definitions or {}, graph
    )


def test_assignment_is_checked_without_equations_or_inference_from_names_and_prose():
    b = brief(True)
    d = apply(public=b, variables=[variable("y", "algebraic"), variable("memory")])
    result = rows(d)
    assert result["blocking_items"] == ["memory_assignment:memory"]
    assert result["explicit_type_rule_count"] == 0
    d = apply(
        d,
        public=b,
        mechanism_bindings=[{"requirement_id": "memory", "memory_states": ["memory"]}],
    )
    result = rows(d)
    assert result["status"] == "ready" and not d.equations
    assert (
        next(r for r in result["items"] if r["code"] == "mechanism_topology")["status"]
        == "deferred"
    )
    # A scientific explanation alone neither assigns memory nor changes a type.
    d = apply(
        d,
        public=b,
        variables=[
            {
                **variable("memory", "algebraic"),
                "scientific_role": "This is a differential memory state",
            }
        ],
    )
    row = next(r for r in rows(d)["items"] if r["code"] == "memory_assignment")
    assert row["status"] == "inconsistent"
    assert row["state_checks"][0]["definition"] == "algebraic"


@pytest.mark.parametrize("state", ["later", "y", "a", "z"])
def test_forward_missing_supplied_algebraic_and_endpoint_memory_refs_block_only_closure(
    state,
):
    d = apply(
        public=brief(True),
        variables=[variable("y"), variable("z", "algebraic")],
        mechanism_bindings=[{"requirement_id": "memory", "memory_states": [state]}],
    )
    assert rows(d)["status"] == "incomplete"
    assert d.mechanism_bindings[0].memory_states == (state,)


def test_optional_memory_unknown_ids_and_explicit_types_are_separate_obligations():
    b = brief(True)
    b = b.model_copy(
        update={
            "requirements": tuple(
                r.model_copy(update={"requires_dynamic_memory": False})
                for r in b.requirements
            )
        }
    )
    d = apply(public=b, variables=[variable("y")])
    assert rows(d, public=b)["status"] == "ready"
    d = apply(
        d,
        public=b,
        mechanism_bindings=[{"requirement_id": "memory", "memory_states": ["y"]}],
    )
    assert rows(d, public=b)["status"] == "ready"  # No mandatory separate state.
    d = apply(
        d,
        public=b,
        mechanism_bindings=[{"requirement_id": "unknown", "memory_states": ["y"]}],
    )
    assert rows(d, public=b)["blocking_items"] == ["memory_assignment:unknown"]
    d = apply(d, public=b, remove_bindings=["unknown"])
    assert rows(d, public=b, definitions={"y": "algebraic"})["blocking_items"] == [
        "explicit_type:y"
    ]


def test_algebraic_readout_needs_chosen_dynamic_coordinates_but_not_yet_a_cycle():
    g = contract("target_feedback")
    d = apply(variables=[variable("y", "algebraic"), variable("x")])
    assert rows(d, public=brief(), graph=g)["blocking_items"] == [
        "feedback_coordinates:y"
    ]
    d = apply(d, feedback_bindings=[{"target": "y", "states": ["x"]}])
    assert rows(d, public=brief(), graph=g)["status"] == "ready"
    d = apply(d, variables=[variable("x", "algebraic")])
    assert rows(d, public=brief(), graph=g)["status"] == "incomplete"
    d = apply(d, variables=[variable("y")], remove_feedback_bindings=["y"])
    assert rows(d, public=brief(), graph=g)["status"] == "ready"
    d = apply(d, feedback_bindings=[{"target": "y", "states": ["x"]}])
    assert rows(d, public=brief(), graph=g)["status"] == "incomplete"


def execute(
    root, send, *, public=None, target_definitions=None, can_start=lambda: True
):
    b = public or brief(True)
    client = ConstructionClient(
        settings=StagedModelSettings(),
        directory=root / "calls",
        namespace="variable-checklist",
        seed=0,
        base_url="http://offline",
        transport=send,
        token_transport=tokenize,
        can_start=can_start,
    )
    return schedules.run(
        b,
        context(),
        target_definitions or {},
        b.model_dump(mode="json"),
        client,
        root / "construction",
        "joint_adaptive",
        prompt_family="minimal",
        process_question="dedicated",
        bookkeeping_policy=checklist.POLICY,
    )


def test_each_turn_checklist_repair_retention_rejection_and_interruption_resume(
    tmp_path,
):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        if p["stage"] == "variables":
            assert "pending" not in p["current_draft"]
            assert "read_only_balances" not in p["current_draft"]
            assert checklist.INSTRUCTION in p["stage_instructions"]
        if len(calls) == 1:
            assert "public_target:y" in p["variable_checklist"]["blocking_items"]
            edit = patch(
                variables=[variable("y"), variable("x", "algebraic")],
                mechanism_bindings=[
                    {"requirement_id": "memory", "memory_states": ["x"]}
                ],
                stage_complete=True,
            )
        elif len(calls) == 2:
            assert p["stage"] == "variables"
            assert p["variable_checklist"]["blocking_items"] == [
                "memory_assignment:memory"
            ]
            assert p["last_edit_result"]["transaction_applied"]
            assert (
                p["runtime_diagnostics"]["stage_completion"]["status"] == "incomplete"
            )
            # Invalid supplied-source edit: none of this batch is committed.
            edit = patch(variables=[variable("x"), variable("u")], stage_complete=True)
        elif len(calls) == 3:
            assert p["last_edit_result"]["status"] == "rejected"
            assert p["variable_checklist"]["blocking_items"] == [
                "memory_assignment:memory"
            ]
            edit = patch(variables=[variable("x")], stage_complete=True)
        elif p["stage"] == "shared_laws":
            assert p["variable_checklist"]["status"] == "ready"
            edit = patch(stage_complete=True)
        else:
            edit = patch(
                equations=[equation("y", "x"), equation("x", "u")], stage_complete=True
            )
        return response(edit.model_dump(mode="json"))

    with pytest.raises(DeferredCall):
        execute(tmp_path, send, can_start=lambda: len(calls) < 2)
    result = execute(tmp_path, send)
    assert result["status"] == "topology_complete"
    assert len(calls) == 5 and execute(tmp_path, send) == result and len(calls) == 5
    event = sealed_read(tmp_path / "construction/events/000.json")
    assert event["accepted"] and not event["stage_complete"]
    assert event["variable_checklist_after"]["blocking_items"] == [
        "memory_assignment:memory"
    ]


def test_explicit_type_mismatch_and_missing_binding_do_not_consume_unbounded_calls(
    tmp_path,
):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        return response(
            patch(variables=[variable("y")], stage_complete=True).model_dump(
                mode="json"
            )
        )

    result = execute(tmp_path, send, target_definitions={"y": "algebraic"})
    assert [p["stage"] for p in calls] == ["variables"] * 3 + ["repair"] * 3
    assert result["status"] == "topology_incomplete"
    assert not result["before_repair"]["stage_outcomes"][0]["completed"]


def test_later_type_edit_reopens_declaration_problem_and_assignment_is_not_graph_pass(
    tmp_path,
):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        if p["stage"] == "variables":
            edit = patch(
                variables=[variable("y"), variable("x")],
                mechanism_bindings=[
                    {"requirement_id": "memory", "memory_states": ["x"]}
                ],
                stage_complete=True,
            )
        elif p["stage"] == "shared_laws":
            edit = patch(stage_complete=True)
        elif p["stage"] == "equations":
            edit = patch(
                variables=[variable("x", "algebraic")],
                equations=[equation("y", "u"), equation("x", "u")],
                stage_complete=True,
            )
        else:
            assert p["variable_checklist"]["status"] == "incomplete"
            codes = {r["code"] for r in p["runtime_diagnostics"]["structural_failures"]}
            assert {"variable_declaration", "memory_target_path"} <= codes
            edit = patch(
                variables=[variable("x")],
                equations=[equation("y", "x")],
                stage_complete=True,
            )
        return response(edit.model_dump(mode="json"))

    result = execute(tmp_path, send)
    assert result["status"] == "topology_complete" and len(calls) == 4


@pytest.mark.parametrize(
    "study, policy",
    [
        ("variable_checklist_confirmation", checklist.POLICY),
        ("stage_check_confirmation", bookkeeping.STAGE_POLICY),
        ("deferred_interaction_confirmation", bookkeeping.DEFERRED_POLICY),
    ],
)
def test_eight_case_study_reports_real_stage_evidence_and_resumes(
    tmp_path, study, policy
):
    from tests.test_construction_comparison import source_fixture, transport

    source_fixture(tmp_path)
    root = tmp_path / "study"
    plan = campaign.freeze(tmp_path / "new", root, study=study)
    assert campaign.verify(root) == plan
    assert len(plan["tasks"]) == 8
    assert sum("detention" in t["benchmark_id"] for t in plan["tasks"]) == 2
    assert plan["config"]["bookkeeping_policy"] == policy
    campaign.report(root, plan)
    pending = json.loads((root / "VARIABLES.json").read_text())
    assert pending["checkpoints"]["first_retained"]["available"] == 0
    assert all(r["variable_stage_completed"] is None for r in pending["rows"])
    if study in campaign.SHARED_STAGE_STUDIES:
        shared = json.loads((root / "SHARED_PROCESSES.json").read_text())
        assert shared["checkpoints"]["first_retained"]["available"] == 0
        assert all(r["stage_completed"] is None for r in shared["rows"])
    calls = []
    base = transport(calls)

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        if p["stage"] == "shared_laws":
            calls.append(p)
            return response({**p["response_template"], "stage_complete": True})
        result = base(url, body, timeout)
        message = result["choices"][0]["message"]
        edits = json.loads(message["content"])
        for e in edits.get("equations", []):
            if e["name"] in {"T", "h_down"}:
                e["terms"][0]["sources"].append(e["name"])
        message["content"] = json.dumps(edits)
        return result

    for task in plan["tasks"]:
        result = campaign.propose(
            root, plan, task, "http://offline", transport=send, token_transport=tokenize
        )
        assert result["status"] == "topology_complete"
    n = len(calls)
    if study == "deferred_interaction_confirmation":
        assert all(p["bookkeeping_policy"] == policy for p in calls)
        assert all(p["requirement_status"]["items"] for p in calls)
        with pytest.raises(ValueError, match="fixes its minimal policy"):
            campaign.freeze(
                tmp_path / "new",
                tmp_path / "wrong-policy",
                study=study,
                bookkeeping_policy=bookkeeping.STAGE_POLICY,
            )
    campaign.report(root, plan)
    values = json.loads((root / "VARIABLES.json").read_text())
    assert values["completed_variable_stages"] == 8
    assert values["checkpoints"]["first_retained"]["declaration_ready"] == 8
    assert values["checkpoints"]["after_topology_and_global_repair"]["available"] == 8
    assert "not scientific correctness" in (root / "VARIABLES.md").read_text()
    if study in campaign.SHARED_STAGE_STUDIES:
        shared = json.loads((root / "SHARED_PROCESSES.json").read_text())
        assert shared["completed_shared_stages"] == 8
        assert shared["checkpoints"]["first_retained"]["empty_decisions"] == 8
        assert shared["checkpoints"]["after_global_repair"]["locally_consistent"] == 8
    for task in plan["tasks"]:
        campaign.propose(
            root, plan, task, "http://offline", transport=send, token_transport=tokenize
        )
    assert len(calls) == n
    with pytest.raises(ValueError):
        bookkeeping.validate_policy(policy, "current")
    with pytest.raises(ValueError):
        campaign.freeze(
            tmp_path / "new",
            tmp_path / "invalid",
            study="prompt_comparison",
            bookkeeping_policy=policy,
        )

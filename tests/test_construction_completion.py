"""Specific completion decisions without scientific inference or budget changes."""

import json
from copy import deepcopy

import pytest

from autoformalism.llm.construction import ConstructionClient
from autoformalism.llm.staged_topology import StagedModelSettings
from autoformalism.rebuttal.repair_comparison import RepairBudgetExceeded
from autoformalism.search import construction_completion as completion
from autoformalism.search import construction_coverage as coverage
from autoformalism.search import construction_deferred as deferred
from autoformalism.search import construction_ledger as ledger
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


def packet(draft, stage="repair", public=None):
    b = public or brief()
    return completion.payload(
        stage, b, context(), {}, b.model_dump(mode="json"), draft, None
    )


def test_required_assignment_is_a_choice_not_inferred_from_same_name():
    b = brief(True)
    d = apply(
        public=b, variables=[variable("y"), variable("memory"), variable("other")]
    )
    before = deepcopy(d)
    p = packet(d, "variables", b)
    q = next(
        i for i in p["feedback"]["issues"] if i["code"] == "confirm_memory_assignment"
    )
    assert q["retained_entry"] is None
    assert q["choices_by_type_only"] == ["memory", "other"]
    assert len(q["conditional_answers"]) == 2
    assert "not by declaring" in q["question"]
    assert d == before and not d.mechanism_bindings
    changed = apply(d, public=b, **q["conditional_answers"][1]["edit"])
    assert changed.mechanism_bindings[0].memory_states == ("other",)
    assert not any(
        i["code"] == "confirm_memory_assignment"
        for i in packet(changed, "variables", b)["feedback"]["issues"]
    )
    assert p["public_brief"] == b.model_dump(mode="json")


def test_no_eligible_state_does_not_invent_assignment_or_relax_check():
    b = brief(True)
    d = apply(public=b, variables=[variable("y")])
    q = packet(d, "variables", b)["feedback"]["issues"][0]
    assert q["code"] == "confirm_memory_assignment"
    assert q["choices_by_type_only"] == q["conditional_answers"] == []
    assert not deferred.assessment(b, context(), {}, d, None)["eligible"]


def test_optional_binding_withdrawal_preserves_required_path_and_other_state():
    b = brief(True)
    b = b.model_copy(
        update={
            "requirements": (
                b.requirements[0].model_copy(update={"requires_dynamic_memory": False}),
            )
        }
    )
    d = apply(
        public=b,
        variables=[variable("y"), variable("m")],
        equations=[equation("m", "m"), equation("y", "u")],
        mechanism_bindings=[{"requirement_id": "memory", "memory_states": ["m"]}],
    )
    q = next(
        i["assignment_decision"]
        for i in packet(d, public=b)["feedback"]["issues"]
        if "assignment_decision" in i
    )
    assert q["retained_entry"]["memory_states"] == ["m"]
    edit = q["if_optional_assignment_was_unintended"]["edit"]
    changed = apply(d, public=b, **edit)
    assert not changed.mechanism_bindings
    assert changed.variables == d.variables and changed.equations == d.equations
    assert deferred.assessment(b, context(), {}, changed, None)["eligible"]
    # Removing a binding does not remove the public driver-to-target obligation.
    changed = apply(changed, public=b, equations=[equation("y", "m")])
    assert not deferred.assessment(b, context(), {}, changed, None)["eligible"]
    required = next(
        i["assignment_decision"]
        for i in packet(d, public=brief(True))["feedback"]["issues"]
        if "assignment_decision" in i
    )
    assert "if_optional_assignment_was_unintended" not in required


def test_missing_equation_names_exact_quantity_and_offers_supplied_alternative():
    d = apply(
        variables=[variable("y"), variable("a", "algebraic")],
        equations=[equation("y", "a"), {"name": "a", "terms": []}],
    )
    q = next(
        i
        for i in packet(d)["feedback"]["issues"]
        if i["code"] == "equation_destination"
    )
    assert q["exact_name"] == "a" and q["retained_entry"] == {"name": "a", "terms": []}
    assert "name='a'" in q["action"] and "different name" in q["action"]
    edit = q["if_you_intended_to_use_the_supplied_observation"]["edit"]
    changed = apply(d, **edit)
    assert deferred.assessment(brief(), context(), {}, changed, None)["eligible"]
    target = packet(apply(variables=[variable("y")]))["feedback"]["issues"]
    q = next(i for i in target if i["code"] == "equation_destination")
    assert "if_you_intended_to_use_the_supplied_observation" not in q
    assert not any(
        i["code"] == "equation_destination"
        for i in packet(d, "variables")["feedback"]["issues"]
    )


def test_shared_only_missing_entry_is_different_from_empty_assembled_rhs():
    d = apply(
        variables=[variable("x"), variable("y")],
        processes=[process()],
        equations=[equation("x", "u")],
    )
    q = next(
        i
        for i in packet(d)["feedback"]["issues"]
        if i["code"] == "equation_destination"
    )
    assert q["exact_name"] == "y" and q["retained_entry"] is None
    assert q["already_inserted_uses"][0]["sign"] == "positive"
    changed = apply(d, **q["if_shared_uses_are_the_entire_intended_rhs"])
    assert deferred.assessment(brief(), context(), {}, changed, None)["eligible"]


@pytest.mark.parametrize("stage", ["equations", "repair"])
@pytest.mark.parametrize("local", [False, True])
def test_process_owned_variable_never_requests_a_second_equation(stage, local):
    declared = process()
    if local:
        declared.update(kind="influence", uses=[{"target": "y", "sign": "negative"}])
    d = apply(
        variables=[variable("x"), variable("y"), variable("p", "algebraic")],
        processes=[declared],
        equations=[equation("x", "u")],
    )
    before = deepcopy(d)
    issues = packet(d, stage)["feedback"]["issues"]
    destinations = [i for i in issues if i["code"] == "equation_destination"]
    assert [i["exact_name"] for i in destinations] == ["y"]
    assert ledger.pending(brief(), d)["equations_to_define"] == ["y"]
    assert d == before
    changed = apply(d, equations=[{"name": "y", "terms": []}])
    assert deferred.assessment(brief(), context(), {}, changed, None)["eligible"]
    # An incompatible extra definition still requires a proposer decision.
    with pytest.raises(ValueError, match="runtime-generated definition"):
        apply(changed, equations=[equation("p", "u")])
    # Removing the process does not silently remove the variable declaration.
    removed = apply(changed, remove_processes=["p"])
    assert any(
        i["code"] == "equation_destination" and i["exact_name"] == "p"
        for i in packet(removed, stage)["feedback"]["issues"]
    )


@pytest.mark.parametrize(
    "uses",
    [
        [{"target": "y", "sign": "negative", "conversion": "1/area"}],
        [{"target": "y", "sign": "negative"}, {"target": "u", "sign": "positive"}],
        [{"target": "y", "sign": "negative"}, {"target": "y", "sign": "positive"}],
    ],
)
def test_flow_choice_preserves_all_signs_dependencies_and_conversions(uses):
    d = apply(
        variables=[variable("y")], processes=[process(depends_on=["y", "u"], uses=uses)]
    )
    before = deepcopy(d)
    p = packet(d, "shared_laws")
    q = next(
        i
        for i in p["feedback"]["issues"]
        if i["code"] == "flow_classification_decision"
    )
    edit = q["if_boundary_flow_or_other_nontransfer_effect"]["edit"]
    assert edit["processes"] == [
        {**d.processes[0].model_dump(mode="json"), "kind": "influence"}
    ]
    assert d == before
    assert "not fixed" in q["if_boundary_flow_or_other_nontransfer_effect"]["scope"]
    assert q["generated_variables"] == ["y"]


def make_client(tmp_path, send, tokens=524288, requests=128):
    return ConstructionClient(
        settings=StagedModelSettings(
            maximum_requests=requests, maximum_total_tokens=tokens
        ),
        base_url="http://offline",
        directory=tmp_path / "calls",
        namespace="test",
        seed=0,
        transport=send,
        token_transport=tokenize,
    )


def test_budget_preflight_distinguishes_context_from_reservation_and_never_sends(
    tmp_path,
):
    def forbidden(*args):
        raise AssertionError("generation must not start")

    c = make_client(tmp_path, forbidden, tokens=10000)
    c.budget_diagnostics_enabled = True
    for _ in range(2):
        with pytest.raises(RepairBudgetExceeded, match="token budget"):
            c.call(
                system="x",
                user="x" * 5000,
                response_model=ledger.DraftPatch,
                step="repair",
                attempt=0,
            )
        d = c.last_budget_diagnostic
        assert d["status"] == "blocked" and d["context_preflight"]["status"] == "ready"
        assert d["reservation"] == d["request_bytes"] + 8192
        assert d["reservation_shortfall"] == d["reservation"] - 10000
        assert d["blocking_reasons"] == ["token_reservation"]
    assert c.records == [] and not list(c.directory.glob("*.json"))
    assert len(list((c.directory / "budget_preflight").glob("*.json"))) == 1


def test_started_requests_keep_existing_charging_and_cached_resume(tmp_path):
    calls = []

    def send(*args):
        calls.append(args)
        return response({"stage_complete": True})

    c = make_client(tmp_path, send)
    c.budget_diagnostics_enabled = True
    kw = {
        "system": "x",
        "user": "x",
        "response_model": ledger.DraftPatch,
        "step": "variables",
        "attempt": 0,
    }
    r = c.call(**kw)
    assert r["budget_charge_basis"] == "provider_total_tokens"
    assert c.last_budget_diagnostic["status"] == "ready"
    restored = make_client(tmp_path, send)
    restored.budget_diagnostics_enabled = True
    assert restored.call(**kw) == r and len(calls) == 1
    assert len(restored.records) == 1
    # A later blocked call cannot be confused with the earlier ready preflight.
    restored.settings = restored.settings.model_copy(update={"maximum_requests": 1})
    with pytest.raises(RepairBudgetExceeded, match="request budget"):
        restored.call(**{**kw, "attempt": 1})
    assert restored.last_budget_diagnostic["blocking_reasons"] == ["request_limit"]


@pytest.mark.parametrize("policy", [completion.POLICY, coverage.POLICY])
def test_live_binding_confirmation_uses_existing_repair_allowance_and_resumes(
    tmp_path, policy
):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        edit = {**p["response_template"], "stage_complete": True}
        if p["stage"] == "variables":
            if len(calls) == 1:
                edit["variables"] = [variable("y"), variable("m")]
            else:
                q = next(
                    i
                    for i in p["feedback"]["issues"]
                    if i["code"] == "confirm_memory_assignment"
                )
                edit.update(q["conditional_answers"][0]["edit"])
        if p["stage"] == "equations":
            edit["equations"] = [equation("m", "u"), equation("y", "m")]
        return response(edit)

    def run():
        b = brief(True)
        return schedules.run(
            b,
            context(),
            {},
            b.model_dump(mode="json"),
            make_client(tmp_path, send),
            tmp_path / "construction",
            "joint_adaptive",
            prompt_family="minimal",
            process_question="dedicated",
            bookkeeping_policy=policy,
        )

    result = run()
    assert result["status"] == "topology_complete"
    assert [p["stage"] for p in calls] == [
        "variables",
        "variables",
        "shared_laws",
        "equations",
    ]
    assert result["budget_stop"] is None
    assert run() == result and len(calls) == 4


@pytest.mark.parametrize("policy", [completion.POLICY, coverage.POLICY])
def test_global_budget_prompt_replays_historical_charge_not_future_cached_calls(
    tmp_path,
    policy,
):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(body)
        edit = {**p["response_template"], "stage_complete": True}
        if p["stage"] == "variables":
            edit["variables"] = [variable("y")]
        elif p["stage"] == "equations":
            edit["equations"] = [{"name": "y", "terms": []}]
        elif p["stage"] == "repair":
            budget = p["feedback"]["repair_budget"]
            assert budget["maximum_global_calls"] == 5
            assert budget["maximum_global_repair_tokens"] == 131072
            assert budget["available_before_reservation"] == 131072
            edit["equations"] = [equation("y", "u")]
        return response(edit)

    def run():
        return schedules.run(
            brief(),
            context(),
            {},
            brief().model_dump(mode="json"),
            make_client(tmp_path, send),
            tmp_path / "construction",
            "joint_adaptive",
            prompt_family="minimal",
            process_question="dedicated",
            bookkeeping_policy=policy,
        )

    result = run()
    assert result["status"] == "topology_complete" and len(calls) == 4
    assert run() == result and len(calls) == 4

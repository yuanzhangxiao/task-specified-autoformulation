"""Regression cases for ambiguous binding fields and explicit named-law consumers."""

import json

import pytest

from autoformalism.llm.construction import ConstructionClient
from autoformalism.llm.staged_topology import StagedModelSettings
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.search import construction_handoff as handoff
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import construction_schedules as schedules
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


def signed_term(target, source, sign):
    e = equation(target, source)
    e["terms"][0]["outer_weight_sign"] = sign
    return e


def local_law():
    return apply(
        variables=[variable("x"), variable("y")],
        processes=[
            process(
                kind="influence",
                uses=[{"target": "y", "sign": "positive", "conversion": "1/area"}],
            )
        ],
        equations=[equation("x", "u"), {"name": "y", "terms": []}],
    )


def normalize(draft, edits):
    return handoff.normalize_consumers(
        ledger.apply_patch(brief(), draft, edits), edits, draft
    )


def test_new_consumer_preserves_upstream_depletion_and_one_shared_law():
    d = local_law()
    e = signed_term("x", "p", "negative")
    e["terms"].insert(0, equation("x", "u")["terms"][0])
    updated, log = normalize(d, patch(equations=[e]))
    assert d.processes[0].uses[0].target == "y" and len(d.processes[0].uses) == 1
    assert [(u.target, u.sign, u.conversion) for u in updated.processes[0].uses] == [
        ("y", "positive", "1/area"),
        ("x", "negative", None),
    ]
    assert updated.processes[0].kind == "influence"  # No inferred transfer type.
    a = ledger.assess(brief(), context(), {}, updated)
    assert a["eligible"], a["errors"]
    assert len(a["shared_process_bindings"]) == 1
    assembled = next(e for e in a["equations"] if e["name"] == "x")
    assert assembled["terms"][-1]["sources"] == ["p"]
    assert assembled["terms"][-1]["outer_weight_sign"] == "negative"
    assert log[0]["code"] == "explicit_process_consumer"
    assert not log[0]["conversion_inferred"]
    # Applying the same intended reference again cannot create two consumers.
    again, _ = normalize(updated, patch(equations=[e]))
    assert again == updated


def test_existing_identity_repeat_preserves_conversion_and_is_counted_once():
    updated, log = normalize(
        local_law(), patch(equations=[signed_term("y", "p", "positive")])
    )
    assert updated.processes[0].uses[0].conversion == "1/area"
    assert updated.equations[-1].terms == ()
    assert len(ledger.assembled_equations(brief(), updated)[-1].terms) == 1
    assert log[0]["code"] == "repeated_process_use"


@pytest.mark.parametrize(
    "variant",
    ["sign_conflict", "unrestricted", "mixed", "repeated", "removed_consumer"],
)
def test_ambiguous_references_are_not_silently_assigned(variant):
    d = local_law()
    e = signed_term("x", "p", "negative")
    processes = []
    if variant == "sign_conflict":
        e["name"] = "y"
    elif variant == "unrestricted":
        e["terms"][0]["outer_weight_sign"] = "unrestricted"
    elif variant == "mixed":
        e["terms"][0]["sources"].append("area")
    elif variant == "repeated":
        e["terms"].append(dict(e["terms"][0]))
    else:
        d, _ = normalize(d, patch(equations=[e]))
        processes = list(local_law().processes)  # Explicitly removes the new use.
    edits = patch(equations=[e], processes=processes)
    candidate = ledger.apply_patch(brief(), d, edits)
    updated, log = handoff.normalize_consumers(candidate, edits, d)
    assert updated == candidate and log == []
    issue = handoff.process_reference_issues(updated)[0]
    assert issue["code"] == "process_consumer_decision"
    assert issue["process"] == "p"
    assert "preserve other uses" in issue["repair_options"][0]
    assert not ledger.assess(brief(), context(), {}, updated)["eligible"]


def test_inapplicable_binding_cleanup_does_not_assign_scientific_memory():
    b = brief(True)
    d = apply(
        public=b,
        variables=[variable("y", "algebraic"), variable("m")],
        equations=[equation("y", "m"), equation("m", "u")],
        feedback_bindings=[{"target": "y", "states": ["m"]}],
    )
    raw = patch(stage_complete=True).model_dump(mode="json")
    empty = contract().model_copy(update={"obligations": ()})
    normalized, log = ledger.normalize_reply(b, raw, graph_contract=empty, draft=d)
    assert raw["remove_feedback_bindings"] == []
    assert normalized["remove_feedback_bindings"] == ["y"]
    d = ledger.apply_patch(b, d, ledger.DraftPatch.model_validate(normalized))
    assert not d.feedback_bindings and not d.mechanism_bindings
    assert log[0]["replacement_assignment"] is None
    assert not ledger.assess(b, context(), {}, d, graph_contract=empty)["eligible"]
    d = ledger.apply_patch(
        b,
        d,
        patch(
            mechanism_bindings=[{"requirement_id": "memory", "memory_states": ["m"]}]
        ),
    )
    assert ledger.assess(b, context(), {}, d, graph_contract=empty)["eligible"]
    raw["feedback_bindings"] = [{"target": "not_a_target", "states": ["m"]}]
    assert ledger.normalize_reply(b, raw, graph_contract=empty)[0] == raw


def test_applicable_coordinates_remain_proposer_owned_with_exact_removal_key():
    d = apply(
        variables=[variable("y"), variable("m")],
        equations=[equation("y", "y", "u"), equation("m", "m")],
        feedback_bindings=[{"target": "y", "states": ["m"]}],
    )
    c = contract("target_feedback")
    raw = patch(stage_complete=True).model_dump(mode="json")
    assert ledger.normalize_reply(brief(), raw, graph_contract=c, draft=d) == (raw, [])
    error = next(
        e
        for e in ledger.assess(brief(), context(), {}, d, graph_contract=c)["errors"]
        if e["code"] == "differential_target_is_own_coordinate"
    )
    assert error["repair"] == {"remove_feedback_bindings": ["y"]}
    with pytest.raises(ValueError, match="uses target keys; available keys=\\['y'\\]"):
        apply(d, remove_feedback_bindings=["m"])
    updated = apply(d, **error["repair"])
    assert ledger.assess(brief(), context(), {}, updated, graph_contract=c)["eligible"]


def test_binding_help_is_explicit_but_does_not_preselect_scientific_assignment():
    d = apply(variables=[variable("y"), variable("x"), variable("z")])
    help = handoff.binding_context(brief(True), d, contract("target_feedback"))
    question = help["mechanism_bindings"][0]
    assert question["eligible_declared_states_by_type_only"] == ["x", "z"]
    assert question["assignment_missing"]
    assert question["reply_example_replace_placeholder"] == {
        "mechanism_bindings": [
            {"requirement_id": "memory", "memory_states": ["CHOSEN_STATE"]}
        ]
    }
    assert help["feedback_bindings"]["needs_proposer_coordinates"] == []
    assert help["feedback_bindings"]["reply_examples_replace_placeholder"] == []
    assert (
        handoff.binding_context(brief(True), d, None)["feedback_bindings"][
            "applicable_targets"
        ]
        == []
    )


def test_live_handoff_prompt_and_canonical_transactions_resume_without_new_calls(
    tmp_path,
):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        assert "binding_context" in p
        if p["stage"] == "relationships":
            d = local_law()
            edits = patch(
                variables=d.variables, processes=d.processes, stage_complete=True
            )
        else:
            edits = patch(
                equations=[signed_term("x", "p", "negative"), equation("y", "u")],
                stage_complete=True,
            )
        return response(edits.model_dump(mode="json"))

    client = ConstructionClient(
        settings=StagedModelSettings(maximum_requests=128, maximum_total_tokens=524288),
        directory=tmp_path / "calls",
        namespace="handoff",
        seed=0,
        base_url="http://offline",
        transport=send,
        token_transport=tokenize,
    )

    def run():
        return schedules.run(
            brief(),
            context(),
            {},
            brief().model_dump(mode="json"),
            client,
            tmp_path / "construction",
            "joint_adaptive",
        )

    result = run()
    assert result["status"] == "topology_complete"
    assert len(result["draft"]["processes"][0]["uses"]) == 2
    event = sealed_read(tmp_path / "construction/events/001.json")
    assert event["normalizations"][0]["code"] == "explicit_process_consumer"
    assert run() == result and len(calls) == 2
    changes = handoff.edit_effects(
        ledger.Draft.model_validate(event["before"]),
        ledger.Draft.model_validate(event["after"]),
    )
    assert len(changes["processes"][0]["after"]["uses"]) == 2


def test_missing_memory_assignment_gets_exact_field_and_requires_proposer_choice(
    tmp_path,
):
    calls = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        calls.append(p)
        if p["stage"] == "relationships":
            edits = patch(variables=[variable("y"), variable("m")], stage_complete=True)
        elif p["stage"] == "equations":
            edits = patch(
                equations=[equation("y", "m"), equation("m", "u")],
                feedback_bindings=[{"target": "y", "states": ["m"]}],
                stage_complete=True,
            )
        else:
            assert not p["current_draft"]["declarations"]["mechanism_bindings"]
            assert not p["current_draft"]["declarations"]["feedback_bindings"]
            q = p["binding_context"]["mechanism_bindings"][0]
            assert q["assignment_missing"]
            assert q["eligible_declared_states_by_type_only"] == ["m"]
            assert q["reply_example_replace_placeholder"]["mechanism_bindings"][0][
                "memory_states"
            ] == ["CHOSEN_STATE"]
            edits = patch(
                mechanism_bindings=[
                    {"requirement_id": "memory", "memory_states": ["m"]}
                ],
                stage_complete=True,
            )
        return response(edits.model_dump(mode="json"))

    b = brief(True)
    client = ConstructionClient(
        settings=StagedModelSettings(maximum_requests=128, maximum_total_tokens=524288),
        directory=tmp_path / "calls",
        namespace="memory-choice",
        seed=0,
        base_url="http://offline",
        transport=send,
        token_transport=tokenize,
    )
    result = schedules.run(
        b,
        context(),
        {},
        b.model_dump(mode="json"),
        client,
        tmp_path / "construction",
        "joint_adaptive",
        graph_contract=contract().model_copy(update={"obligations": ()}),
    )
    assert result["status"] == "topology_complete" and len(calls) == 3
    assert not result["before_repair"]["assessment"]["eligible"]
    assert result["draft"]["mechanism_bindings"] == [
        {"requirement_id": "memory", "memory_states": ["m"]}
    ]


def test_saved_reply_audit_is_read_only_and_retains_actual_predecessors(tmp_path):
    from autoformalism.research import construction_comparison as campaign
    from scripts.audit_construction_handoff import audit
    from tests.test_construction_comparison import source_fixture, transport

    source_fixture(tmp_path)
    root = tmp_path / "comparison"
    plan = campaign.freeze(tmp_path / "new", root, study="live_confirmation")
    campaign.propose(
        root,
        plan,
        plan["tasks"][0],
        "http://offline",
        transport=transport([]),
        token_transport=tokenize,
    )
    original = {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    result = audit(root, tmp_path / "audit.json")
    assert result["saved_attempts"] > 0
    assert result["unexpected_acceptance_regressions"] == []
    assert result["llm_calls"] == result["optimizer_calls"] == 0
    assert audit(root, tmp_path / "audit.json") == result
    assert {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    } == original
    with pytest.raises(ValueError, match="outside"):
        audit(root, root / "audit.json")
    event = next((root / "results").glob("*/construction/events/*.json"))
    event.write_text(event.read_text().replace('"accepted": true', '"accepted": false'))
    with pytest.raises(ValueError):
        audit(root, tmp_path / "bad.json")

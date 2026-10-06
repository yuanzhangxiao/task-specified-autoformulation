"""Local feedback, explicit overlap choices and namespace bookkeeping regressions."""

import json

import pytest

from autoformalism.llm.construction import ConstructionClient
from autoformalism.llm.staged_topology import StagedModelSettings
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import construction_schedules as schedules
from tests.test_construction_ledger import (
    apply,
    brief,
    context,
    equation,
    optional_memory_brief,
    patch,
    process,
    variable,
)
from tests.test_explicit_variable_bindings import response
from tests.test_phase_c_construction_baseline import tokenize
from tests.test_public_graph_obligations import contract


def assess(draft):
    return ledger.assess(
        brief(),
        context(),
        {},
        draft,
        graph_contract=contract("target_feedback"),
        clarify_overlaps=True,
    )


def test_upstream_loop_cannot_replace_local_feedback():
    d = apply(
        variables=[variable("y"), variable("up")],
        equations=[equation("y", "up", "u"), equation("up", "up")],
    )
    legacy = ledger.assess(brief(), context(), {}, d, graph_contract=contract())
    assert legacy["eligible"]
    assert not assess(d)["eligible"]
    assert assess(d)["reviewed_public_graph_checks"][0]["passed"] is False
    d = apply(d, equations=[equation("y", "up", "u", "y")])
    assert assess(d)["eligible"]


@pytest.mark.parametrize("mode", ["direct", "coupled", "algebraic_readout"])
def test_target_feedback_allows_explicit_equivalent_coordinates(mode):
    d = apply(
        variables=[
            variable("y", "algebraic"),
            variable("volume"),
            variable("level", "algebraic"),
        ],
        equations=[
            equation("y", "level"),
            equation("level", "volume"),
            equation("volume", "volume", "u"),
        ],
    )
    assert not assess(d)["eligible"]  # No runtime guessing of scientific assignment.
    d = apply(d, feedback_bindings=[{"target": "y", "states": ["volume"]}])
    if mode != "algebraic_readout":
        d = apply(
            d,
            variables=[variable("y")],
            remove_feedback_bindings=["y"],
            equations=[
                equation("y", "u", "y" if mode == "direct" else "volume"),
                equation("volume", "y"),
            ],
        )
    assert assess(d)["eligible"], assess(d)["errors"]


@pytest.mark.parametrize("state", ["up", "readout", "unknown"])
def test_readout_binding_requires_differential_coordinate_without_intervening_state(
    state,
):
    d = apply(
        variables=[
            variable("y", "algebraic"),
            variable("down"),
            variable("up"),
            variable("readout", "algebraic"),
        ],
        equations=[
            equation("y", "down"),
            equation("down", "up", "down"),
            equation("up", "up"),
            equation("readout", "up"),
        ],
        feedback_bindings=[{"target": "y", "states": [state]}],
    )
    assert not assess(d)["eligible"]
    d = apply(d, feedback_bindings=[{"target": "y", "states": ["down"]}])
    assert assess(d)["eligible"]


def test_binding_cannot_bypass_differential_target_feedback():
    d = apply(
        variables=[variable("y"), variable("up")],
        equations=[equation("y", "up"), equation("up", "up")],
        feedback_bindings=[{"target": "y", "states": ["up"]}],
    )
    assert not assess(d)["eligible"]
    assert "differential_target_is_own_coordinate" in str(assess(d)["errors"])


def test_check_id_cleanup_does_not_guess_requirement_or_state():
    d = apply(
        variables=[variable("y")],
        equations=[equation("y", "y")],
        mechanism_bindings=[{"requirement_id": "public_rule", "memory_states": ["y"]}],
    )
    raw = patch(
        mechanism_bindings=[{"requirement_id": "memory", "memory_states": ["y"]}],
        stage_complete=True,
    ).model_dump(mode="json")
    b = optional_memory_brief()
    normalized, log = ledger.normalize_reply(b, raw, graph_contract=contract(), draft=d)
    assert raw["remove_bindings"] == []
    assert normalized["remove_bindings"] == ["public_rule"]
    updated = ledger.apply_patch(b, d, ledger.DraftPatch.model_validate(normalized))
    assert [v.requirement_id for v in updated.mechanism_bindings] == ["memory"]
    assert updated.mechanism_bindings[0].memory_states == ("y",)
    assert log[0]["replacement_assignment"] is None
    raw["mechanism_bindings"] = [{"requirement_id": "unknown", "memory_states": ["y"]}]
    assert ledger.normalize_reply(b, raw, graph_contract=contract())[0] == raw
    raw["mechanism_bindings"][0]["requirement_id"] = "public_rule"
    assert (
        ledger.normalize_reply(b, raw, graph_contract=contract())[0][
            "mechanism_bindings"
        ]
        == []
    )
    # Exact collision with a genuine public requirement must preserve its semantics.
    b = b.model_copy(
        update={
            "requirements": (
                b.requirements[0].model_copy(update={"id": "public_rule"}),
            )
        }
    )
    assert ledger.normalize_reply(b, raw, graph_contract=contract()) == (raw, [])


def overlapping_draft():
    return apply(
        variables=[variable("y")],
        processes=[
            process(
                depends_on=["u"],
                kind="influence",
                uses=[{"target": "y", "sign": "positive", "conversion": None}],
            )
        ],
        equations=[equation("y", "u")],
    )


def test_preview_shows_automatic_rhs_before_ordinary_declaration():
    d = overlapping_draft().model_copy(update={"equations": ()})
    view = ledger.snapshot(brief(), d)["equation_views"][0]
    assert not view["ordinary_rhs_declared"]
    assert len(view["assembled_terms_including_process_uses"]) == 1
    assert "terms=[]" in view["next_action"]
    assert ledger.pending(brief(), d)["equations_to_define"] == ["y"]


def test_overlap_is_a_question_not_an_automatic_scientific_rejection():
    d = overlapping_draft()
    check = ledger.assess(brief(), context(), {}, d, clarify_overlaps=True)
    assert not check["eligible"] and not check["errors"]
    overlap = check["clarification_requests"][0]
    d = apply(
        d,
        overlap_confirmations=[
            {
                "overlap_id": overlap["overlap_id"],
                "scientific_distinction": "Separate effects with a common driver.",
            }
        ],
    )
    assert ledger.assess(brief(), context(), {}, d, clarify_overlaps=True)["eligible"]
    # Changed signs/laws invalidate the old confirmation; no permanent bypass.
    d = apply(
        d,
        equations=[
            {
                **equation("y", "u"),
                "terms": [
                    {**equation("y", "u")["terms"][0], "outer_weight_sign": "negative"}
                ],
            }
        ],
    )
    assert not ledger.assess(brief(), context(), {}, d, clarify_overlaps=True)[
        "eligible"
    ]
    d = apply(d, equations=[{"name": "y", "terms": []}])
    assert ledger.assess(brief(), context(), {}, d, clarify_overlaps=True)["eligible"]
    assert len(ledger.assembled_equations(brief(), d)[-1].terms) == 1


def test_explicit_process_reference_is_not_an_expanded_duplicate():
    d = overlapping_draft()
    d = apply(
        d,
        equations=[
            {
                **equation("y", "p"),
                "terms": [
                    {**equation("y", "p")["terms"][0], "outer_weight_sign": "positive"}
                ],
            }
        ],
    )
    assert ledger.contribution_overlaps(d) == []
    assert len(ledger.assembled_equations(brief(), d)[-1].terms) == 1


def test_transfer_feedback_offers_reclassification_without_runtime_sign_edits():
    p = process(
        uses=[{"target": n, "sign": "positive", "conversion": None} for n in ("x", "y")]
    )
    d = apply(
        variables=[variable("x"), variable("y")],
        processes=[p],
        equations=[{"name": "x", "terms": []}, {"name": "y", "terms": []}],
    )
    error = next(
        e
        for e in ledger.assess(brief(), context(), {}, d)["errors"]
        if e["code"] == "process_uses"
    )
    assert any("influence" in option for option in error["repair_options"])
    assert any("opposite signs" in option for option in error["repair_options"])
    d = apply(d, processes=[{**p, "kind": "influence"}])
    assert ledger.assess(brief(), context(), {}, d)["eligible"]
    assert {u.sign for u in d.processes[0].uses} == {"positive"}


def test_bounded_clarification_uses_proposer_choice_and_cache(tmp_path):
    calls = []

    def send(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"])
        calls.append(payload)
        if payload["stage"] == "relationships":
            d = overlapping_draft()
            reply = patch(
                variables=d.variables, processes=d.processes, stage_complete=True
            )
        elif payload["stage"] == "equations":
            assert payload["current_draft"]["equation_views"][0][
                "assembled_terms_including_process_uses"
            ]
            reply = patch(equations=[equation("y", "u")], stage_complete=True)
        else:
            assert payload["runtime_diagnostics"]["structural_failures"] == []
            question = payload["runtime_diagnostics"]["clarification_requests"][0]
            reply = patch(
                overlap_confirmations=[
                    {
                        "overlap_id": question["overlap_id"],
                        "scientific_distinction": "Intentionally separate effects.",
                    }
                ],
                stage_complete=True,
            )
        return response(reply.model_dump(mode="json"))

    client = ConstructionClient(
        settings=StagedModelSettings(maximum_requests=128, maximum_total_tokens=524288),
        directory=tmp_path / "calls",
        namespace="clarification",
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
    assert len(calls) == 3
    assert run() == result and len(calls) == 3
    assert sealed_read(tmp_path / "construction/events/002.json")["accepted"]


def test_accepted_repair_is_not_presented_as_rejected_reply(tmp_path):
    calls = []

    def send(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"])
        calls.append(payload)
        if payload["stage"] == "relationships":
            edits = patch(variables=[variable("y")], stage_complete=True)
        elif payload["stage"] == "equations":
            edits = patch(equations=[equation("y", "u")], stage_complete=True)
        elif len(calls) == 3:
            # Accepted edit that deliberately leaves the public graph unresolved.
            edits = patch(
                variables=[{**variable("y"), "scientific_role": "Storage"}],
                stage_complete=True,
            )
        else:
            feedback = payload["runtime_diagnostics"]["last_edit_result"]
            assert feedback["status"] == "accepted"
            assert feedback["error"] is None
            assert feedback["rejected_reply"] is None
            edits = patch(equations=[equation("y", "u", "y")], stage_complete=True)
        return response(edits.model_dump(mode="json"))

    client = ConstructionClient(
        settings=StagedModelSettings(maximum_requests=128, maximum_total_tokens=524288),
        directory=tmp_path / "calls",
        namespace="accepted-edit",
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
        "joint_adaptive",
        graph_contract=contract("target_feedback"),
    )
    assert result["status"] == "topology_complete"
    assert len(calls) == 4

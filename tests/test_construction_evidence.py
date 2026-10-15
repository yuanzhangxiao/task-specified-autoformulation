"""Evidence feedback distinguishes missing records from missing scientific paths."""

import json

import pytest
from pydantic import ValidationError

from autoformalism.research import (
    construction_boundary_contracts,
    construction_obligations,
)
from autoformalism.search import construction_evidence as core
from autoformalism.search import mechanism_realizations as old
from autoformalism.staged_topology import content_hash
from tests.test_construction_ledger import context, equation, process, variable
from tests.test_construction_realizations import public, record


def patch(**changes):
    """Validate the actual new reply contract."""
    return core.DraftPatch(stage_complete=False, **changes)


def apply(draft=None, **changes):
    """Commit using the same normalization order as the construction scheduler."""
    before = draft or core.Draft()
    edit = patch(**changes)
    updated = core.apply_patch(public(), before, edit)
    return core.normalize_consumers(updated, edit, before)[0]


def draft(*components):
    """An input effect and a state-dependent contribution in a single balance."""
    return apply(
        variables=[variable("y")],
        equations=[
            {
                "name": "y",
                "terms": [
                    equation("y", "u")["terms"][0],
                    equation("y", "y")["terms"][0],
                ],
            }
        ],
        mechanism_realizations=[record(*(components or ("y",)))],
    )


def boundary(b=None):
    """A test-only reviewed role supported by exact public text."""
    b = b or public()
    return core.BoundaryContract(
        source_brief_sha256=content_hash(b.model_dump(mode="json")),
        obligations=[
            {
                "id": "loss",
                "target": "y",
                "public_quote": "Generate y from u",
                "interpretation": "For this fixture, identify the external loss.",
            }
        ],
    )


def choice(**updates):
    """Choose the existing second local contribution, without guessing its law."""
    return {
        "obligation_id": "loss",
        "receiver": "y",
        "local_term_index": 1,
        "explanation": "This term represents external removal.",
        **updates,
    }


def assess(d, contract=None):
    return core.assess(public(), context(), {}, d, boundary_contract=contract)


def test_source_only_record_is_provisional_then_actionable_without_false_path_claim():
    d = draft("u")
    assert core.variable_checklist(public(), d, {})["status"] == "ready"
    historical = old.assess(public(), context(), {}, core.base(d))
    assert historical["eligible"]
    result = assess(d)
    assert not result["eligible"]
    issue = next(
        e for e in result["errors"] if e["code"] == "mechanism_realization_source_only"
    )
    assert issue["public_path_status"] == "passed"
    assert issue["path_checks"][0]["passed"]
    assert (
        "incomplete implementation record, not a missing-path claim" in issue["action"]
    )
    assert "Do not invent another state" in issue["action"]
    repaired = apply(d, mechanism_realizations=[record("u", "y")])
    assert assess(repaired)["eligible"]
    assert repaired.equations == d.equations


def test_source_only_failed_path_is_not_reported_as_existing():
    d = apply(draft("u"), equations=[equation("y", "y")])
    issue = next(
        e
        for e in assess(d)["errors"]
        if e["code"] == "mechanism_realization_source_only"
    )
    assert issue["public_path_status"] == "failed"
    assert "path exists" not in issue["action"]


def test_external_loss_is_not_inferred_from_target_feedback():
    result = assess(draft(), boundary())
    assert not result["eligible"]
    row = result["boundary_checks"][0]
    assert row["issues"] == ["missing_external_exchange_assignment"]
    assert row["behavior_status"] == "unverified"


@pytest.mark.parametrize("sign", ["positive", "negative", "unrestricted"])
def test_local_external_assignment_is_coverage_not_a_sign_claim(sign):
    d = draft()
    entries = d.equations[0].model_dump(mode="json")
    entries["terms"][1]["outer_weight_sign"] = sign
    d = apply(d, equations=[entries], external_exchanges=[choice()])
    result = assess(d, boundary())
    assert result["eligible"]
    assert result["boundary_checks"][0]["status"] == "reference_covered"
    assert result["boundary_checks"][0]["behavior_status"] == "unverified"
    assert d.external_exchanges[0].reference_sha256
    assert not d.processes


def test_internal_transfer_cannot_serve_as_distinct_external_loss():
    d = apply(
        variables=[variable("y"), variable("x")],
        equations=[equation("x", "u"), {"name": "y", "terms": []}],
        processes=[process(depends_on=["x", "y"])],
        mechanism_realizations=[record("y")],
        external_exchanges=[choice(local_term_index=None, shared_process="p")],
    )
    row = assess(d, boundary())["boundary_checks"][0]
    assert "internal_transfer_is_not_distinct_external_loss" in row["issues"]
    assert "Do not relabel a genuine internal transfer" in row["action"]


def test_equivalent_storage_coordinate_uses_algebraic_target_readout():
    d = apply(
        variables=[variable("y", "algebraic"), variable("storage")],
        equations=[equation("y", "storage"), equation("storage", "storage", "u")],
        mechanism_realizations=[record("storage", "y")],
        external_exchanges=[choice(receiver="storage", local_term_index=0)],
    )
    row = assess(d, boundary())["boundary_checks"][0]
    assert row["status"] == "reference_covered"
    # An unrelated state merely driving another dynamic balance is not a readout.
    d = apply(d, variables=[variable("y")])
    row = assess(d, boundary())["boundary_checks"][0]
    assert "balance_coordinate_has_no_algebraic_target_readout" in row["issues"]


@pytest.mark.parametrize("operation", ["change", "reorder", "remove"])
def test_local_reference_does_not_silently_move_with_later_edits(operation):
    d = apply(draft(), external_exchanges=[choice()])
    equation = d.equations[0].model_dump(mode="json")
    if operation == "change":
        equation["terms"][1]["sources"] = ["a"]
    elif operation == "reorder":
        equation["terms"].reverse()
    else:
        equation["terms"].pop()
    updated = apply(d, equations=[equation])
    row = assess(updated, boundary())["boundary_checks"][0]
    assert row["status"] == "incomplete"
    expected = (
        "contribution_reference_unresolved"
        if operation == "remove"
        else ("contribution_reference_changed")
    )
    assert expected in row["issues"]
    if operation != "remove":
        refreshed = apply(updated, external_exchanges=[choice()])
        assert assess(refreshed, boundary())["boundary_checks"][0]["status"] == (
            "reference_covered"
        )


def test_irrelevant_edits_do_not_invalidate_external_reference():
    d = apply(draft(), external_exchanges=[choice()])
    updated = apply(d, variables=[variable("z")], equations=[equation("z", "u")])
    assert d.external_exchanges == updated.external_exchanges
    assert assess(updated, boundary())["boundary_checks"][0]["status"] == (
        "reference_covered"
    )


def test_unknown_or_supplied_boundary_receiver_remains_actionable():
    d = apply(draft(), external_exchanges=[choice(receiver="u")])
    issues = assess(d, boundary())["boundary_checks"][0]["issues"]
    assert "receiver_is_not_generated_balance_state" in issues
    assert "contribution_reference_unresolved" in issues


def test_external_reference_schema_and_atomic_edit_ownership():
    with pytest.raises(ValidationError, match="exactly one"):
        patch(external_exchanges=[choice(shared_process="p")])
    with pytest.raises(ValidationError):
        patch(external_exchanges=[choice(reference_sha256="0" * 64)])
    d = apply(draft(), external_exchanges=[choice()])
    with pytest.raises(ValueError, match="ambiguous"):
        apply(d, external_exchanges=[choice()], remove_external_exchanges=["loss"])
    with pytest.raises(ValueError, match="obligation_id"):
        apply(d, remove_external_exchanges=["y"])
    removed = apply(d, remove_external_exchanges=["loss"])
    assert not removed.external_exchanges
    assert not assess(removed, boundary())["eligible"]


def test_compatibility_controls_stay_out_of_model_and_reject_empty_explanation():
    answer = {
        "question_id": "q1",
        "disposition": "retain_approximation",
        "explanation": "The discrepancy is accepted pending fitting.",
    }
    p = patch(compatibility_decisions=[answer])
    d = draft()
    assert core.apply_patch(public(), d, p) == d
    assert "compatibility_decisions" not in d.model_dump()
    with pytest.raises(ValidationError):
        patch(compatibility_decisions=[{**answer, "explanation": "  "}])
    with pytest.raises(ValueError, match="reply-control"):
        apply(d, variables=[variable("compatibility_decisions")])


def test_historical_artifact_loading_is_not_an_implicit_upgrade():
    d = draft()
    assert isinstance(core.load_draft(d.model_dump(mode="json")), core.Draft)
    previous = core.base(d)
    assert type(core.load_draft(previous.model_dump(mode="json"))) is old.Draft
    assert "external_exchanges" not in old.response_model("repair").model_fields


def test_public_boundary_contract_is_quote_and_digest_bound():
    boundary().validate_public(public())
    with pytest.raises(ValueError, match="hash"):
        boundary().validate_public(
            public().model_copy(update={"scientific_context": "changed"})
        )
    bad = boundary().model_copy(
        update={
            "obligations": (
                boundary()
                .obligations[0]
                .model_copy(update={"public_quote": "hidden equation"}),
            )
        }
    )
    with pytest.raises(ValueError, match="quote"):
        bad.validate_public(public())
    empty = construction_boundary_contracts.reviewed_contract(
        construction_obligations.DALLA[0], public()
    )
    assert not empty.obligations
    with pytest.raises(ValueError, match="profile"):
        construction_boundary_contracts.reviewed_contract("unknown", public())


def test_payload_exposes_actionable_coverage_and_compact_indexed_meanings():
    d = draft("u")
    packet = core.payload(
        "repair",
        public(),
        context(),
        {},
        public().model_dump(mode="json"),
        d,
        None,
        boundary_contract=boundary(),
        training_highlights="TRAIN: y varies.",
        compatibility_questions=[{"question_id": "q1", "question": "Explain?"}],
    )
    assert packet["training_highlights"] == "TRAIN: y varies."
    assert packet["compatibility_questions"][0]["question_id"] == "q1"
    codes = {row["code"] for row in packet["feedback"]["issues"]}
    assert "mechanism_realization_source_only" in codes
    assert "external_exchange_coverage" in codes
    assert packet["model_view"]["balances"][0]["local_process_meanings_by_index"] == {
        "0": "joint effect",
        "1": "joint effect",
    }
    row = next(
        row
        for row in packet["requirements_checklist"]
        if row["id"] == "external_exchange:loss"
    )
    assert row["status"] == "incomplete"
    assert "reference_sha256" not in json.dumps(packet["response_template"])
    assert "compatibility_decisions" in packet["response_template"]


@pytest.mark.parametrize("stage", ["variables", "shared_laws", "equations", "repair"])
def test_new_schema_fields_are_stage_specific(stage):
    fields = core.response_model(stage).model_fields
    assert ("compatibility_decisions" in fields) == (stage == "repair")
    assert ("external_exchanges" in fields) == (stage != "variables")
    assert "reference_sha256" not in json.dumps(
        core.response_model(stage).model_json_schema()
    )


def test_interaction_handoff_retains_exact_roles_and_unknown_conversions():
    d = apply(
        draft(),
        variables=[variable("x")],
        equations=[equation("x", "u")],
        processes=[
            process(
                uses=[
                    {"target": "x", "sign": "negative", "conversion": None},
                    {"target": "y", "sign": "positive", "conversion": "1/area"},
                ]
            )
        ],
        external_exchanges=[choice()],
    )
    before = d.model_dump(mode="json")
    result = core.interaction_handoff(public(), d)
    assert d.model_dump(mode="json") == before
    assert result["status"] == "pending_interaction_review"
    assert result["topology_blocking"] is False
    transfer = result["internal_transfer_candidates"][0]
    assert transfer["process"] == d.processes[0].model_dump(mode="json")
    assert transfer["process"]["uses"][0]["conversion"] is None
    assert transfer["process"]["uses"][1]["conversion"] == "1/area"
    assert "every gain and conversion" in transfer["question_for_interaction"]
    external = result["external_role_candidates"][0]
    assert external["assignment"] == d.external_exchanges[0].model_dump(mode="json")
    assert external["reference_matches_assignment"] is True
    assert external["current_reference"]["local_term"] == (
        d.equations[0].terms[1].model_dump(mode="json")
    )
    assert result["function_checks_executed"] == 0
    assert result["scientific_clarification_calls"] == 0
    assert "same_dependency_candidates" in result
    assert "signed_quantity_candidates" in result

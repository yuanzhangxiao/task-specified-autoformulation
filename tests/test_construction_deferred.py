"""Defer ambiguous functions without weakening declared structural requirements."""

import json

import pytest

from autoformalism.llm.construction import ConstructionClient
from autoformalism.llm.staged_topology import DeferredCall, StagedModelSettings
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


def overlapping():
    return apply(
        variables=[variable("y")],
        processes=[
            process(
                kind="influence",
                depends_on=["u"],
                uses=[{"target": "y", "sign": "positive"}],
            )
        ],
        equations=[equation("y", "u")],
    )


def test_overlap_deferred_without_merging_or_changing_legacy_acceptance():
    draft = overlapping()
    before = draft.model_dump()
    old = ledger.assess(brief(), context(), {}, draft, clarify_overlaps=True)
    new = deferred.assessment(brief(), context(), {}, draft, None)
    assert not old["eligible"] and old["clarification_requests"]
    assert new["eligible"] and not new["clarification_requests"]
    assert draft.model_dump() == before
    assert new["contribution_overlaps"][0]["status"] == "deferred_to_interaction"
    assert "question" not in new["contribution_overlaps"][0]
    assert len(new["equations"][-1]["terms"]) == 2


def test_missing_memory_path_still_blocks_and_status_tracks_each_stage():
    b = brief(True)
    draft = apply(
        public=b,
        variables=[variable("y"), variable("x")],
        equations=[equation("x", "u"), equation("y", "u")],
        mechanism_bindings=[{"requirement_id": "memory", "memory_states": ["x"]}],
    )
    check = deferred.assessment(b, context(), {}, draft, None)
    assert not check["eligible"]
    assert any(e["code"] == "memory_target_path" for e in check["errors"])
    for stage, expected in [("variables", "deferred"), ("repair", "failed")]:
        rows = deferred.requirement_status(b, draft, {}, None, stage, check)["items"]
        assert (
            next(r for r in rows if r["id"] == "mechanism_topology:memory")["status"]
            == expected
        )
        assert (
            next(r for r in rows if r["id"] == "remaining_public_prose")["status"]
            == "unassessed"
        )
    repaired = apply(draft, public=b, equations=[equation("y", "x")])
    check = deferred.assessment(b, context(), {}, repaired, None)
    rows = deferred.requirement_status(b, repaired, {}, None, "repair", check)["items"]
    assert (
        next(r for r in rows if r["id"] == "mechanism_topology:memory")["status"]
        == "passed"
    )


def test_uncompiled_topology_is_unavailable_not_a_missing_path_verdict():
    draft = apply(public=brief(True), variables=[variable("y")])
    check = deferred.assessment(brief(True), context(), {}, draft, None)
    rows = deferred.requirement_status(brief(True), draft, {}, None, "repair", check)[
        "items"
    ]
    assert (
        next(r for r in rows if r["id"] == "mechanism_topology:memory")["status"]
        == "unavailable"
    )


def test_sign_interpretation_handoff_is_generic_nonblocking_and_does_not_flip():
    negative = lambda name, source: {  # noqa: E731
        "name": name,
        "terms": [
            {
                "sources": [source],
                "outer_weight_sign": "negative",
                "scientific_role": "signed effect",
            }
        ],
    }
    draft = apply(
        variables=[variable("y"), variable("q", "algebraic")],
        equations=[negative("q", "y"), negative("y", "q")],
    )
    check = deferred.assessment(brief(), context(), {}, draft, None)
    assert check["eligible"]
    handoff = check["interaction_review_handoff"]
    assert not handoff["topology_blocking"]
    assert handoff["signed_quantity_candidates"][0]["quantity"] == "q"
    assert (
        "positive outward rate"
        in handoff["signed_quantity_candidates"][0]["question_for_interaction"]
    )
    assert all(
        t.outer_weight_sign.value == "negative"
        for e in draft.equations
        for t in e.terms
    )


def test_duplicate_receiver_is_not_confused_with_same_dependency_laws():
    draft = apply(
        variables=[variable("y")],
        processes=[
            process(
                depends_on=["u"],
                uses=[
                    {"target": "y", "sign": "positive"},
                    {"target": "y", "sign": "negative"},
                ],
            )
        ],
        equations=[{"name": "y", "terms": []}],
    )
    check = deferred.assessment(brief(), context(), {}, draft, None)
    assert not check["eligible"]
    assert any(e["code"] == "process_uses" for e in check["errors"])


def test_new_policy_completes_without_overlap_repair_and_resumes_cached_calls(tmp_path):
    calls = []

    def send(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"])
        calls.append(packet)
        assert "requirement_status" in packet
        assert "potential_contribution_overlaps" not in packet["current_draft"]
        stage = packet["stage"]
        edit = {**packet["response_template"], "stage_complete": True}
        if stage == "variables":
            edit["variables"] = [variable("y")]
        elif stage == "shared_laws":
            edit["processes"] = overlapping().model_dump(mode="json")["processes"]
        elif stage == "equations":
            edit["equations"] = [equation("y", "u")]
        else:
            pytest.fail("Same-dependency ambiguity must not cause global repair")
        return response(edit)

    def run(can_start=lambda: True):
        client = ConstructionClient(
            settings=StagedModelSettings(),
            directory=tmp_path / "calls",
            namespace="deferred",
            seed=0,
            base_url="http://offline",
            transport=send,
            token_transport=tokenize,
            can_start=can_start,
        )
        return schedules.run(
            brief(),
            context(),
            {},
            brief().model_dump(mode="json"),
            client,
            tmp_path / "construction",
            "joint_adaptive",
            prompt_family="minimal",
            process_question="dedicated",
            bookkeeping_policy=deferred.POLICY,
        )

    with pytest.raises(DeferredCall):
        run(lambda: len(calls) < 1)
    result = run()
    assert result["status"] == "topology_complete"
    assert len(calls) == 3
    assert result["assessment"]["interaction_review_handoff"][
        "same_dependency_candidates"
    ]
    assert run() == result
    assert len(calls) == 3


@pytest.mark.parametrize("stage", ["variables", "shared_laws", "equations", "repair"])
def test_request_adapter_keeps_stage_wording_and_shows_requirement_status(stage):
    draft = overlapping()
    old = {
        "stage": stage,
        "stage_instructions": "Original stage instructions.",
        "current_draft": ledger.snapshot(brief(), draft),
    }
    new = deferred.update_payload(old, brief(), context(), {}, draft, None)
    assert new["stage_instructions"] == old["stage_instructions"]
    assert new["requirement_status"]["items"]


def test_request_preview_removes_old_overlap_questions_but_preserves_other_failures():
    draft = overlapping()
    old = {
        "stage": "repair",
        "stage_instructions": "Repair the topology.",
        "current_draft": ledger.snapshot(brief(), draft),
        "runtime_diagnostics": {
            "clarification_requests": [{"question": "old overlap question"}],
            "structural_failures": [{"code": "memory_target_path"}],
        },
    }
    new = deferred.update_payload(old, brief(), context(), {}, draft, None)
    assert "old overlap question" not in json.dumps(new)
    assert new["runtime_diagnostics"]["structural_failures"] == [
        {"code": "memory_target_path"}
    ]
    assert old["runtime_diagnostics"]["clarification_requests"]
    assert new["current_draft"]["declarations"] == old["current_draft"]["declarations"]


def test_composition_requirement_is_displayed_and_failure_is_not_deferred():
    b = brief().model_copy(update={"target_dependencies": ()})
    from autoformalism.schemas.staged_topology import PublicScientificBrief

    b = PublicScientificBrief.model_validate(
        {
            **b.model_dump(mode="json"),
            "target_dependencies": [
                {
                    "target": "y",
                    "acceptable_sources": ["a"],
                    "public_requirement": "y includes supplied a",
                }
            ],
        }
    )
    d = apply(variables=[variable("y")], equations=[equation("y", "u")])
    check = deferred.assessment(b, context(), {}, d, None)
    assert not check["eligible"]
    rows = deferred.requirement_status(b, d, {}, None, "repair", check)["items"]
    assert (
        next(r for r in rows if r["id"] == "target_composition:0")["status"] == "failed"
    )


def test_review_diff_escapes_untrusted_prompt_text_and_path_identity_ignores_witness():
    from autoformalism.research.construction_deferred_review import (
        marked_diff,
        passed_paths,
    )

    rendered = marked_diff("old", "<script>alert(1)</script>")
    assert "<script>" not in rendered and "&lt;script&gt;" in rendered
    assert "<ins>" in rendered and "<del>" in rendered
    check = {
        "public_structure_checks": [],
        "memory_binding_checks": [],
        "reviewed_public_graph_checks": [
            {"id": "feedback", "passed": True, "target": "y", "witness": ["a", "a"]}
        ],
    }
    first = passed_paths(check)
    check["reviewed_public_graph_checks"][0]["witness"] = ["b", "b"]
    assert first.keys() == passed_paths(check).keys()
    check["reviewed_public_graph_checks"][0]["passed"] = None
    assert not passed_paths(check)

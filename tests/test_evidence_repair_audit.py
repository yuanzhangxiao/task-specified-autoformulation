"""Offline feedback previews keep scientific evidence and historical state separate."""

from copy import deepcopy

import pytest

from autoformalism.search import construction_evidence as historical
from autoformalism.search import construction_evidence_repair as revised
from scripts import audit_evidence_repair_feedback as audit
from tests.test_construction_evidence import apply, boundary, choice, draft
from tests.test_construction_ledger import context
from tests.test_construction_realizations import public


def question(identity="old", target="y"):
    return {
        "question_id": identity,
        "target": target,
        "property": "closed_subsystem",
        "witnesses": [{"trajectories": ["one", "two"], "observations": [1, 3]}],
    }


def test_saved_answers_translate_only_on_their_actual_scientific_question():
    old, new = question(), question("semantic")
    decision = {
        "question_id": "old",
        "disposition": "retain_approximation",
        "explanation": "An acknowledged approximation, not verified accuracy.",
    }
    translated = audit.translate_decisions([old], [new], [decision])
    assert translated == [{**decision, "question_id": "semantic"}]
    assert decision["question_id"] == "old"
    with pytest.raises(ValueError, match="was not displayed"):
        audit.translate_decisions([], [new], [decision])
    with pytest.raises(ValueError, match="changed scientific question"):
        audit.translate_decisions([old], [question("semantic", "x")], [decision])
    with pytest.raises(ValueError, match="changed scientific question"):
        audit.translate_decisions([old], [{**new, "assumptions": ["new"]}], [decision])


def test_unchanged_questions_keep_answers_but_not_scientific_approval():
    q = question("semantic")
    report = {"advisory_questions": [q], "acceptance_gate": False}
    answer = {"question_id": q["question_id"], "disposition": "retain_approximation"}
    state = audit.advisory_state(report, {q["question_id"]: answer})
    assert not state["pending_questions"]
    assert state["review_status"] == "addressed"
    assert state["scientific_correctness"] == "unverified"
    # A stale identity or a promised revision cannot resolve a current conflict.
    assert audit.advisory_state(report, {"unrelated": answer})["pending_questions"]
    assert audit.advisory_state(
        report, {q["question_id"]: {**answer, "disposition": "revise"}}
    )["pending_questions"]


def test_structurally_accepted_conflict_is_classified_as_scientific_question():
    state = {
        "eligible": True,
        "training_compatibility": {"pending_questions": [question()]},
    }
    assert audit.entry_reason(state) == "conditional_scientific_question"
    state["eligible"] = False
    assert audit.entry_reason(state) == "structural_obligation"
    assert audit.entry_reason({"eligible": True}) == "completion_control"


@pytest.mark.parametrize("valid", [False, True])
def test_same_saved_draft_keeps_its_structural_assessment(valid):
    original = draft()
    if valid:
        original = apply(original, external_exchanges=[choice()])
    raw = original.model_dump(mode="json")
    args = (public(), context(), {}, None, boundary(), original)
    old = audit.full_assessment(historical, *args)
    new = audit.full_assessment(revised, *args)
    audit.same_assessment(old, new)
    assert old["eligible"] is valid
    assert original.model_dump(mode="json") == raw
    tampered = deepcopy(new)
    tampered["eligible"] = not old["eligible"]
    with pytest.raises(ValueError, match="structural assessment changed"):
        audit.same_assessment(old, tampered)


def test_html_escapes_untrusted_scientific_text(tmp_path):
    preview = {
        "task": "<script>alert('x')</script>",
        "event": 1,
        "before_structural": True,
        "after_structural": False,
        "revised_advice": {"pending_questions": [question()]},
        "revised_before": {"scientific_role": "<script>bad()</script>"},
        "historical_before": {},
        "historical_after": {},
        "required_predicates_lost": [],
    }
    audit.render({"counts": {"repair_requests": 1}}, [preview], tmp_path)
    rendered = (tmp_path / "request-000.html").read_text()
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "No new construction chain" in rendered

"""Opt-in retained-witness IDs ignore role prose while preserving mathematical edits."""

from copy import deepcopy

import pytest

from autoformalism.research import construction_training as training
from autoformalism.research import initialization_handoff_audit as audit
from autoformalism.research.repair_stagnation import model_signature
from autoformalism.search.construction_ledger import Draft
from autoformalism.search.topology_initialization import InitializerDependencyHandoff
from autoformalism.staged_topology import content_hash
from tests.test_construction_semantic_advice import ids
from tests.test_topology_initialization import prefix_fixture


def retained_fixture():
    """Retain an early discrepancy despite a later difference in forcing."""
    draft, evidence, question = prefix_fixture()
    raw = draft.model_dump(mode="json")
    raw["equations"] = []
    raw["processes"] = [
        {
            "name": "P",
            "depends_on": ["u", "v"],
            "kind": "influence",
            "uses": [{"target": "a", "sign": "positive", "conversion": "1"}],
            "scientific_meaning": "External infusion contributes directly to the state",
        }
    ]
    return Draft.model_validate(raw), evidence, question


def qualified(draft, evidence, question, **kwargs):
    """Exercise the same family-qualified retained-witness path as live repair."""
    return audit.qualified_feedback(
        draft,
        evidence,
        audit.information_contract(evidence, "causal_runtime"),
        InitializerDependencyHandoff(),
        family=True,
        previous_questions=[question],
        **kwargs,
    )


def mathematical(draft, evidence, question):
    return qualified(
        draft,
        evidence,
        question,
        identity_policy=training.MATHEMATICAL_QUESTION_IDENTITY,
    )


def test_retained_witness_default_preserves_historical_full_draft_identity():
    draft, evidence, question = retained_fixture()
    old = qualified(draft, evidence, question)
    assert old == qualified(
        draft,
        evidence,
        question,
        identity_policy=training.SEMANTIC_QUESTION_IDENTITY,
    )
    assert len(old["advisory_questions"]) == 1
    recheck = old["saved_witness_recheck"]
    assert ids(old) == {
        row["property"]: content_hash(
            {
                "policy": recheck["protocol"],
                "evidence_sha256": old["initialization_evidence_sha256"],
                "topology_sha256": old["topology_sha256"],
                "diagnostic": row,
            }
        )[:20]
        for row in recheck["diagnostics"]
    }
    assert "semantic_topology_sha256" not in old["advisory_questions"][0]
    assert old["full_report"] == mathematical(draft, evidence, question)["full_report"]


@pytest.mark.parametrize("field", ["scientific_meaning", "scientific_role"])
def test_seed1_period_only_edit_keeps_new_question_and_model_signature(field):
    """Seed 1 event 007 changed only a process meaning's trailing period."""
    draft, evidence, question = retained_fixture()
    raw = draft.model_dump(mode="json")
    if field == "scientific_meaning":
        raw["processes"][0][field] += "."
    else:
        raw["variables"][0][field] += "."
    edited = Draft.model_validate(raw)
    old_before = qualified(draft, evidence, question)
    old_after = qualified(edited, evidence, question)
    assert ids(old_before) != ids(old_after)
    before = mathematical(draft, evidence, question)
    after = mathematical(edited, evidence, question)
    assert before["topology_sha256"] != after["topology_sha256"]
    assert ids(before) == ids(after)
    assert before["advisory_questions"] == after["advisory_questions"]
    assert before["saved_witness_recheck"]["checks"] == after[
        "saved_witness_recheck"
    ]["checks"]
    check = {"eligible": True, "errors": []}
    assert model_signature(draft, check, before) == model_signature(
        edited, check, after
    )
    assert model_signature(draft, check, old_before) != model_signature(
        edited, check, old_after
    )


@pytest.mark.parametrize("edit", ["dependency", "sign", "conversion", "multiplicity"])
def test_retained_witness_mathematical_edits_change_identity_and_model_signature(edit):
    draft, evidence, question = retained_fixture()
    before = mathematical(draft, evidence, question)
    raw = draft.model_dump(mode="json")
    process = raw["processes"][0]
    if edit == "dependency":
        process["depends_on"] = ["v"]
    elif edit == "sign":
        process["uses"][0]["sign"] = "negative"
    elif edit == "conversion":
        process["uses"][0]["conversion"] = "2"
    else:
        process["uses"].append(dict(process["uses"][0]))
    edited = Draft.model_validate(raw)
    after = mathematical(edited, evidence, question)
    assert ids(before).keys() == ids(after).keys() == {"closed_subsystem"}
    assert ids(before) != ids(after)
    assert before["advisory_questions"][0]["witnesses"] == after[
        "advisory_questions"
    ][0]["witnesses"]
    check = {"eligible": True, "errors": []}
    assert model_signature(draft, check, before) != model_signature(
        edited, check, after
    )


def test_retained_identity_preserves_witness_coordinates_and_observations():
    draft, evidence, question = retained_fixture()
    raw = evidence.model_dump(mode="json", exclude={"sha256"})
    raw["trajectories"][1]["targets"]["a"][2] = 2.0
    evidence = training.load_evidence({**raw, "sha256": content_hash(raw)})
    later = deepcopy(question)
    later["witnesses"][0]["sample_indices"] = [2, 2]
    later["witnesses"][0]["times"] = [2, 2]
    before = mathematical(draft, evidence, question)
    after = mathematical(draft, evidence, later)
    assert before["topology_sha256"] == after["topology_sha256"]
    assert before["evidence_sha256"] == after["evidence_sha256"]
    assert ids(before) != ids(after)
    assert before["advisory_questions"][0]["witnesses"][0]["sample_indices"] == [1, 1]
    assert after["advisory_questions"][0]["witnesses"][0]["sample_indices"] == [2, 2]
    assert after["advisory_questions"][0]["witnesses"][0]["observations"] == [1, 2]

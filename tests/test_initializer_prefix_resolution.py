"""Initializer edits resolve saved witnesses only through complete causal rechecks."""

from copy import deepcopy

import pytest

from autoformalism.research.actionable_repair_status import build_state
from autoformalism.research.initialization_handoff_audit import qualified_feedback
from tests.test_topology_initialization import choice, fixture, prefix_fixture


def initialized_pair():
    """One unchanged graph, two explicit initializer dependency choices."""
    model, evidence, contract = fixture(extra_initial=True)
    before = qualified_feedback(
        model, evidence, contract, choice("shared_parameter", ())
    )
    questions = before["advisory_questions"]
    assert any(q["property"] == "closed_subsystem" for q in questions)
    after = qualified_feedback(
        model,
        evidence,
        contract,
        choice(arguments=("b",)),
        previous_questions=questions,
    )
    return before, after


def test_initializer_change_resolves_saved_witness_without_a_graph_scope_change():
    before, after = initialized_pair()
    originals = deepcopy((before, after))
    assert before["topology_sha256"] == after["topology_sha256"]
    assert before["evidence_sha256"] == after["evidence_sha256"]
    assert (
        before["initialization_evidence_sha256"]
        != after["initialization_evidence_sha256"]
    )
    assert {c["status"] for c in after["saved_witness_recheck"]["checks"]} == {
        "conditioning_distinction"
    }
    state = build_state(after, before, True)
    assert state["completion_allowed"]
    assert state["repair_status"] == "checks_clear"
    assert (
        state["baseline_outcomes"][0]["status"] == "conditioning_distinction_verified"
    )
    assert state["scientific_correctness"] == "unverified"
    assert (before, after) == originals
    assert not build_state(after, before, False)["completion_allowed"]


@pytest.mark.parametrize(
    "change",
    [
        "protocol",
        "evidence",
        "topology",
        "full_evidence",
        "full_topology",
        "empty",
        "question",
        "target",
        "samples",
        "time",
        "trajectories",
        "preserved",
        "unavailable",
        "uncertain",
        "malformed",
        "missing_diagnostic",
        "unavailable_assumption",
    ],
)
def test_incomplete_or_stale_prefix_checks_do_not_resolve(change):
    before, after = initialized_pair()
    recheck = after["saved_witness_recheck"]
    if change == "protocol":
        recheck["protocol"] = "different-policy"
    elif change in {"evidence", "topology"}:
        recheck[change + "_sha256"] = "different"
    elif change in {"full_evidence", "full_topology"}:
        after["full_report"][change.removeprefix("full_") + "_sha256"] = "different"
    elif change == "empty":
        recheck["checks"] = []
    elif change == "question":
        recheck["checks"][0]["previous_question_id"] = "different-question"
    elif change == "target":
        recheck["checks"][0]["target"] = "different-target"
    elif change == "samples":
        recheck["checks"][0]["sample_indices"] = [0, 0]
    elif change == "time":
        recheck["checks"][0]["conditioning_end_time"] = 100
    elif change == "trajectories":
        recheck["checks"][0]["trajectories"] = ["other", "pair"]
    elif change in {"preserved", "unavailable", "uncertain"}:
        recheck["checks"][0]["status"] = (
            "uncertainty_limited" if change == "uncertain" else change
        )
    elif change == "malformed":
        recheck["checks"][0]["trajectories"] = None
    elif change == "missing_diagnostic":
        after["full_report"]["diagnostics"] = []
    elif change == "unavailable_assumption":
        diagnostic = next(
            row
            for row in after["full_report"]["diagnostics"]
            if row["property"] == "closed_subsystem"
        )
        diagnostic["unavailable"] = ["unknown initializer"]
    state = build_state(after, before, True)
    assert not state["completion_allowed"]
    assert state["repair_status"] == "review_deferred"


def test_all_prior_witnesses_require_rechecks_not_just_one_success():
    before, after = initialized_pair()
    old = before["advisory_questions"][0]
    another = deepcopy(old["witnesses"][0])
    another["sample_indices"] = [2, 2]
    another["times"] = [2, 2]
    old["witnesses"].append(another)
    assert not build_state(after, before, True)["completion_allowed"]


def test_all_prior_question_records_require_rechecks_even_for_the_same_target():
    before, after = initialized_pair()
    another = deepcopy(before["advisory_questions"][0])
    another["question_id"] = "another-saved-question"
    before["advisory_questions"].insert(0, another)
    assert not build_state(after, before, True)["completion_allowed"]


def test_missing_saved_witness_coordinates_do_not_resolve():
    before, after = initialized_pair()
    before["advisory_questions"][0]["witnesses"][0].pop("times")
    assert not build_state(after, before, True)["completion_allowed"]


def test_incomplete_prefix_recheck_cannot_fall_back_to_changed_graph_scope():
    from autoformalism.research.initialization_handoff_audit import information_contract
    from autoformalism.search.topology_initialization import (
        InitializerDependencyHandoff,
    )

    model, evidence, old = prefix_fixture(early_forcing=True)
    old["scope"] = ["a", "u"]
    baseline = {"evidence_sha256": evidence.sha256, "advisory_questions": [old]}
    feedback = qualified_feedback(
        model,
        evidence,
        information_contract(evidence, "causal_runtime"),
        InitializerDependencyHandoff(),
        family=True,
        previous_questions=[old],
    )
    assert build_state(feedback, baseline, True)["completion_allowed"]
    feedback["saved_witness_recheck"]["checks"] = []
    assert not build_state(feedback, baseline, True)["completion_allowed"]


def test_current_conflict_outweighs_a_saved_conditioning_distinction():
    before, after = initialized_pair()
    after["advisory_questions"] = deepcopy(before["advisory_questions"])
    state = build_state(after, before, True)
    assert state["repair_status"] == "repair_required"
    assert not state["completion_allowed"]


def test_absent_prefix_receipt_keeps_historical_same_scope_deferral():
    before, after = initialized_pair()
    after.pop("saved_witness_recheck")
    assert not build_state(after, before, True)["completion_allowed"]
